"""
train.py
--------
Full training loop for the from-scratch GPT model.

Optimizations applied:
  - Mixed precision training (float16) via torch.amp — ~2x memory saving
  - Gradient accumulation — larger effective batch without more VRAM
  - Adaptive max_chars — tokenizer samples as much corpus as RAM allows
  - pin_memory + prefetch DataLoader — overlaps CPU/GPU transfers
  - Memory monitoring every N steps
  - Gradient clipping, cosine LR schedule, AdamW, checkpointing

Usage:
    python train.py --corpus corpus.txt --save_dir checkpoints

    # With mixed precision + gradient accumulation:
    python train.py --corpus corpus.txt --mixed_precision --grad_accum 4
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from tokenizer import BPETokenizer
from model     import GPT, GPTConfig
from dataset   import TextDataset, build_dataloader


# ============================================================================
# Adaptive RAM helper
# ============================================================================

def get_adaptive_max_chars(fallback: int = 5_000_000) -> int:
    """
    Set max_chars for tokenizer training to 80% of available RAM.
    Falls back to 5 MB if psutil is not installed.
    """
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / 1e6
        adaptive     = int(available_mb * 0.8 * 1e6)
        result       = max(5_000_000, min(adaptive, 500_000_000))
        print(f"[RAM]  Available: {available_mb:.0f} MB  →  "
              f"tokenizer sample: {result/1e6:.0f} MB")
        return result
    except ImportError:
        return fallback


# ============================================================================
# Memory monitor
# ============================================================================

def print_memory_stats(device: torch.device) -> None:
    """Print current VRAM and RAM usage."""
    if device.type == "cuda":
        vram_used  = torch.cuda.memory_allocated(device) / 1e9
        vram_total = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  [MEM] VRAM {vram_used:.2f}/{vram_total:.1f} GB", end="")
    try:
        import psutil
        ram = psutil.Process().memory_info().rss / 1e9
        print(f"  RAM {ram:.2f} GB", end="")
    except ImportError:
        pass
    print()


# ============================================================================
# Learning rate schedule
# ============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


# ============================================================================
# Checkpoint helpers
# ============================================================================

def save_checkpoint(save_dir, step, model, optimizer, scaler, val_loss, cfg_dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"ckpt_{step:07d}.pt"
    torch.save({
        "step":            step,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state":    scaler.state_dict() if scaler else None,
        "val_loss":        val_loss,
        "cfg":             cfg_dict,
    }, ckpt_path)
    (save_dir / "latest.txt").write_text(str(ckpt_path))
    print(f"[CKPT] Saved → '{ckpt_path}'  (val_loss={val_loss:.4f})")


def load_checkpoint(save_dir, model, optimizer=None, scaler=None):
    latest_file = save_dir / "latest.txt"
    if not latest_file.exists():
        return 0, float("inf")
    ckpt_path = Path(latest_file.read_text().strip())
    if not ckpt_path.exists():
        return 0, float("inf")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    step     = ckpt["step"]
    val_loss = ckpt.get("val_loss", float("inf"))
    print(f"[CKPT] Resumed from '{ckpt_path}'  (step={step}, val_loss={val_loss:.4f})")
    return step, val_loss


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=20, use_amp=False):
    model.eval()
    total = 0.0
    n     = 0
    for x, y in val_loader:
        if n >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast(device_type=device.type, enabled=use_amp):
            _, loss = model(x, targets=y)
        total += loss.item()
        n     += 1
    model.train()
    return total / max(n, 1)


# ============================================================================
# Training loop
# ============================================================================

def train(args: argparse.Namespace) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] CUDA  –  {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[DEVICE] MPS  –  Apple Silicon")
    else:
        device = torch.device("cpu")
        print("[DEVICE] CPU")

    use_amp = args.mixed_precision and device.type == "cuda"
    if use_amp:
        print("[AMP]  Mixed precision training enabled (float16)")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tok_path = save_dir / "tokenizer.json"
    if tok_path.exists() and not args.retrain_tokenizer:
        tokenizer = BPETokenizer.load(str(tok_path))
    else:
        print("\n[STEP 1/4] Training BPE tokenizer …")
        # Use adaptive max_chars unless user specified one explicitly
        max_chars = (args.tok_max_chars
                     if args.tok_max_chars
                     else get_adaptive_max_chars())
        tokenizer = BPETokenizer()
        tokenizer.train(
            corpus_path = args.corpus,
            vocab_size  = args.vocab_size,
            verbose     = True,
            max_chars   = max_chars,
        )
        tokenizer.save(str(tok_path))

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\n[STEP 2/4] Building datasets …")
    pin = (device.type == "cuda")

    train_ds = TextDataset(
        args.corpus, tokenizer,
        context_len  = args.context_len,
        split        = "train",
        val_fraction = args.val_fraction,
        num_workers  = args.encode_workers,
    )
    val_ds = TextDataset(
        args.corpus, tokenizer,
        context_len  = args.context_len,
        split        = "val",
        val_fraction = args.val_fraction,
        num_workers  = args.encode_workers,
        verbose      = True,
    )

    train_loader = build_dataloader(
        train_ds,
        batch_size          = args.batch_size,
        shuffle             = True,
        num_workers         = args.loader_workers,
        pin_memory          = pin,
        prefetch_factor     = 2 if args.loader_workers > 0 else None,
        persistent_workers  = args.loader_workers > 0,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size          = args.batch_size,
        shuffle             = False,
        num_workers         = args.loader_workers,
        pin_memory          = pin,
        prefetch_factor     = 2 if args.loader_workers > 0 else None,
        persistent_workers  = args.loader_workers > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[STEP 3/4] Building model …")
    cfg = GPTConfig(
        vocab_size  = tokenizer.vocab_size,
        context_len = args.context_len,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        d_ff        = args.d_ff,
        dropout     = args.dropout,
    )
    model = GPT(cfg).to(device)
    print(model)

    # ── Optimiser ─────────────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # Mixed precision scaler — no-op when use_amp=False
    scaler = GradScaler(enabled=use_amp)

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    best_val   = float("inf")
    if args.resume:
        start_step, best_val = load_checkpoint(save_dir, model, optimizer, scaler)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # ── CSV log ───────────────────────────────────────────────────────────
    log_path   = save_dir / "training_log.csv"
    log_exists = log_path.exists() and args.resume
    log_file   = open(log_path, "a", newline="")
    writer     = csv.writer(log_file)
    if not log_exists:
        writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])

    cfg_dict = vars(cfg)
    (save_dir / "model_config.json").write_text(
        json.dumps(cfg_dict, indent=2, default=str)
    )

    # ================================================================== #
    # Training loop                                                       #
    # ================================================================== #
    print(f"\n[STEP 4/4] Training for {args.max_steps:,} steps …")
    if args.grad_accum > 1:
        print(f"[INFO]  Gradient accumulation: {args.grad_accum} steps  "
              f"(effective batch = {args.batch_size * args.grad_accum})")

    model.train()
    train_iter    = iter(train_loader)
    step          = start_step
    tokens_seen   = 0
    t_start       = time.perf_counter()
    accum_loss    = 0.0

    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:

        # Fetch batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ── Forward + backward (mixed precision) ──────────────────────────
        with autocast(device_type=device.type, enabled=use_amp):
            _, loss = model(x, targets=y)
            # Scale loss for gradient accumulation
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()
        accum_loss += loss.item()

        # ── Optimizer step (every grad_accum mini-steps) ──────────────────
        if (step + 1) % args.grad_accum == 0 or step == args.max_steps - 1:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        tokens_seen += x.numel()
        step        += 1

        # ── Logging ───────────────────────────────────────────────────────
        if step % args.log_every == 0:
            elapsed     = time.perf_counter() - t_start
            tok_per_sec = tokens_seen / elapsed
            train_loss  = accum_loss * args.grad_accum   # unscale for display
            accum_loss  = 0.0

            print(f"step {step:>7,}/{args.max_steps:,}  "
                  f"loss={train_loss:.4f}  "
                  f"lr={lr:.2e}  "
                  f"tok/s={tok_per_sec:,.0f}")
            writer.writerow([step, f"{train_loss:.6f}", "", f"{lr:.6e}",
                             f"{tok_per_sec:.1f}"])
            log_file.flush()

        # ── Memory monitor ────────────────────────────────────────────────
        if args.mem_every and step % args.mem_every == 0:
            print_memory_stats(device)

        # ── Validation ────────────────────────────────────────────────────
        if step % args.val_every == 0:
            val_loss = evaluate(model, val_loader, device,
                                max_batches=args.val_batches,
                                use_amp=use_amp)
            print(f"  [VAL] step={step:,}  val_loss={val_loss:.4f}  "
                  f"perplexity={math.exp(val_loss):.2f}")
            writer.writerow([step, "", f"{val_loss:.6f}", "", ""])
            log_file.flush()

            save_checkpoint(save_dir, step, model, optimizer, scaler,
                            val_loss, cfg_dict)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), save_dir / "best_model.pt")
                print(f"  [BEST] val_loss={best_val:.4f}")

    log_file.close()
    elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] {elapsed/60:.1f} min  |  best val_loss={best_val:.4f}  "
          f"(perplexity {math.exp(best_val):.2f})")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a GPT LLM from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    data = p.add_argument_group("Data")
    data.add_argument("--corpus",       required=True)
    data.add_argument("--save_dir",     default="checkpoints")
    data.add_argument("--val_fraction", type=float, default=0.05)

    # Tokenizer
    tok = p.add_argument_group("Tokenizer")
    tok.add_argument("--vocab_size",         type=int,  default=4096)
    tok.add_argument("--retrain_tokenizer",  action="store_true")
    tok.add_argument("--tok_max_chars",      type=int,  default=None,
                     help="Max chars for tokenizer training. "
                          "Default: auto-detect from available RAM.")

    # Model
    arch = p.add_argument_group("Model architecture")
    arch.add_argument("--context_len", type=int,   default=256)
    arch.add_argument("--d_model",     type=int,   default=512)
    arch.add_argument("--n_heads",     type=int,   default=8)
    arch.add_argument("--n_layers",    type=int,   default=6)
    arch.add_argument("--d_ff",        type=int,   default=2048)
    arch.add_argument("--dropout",     type=float, default=0.1)

    # Training
    tr = p.add_argument_group("Training")
    tr.add_argument("--max_steps",    type=int,   default=5000)
    tr.add_argument("--batch_size",   type=int,   default=32)
    tr.add_argument("--max_lr",       type=float, default=3e-4)
    tr.add_argument("--min_lr",       type=float, default=3e-5)
    tr.add_argument("--warmup_steps", type=int,   default=200)
    tr.add_argument("--weight_decay", type=float, default=0.1)
    tr.add_argument("--grad_clip",    type=float, default=1.0)
    tr.add_argument("--grad_accum",   type=int,   default=1,
                    help="Gradient accumulation steps. "
                         "Effective batch = batch_size × grad_accum.")
    tr.add_argument("--mixed_precision", action="store_true",
                    help="Enable float16 mixed precision (CUDA only). "
                         "~2x memory saving, faster on Tensor Core GPUs.")

    # Workers
    wk = p.add_argument_group("Workers")
    wk.add_argument("--encode_workers", type=int, default=4,
                    help="CPU threads for parallel dataset encoding.")
    wk.add_argument("--loader_workers", type=int, default=2,
                    help="DataLoader background workers.")

    # Logging
    log = p.add_argument_group("Logging")
    log.add_argument("--log_every",   type=int, default=50)
    log.add_argument("--val_every",   type=int, default=500)
    log.add_argument("--val_batches", type=int, default=20)
    log.add_argument("--mem_every",   type=int, default=None,
                     help="Print VRAM/RAM stats every N steps. None = disabled.")
    log.add_argument("--resume",      action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
