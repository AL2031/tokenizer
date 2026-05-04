"""
train.py
--------
Full training loop for the from-scratch GPT model.

Features:
  - Cosine learning-rate schedule with linear warmup
  - Gradient clipping
  - Periodic validation loss evaluation
  - Checkpoint save/resume
  - Training log written to CSV
  - Works on CUDA, MPS (Apple Silicon), or CPU

Usage:
    python train.py \\
        --corpus    corpus.txt \\
        --save_dir  checkpoints \\
        --vocab_size 4096 \\
        --context_len 256 \\
        --batch_size 32 \\
        --max_steps 5000

    # Resume from latest checkpoint:
    python train.py --corpus corpus.txt --save_dir checkpoints --resume
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

from tokenizer import BPETokenizer
from model     import GPT, GPTConfig
from dataset   import TextDataset, build_dataloader


# ============================================================================
# Learning rate schedule
# ============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float,
           min_lr: float) -> float:
    """
    Linear warmup then cosine decay to min_lr.

    Schedule:
        0 .. warmup_steps          : linear ramp from 0 -> max_lr
        warmup_steps .. max_steps  : cosine decay from max_lr -> min_lr
        > max_steps                : constant min_lr
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


# ============================================================================
# Checkpoint helpers
# ============================================================================

def save_checkpoint(
    save_dir: Path,
    step:      int,
    model:     GPT,
    optimizer: torch.optim.Optimizer,
    val_loss:  float,
    cfg_dict:  dict,
) -> None:
    """Save model weights, optimizer state, and metadata to save_dir/ckpt_{step}.pt"""
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"ckpt_{step:07d}.pt"
    torch.save(
        {
            "step":           step,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss":       val_loss,
            "cfg":            cfg_dict,
        },
        ckpt_path,
    )
    # Write a pointer to the latest checkpoint so we can resume easily
    (save_dir / "latest.txt").write_text(str(ckpt_path))
    print(f"[CKPT] Saved  → '{ckpt_path}'  (val_loss={val_loss:.4f})")


def load_checkpoint(save_dir: Path, model: GPT, optimizer=None):
    """
    Load the latest checkpoint from save_dir.
    Returns the step number, or 0 if no checkpoint exists.
    """
    latest_file = save_dir / "latest.txt"
    if not latest_file.exists():
        return 0, float("inf")

    ckpt_path = Path(latest_file.read_text().strip())
    if not ckpt_path.exists():
        return 0, float("inf")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    step     = ckpt["step"]
    val_loss = ckpt.get("val_loss", float("inf"))
    print(f"[CKPT] Resumed from '{ckpt_path}'  (step={step}, val_loss={val_loss:.4f})")
    return step, val_loss


# ============================================================================
# Validation
# ============================================================================

@torch.no_grad()
def evaluate(model: GPT, val_loader, device: torch.device, max_batches: int = 20) -> float:
    """
    Compute mean validation loss over up to max_batches batches.
    Uses torch.no_grad() for memory efficiency.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for x, y in val_loader:
        if n_batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        total_loss += loss.item()
        n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ============================================================================
# Main training loop
# ============================================================================

def train(args: argparse.Namespace) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] CUDA  –  {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[DEVICE] MPS  –  Apple Silicon")
    else:
        device = torch.device("cpu")
        print("[DEVICE] CPU  (training will be slow)")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = save_dir / "tokenizer.json"

    if tok_path.exists() and not args.retrain_tokenizer:
        tokenizer = BPETokenizer.load(str(tok_path))
    else:
        print("\n[STEP 1/4] Training BPE tokenizer …")
        tokenizer = BPETokenizer()
        tokenizer.train(
            corpus_path = args.corpus,
            vocab_size  = args.vocab_size,
            verbose     = True,
        )
        tokenizer.save(str(tok_path))

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\n[STEP 2/4] Building datasets …")
    train_ds = TextDataset(
        args.corpus, tokenizer,
        context_len  = args.context_len,
        split        = "train",
        val_fraction = args.val_fraction,
    )
    val_ds = TextDataset(
        args.corpus, tokenizer,
        context_len  = args.context_len,
        split        = "val",
        val_fraction = args.val_fraction,
        verbose      = True,
    )

    # pin_memory only makes sense with CUDA
    pin = (device.type == "cuda")
    train_loader = build_dataloader(train_ds, batch_size=args.batch_size,
                                    shuffle=True,  pin_memory=pin)
    val_loader   = build_dataloader(val_ds,   batch_size=args.batch_size,
                                    shuffle=False, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────────
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

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Use AdamW – Adam with weight decay properly applied only to weights,
    # not to biases or LayerNorm parameters.
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr   = args.max_lr,
        betas = (0.9, 0.95),
        eps  = 1e-8,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    best_val   = float("inf")
    if args.resume:
        start_step, best_val = load_checkpoint(save_dir, model, optimizer)
        # Move optimizer state tensors to the right device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path   = save_dir / "training_log.csv"
    log_exists = log_path.exists() and args.resume
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens_per_sec"])

    # ── Save config ───────────────────────────────────────────────────────────
    cfg_dict = vars(cfg)
    (save_dir / "model_config.json").write_text(
        json.dumps(cfg_dict, indent=2, default=str)
    )

    # ================================================================== #
    # Training loop                                                       #
    # ================================================================== #
    print(f"\n[STEP 4/4] Training for {args.max_steps:,} steps …\n")
    model.train()

    step           = start_step
    train_iter     = iter(train_loader)
    t_start        = time.perf_counter()
    tokens_seen    = 0

    while step < args.max_steps:

        # ── Fetch batch (cycle the iterator) ─────────────────────────────
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # ── Learning rate schedule ────────────────────────────────────────
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # ── Forward + backward ────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

        logits, loss = model(x, targets=y)
        loss.backward()

        # Gradient clipping prevents exploding gradients early in training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        optimizer.step()

        step        += 1
        tokens_seen += x.numel()

        # ── Logging ───────────────────────────────────────────────────────
        if step % args.log_every == 0:
            elapsed      = time.perf_counter() - t_start
            tok_per_sec  = tokens_seen / elapsed
            train_loss   = loss.item()

            print(
                f"step {step:>7,}/{args.max_steps:,}  "
                f"loss={train_loss:.4f}  "
                f"lr={lr:.2e}  "
                f"tok/s={tok_per_sec:,.0f}"
            )
            log_writer.writerow([step, f"{train_loss:.6f}", "", f"{lr:.6e}",
                                  f"{tok_per_sec:.1f}"])
            log_file.flush()

        # ── Validation ────────────────────────────────────────────────────
        if step % args.val_every == 0:
            val_loss = evaluate(model, val_loader, device,
                                max_batches=args.val_batches)
            print(f"  [VAL] step={step:,}  val_loss={val_loss:.4f}  "
                  f"perplexity={math.exp(val_loss):.2f}")
            log_writer.writerow([step, "", f"{val_loss:.6f}", "", ""])
            log_file.flush()

            # Save checkpoint (always save; mark best separately)
            save_checkpoint(save_dir, step, model, optimizer, val_loss, cfg_dict)
            if val_loss < best_val:
                best_val = val_loss
                # Symlink / copy as "best"
                best_path = save_dir / "best_model.pt"
                torch.save(model.state_dict(), best_path)
                print(f"  [BEST] New best val_loss={best_val:.4f}  → '{best_path}'")

    # ── Done ─────────────────────────────────────────────────────────────────
    log_file.close()
    total_time = time.perf_counter() - t_start
    print(f"\n[DONE] Training complete in {total_time/60:.1f} min")
    print(f"       Best val loss : {best_val:.4f}  (perplexity {math.exp(best_val):.2f})")
    print(f"       Checkpoints   : '{save_dir}'")


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
    data.add_argument("--corpus",       required=True, help="Path to plain-text training corpus.")
    data.add_argument("--save_dir",     default="checkpoints", help="Directory for checkpoints and logs.")
    data.add_argument("--val_fraction", type=float, default=0.05, help="Fraction of corpus held out for validation.")

    # Tokenizer
    tok = p.add_argument_group("Tokenizer")
    tok.add_argument("--vocab_size",         type=int,  default=4096, help="BPE vocabulary size.")
    tok.add_argument("--retrain_tokenizer",  action="store_true",     help="Retrain tokenizer even if tokenizer.json exists.")

    # Model architecture
    arch = p.add_argument_group("Model architecture")
    arch.add_argument("--context_len", type=int,   default=256,  help="Tokens per training sample.")
    arch.add_argument("--d_model",     type=int,   default=512,  help="Embedding / residual stream dimension.")
    arch.add_argument("--n_heads",     type=int,   default=8,    help="Number of attention heads.")
    arch.add_argument("--n_layers",    type=int,   default=6,    help="Number of transformer blocks.")
    arch.add_argument("--d_ff",        type=int,   default=2048, help="Feed-forward hidden dimension.")
    arch.add_argument("--dropout",     type=float, default=0.1,  help="Dropout probability.")

    # Training
    tr = p.add_argument_group("Training")
    tr.add_argument("--max_steps",    type=int,   default=5000,  help="Total number of gradient steps.")
    tr.add_argument("--batch_size",   type=int,   default=32,    help="Samples per batch.")
    tr.add_argument("--max_lr",       type=float, default=3e-4,  help="Peak learning rate.")
    tr.add_argument("--min_lr",       type=float, default=3e-5,  help="Minimum LR at end of cosine decay.")
    tr.add_argument("--warmup_steps", type=int,   default=200,   help="Linear LR warmup steps.")
    tr.add_argument("--weight_decay", type=float, default=0.1,   help="AdamW weight decay.")
    tr.add_argument("--grad_clip",    type=float, default=1.0,   help="Gradient clipping max norm.")

    # Logging / checkpointing
    log = p.add_argument_group("Logging")
    log.add_argument("--log_every",   type=int, default=50,  help="Print training loss every N steps.")
    log.add_argument("--val_every",   type=int, default=500, help="Run validation every N steps.")
    log.add_argument("--val_batches", type=int, default=20,  help="Number of val batches per evaluation.")
    log.add_argument("--resume",      action="store_true",   help="Resume from latest checkpoint in save_dir.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
