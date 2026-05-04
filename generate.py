"""
generate.py
-----------
Load a trained checkpoint and generate text interactively.

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt
    python generate.py --checkpoint checkpoints/best_model.pt --temperature 0.8 --top_p 0.9
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from tokenizer import BPETokenizer
from model     import GPT, GPTConfig


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_dir: str, device: torch.device):
    """
    Load tokenizer and model from a checkpoint directory.

    Expects:
        checkpoints/
          tokenizer.json
          model_config.json
          best_model.pt   (or any ckpt_*.pt file)
    """
    ckpt_dir = Path(checkpoint_dir)

    # Load tokenizer
    tok_path = ckpt_dir / "tokenizer.json"
    if not tok_path.exists():
        sys.exit(f"[ERROR] tokenizer.json not found in '{ckpt_dir}'")
    tokenizer = BPETokenizer.load(str(tok_path))

    # Load model config
    cfg_path = ckpt_dir / "model_config.json"
    if not cfg_path.exists():
        sys.exit(f"[ERROR] model_config.json not found in '{ckpt_dir}'")
    cfg_dict = json.loads(cfg_path.read_text())
    cfg = GPTConfig(**{k: v for k, v in cfg_dict.items() if hasattr(GPTConfig, k)})

    # Load weights
    best_path = ckpt_dir / "best_model.pt"
    latest    = ckpt_dir / "latest.txt"
    if best_path.exists():
        weights_path = best_path
    elif latest.exists():
        weights_path = Path(latest.read_text().strip())
    else:
        sys.exit(f"[ERROR] No model weights found in '{ckpt_dir}'")

    model = GPT(cfg)
    state = torch.load(weights_path, map_location="cpu")
    # best_model.pt saves only the state_dict directly
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"[OK] Loaded model ({model.num_parameters()/1e6:.1f}M params) from '{weights_path}'")
    return tokenizer, model


def generate_interactive(tokenizer, model, device, args):
    """Interactive REPL for text generation."""
    print(f"\n{'='*60}")
    print("  From-Scratch LLM  |  type 'quit' to exit")
    print(f"{'='*60}\n")

    while True:
        try:
            prompt = input("Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt or prompt.lower() == "quit":
            break

        ids     = tokenizer.encode(prompt, add_bos=True)
        inp     = torch.tensor([ids], dtype=torch.long, device=device)

        out_ids = model.generate(
            inp,
            max_new_tokens     = args.max_new_tokens,
            temperature        = args.temperature,
            top_k              = args.top_k,
            top_p              = args.top_p,
            repetition_penalty = args.repetition_penalty,
            eos_id             = tokenizer.eos_id,
        )

        # Decode only the newly generated tokens
        new_ids = out_ids[0, len(ids):].tolist()
        print(f"\nModel: {prompt}{tokenizer.decode(new_ids)}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    p.add_argument("--checkpoint",         required=True,       help="Path to checkpoint directory.")
    p.add_argument("--max_new_tokens",     type=int,   default=200)
    p.add_argument("--temperature",        type=float, default=1.0)
    p.add_argument("--top_k",             type=int,   default=50)
    p.add_argument("--top_p",             type=float, default=1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = detect_device()
    print(f"[DEVICE] {device}")
    tokenizer, model = load_model(args.checkpoint, device)
    generate_interactive(tokenizer, model, device, args)
