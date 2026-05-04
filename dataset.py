"""
dataset.py
----------
Dataset and data-loading utilities built from scratch with PyTorch only.
No HuggingFace datasets, no external data libraries.

The dataset treats the entire corpus as one long sequence of token ids,
then slices it into overlapping (or non-overlapping) chunks of `context_len`
tokens.  Each chunk becomes one training sample:

    input  = tokens[i : i + context_len]
    target = tokens[i+1 : i + context_len + 1]   (shifted by one position)

This is the standard autoregressive language-modelling objective.

Usage:
    from dataset import TextDataset, build_dataloader

    dataset    = TextDataset("corpus.txt", tokenizer, context_len=256)
    train_dl   = build_dataloader(dataset, batch_size=32, split="train")
"""

import math
import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    Flat token-id dataset for causal language modelling.

    The full corpus is tokenised once and stored as a single 1-D LongTensor.
    __getitem__ slices out one context window and returns (input, target).
    """

    def __init__(
        self,
        corpus_path: str,
        tokenizer,                  # BPETokenizer instance
        context_len:  int   = 256,
        stride:       int   = None, # step between windows (None = non-overlapping)
        split:        str   = "train",
        val_fraction: float = 0.05, # fraction of tokens held out for validation
        verbose:      bool  = True,
    ):
        """
        Args:
            corpus_path:   Path to the plain-text corpus file.
            tokenizer:     Trained BPETokenizer instance.
            context_len:   Number of tokens per training sample.
            stride:        How many tokens to advance per sample.
                           None (default) = context_len (non-overlapping).
                           < context_len  = overlapping windows.
            split:         "train" or "val".
            val_fraction:  Fraction of the corpus reserved for validation.
            verbose:       Print dataset stats on construction.
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: '{path}'")

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("Corpus file is empty.")

        self.context_len = context_len
        self.stride      = stride if stride is not None else context_len

        # ── Tokenise the full corpus ──────────────────────────────────────
        if verbose:
            print(f"[Dataset] Tokenising corpus ({len(text):,} chars) …")
        all_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        tokens  = torch.tensor(all_ids, dtype=torch.long)

        # ── Train / val split ─────────────────────────────────────────────
        n_val   = max(1, int(len(tokens) * val_fraction))
        n_train = len(tokens) - n_val

        if split == "train":
            self.tokens = tokens[:n_train]
        elif split == "val":
            self.tokens = tokens[n_train:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # ── Pre-compute valid window start indices ────────────────────────
        # A window is valid if it has at least context_len + 1 tokens after it
        # (we need one extra token for the target label).
        self.starts = list(range(
            0,
            len(self.tokens) - context_len,
            self.stride,
        ))

        if len(self.starts) == 0:
            raise ValueError(
                f"Corpus too small for context_len={context_len}.  "
                f"Need at least {context_len + 1} tokens, got {len(self.tokens)}."
            )

        if verbose:
            print(
                f"[Dataset] {split.upper()}  |  "
                f"{len(self.tokens):,} tokens  |  "
                f"{len(self.starts):,} samples  |  "
                f"context={context_len}, stride={self.stride}"
            )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: input token ids  shape (context_len,)
            y: target token ids shape (context_len,)  — x shifted right by 1
        """
        start = self.starts[idx]
        chunk = self.tokens[start : start + self.context_len + 1]
        x     = chunk[:-1]   # input:  tokens 0 .. context_len-1
        y     = chunk[1:]    # target: tokens 1 .. context_len
        return x, y


# ============================================================================
# Dataloader factory
# ============================================================================

def build_dataloader(
    dataset:     TextDataset,
    batch_size:  int  = 32,
    shuffle:     bool = True,
    num_workers: int  = 0,
    pin_memory:  bool = False,
) -> DataLoader:
    """
    Wrap a TextDataset in a DataLoader with sensible defaults.

    Args:
        dataset:     A TextDataset instance.
        batch_size:  Samples per batch.
        shuffle:     Shuffle between epochs (set False for val).
        num_workers: Parallel data loading workers (0 = main process).
        pin_memory:  Page-lock host memory for faster GPU transfer.

    Returns:
        A torch.utils.data.DataLoader ready to iterate.
    """
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,   # keep all batches the same size
    )


# ============================================================================
# Corpus statistics helper
# ============================================================================

def corpus_stats(corpus_path: str, tokenizer) -> dict:
    """
    Print and return basic statistics about a corpus.

    Useful for deciding vocab_size, context_len, and training duration.
    """
    text   = Path(corpus_path).read_text(encoding="utf-8")
    ids    = tokenizer.encode(text)
    words  = text.split()
    lines  = text.splitlines()

    stats = {
        "chars":          len(text),
        "words":          len(words),
        "lines":          len(lines),
        "tokens":         len(ids),
        "chars_per_tok":  len(text) / max(len(ids), 1),
        "unique_tokens":  len(set(ids)),
    }

    print("\n[Corpus Stats]")
    for k, v in stats.items():
        print(f"  {k:<20}: {v:,.2f}" if isinstance(v, float) else f"  {k:<20}: {v:,}")

    return stats
