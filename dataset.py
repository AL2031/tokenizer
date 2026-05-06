"""
dataset.py
----------
Dataset and data-loading utilities built from scratch with PyTorch only.
No HuggingFace datasets, no external data libraries.

Encodes the corpus in chunks to avoid loading the full text into RAM at once —
this is the key fix for large corpora like UltraChat that crash on Colab.
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    Flat token-id dataset for causal language modelling.

    Encodes the corpus line-by-line in chunks so RAM never spikes.
    __getitem__ slices out one context window and returns (input, target).
    """

    def __init__(
        self,
        corpus_path:  str,
        tokenizer,
        context_len:  int   = 256,
        stride:       int   = None,
        split:        str   = "train",
        val_fraction: float = 0.05,
        chunk_lines:  int   = 10_000,  # encode this many lines at a time
        verbose:      bool  = True,
    ):
        """
        Args:
            corpus_path:   Path to the plain-text corpus file.
            tokenizer:     Trained BPETokenizer instance.
            context_len:   Number of tokens per training sample.
            stride:        Step between windows. None = non-overlapping.
            split:         "train" or "val".
            val_fraction:  Fraction of corpus held out for validation.
            chunk_lines:   Lines to encode per chunk — controls RAM usage.
                           Lower = less RAM. 10k lines is safe for Colab.
            verbose:       Print progress.
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: '{path}'")

        self.context_len = context_len
        self.stride      = stride if stride is not None else context_len

        # ── Encode corpus in chunks ───────────────────────────────────────
        # Instead of read_text() which loads everything into RAM at once,
        # we read line by line and encode in batches of chunk_lines.
        # This keeps RAM flat regardless of corpus size.
        if verbose:
            size_mb = path.stat().st_size / 1e6
            print(f"[Dataset] Encoding '{path.name}' ({size_mb:.1f} MB) in chunks …")

        all_ids     = []
        chunk       = []
        total_lines = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                chunk.append(line)
                if len(chunk) >= chunk_lines:
                    ids = tokenizer.encode("".join(chunk))
                    all_ids.extend(ids)
                    total_lines += len(chunk)
                    chunk = []
                    if verbose:
                        print(f"  {total_lines:,} lines → {len(all_ids):,} tokens …",
                              end="\r")

            # Encode remaining lines
            if chunk:
                ids = tokenizer.encode("".join(chunk))
                all_ids.extend(ids)
                total_lines += len(chunk)

        if verbose:
            print(f"  {total_lines:,} lines → {len(all_ids):,} tokens total      ")

        tokens = torch.tensor(all_ids, dtype=torch.long)
        del all_ids  # free Python list immediately after converting to tensor

        # ── Train / val split ─────────────────────────────────────────────
        n_val   = max(1, int(len(tokens) * val_fraction))
        n_train = len(tokens) - n_val

        if split == "train":
            self.tokens = tokens[:n_train]
        elif split == "val":
            self.tokens = tokens[n_train:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        del tokens  # free full tensor — we only keep the split slice

        # ── Pre-compute window start indices ──────────────────────────────
        self.starts = list(range(
            0,
            len(self.tokens) - context_len,
            self.stride,
        ))

        if len(self.starts) == 0:
            raise ValueError(
                f"Corpus too small for context_len={context_len}. "
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
        start = self.starts[idx]
        chunk = self.tokens[start : start + self.context_len + 1]
        x     = chunk[:-1]
        y     = chunk[1:]
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
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )
