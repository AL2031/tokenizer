"""
dataset.py
----------
Dataset and data-loading utilities built from scratch with PyTorch only.

Optimizations applied:
  - Parallel CPU encoding using ThreadPoolExecutor
  - Chunked line reading so RAM stays flat on large corpora
  - DataLoader with pin_memory, prefetch_factor, persistent_workers
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    Flat token-id dataset for causal language modelling.

    Encodes the corpus in parallel chunks so RAM never spikes and
    CPU cores are fully utilised during the encoding phase.
    """

    def __init__(
        self,
        corpus_path:  str,
        tokenizer,
        context_len:  int   = 256,
        stride:       int   = None,
        split:        str   = "train",
        val_fraction: float = 0.05,
        chunk_lines:  int   = 5_000,   # lines per encoding chunk
        num_workers:  int   = 4,        # parallel encoding threads
        verbose:      bool  = True,
    ):
        """
        Args:
            corpus_path:   Path to the plain-text corpus file.
            tokenizer:     Trained BPETokenizer instance.
            context_len:   Tokens per training sample.
            stride:        Step between windows. None = non-overlapping.
            split:         "train" or "val".
            val_fraction:  Fraction of corpus held out for validation.
            chunk_lines:   Lines per encoding chunk — lower = less RAM.
            num_workers:   CPU threads for parallel encoding.
            verbose:       Print progress.
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: '{path}'")

        self.context_len = context_len
        self.stride      = stride if stride is not None else context_len

        # ── Read corpus into ordered chunks ───────────────────────────────
        if verbose:
            size_mb = path.stat().st_size / 1e6
            print(f"[Dataset] Reading '{path.name}' ({size_mb:.1f} MB) …")

        chunks       = []
        current      = []
        total_lines  = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                current.append(line)
                if len(current) >= chunk_lines:
                    chunks.append("".join(current))
                    total_lines += len(current)
                    current = []
            if current:
                chunks.append("".join(current))
                total_lines += len(current)

        if verbose:
            print(f"[Dataset] {total_lines:,} lines  →  "
                  f"{len(chunks):,} chunks  →  "
                  f"encoding with {num_workers} threads …")

        # ── Parallel encoding ─────────────────────────────────────────────
        # ThreadPoolExecutor is safe here because our tokenizer is read-only
        # during encoding (no shared mutable state).
        results     = [None] * len(chunks)
        completed   = 0

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            future_to_idx = {
                pool.submit(tokenizer.encode, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx          = future_to_idx[future]
                results[idx] = future.result()
                completed   += 1
                if verbose:
                    print(f"  chunk {completed:,}/{len(chunks):,} done …",
                          end="\r")

        if verbose:
            print()

        # ── Flatten and convert to tensor ─────────────────────────────────
        all_ids = [tid for chunk_ids in results for tid in chunk_ids]
        del results, chunks

        tokens = torch.tensor(all_ids, dtype=torch.long)
        del all_ids

        if verbose:
            print(f"[Dataset] {len(tokens):,} tokens total")

        # ── Train / val split ─────────────────────────────────────────────
        n_val   = max(1, int(len(tokens) * val_fraction))
        n_train = len(tokens) - n_val

        if split == "train":
            self.tokens = tokens[:n_train]
        elif split == "val":
            self.tokens = tokens[n_train:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        del tokens

        # ── Window indices ────────────────────────────────────────────────
        self.starts = list(range(
            0,
            len(self.tokens) - context_len,
            self.stride,
        ))

        if len(self.starts) == 0:
            raise ValueError(
                f"Corpus too small for context_len={context_len}. "
                f"Got {len(self.tokens)} tokens, need at least {context_len + 1}."
            )

        if verbose:
            print(
                f"[Dataset] {split.upper()}  |  "
                f"{len(self.tokens):,} tokens  |  "
                f"{len(self.starts):,} samples  |  "
                f"context={context_len}"
            )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        chunk = self.tokens[start : start + self.context_len + 1]
        return chunk[:-1], chunk[1:]


# ============================================================================
# DataLoader factory
# ============================================================================

def build_dataloader(
    dataset:            TextDataset,
    batch_size:         int  = 32,
    shuffle:            bool = True,
    num_workers:        int  = 2,
    pin_memory:         bool = True,   # pre-pin host memory for faster GPU transfer
    prefetch_factor:    int  = 2,      # batches to prefetch ahead
    persistent_workers: bool = True,   # keep workers alive between epochs
) -> DataLoader:
    """
    Build a DataLoader with GPU-friendly defaults.

    pin_memory + prefetch_factor together overlap CPU→GPU transfers with
    GPU compute, which can give 10-20% throughput improvement.
    persistent_workers avoids re-spawning worker processes each epoch.
    """
    # num_workers=0 disables prefetch_factor and persistent_workers
    if num_workers == 0:
        return DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle    = shuffle,
            drop_last  = True,
        )

    return DataLoader(
        dataset,
        batch_size          = batch_size,
        shuffle             = shuffle,
        num_workers         = num_workers,
        pin_memory          = pin_memory,
        prefetch_factor     = prefetch_factor,
        persistent_workers  = persistent_workers,
        drop_last           = True,
    )
