"""
dataset.py
----------
Optimized dataset with:
  - Token cache: encodes once, saves to .npy, instant reload on next run
  - Memory-mapped arrays: train on 100GB corpora with 8GB RAM
  - ProcessPoolExecutor: real parallelism (GIL-free) for CPU-bound encoding
  - Corpus size limit: --max_dataset_mb caps RAM and encoding time
"""

import os
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Worker function — must be at module level for pickling by ProcessPoolExecutor
# ============================================================================

def _encode_chunk(args):
    """
    Encode a single text chunk.
    Reloads the tokenizer in each worker process — required because
    ProcessPoolExecutor spawns separate processes with no shared memory.
    """
    chunk_text, tokenizer_path = args
    # Import here so each worker process gets its own copy
    import sys
    sys.path.insert(0, str(Path(tokenizer_path).parent.parent))
    from tokenizer import BPETokenizer
    tok = BPETokenizer.load(tokenizer_path)
    return tok.encode(chunk_text)


# ============================================================================
# Dataset
# ============================================================================

class TextDataset(Dataset):
    """
    Flat token-id dataset for causal language modelling.

    Key optimizations:
      - Encodes corpus once and caches tokens to a .npy file on disk.
        Subsequent runs skip encoding entirely — instant startup.
      - Stores tokens as a numpy memmap so only accessed slices are
        loaded into RAM, not the full tensor.
      - Uses ProcessPoolExecutor for real CPU parallelism during encoding.
    """

    def __init__(
        self,
        corpus_path:     str,
        tokenizer,
        tokenizer_path:  str   = None,  # path to tokenizer.json for worker processes
        context_len:     int   = 256,
        stride:          int   = None,
        split:           str   = "train",
        val_fraction:    float = 0.05,
        chunk_lines:     int   = 5_000,
        num_workers:     int   = 4,
        max_dataset_mb:  float = 200.0,  # cap corpus reading at this many MB
        cache_dir:       str   = None,   # where to store .npy token cache
        verbose:         bool  = True,
    ):
        """
        Args:
            corpus_path:    Path to plain-text corpus file.
            tokenizer:      Trained BPETokenizer instance.
            tokenizer_path: Path to saved tokenizer.json (needed for workers).
            context_len:    Tokens per training sample.
            stride:         Step between windows. None = non-overlapping.
            split:          "train" or "val".
            val_fraction:   Fraction held out for validation.
            chunk_lines:    Lines per encoding chunk.
            num_workers:    Parallel worker processes for encoding.
            max_dataset_mb: Max MB of corpus to read. Caps RAM + encoding time.
            cache_dir:      Directory to store cached .npy token files.
                            Defaults to same directory as corpus.
            verbose:        Print progress.
        """
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: '{corpus_path}'")

        self.context_len = context_len
        self.stride      = stride if stride is not None else context_len

        cache_dir   = Path(cache_dir) if cache_dir else corpus_path.parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ── Cache key: hash of corpus path + vocab size + max_mb ──────────
        # If any of these change, the cache is invalidated automatically.
        cache_key  = hashlib.md5(
            f"{corpus_path}{tokenizer.vocab_size}{max_dataset_mb}".encode()
        ).hexdigest()[:8]
        cache_file = cache_dir / f"tokens_{cache_key}.npy"

        # ── Load from cache or encode ──────────────────────────────────────
        if cache_file.exists():
            if verbose:
                print(f"[Dataset] Loading token cache '{cache_file.name}' …")
            all_tokens = np.load(str(cache_file), mmap_mode='r')
            if verbose:
                print(f"[Dataset] {len(all_tokens):,} tokens loaded instantly.")
        else:
            all_tokens = self._encode_corpus(
                corpus_path    = corpus_path,
                tokenizer      = tokenizer,
                tokenizer_path = tokenizer_path,
                chunk_lines    = chunk_lines,
                num_workers    = num_workers,
                max_dataset_mb = max_dataset_mb,
                verbose        = verbose,
            )
            # Save to disk for next run
            if verbose:
                print(f"[Dataset] Saving token cache → '{cache_file}' …")
            np.save(str(cache_file), all_tokens)
            if verbose:
                print(f"[Dataset] Cache saved. Future runs will load instantly.")

        # ── Train / val split ─────────────────────────────────────────────
        n_total = len(all_tokens)
        n_val   = max(1, int(n_total * val_fraction))
        n_train = n_total - n_val

        if split == "train":
            tokens = all_tokens[:n_train]
        elif split == "val":
            tokens = all_tokens[n_train:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # ── Memory-mapped storage ──────────────────────────────────────────
        # Store as numpy array (memmap-backed if loaded from cache).
        # __getitem__ reads only the needed slice — RAM stays flat.
        self.tokens = tokens

        # ── Window indices ─────────────────────────────────────────────────
        self.starts = list(range(
            0,
            len(self.tokens) - context_len,
            self.stride,
        ))

        if len(self.starts) == 0:
            raise ValueError(
                f"Corpus too small for context_len={context_len}. "
                f"Got {len(self.tokens):,} tokens."
            )

        if verbose:
            print(
                f"[Dataset] {split.upper()}  |  "
                f"{len(self.tokens):,} tokens  |  "
                f"{len(self.starts):,} samples  |  "
                f"context={context_len}"
            )

    @staticmethod
    def _encode_corpus(
        corpus_path,
        tokenizer,
        tokenizer_path,
        chunk_lines,
        num_workers,
        max_dataset_mb,
        verbose,
    ) -> np.ndarray:
        """
        Read corpus up to max_dataset_mb, encode in parallel chunks,
        return a flat numpy int64 array.
        """
        max_bytes = int(max_dataset_mb * 1e6)
        size_mb   = corpus_path.stat().st_size / 1e6

        if verbose:
            if size_mb > max_dataset_mb:
                print(f"[Dataset] Corpus {size_mb:.0f} MB → "
                      f"reading first {max_dataset_mb:.0f} MB …")
            else:
                print(f"[Dataset] Reading '{corpus_path.name}' "
                      f"({size_mb:.1f} MB) …")

        # Read corpus up to max_bytes
        chunks      = []
        current     = []
        bytes_read  = 0
        total_lines = 0
        stop        = False

        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                current.append(line)
                bytes_read += len(line.encode("utf-8"))
                if len(current) >= chunk_lines:
                    chunks.append("".join(current))
                    total_lines += len(current)
                    current = []
                if bytes_read >= max_bytes:
                    stop = True
                    break
            if current:
                chunks.append("".join(current))
                total_lines += len(current)

        if verbose:
            print(f"[Dataset] {total_lines:,} lines  →  "
                  f"{len(chunks):,} chunks  →  "
                  f"encoding with {num_workers} workers …")

        # ── Parallel encoding ─────────────────────────────────────────────
        # Use ProcessPoolExecutor for real parallelism (no GIL).
        # Each worker reloads the tokenizer from disk.
        results   = [None] * len(chunks)
        completed = 0

        if tokenizer_path and num_workers > 1:
            worker_args = [(chunk, str(tokenizer_path)) for chunk in chunks]
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                future_to_idx = {
                    pool.submit(_encode_chunk, arg): i
                    for i, arg in enumerate(worker_args)
                }
                for future in as_completed(future_to_idx):
                    idx           = future_to_idx[future]
                    results[idx]  = future.result()
                    completed    += 1
                    if verbose:
                        print(f"  chunk {completed:,}/{len(chunks):,} …",
                              end="\r")
        else:
            # Fallback: single-process encoding
            for i, chunk in enumerate(chunks):
                results[i] = tokenizer.encode(chunk)
                completed += 1
                if verbose:
                    print(f"  chunk {completed:,}/{len(chunks):,} …",
                          end="\r")

        if verbose:
            print()

        # Flatten to numpy array
        all_ids = [tid for chunk_ids in results for tid in chunk_ids]
        del results, chunks
        return np.array(all_ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        # .copy() needed for memmap slices before converting to tensor
        chunk = torch.from_numpy(
            self.tokens[start : start + self.context_len + 1].copy()
        ).long()
        return chunk[:-1], chunk[1:]


# ============================================================================
# DataLoader factory
# ============================================================================

def build_dataloader(
    dataset:            TextDataset,
    batch_size:         int  = 32,
    shuffle:            bool = True,
    num_workers:        int  = 0,    # 0 = main process (fastest for memmap datasets)
    pin_memory:         bool = True,
    prefetch_factor:    int  = 2,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Build a DataLoader optimized for memmap-backed datasets.

    num_workers=0 is often fastest when tokens are already in RAM or memmap,
    because worker startup overhead outweighs any parallel loading benefit.
    Set num_workers > 0 only if __getitem__ is slow (e.g., heavy augmentation).
    """
    if num_workers == 0:
        return DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle    = shuffle,
            pin_memory = pin_memory and torch.cuda.is_available(),
            drop_last  = shuffle,
        )

    return DataLoader(
        dataset,
        batch_size          = batch_size,
        shuffle             = shuffle,
        num_workers         = num_workers,
        pin_memory          = pin_memory and torch.cuda.is_available(),
        prefetch_factor     = prefetch_factor,
        persistent_workers  = persistent_workers,
        drop_last           = shuffle,
    )
