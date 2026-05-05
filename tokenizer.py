"""
tokenizer.py
------------
A Byte-Pair Encoding (BPE) tokenizer built entirely from scratch.
No HuggingFace tokenizers, no SentencePiece, no external tokenizer libs.

How BPE works:
  1. Start with every character in the corpus as its own token.
  2. Count every adjacent pair of tokens in the corpus.
  3. Merge the most frequent pair into a new single token.
  4. Repeat steps 2-3 for `vocab_size` iterations.
  5. Save the merge rules — at inference time, apply them in the same order.

Usage:
    from tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.train("my_corpus.txt", vocab_size=4096)
    tok.save("tokenizer.json")

    tok2 = BPETokenizer.load("tokenizer.json")
    ids  = tok2.encode("Hello world!")
    text = tok2.decode(ids)
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class BPETokenizer:
    # ------------------------------------------------------------------ #
    # Special tokens                                                       #
    # ------------------------------------------------------------------ #
    PAD_TOKEN   = "<|pad|>"
    UNK_TOKEN   = "<|unk|>"
    BOS_TOKEN   = "<|bos|>"
    EOS_TOKEN   = "<|eos|>"
    SPACE_TOKEN = "Ġ"          # marks a space-prefixed word (GPT-2 convention)

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self):
        # token -> id  and  id -> token
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Ordered list of merge rules: (left_token, right_token) -> merged
        # Order matters — merges are applied in training order at inference time.
        self.merges: List[Tuple[str, str]] = []

        # Convenience ids
        self.pad_id = self.unk_id = self.bos_id = self.eos_id = None

    # ================================================================== #
    # TRAINING                                                            #
    # ================================================================== #

    def train(self, corpus_path: str, vocab_size: int = 4096,
              min_frequency: int = 2, verbose: bool = True,
              max_chars: int = 5_000_000) -> None:
        """
        Train the BPE tokenizer on a plain-text corpus file.

        Args:
            corpus_path:   Path to a UTF-8 text file.
            vocab_size:    Target vocabulary size (including special tokens).
            min_frequency: Stop merging when the best pair appears fewer than
                           this many times (early-stop for sparse corpora).
            verbose:       Print progress every 100 merges.
            max_chars:     Max characters to read for BPE training.
                           Caps RAM usage — 5 MB is enough for a solid vocab.
                           The full corpus is still used for model training.
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: '{path}'")

        # Read only up to max_chars — BPE only needs a representative sample.
        # Loading a 500 MB corpus into word-frequency dicts explodes RAM.
        with open(path, "r", encoding="utf-8") as f:
            text = f.read(max_chars)

        if not text.strip():
            raise ValueError("Corpus file is empty.")

        actual_size = path.stat().st_size
        if verbose:
            if actual_size > max_chars:
                print(f"[BPE] Large corpus: {actual_size/1e6:.0f} MB total  →  "
                      f"sampling first {max_chars/1e6:.0f} MB for vocab building")
            print(f"[BPE] {len(text):,} chars  |  target vocab: {vocab_size:,}")

        # ── Step 1: character-level pre-tokenisation ──────────────────────
        # Split on whitespace; mark EVERY word with SPACE_TOKEN prefix.
        # This includes the first word so training and encoding are identical.
        word_freq = defaultdict(int)
        for word in text.split():
            marked = self.SPACE_TOKEN + word   # "hello" -> "Ġhello"
            word_freq[tuple(marked)] += 1      # ('Ġ','h','e','l','l','o')

        # ── Step 2: build the initial base vocabulary ─────────────────────
        # Every unique character that appears in the corpus becomes a token.
        base_vocab: set = set()
        for word_tuple in word_freq:
            base_vocab.update(word_tuple)

        # Special tokens first (ids 0-3), then sorted characters
        vocab = self.SPECIAL_TOKENS + sorted(base_vocab)
        self._build_lookup(vocab)

        if verbose:
            print(f"[BPE] Base vocab: {len(vocab):,} tokens  "
                  f"(4 special + {len(base_vocab):,} chars)")

        # ── Step 3: BPE merge loop ────────────────────────────────────────
        num_merges = vocab_size - len(vocab)
        if num_merges <= 0:
            if verbose:
                print("[BPE] Vocab already at target size — no merges needed.")
            return

        for merge_idx in range(num_merges):
            # Count all adjacent pairs across the corpus
            pair_freq = defaultdict(int)
            for word_tuple, freq in word_freq.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_freq[pair] += freq

            if not pair_freq:
                break   # Nothing left to merge

            best_pair, best_freq = max(pair_freq.items(), key=lambda x: x[1])

            if best_freq < min_frequency:
                if verbose:
                    print(f"[BPE] Early stop at merge {merge_idx}: "
                          f"best pair frequency {best_freq} < {min_frequency}")
                break

            # Create the merged token and register it
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            new_id = len(self.token_to_id)
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id]    = new_token

            # Apply the merge to the word frequency dictionary
            word_freq = self._apply_merge(word_freq, best_pair, new_token)

            if verbose and (merge_idx + 1) % 100 == 0:
                print(f"[BPE] Merge {merge_idx + 1:,}/{num_merges:,}  "
                      f"vocab={len(self.token_to_id):,}  "
                      f"merged='{new_token}'  freq={best_freq:,}")

        if verbose:
            print(f"[BPE] Training complete. Final vocab size: "
                  f"{len(self.token_to_id):,}")

    @staticmethod
    def _apply_merge(
        word_freq: Dict[Tuple, int],
        pair: Tuple[str, str],
        new_token: str,
    ) -> Dict[Tuple, int]:
        """
        Replace every occurrence of `pair` in all word tuples with `new_token`.
        Returns a new word_freq dictionary with updated tuples.
        """
        updated: Dict[Tuple, int] = {}
        a, b = pair
        for word_tuple, freq in word_freq.items():
            new_word: List[str] = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == a and word_tuple[i + 1] == b:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            updated[tuple(new_word)] = freq
        return updated

    def _build_lookup(self, vocab: List[str]) -> None:
        """Rebuild token<->id maps from a flat vocab list."""
        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(vocab)}
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]

    # ================================================================== #
    # ENCODING                                                            #
    # ================================================================== #

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int = None,
    ) -> List[int]:
        """
        Convert a string to a list of token ids.

        Args:
            text:       Input string.
            add_bos:    Prepend the BOS token id.
            add_eos:    Append the EOS token id.
            max_length: Truncate to this many tokens (applied after BOS/EOS).

        Returns:
            List of integer token ids.
        """
        ids: List[int] = []

        # Every word gets the SPACE_TOKEN prefix — identical to training.
        # This means the vocab always sees "Ġword" never a bare "word",
        # which eliminates UNK for any character seen during training.
        for word in text.split():
            marked = self.SPACE_TOKEN + word   # always prefix, including first word
            chars  = list(marked)
            tokens = self._bpe_encode_word(chars)
            for tok in tokens:
                ids.append(self.token_to_id.get(tok, self.unk_id))

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def _bpe_encode_word(self, chars: List[str]) -> List[str]:
        """
        Apply the trained merge rules to a single pre-tokenised word
        (given as a list of characters).
        """
        if len(chars) == 1:
            return chars

        # Apply each merge rule in training order
        for left, right in self.merges:
            i = 0
            merged: List[str] = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == left and chars[i + 1] == right:
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(chars[i])
                    i += 1
            chars = merged
            if len(chars) == 1:
                break   # fully merged — no point continuing

        return chars

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int = None,
        pad: bool = True,
    ) -> List[List[int]]:
        """
        Encode a list of strings, optionally padding to the longest sequence.

        Returns:
            List of token-id lists (all same length if pad=True).
        """
        encoded = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
            for t in texts
        ]
        if pad:
            max_len = max(len(e) for e in encoded)
            encoded = [e + [self.pad_id] * (max_len - len(e)) for e in encoded]
        return encoded

    # ================================================================== #
    # DECODING                                                            #
    # ================================================================== #

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert a list of token ids back to a string.

        Args:
            ids:                   List of integer token ids.
            skip_special_tokens:   If True, strip PAD/BOS/EOS/UNK tokens.

        Returns:
            Decoded string.
        """
        special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        tokens: List[str] = []

        for tid in ids:
            if skip_special_tokens and tid in special:
                continue
            tok = self.id_to_token.get(tid, self.UNK_TOKEN)
            tokens.append(tok)

        # Join tokens and replace the Ġ space marker with a real space.
        # Since every word is prefixed with Ġ, the result starts with a
        # leading space — strip it off.
        text = "".join(tokens).replace(self.SPACE_TOKEN, " ").lstrip(" ")
        return text

    # ================================================================== #
    # SAVE / LOAD                                                         #
    # ================================================================== #

    def save(self, path: str) -> None:
        """Serialise the full tokenizer state to a JSON file."""
        state = {
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"[BPE] Saved to '{path}'  "
              f"(vocab={len(self.token_to_id):,}, merges={len(self.merges):,})")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Restore a tokenizer from a JSON file produced by `save()`."""
        data   = json.loads(Path(path).read_text(encoding="utf-8"))
        tok    = cls()
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tok.merges      = [tuple(pair) for pair in data["merges"]]
        tok.pad_id = tok.token_to_id.get(cls.PAD_TOKEN)
        tok.unk_id = tok.token_to_id.get(cls.UNK_TOKEN)
        tok.bos_id = tok.token_to_id.get(cls.BOS_TOKEN)
        tok.eos_id = tok.token_to_id.get(cls.EOS_TOKEN)
        print(f"[BPE] Loaded from '{path}'  "
              f"(vocab={len(tok.token_to_id):,}, merges={len(tok.merges):,})")
        return tok

    # ================================================================== #
    # HELPERS                                                             #
    # ================================================================== #

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (f"BPETokenizer(vocab_size={self.vocab_size}, "
                f"merges={len(self.merges)})")
