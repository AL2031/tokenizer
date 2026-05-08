"""
model.py
--------
A GPT-style decoder-only Transformer built from scratch using only PyTorch.
No HuggingFace models, no pre-built attention layers — every operation is
written explicitly so you can see exactly what is happening.

Architecture:
  Embedding (token + position)
    └─> N x TransformerBlock
          ├─ LayerNorm
          ├─ CausalSelfAttention  (multi-head, with causal mask)
          ├─ Residual connection
          ├─ LayerNorm
          ├─ FeedForward  (GELU activation, 4x expansion)
          └─ Residual connection
    └─> LayerNorm
    └─> Linear head (projects to vocab logits)

Default config targets a small but trainable model (~25 M parameters):
  vocab_size  : from tokenizer
  context_len : 256   (tokens per sample)
  d_model     : 512   (embedding dimension)
  n_heads     : 8     (attention heads)
  n_layers    : 6     (transformer blocks)
  d_ff        : 2048  (feed-forward hidden size)
  dropout     : 0.1

Usage:
    from model import GPTConfig, GPT

    cfg   = GPTConfig(vocab_size=4096)
    model = GPT(cfg)
    print(model.num_parameters())
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration dataclass
# ============================================================================

@dataclass
class GPTConfig:
    """All hyper-parameters for the GPT model in one place."""

    vocab_size:  int   = 4096    # set to tokenizer.vocab_size after training
    context_len: int   = 256     # maximum sequence length (context window)
    d_model:     int   = 512     # embedding / residual stream dimension
    n_heads:     int   = 8       # number of attention heads
    n_layers:    int   = 6       # number of transformer blocks
    d_ff:        int   = 2048    # feed-forward hidden dimension (usually 4 * d_model)
    dropout:     float = 0.1     # applied after attention, FF, and embeddings
    bias:        bool  = True    # use bias in Linear and LayerNorm layers

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        self.head_dim = self.d_model // self.n_heads


# ============================================================================
# Building blocks
# ============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal (autoregressive) mask.

    Each token can only attend to itself and earlier tokens — this is what
    makes the model generative: it cannot "see the future" during training.

    The causal mask is registered as a buffer so it moves to the correct
    device automatically with .to(device) / .cuda().
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model  = cfg.d_model
        self.dropout  = cfg.dropout

        # Single fused projection for Q, K, V — 3x more efficient than
        # three separate Linear layers because it's one matrix multiply.
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)

        # Output projection after concatenating all heads
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Causal mask: a lower-triangular matrix of ones.
        # Shape: (1, 1, context_len, context_len) for broadcasting over
        # the batch and head dimensions.
        mask = torch.tril(torch.ones(cfg.context_len, cfg.context_len))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.context_len, cfg.context_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape   # batch, seq_len, d_model

        # ── Project to Q, K, V ───────────────────────────────────────────
        # qkv shape: (B, T, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Split along the last dimension into three equal chunks
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape for multi-head attention:
        # (B, T, d_model) -> (B, n_heads, T, head_dim)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # ── Scaled dot-product attention ──────────────────────────────────
        # Use F.scaled_dot_product_attention (PyTorch 2.0+) which
        # automatically uses Flash Attention when available on CUDA.
        # Flash Attention is 2-4x faster and uses O(sqrt(N)) memory
        # instead of O(N^2) — a huge win for long sequences.
        try:
            # is_causal=True handles the causal mask internally (faster)
            attended = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.0,
                is_causal = True,
            )
        except Exception:
            # Fallback for older PyTorch versions
            scale   = 1.0 / math.sqrt(self.head_dim)
            scores  = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores  = scores.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )
            weights  = F.softmax(scores, dim=-1)
            weights  = self.attn_dropout(weights)
            attended = torch.matmul(weights, v)

        # ── Merge heads ───────────────────────────────────────────────────
        # (B, n_heads, T, head_dim) -> (B, T, d_model)
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(attended))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network applied identically to each token.

    Architecture:
        Linear(d_model -> d_ff)  +  GELU  +  Linear(d_ff -> d_model)

    GELU (Gaussian Error Linear Unit) is the standard activation for GPT
    models — it is smoother than ReLU and empirically trains better.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One full transformer decoder block:

        x = x + Attention(LayerNorm(x))     ← residual around attention
        x = x + FeedForward(LayerNorm(x))   ← residual around FF

    Pre-norm (LayerNorm BEFORE each sub-layer) is used here because it
    trains more stably than the original post-norm formulation.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1   = nn.LayerNorm(cfg.d_model, elementwise_affine=cfg.bias)
        self.attn    = CausalSelfAttention(cfg)
        self.norm2   = nn.LayerNorm(cfg.d_model, elementwise_affine=cfg.bias)
        self.ff      = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # attention sub-layer + residual
        x = x + self.ff(self.norm2(x))     # feed-forward sub-layer + residual
        return x


# ============================================================================
# Full GPT model
# ============================================================================

class GPT(nn.Module):
    """
    GPT-style decoder-only language model.

    During training:
        logits, loss = model(input_ids, targets=shifted_input_ids)

    During inference:
        logits, _ = model(input_ids)
        next_token = sample(logits[:, -1, :])
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict(dict(
            # Token embedding: maps each token id to a d_model-dimensional vector
            tok_emb  = nn.Embedding(cfg.vocab_size, cfg.d_model),

            # Position embedding: learned vector for each position 0..context_len-1
            pos_emb  = nn.Embedding(cfg.context_len, cfg.d_model),

            drop     = nn.Dropout(cfg.dropout),

            # Stack of N identical transformer blocks
            blocks   = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]),

            # Final layer norm before the output projection
            norm_out = nn.LayerNorm(cfg.d_model, elementwise_affine=cfg.bias),
        ))

        # Language model head: projects d_model -> vocab_size to get logits
        # We tie the weights with the token embedding matrix — this is standard
        # practice (Press & Wolf, 2017) and reduces parameters by ~10-20%.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.tok_emb.weight   # weight tying

        # Initialise weights using GPT-style scaled init
        self.apply(self._init_weights)

        # Scale down residual projections by 1/sqrt(n_layers) so that the
        # variance of the residual stream doesn't blow up with depth.
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(param, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Standard GPT weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,              # (B, T)
        targets:   Optional[torch.Tensor] = None,  # (B, T)  — shifted input_ids
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids:  Long tensor of shape (batch, seq_len).
            targets:    Optional long tensor of same shape.  When provided,
                        cross-entropy loss is computed and returned.

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar tensor if targets provided, else None
        """
        B, T = input_ids.shape
        assert T <= self.cfg.context_len, (
            f"Sequence length {T} exceeds model context_len {self.cfg.context_len}"
        )

        device = input_ids.device

        # ── Embeddings ───────────────────────────────────────────────────
        # Token embeddings: (B, T, d_model)
        tok = self.transformer.tok_emb(input_ids)

        # Position embeddings: (1, T, d_model) — broadcastable over batch
        pos = self.transformer.pos_emb(
            torch.arange(T, device=device).unsqueeze(0)
        )

        x = self.transformer.drop(tok + pos)

        # ── Transformer blocks ───────────────────────────────────────────
        for block in self.transformer.blocks:
            x = block(x)

        # ── Output projection ────────────────────────────────────────────
        x      = self.transformer.norm_out(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        # ── Loss (optional) ──────────────────────────────────────────────
        loss = None
        if targets is not None:
            # Cross-entropy expects (N, C) logits and (N,) targets
            # Flatten: (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,   # positions padded with -1 don't contribute
            )

        return logits, loss

    # ================================================================== #
    # Inference helpers                                                   #
    # ================================================================== #

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids:          torch.Tensor,   # (1, T_prompt)
        max_new_tokens:      int   = 200,
        temperature:         float = 1.0,
        top_k:               int   = 50,
        top_p:               float = 1.0,
        repetition_penalty:  float = 1.0,
        eos_id:              Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation: append one token at a time.

        Args:
            prompt_ids:         Starting token ids, shape (1, T).
            max_new_tokens:     How many tokens to generate.
            temperature:        Sampling temperature (1.0 = unchanged).
            top_k:              Keep only top-k logits before sampling.
            top_p:              Nucleus sampling threshold.
            repetition_penalty: Penalise already-generated tokens (>1 = less repeat).
            eos_id:             Stop early when this token is generated.

        Returns:
            Token id tensor of shape (1, T + max_new_tokens).
        """
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate if the context is getting too long
            context = ids if ids.shape[1] <= self.cfg.context_len else \
                      ids[:, -self.cfg.context_len:]

            logits, _ = self(context)
            # Take logits for the very last position only
            logits = logits[:, -1, :]   # (1, vocab_size)

            # ── Repetition penalty ────────────────────────────────────────
            if repetition_penalty != 1.0:
                for token_id in ids[0].tolist():
                    logits[0, token_id] /= repetition_penalty

            # ── Temperature ───────────────────────────────────────────────
            if temperature != 1.0:
                logits = logits / temperature

            # ── Top-k filter ──────────────────────────────────────────────
            if top_k > 0:
                top_vals = torch.topk(logits, min(top_k, logits.size(-1))).values
                logits[logits < top_vals[:, -1:]] = float("-inf")

            # ── Top-p (nucleus) filter ────────────────────────────────────
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens whose cumulative prob exceeds top_p
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                # Scatter back to original ordering
                logits = torch.zeros_like(logits).scatter_(
                    1, sorted_idx, sorted_logits
                )

            # ── Sample ────────────────────────────────────────────────────
            probs    = F.softmax(logits, dim=-1)
            next_id  = torch.multinomial(probs, num_samples=1)   # (1, 1)
            ids      = torch.cat([ids, next_id], dim=1)

            if eos_id is not None and next_id.item() == eos_id:
                break

        return ids

    # ================================================================== #
    # Utilities                                                           #
    # ================================================================== #

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total (or only trainable) parameter count."""
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)

    def __repr__(self) -> str:
        n = self.num_parameters()
        return (
            f"GPT(\n"
            f"  vocab={self.cfg.vocab_size}, ctx={self.cfg.context_len}, "
            f"  d={self.cfg.d_model}, heads={self.cfg.n_heads}, "
            f"  layers={self.cfg.n_layers}, ff={self.cfg.d_ff}\n"
            f"  params={n/1e6:.2f}M\n"
            f")"
        )
