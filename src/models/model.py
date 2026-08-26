"""EconomyEncoder V1 — bidirectional transformer encoder for economic impact.

Architecture (fixed in MODEL_SPEC.md):
    8 layers, 384 hidden, 6 heads (64 per head), RoPE, Pre-RMSNorm,
    SwiGLU FFN (1024), ~20-25M parameters.

Input is a single token sequence:
    [CLS] [EVENT] ... [CASE] ... [CONTEXT] ... [SEP]

Output is one scalar in [-1, +1] from the [CLS] vector:
    Linear(384->128) -> SiLU -> Dropout -> Linear(128->1) -> tanh
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class RoPE(nn.Module):
    """Rotary positional embeddings applied to attention scores."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)
        cos = angles.cos()
        sin = angles.sin()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len > self.cos.shape[0]:
            raise ValueError(
                f"sequence length {seq_len} exceeds RoPE capacity {self.cos.shape[0]}"
            )
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rotated = torch.stack(
            (x1 * cos - x2 * sin, x1 * sin + x2 * cos),
            dim=-1,
        )
        return rotated.flatten(-2)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: w2(silu(w1(x)) * w3(x))."""

    def __init__(self, dim: int, ff_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class EncoderBlock(nn.Module):
    """Pre-RMSNorm transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, max_seq_len)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class EconomyEncoder(nn.Module):
    """EconomyEncoder V1: text -> single economic impact score.

    Parameters (~23M with vocab=24000, dim=384, ff=1024, layers=8):
        token_embedding:    24000 * 384          = 9.2M
        8 encoder blocks:
          attention (QKVO):  8 * 4 * 384^2       = 4.7M
          SwiGLU:            8 * 3 * 384 * 1024  = 9.4M
        output head:         384*128 + 128       = 0.05M
    """

    def __init__(
        self,
        vocab_size: int = 24_000,
        d_model: int = 384,
        num_heads: int = 6,
        num_layers: int = 8,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        cls_idx: int = 1,
        sep_idx: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, num_heads, ff_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        )
        self.final_norm = RMSNorm(d_model)

        self.score_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence_length]")
        if token_ids.shape[0] == 0:
            raise ValueError("batch must contain at least one sample")
        if token_ids.dtype == torch.bool or torch.is_floating_point(token_ids) or torch.is_complex(token_ids):
            raise TypeError("token_ids must use an integer dtype")
        if attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must have the same shape as token_ids")
        if not torch.logical_or(attention_mask == 0, attention_mask == 1).all():
            raise ValueError("attention_mask must contain only 0 or 1")
        if not attention_mask.bool().any(dim=1).all():
            raise ValueError("every sample must contain at least one non-padding token")

        batch, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_embedding(token_ids)

        key_padding_mask = ~attention_mask.bool()

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        cls_mask = token_ids == self.cls_idx
        has_cls = cls_mask.any(dim=1)
        if not has_cls.all():
            raise ValueError("every sample must contain a [CLS] token")

        cls_indices = cls_mask.float().argmax(dim=1)
        batch_indices = torch.arange(batch, device=token_ids.device)
        cls_vectors = x[batch_indices, cls_indices]

        score = self.score_head(cls_vectors)
        score = torch.tanh(score)

        return {"score": score}
