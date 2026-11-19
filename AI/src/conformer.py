from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSustainLayer(nn.Module):
    """Differentiable sustain detector separating fast passing melody (<350ms) from sustained chord tones (>=500ms)."""

    def __init__(self, short_frames: int = 7, sustain_frames: int = 39):
        super().__init__()
        self.short_frames = short_frames
        self.sustain_frames = sustain_frames
        self.alpha = nn.Parameter(torch.tensor(8.0))
        self.threshold = nn.Parameter(torch.tensor(0.28))

    def forward(self, chroma_logits: torch.Tensor) -> torch.Tensor:
        orig_dtype = chroma_logits.dtype
        # Compute in float32 for log/exp numerical stability under AMP fp16
        c_prob = torch.sigmoid(chroma_logits.float())
        x = c_prob.transpose(1, 2)  # (B, 12, T)

        sh_pad = self.short_frames // 2
        c_short = F.avg_pool1d(
            x, kernel_size=self.short_frames, stride=1, padding=sh_pad, count_include_pad=False
        )

        # Geometric mean over ~520ms (39 frames @ 75 Hz): stays high (~0.8) for sustained tones,
        # collapses (<0.08) for transient 120-300ms passing melody notes.
        su_pad = self.sustain_frames // 2
        log_c = torch.log(c_short.clamp(min=1e-4))
        geo_sustain = torch.exp(
            F.avg_pool1d(
                log_c, kernel_size=self.sustain_frames, stride=1, padding=su_pad, count_include_pad=False
            )
        )
        gate = torch.sigmoid(self.alpha.float() * (geo_sustain - self.threshold.float()))
        sustain = (geo_sustain * gate).transpose(1, 2)
        return sustain.to(orig_dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embedding to `x` of shape `(B, H, T, D)`."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)


class RoPEMultiHeadAttention(nn.Module):
    """Pure-tensor Multi-Head Self-Attention with Rotary Position Embedding (100% ONNX opset-17 native)."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if dim % (n_heads * 2) != 0:
            raise ValueError(f"dim ({dim}) must be divisible by 2 * n_heads ({2 * n_heads})")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # (B, H, T, D)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]  # (T, D/2)
        cos = torch.cos(freqs)[None, None, :, :].to(q.dtype)
        sin = torch.sin(freqs)[None, None, :, :].to(q.dtype)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if not torch.onnx.is_in_onnx_export():
            drop_p = self.attn_drop.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
        else:
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1).to(v.dtype)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(bsz, seq_len, self.dim)
        return self.out_proj(out)


class ConformerConvModule(nn.Module):
    """Depthwise-separable GLU Convolution module using GroupNorm (batch-size & sequence-length agnostic)."""

    def __init__(self, dim: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw_in = nn.Linear(dim, 2 * dim)
        self.dw_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        n_groups = 16 if dim % 16 == 0 else 1
        self.gn = nn.GroupNorm(n_groups, dim)
        self.act = nn.SiLU()
        self.pw_out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw_in(self.norm(x))
        a, b = y.chunk(2, dim=-1)
        y = a * torch.sigmoid(b)  # GLU
        y = self.dw_conv(y.transpose(1, 2))
        y = self.act(self.gn(y)).transpose(1, 2)
        return self.drop(self.pw_out(y))


class FeedForwardMacaron(nn.Module):

    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiConformerBlock(nn.Module):
    """Macaron-style Conformer block capturing both bar-level harmonic progression and beat-level rhythm."""

    def __init__(self, dim: int = 256, n_heads: int = 4, conv_kernel: int = 15, dropout: float = 0.15):
        super().__init__()
        self.ff1 = FeedForwardMacaron(dim, mult=2, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = RoPEMultiHeadAttention(dim, n_heads=n_heads, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardMacaron(dim, mult=2, dropout=dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn_drop(self.attn(self.attn_norm(x)))
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class BiConformerEncoder(nn.Module):

    def __init__(
        self,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BiConformerBlock(dim=dim, n_heads=n_heads, conv_kernel=conv_kernel, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
