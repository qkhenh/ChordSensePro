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


# ─────────────────────────────────────────────────────────────────────────────
# Model v6 Modules: DropPath, DualScalePassingSustainLayer, Parallel-Kernel
# Bi-Conformer (k=3 || k=15) & FullSongCrossSectionAttention (stride=15)
# ─────────────────────────────────────────────────────────────────────────────


class DropPath(nn.Module):
    """Stochastic Depth per sample along the batch dimension (no-op at eval / ONNX export)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training or torch.onnx.is_in_onnx_export():
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * rand.floor()


class DualScalePassingSustainLayer(nn.Module):
    """Dual-scale harmonic sustain & onset-gated passing-chord detector.

    Computes:
    - `sustain_env (B, T, 12)`: geometric mean over `~520ms` (`39 frames`) rejecting 1-note melody runs.
    - `fast_chord_env (B, T, 12)`: short `~120ms` (`9 frames`) harmonic envelope gated by
      `passing_gate(t) = max(sigmoid(3.0 * (harm_onset_count - 1.65)), p_bass_chg)`:
        * `>= 2` simultaneous pitch onsets in `C2..G4` (passing chord!) -> `passing_gate -> ~0.75..0.99`.
        * `1` single-note upper-register melody run (`C4+`) with steady bass -> `passing_gate -> ~0.10`.
        * `1` single bass floor step-down (`G -> D/F# -> Em7`) -> `p_bass_chg` opens `passing_gate`!
    """

    def __init__(self, short_frames: int = 9, sustain_frames: int = 39):
        super().__init__()
        self.short_frames = short_frames
        self.sustain_frames = sustain_frames
        self.alpha = nn.Parameter(torch.tensor(8.0))
        self.threshold = nn.Parameter(torch.tensor(0.28))
        self.passing_slope = nn.Parameter(torch.tensor(3.0))
        self.passing_thresh = nn.Parameter(torch.tensor(1.65))

    def forward(
        self,
        chroma_logits: torch.Tensor,
        harm_onset_count: torch.Tensor | None = None,
        p_bass_chg: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_dtype = chroma_logits.dtype
        bsz, seq_len, _ = chroma_logits.shape
        c_prob = torch.sigmoid(chroma_logits.float())
        x = c_prob.transpose(1, 2)  # (B, 12, T)

        sh_pad = self.short_frames // 2
        c_short = F.avg_pool1d(
            x, kernel_size=self.short_frames, stride=1, padding=sh_pad, count_include_pad=False
        )

        su_pad = self.sustain_frames // 2
        log_c = torch.log(c_short.clamp(min=1e-4))
        geo_sustain = torch.exp(
            F.avg_pool1d(
                log_c, kernel_size=self.sustain_frames, stride=1, padding=su_pad, count_include_pad=False
            )
        )
        gate_sus = torch.sigmoid(self.alpha.float() * (geo_sustain - self.threshold.float()))
        sustain_env = (geo_sustain * gate_sus).transpose(1, 2)  # (B, T, 12)

        # Smooth onset trigger over short chord window (9 frames = 120ms) so fast 240-400ms chords open cleanly
        if harm_onset_count is None:
            harm_cnt = torch.zeros((bsz, seq_len, 1), device=chroma_logits.device, dtype=torch.float32)
        else:
            harm_cnt = F.max_pool1d(
                harm_onset_count.float().transpose(1, 2),
                kernel_size=self.short_frames,
                stride=1,
                padding=sh_pad,
            ).transpose(1, 2)

        if p_bass_chg is None:
            b_chg = torch.zeros((bsz, seq_len, 1), device=chroma_logits.device, dtype=torch.float32)
        else:
            b_chg = F.max_pool1d(
                p_bass_chg.float().transpose(1, 2),
                kernel_size=self.short_frames,
                stride=1,
                padding=sh_pad,
            ).transpose(1, 2)

        gate_harm = torch.sigmoid(self.passing_slope.float() * (harm_cnt - self.passing_thresh.float()))
        passing_gate = torch.maximum(gate_harm, b_chg).clamp(0.0, 1.0)  # (B, T, 1)

        fast_chord_env = c_short.transpose(1, 2) * passing_gate  # (B, T, 12)
        return sustain_env.to(orig_dtype), fast_chord_env.to(orig_dtype), passing_gate.to(orig_dtype)


class ParallelKernelConvModule(nn.Module):
    """Multi-scale depthwise-separable GLU conv combining BTC-style sharp `k=3` and beat-level `k=15`."""

    def __init__(self, dim: int, k_short: int = 3, k_long: int = 15, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pw_in = nn.Linear(dim, 2 * dim)
        self.dw_short = nn.Conv1d(dim, dim, kernel_size=k_short, padding=k_short // 2, groups=dim)
        self.dw_long = nn.Conv1d(dim, dim, kernel_size=k_long, padding=k_long // 2, groups=dim)
        self.branch_mix = nn.Linear(2 * dim, dim)
        n_groups = 16 if dim % 16 == 0 else 1
        self.gn = nn.GroupNorm(n_groups, dim)
        self.act = nn.SiLU()
        self.pw_out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw_in(self.norm(x))
        a, b = y.chunk(2, dim=-1)
        y = (a * torch.sigmoid(b)).transpose(1, 2)  # (B, D, T)
        y_s = self.dw_short(y).transpose(1, 2)
        y_l = self.dw_long(y).transpose(1, 2)
        y_m = self.branch_mix(torch.cat([y_s, y_l], dim=-1)).transpose(1, 2)
        y_m = self.act(self.gn(y_m)).transpose(1, 2)
        return self.drop(self.pw_out(y_m))


class BiConformerBlockV6(nn.Module):
    """v6 Macaron Conformer block with Parallel-Kernel (`k=3 || k=15`) Conv and Stochastic Depth (`DropPath`)."""

    def __init__(
        self,
        dim: int = 256,
        n_heads: int = 4,
        k_short: int = 3,
        k_long: int = 15,
        dropout: float = 0.15,
        drop_path: float = 0.15,
    ):
        super().__init__()
        self.ff1 = FeedForwardMacaron(dim, mult=2, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = RoPEMultiHeadAttention(dim, n_heads=n_heads, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ParallelKernelConvModule(dim, k_short=k_short, k_long=k_long, dropout=dropout)
        self.ff2 = FeedForwardMacaron(dim, mult=2, dropout=dropout)
        self.dp = DropPath(drop_path)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp(0.5 * self.ff1(x))
        x = x + self.dp(self.attn_drop(self.attn(self.attn_norm(x))))
        x = x + self.dp(self.conv(x))
        x = x + self.dp(0.5 * self.ff2(x))
        return self.final_norm(x)


class BiConformerEncoderV6(nn.Module):

    def __init__(
        self,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        k_short: int = 3,
        k_long: int = 15,
        dropout: float = 0.15,
        drop_path: float = 0.15,
    ):
        super().__init__()
        rates = [
            drop_path * float(i) / max(n_layers - 1, 1)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList([
            BiConformerBlockV6(
                dim=dim,
                n_heads=n_heads,
                k_short=k_short,
                k_long=k_long,
                dropout=dropout,
                drop_path=rates[i],
            )
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FullSongCrossSectionAttention(nn.Module):
    """Lightweight song-wide cross-section attention at `stride=15` (`~0.20s` per bar/beat token).

    Allows high-register Intro Solo-Piano sections (`t=0s`) to attend directly to Full-Band Chorus
    and Outro sections across an entire 4–5 minute track while staying 100% ONNX opset-17 native
    and memory-light (`~1,285` tokens for a 257s song).
    """

    def __init__(
        self,
        dim: int = 256,
        stride: int = 15,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.stride = max(1, int(stride))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": RoPEMultiHeadAttention(dim, n_heads=n_heads, dropout=dropout),
                "drop": nn.Dropout(dropout),
                "ff": FeedForwardMacaron(dim, mult=2, dropout=dropout),
                "dp": DropPath(drop_path),
            })
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)
        # Zero-initialized residual gate so early epochs remain stable and gradually open cross-section context
        self.res_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        if self.stride > 1 and seq_len > self.stride:
            coarse = F.avg_pool1d(
                x.transpose(1, 2),
                kernel_size=self.stride,
                stride=self.stride,
                ceil_mode=True,
            ).transpose(1, 2)
        else:
            coarse = x

        h = coarse
        for blk in self.blocks:
            h = h + blk["dp"](blk["drop"](blk["attn"](blk["norm1"](h))))
            h = h + blk["dp"](blk["ff"](h))
        h = self.out_norm(h)

        if h.shape[1] != seq_len:
            h_up = F.interpolate(
                h.float().transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).to(x.dtype)
        else:
            h_up = h

        return x + torch.tanh(self.res_gate).to(x.dtype) * h_up
