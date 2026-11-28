from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.config import ModelCfg
from src.conformer import (
    BiConformerEncoder,
    BiConformerEncoderV6,
    DualScalePassingSustainLayer,
    FullSongCrossSectionAttention,
    MultiScaleSustainLayer,
)
from src.crf import DynamicBoundaryCRF, MarginalLinearChainCRF
from src.music_theory import (
    N_BASS_CLASSES,
    N_CRF_STATES,
    N_CRF_STATES_V6,
    N_INVERSIONS_V6,
    N_KEYS,
    N_ROOT_CLASSES_V5,
    N_ROOTS,
    N_SKELETONS,
    N_SKELETONS_V6,
    N_TENSIONS_V6,
    N_TYPES,
    REL_PITCH_INDEX,
    SKEL_TYPE_MASK,
    TONNETZ_MATRIX,
    build_functional_crf_prior_v6,
)

# Exact MERT-v1-330M / Wav2Vec2 7-layer conv feature extractor geometry
MERT_CONV_KERNELS: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
MERT_CONV_STRIDES: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
MERT_HOP_SAMPLES: int = 320        # product of MERT_CONV_STRIDES
MERT_RECEPTIVE_SAMPLES: int = 400  # receptive field of the 7 conv layers
NORM_STD_FLOOR: float = 1.8e-3     # -55 dBFS floor preventing silent hiss amplification


def mert_output_length(
    input_length: int,
    conv_kernels: tuple[int, ...] = MERT_CONV_KERNELS,
    conv_strides: tuple[int, ...] = MERT_CONV_STRIDES,
) -> int:
    """Compute exact temporal sequence length output by MERT's CNN feature extractor."""
    length = int(input_length)
    for k, s in zip(conv_kernels, conv_strides):
        length = (length - k) // s + 1
    return max(0, length)


def mert_frame_times_ms(
    crop_start_ms: float,
    n_frames: int,
    sr: int = 24000,
) -> np.ndarray:
    """Return center timestamp (ms) of each MERT output frame."""
    center_samples = np.arange(n_frames, dtype=np.float64) * MERT_HOP_SAMPLES + (MERT_RECEPTIVE_SAMPLES / 2.0)
    return crop_start_ms + center_samples * (1000.0 / float(sr))


def mert_chunked_frame_times_ms(
    crop_start_ms: float,
    n_chunks: int,
    chunk_samples: int,
    sr: int = 24000,
) -> np.ndarray:
    """Return exact center timestamps (ms) across `n_chunks` concatenated MERT chunks."""
    frames_per_chunk = mert_output_length(chunk_samples)
    chunk_ms = chunk_samples * (1000.0 / float(sr))
    parts = [
        mert_frame_times_ms(crop_start_ms + k * chunk_ms, frames_per_chunk, sr=sr)
        for k in range(n_chunks)
    ]
    return np.concatenate(parts, axis=0) if parts else np.empty(0, dtype=np.float64)


def compute_chunked_frame_rms_db(
    wav_float: np.ndarray,
    n_chunks: int,
    chunk_samples: int,
) -> np.ndarray:
    """Compute pre-normalization dBFS RMS per MERT frame `(n_chunks * T_chunk,)` from `wav_float` in `[-1, 1]`."""
    frames_per_chunk = mert_output_length(chunk_samples)
    out = np.full(n_chunks * frames_per_chunk, -80.0, dtype=np.float32)
    for k in range(n_chunks):
        seg = wav_float[k * chunk_samples: (k + 1) * chunk_samples]
        needed = frames_per_chunk * MERT_HOP_SAMPLES
        if len(seg) < needed:
            seg = np.pad(seg, (0, needed - len(seg)))
        frames = seg[:needed].reshape(frames_per_chunk, MERT_HOP_SAMPLES).astype(np.float32)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        # Smooth with a ~40ms (3-frame) moving average so zero-crossings of low bass notes don't dip falsely
        rms_smooth = np.convolve(np.pad(rms, (1, 1), mode="edge"), np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="valid")
        out[k * frames_per_chunk: (k + 1) * frames_per_chunk] = 20.0 * np.log10(np.maximum(rms_smooth, 1e-6))
    return out


class TemporalContextBlock(nn.Module):
    """Lightweight depthwise-grouped Conv1d + channel mixing residual block for frame-level context (v4)."""

    def __init__(self, dim: int, kernel_size: int = 5, groups: int = 16, dropout: float = 0.1):
        super().__init__()
        eff_groups = groups if dim % groups == 0 else 1
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=eff_groups,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.proj(self.drop(self.act(y)))
        return self.norm(residual + self.drop(y))


class ChordHead(nn.Module):

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExtensionHead(nn.Module):

    def __init__(self, in_dim: int, hidden: int, mid: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Linear(mid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChordModel(nn.Module):
    """Legacy clip-level (v2/v3) & 4-head marginal frame-level (v4) model kept for full backward compatibility."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg

        self.mert = AutoModel.from_pretrained(cfg.name, trust_remote_code=True)
        for p in self.mert.parameters():
            p.requires_grad = False

        self._captured: list[torch.Tensor] = []
        self._register_hooks()

        self.layer_weights = nn.Parameter(torch.zeros(cfg.n_layers))

        d = cfg.hidden_dim
        self.temporal_context = (
            TemporalContextBlock(d, kernel_size=5, groups=16, dropout=cfg.dropout)
            if cfg.frame_level
            else None
        )
        self.head_root = ChordHead(d, cfg.head_hidden, cfg.n_root)
        self.head_quality = ChordHead(d + cfg.n_root, cfg.head_hidden, cfg.n_quality)
        self.head_ext = ExtensionHead(
            d + cfg.n_root + cfg.n_quality,
            cfg.ext_head_hidden,
            cfg.ext_head_mid,
            cfg.n_ext,
            cfg.dropout,
        )
        self.head_bass = ChordHead(
            d + cfg.n_root + cfg.n_quality + cfg.n_ext,
            cfg.bass_head_hidden,
            cfg.n_bass,
        )

    def _register_hooks(self) -> None:
        def make_hook():
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if isinstance(h, torch.Tensor):
                    self._captured.append(h.detach() if not h.requires_grad else h)
            return hook

        fp = getattr(self.mert, "feature_projection", None)
        if fp is not None:
            fp.register_forward_hook(make_hook())

        encoder = getattr(self.mert, "encoder", None)
        layers = getattr(encoder, "layers", []) if encoder is not None else []
        for layer in layers:
            layer.register_forward_hook(make_hook())

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._captured = []
        out = self.mert(wav)

        n = self.cfg.n_layers
        if len(self._captured) >= n:
            hidden_states = torch.stack(self._captured[:n])
        elif len(self._captured) > 0:
            pad = [self._captured[-1]] * (n - len(self._captured))
            hidden_states = torch.stack(self._captured + pad)
        else:
            lhs = out.last_hidden_state
            hidden_states = lhs.unsqueeze(0).expand(n, -1, -1, -1)

        w = F.softmax(self.layer_weights, dim=0)
        h = (w[:, None, None, None] * hidden_states).sum(0)

        if self.cfg.frame_level:
            emb = self.temporal_context(h) if self.temporal_context is not None else h
        else:
            emb = self._pool(h)

        root_logits = self.head_root(emb)
        root_src = root_logits.detach() if self.cfg.detach_logits else root_logits
        root_prob = F.softmax(root_src, dim=-1)
        quality_logits = self.head_quality(torch.cat([emb, root_prob], dim=-1))

        qual_src = quality_logits.detach() if self.cfg.detach_logits else quality_logits
        qual_prob = F.softmax(qual_src, dim=-1)
        ext_logits = self.head_ext(torch.cat([emb, root_prob, qual_prob], dim=-1))

        ext_src = ext_logits.detach() if self.cfg.detach_logits else ext_logits
        ext_prob = F.softmax(ext_src, dim=-1)
        bass_logits = self.head_bass(torch.cat([emb, root_prob, qual_prob, ext_prob], dim=-1))

        return root_logits, quality_logits, ext_logits, bass_logits

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if self.cfg.pooling == "mean":
            return h.mean(dim=1)
        elif self.cfg.pooling == "center":
            t = h.shape[1]
            start = t // 4
            end = 3 * t // 4
            return h[:, start:end].mean(dim=1)
        else:
            return h.mean(dim=1)

    def apply_lora(self, lora_cfg) -> None:
        from peft import LoraConfig, get_peft_model
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            bias="none",
        )
        self.mert = get_peft_model(self.mert, peft_config)
        for name, p in self.mert.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    def get_trainable_params(self) -> list[dict]:
        head_params = []
        for name, p in self.named_parameters():
            if p.requires_grad and not name.startswith("mert."):
                head_params.append(p)

        lora_params = []
        for name, p in self.named_parameters():
            if p.requires_grad and name.startswith("mert."):
                lora_params.append(p)

        groups = [{"params": head_params, "lr_key": "lr_head"}]
        if lora_params:
            groups.append({"params": lora_params, "lr_key": "lr_lora"})
        return groups


class HarmonicCQTFilterbank(nn.Module):
    """Exact 5-Octave Constant-Q Wavelet Filterbank (`C1=32.7Hz .. B5=987.8Hz`, 60 semitones)
    sample-center-aligned with MERT's `400`-sample receptive field and `320`-sample hop (`75 Hz`).
    Uses only native PyTorch/ONNX ops (`F.pad`, `F.conv1d`, `amax`, `log1p`).
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = MERT_HOP_SAMPLES,
        receptive_samples: int = MERT_RECEPTIVE_SAMPLES,
        kernel_size: int = 6144,
        n_bins: int = 60,
        q_factor: float = 24.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.kernel_size = kernel_size
        self.n_bins = n_bins
        # Exact symmetric pad so `(N + 2*pad - kernel_size) // 320 + 1 == (N - 400) // 320 + 1`
        self.pad_left = (kernel_size - receptive_samples) // 2
        self.pad_right = kernel_size - receptive_samples - self.pad_left

        midi = np.arange(24, 24 + n_bins, dtype=np.float64)
        freqs = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
        kernels_cos = np.zeros((n_bins, 1, kernel_size), dtype=np.float32)
        kernels_sin = np.zeros((n_bins, 1, kernel_size), dtype=np.float32)
        center = (kernel_size - 1) / 2.0
        t_idx = np.arange(kernel_size, dtype=np.float64) - center

        for i, f_hz in enumerate(freqs):
            win_len = int(min(kernel_size, max(256, round(q_factor * sample_rate / f_hz))))
            half = win_len / 2.0
            mask = np.abs(t_idx) <= half
            local_n = (t_idx[mask] + half) / max(win_len, 1)
            win = 0.5 - 0.5 * np.cos(2.0 * np.pi * local_n)
            win = win / (np.sum(win) + 1e-8) * 2.0
            phase = 2.0 * np.pi * f_hz * t_idx[mask] / sample_rate
            kernels_cos[i, 0, mask] = (win * np.cos(phase)).astype(np.float32)
            kernels_sin[i, 0, mask] = (win * np.sin(phase)).astype(np.float32)

        weight = np.concatenate([kernels_cos, kernels_sin], axis=0)  # (120, 1, K)
        self.register_buffer("weight", torch.from_numpy(weight), persistent=False)
        self._log_norm = float(np.log1p(15.0))

    def forward(self, wav_chunk: torch.Tensor) -> torch.Tensor:
        """Transform `(B_chunks, chunk_samples)` -> `(B_chunks, T_chunk, 60)` in `[0, 1]`."""
        x = F.pad(wav_chunk.float().unsqueeze(1), (self.pad_left, self.pad_right))
        spec = F.conv1d(x, self.weight, stride=self.hop_length)  # (B, 120, T_chunk)
        real = spec[:, : self.n_bins, :]
        imag = spec[:, self.n_bins :, :]
        mag = torch.sqrt(real * real + imag * imag + 1e-12).transpose(1, 2)  # (B, T_chunk, 60)
        ref = mag.amax(dim=-1, keepdim=True).clamp(min=2e-3)
        cqt_log = torch.log1p((mag / ref) * 15.0) / self._log_norm
        return cqt_log.clamp(0.0, 1.0)


class BassPedalHoldLayer(nn.Module):
    """Causal leaky peak-hold envelope (`tau = 1.15s` over `1.60s` horizon) preserving Beat-1
    piano/guitar left-hand bass attacks across Beats 2-4 while reacting with `0ms` lag to new bass notes.
    Uses pure dynamic-time-safe ops (`F.pad`, `F.max_pool1d`, constant negative slices, `amax`).
    """

    def __init__(self, block_frames: int = 8, n_taps: int = 16, tau_frames: float = 86.0):
        super().__init__()
        self.block_frames = block_frames
        self.n_taps = n_taps
        decays = [float(np.exp(-(block_frames * i) / float(tau_frames))) for i in range(n_taps)]
        self.decays = tuple(decays)

    def forward(self, phys_bass_logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(phys_bass_logits.float())          # (B, T, 12)
        x = prob.transpose(1, 2)                                # (B, 12, T)
        p_blk = F.max_pool1d(F.pad(x, (self.block_frames - 1, 0)), kernel_size=self.block_frames, stride=1)
        taps = [x]
        for i in range(1, self.n_taps):
            shift = self.block_frames * i
            shifted = F.pad(p_blk, (shift, 0))[:, :, :-shift] * self.decays[i]
            taps.append(shifted)
        hold = torch.stack(taps, dim=0).amax(dim=0).transpose(1, 2)
        return hold.clamp(0.0, 1.0).to(phys_bass_logits.dtype)


class OctavePitchBassModule(nn.Module):
    """Pitch-equivariant 5-Octave (`C1..B5`) physical bass detector & harmonic register encoder.
    Inspects all 5 octaves relative to each candidate pitch class `p in 0..11` so it sees whether `p`
    has a true low fundamental (`E2` or `E3`), harmonic overtone support (`+12`, `+19`), or another
    note sounding strictly below it (`E3` below `C4`).
    """

    def __init__(self, n_octaves: int = 5, cqt_ctx_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_octaves = n_octaves
        in_per_pc = n_octaves * 12 + n_octaves  # 60 relative octave-semitones + 5 f0 onset-flux bins = 65
        self.phys_mlp = nn.Sequential(
            nn.Linear(in_per_pc, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.cqt_proj = nn.Sequential(
            nn.Linear(n_octaves * 12 + 12 + 12, cqt_ctx_dim),
            nn.GELU(),
            nn.LayerNorm(cqt_ctx_dim),
        )
        self.register_buffer("rel_idx", torch.tensor(REL_PITCH_INDEX, dtype=torch.long), persistent=False)

    def forward(
        self, cqt: torch.Tensor, hold_layer: BassPedalHoldLayer
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = cqt.shape
        cqt_f = cqt.float()
        flux = F.pad(F.relu(cqt_f[:, 1:, :] - cqt_f[:, :-1, :]), (0, 0, 1, 0))  # (B, T, 60)

        cqt_5x12 = cqt_f.reshape(bsz, seq_len, self.n_octaves, 12)
        flux_5x12 = flux.reshape(bsz, seq_len, self.n_octaves, 12)

        # Gather relative 5x12 harmonic spectrum for each candidate pitch class p in 0..11:
        # cqt_5x12[:, :, :, rel_idx] -> (B, T, 5_oct, 12_cand_pc, 12_rel_int) -> (B, T, 12_cand_pc, 60)
        rel_cqt = cqt_5x12[:, :, :, self.rel_idx].permute(0, 1, 3, 2, 4).reshape(bsz, seq_len, 12, self.n_octaves * 12)
        flux_f0 = flux_5x12.permute(0, 1, 3, 2)  # (B, T, 12_cand_pc, 5_oct)

        phys_in = torch.cat([rel_cqt, flux_f0], dim=-1).to(cqt.dtype)
        phys_bass_logits = self.phys_mlp(phys_in).squeeze(-1)  # (B, T, 12)
        phys_bass_prob = torch.sigmoid(phys_bass_logits.float()).to(cqt.dtype)
        bass_hold = hold_layer(phys_bass_logits)               # (B, T, 12)

        cqt_ctx = self.cqt_proj(torch.cat([cqt, phys_bass_prob, bass_hold], dim=-1))
        return phys_bass_logits, phys_bass_prob, bass_hold, cqt_ctx


class ChordModelV5(nn.Module):
    """Jazz-Ready Model v5 / v5.1 (Bass-First Joint Context Architecture):
    - Chunked 5s MERT Acoustic Front-End -> 256d frame sequence + Exact 5-Octave `HarmonicCQTFilterbank` (`C1..B5`)
    - Pitch-Equivariant `OctavePitchBassModule` + Causal `BassPedalHoldLayer` (`0.45s` + `1.50s`)
    - Dual-Register Pitch Chroma (`c_low`, `c_full`) + Multi-Scale Sustain Filter (`80ms` vs `520ms`)
    - 4-Layer RoPE Bi-Conformer (`20s` train / Full-Song serve) + Time-Varying `LocalKeyHead` (24 keys)
    - Bottom-Up `TrueBassHead (0..11 + N)` -> Bass-Conditioned `RootHead` -> `ChordTypeHead` (`493` states)
    - 145-state Guide-Tone Skeleton Marginal Linear-Chain CRF + Slash-Inversion `BassHead`
    """

    def __init__(self, cfg: ModelCfg, sample_rate: int = 24000):
        super().__init__()
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.chunk_samples = cfg.chunk_s * sample_rate
        self.bass_first = bool(getattr(cfg, "bass_first", True))
        self.cqt_bins = int(getattr(cfg, "cqt_bins", 60))

        self.mert = AutoModel.from_pretrained(cfg.name, trust_remote_code=True)
        for p in self.mert.parameters():
            p.requires_grad = False

        self._captured: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

        self.layer_weights = nn.Parameter(torch.zeros(cfg.n_layers))

        c_dim = cfg.context_dim
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, c_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(c_dim),
        )

        # Dual-register pitch-class bottleneck + sustain envelope
        self.head_chroma_low = nn.Sequential(
            nn.Linear(c_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 12),
        )
        self.head_chroma_full = nn.Sequential(
            nn.Linear(c_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 12),
        )
        self.sustain_layer = MultiScaleSustainLayer(short_frames=7, sustain_frames=39)

        cqt_ctx_dim = 64
        if self.bass_first:
            self.cqt_bank = HarmonicCQTFilterbank(
                sample_rate=sample_rate,
                hop_length=MERT_HOP_SAMPLES,
                n_bins=self.cqt_bins,
            )
            self.bass_hold_layer = BassPedalHoldLayer()
            self.oct_bass = OctavePitchBassModule(
                n_octaves=self.cqt_bins // 12,
                cqt_ctx_dim=cqt_ctx_dim,
                dropout=min(cfg.dropout, 0.15),
            )
            conf_in_dim = c_dim + cqt_ctx_dim + 36
        else:
            conf_in_dim = c_dim + 36

        # Long-range musical context encoder
        self.conf_fusion = nn.Sequential(
            nn.Linear(conf_in_dim, c_dim),
            nn.LayerNorm(c_dim),
        )
        self.conformer = BiConformerEncoder(
            dim=c_dim,
            n_layers=cfg.conformer_layers,
            n_heads=cfg.conformer_heads,
            conv_kernel=cfg.conformer_conv_kernel,
            dropout=min(cfg.dropout, 0.2),
        )

        # Time-varying Local Key Head (24 classes: 0..11 Major, 12..23 Minor)
        self.head_key = nn.Sequential(
            nn.Linear(c_dim, 128),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, N_KEYS),
        )

        root_ctx_dim = 128
        self.root_ctx_proj = nn.Linear(c_dim, root_ctx_dim)
        self.root_emb = nn.Embedding(N_ROOTS, root_ctx_dim)

        if self.bass_first:
            # Stage 1 (Bottom-Up): Physical TrueBassHead (0..11 actual sounding bass on 100% of chords, 12 = N)
            tb_in_dim = c_dim + 12 + 12 + 12 + 1  # z + phys_bass_prob + bass_hold + c_low + norm_rms
            self.head_true_bass = nn.Sequential(
                nn.Linear(tb_in_dim, cfg.bass_head_hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.bass_head_hidden, N_ROOT_CLASSES_V5),
            )
            self.phys_bass_res_scale = nn.Parameter(torch.tensor(0.5))

            # Stage 2: Bass-Conditioned Root + Silence (`N`=12) Head
            root_in_dim = c_dim + N_ROOT_CLASSES_V5 + 12 + 12 + 12 + 12 + N_KEYS + 1
            self.head_root = nn.Sequential(
                nn.Linear(root_in_dim, cfg.head_hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_hidden, N_ROOT_CLASSES_V5),
            )

            # Shared Pitch-Equivariant Relative Tensor `u`:
            # z_r(128) + rel(true_bass_12, bass_hold, c_low, c_full, sustain, k_maj, k_min) = 212
            type_in_dim = root_ctx_dim + 7 * 12
            self.head_root_rel = nn.Sequential(
                nn.Linear(type_in_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
            nn.init.normal_(self.head_root_rel[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.head_root_rel[-1].bias)

            self.head_type = nn.Sequential(
                nn.Linear(type_in_dim, cfg.type_head_hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.type_head_hidden, 96),
                nn.SiLU(),
                nn.Linear(96, N_TYPES),
            )

            # Stage 3: Slash-Inversion Head (0..11 when bass != root, 12 = root-position / rootless jazz voicing)
            bass_in_dim = c_dim + N_ROOT_CLASSES_V5 + 12 + 12 + N_ROOT_CLASSES_V5 + N_SKELETONS
            self.head_bass = nn.Sequential(
                nn.Linear(bass_in_dim, cfg.bass_head_hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.bass_head_hidden, N_BASS_CLASSES),
            )
        else:
            root_in_dim = c_dim + 12 + 12 + 12 + N_KEYS + 1
            self.head_root = nn.Sequential(
                nn.Linear(root_in_dim, cfg.head_hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_hidden, N_ROOT_CLASSES_V5),
            )
            type_in_dim = root_ctx_dim + 5 * 12
            self.head_type = nn.Sequential(
                nn.Linear(type_in_dim, cfg.type_head_hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.type_head_hidden, 96),
                nn.SiLU(),
                nn.Linear(96, N_TYPES),
            )
            bass_in_dim = c_dim + 12 + N_ROOT_CLASSES_V5 + N_SKELETONS
            self.head_bass = nn.Sequential(
                nn.Linear(bass_in_dim, cfg.bass_head_hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.bass_head_hidden, N_BASS_CLASSES),
            )

        # 145-state Guide-Tone Skeleton Linear-Chain CRF
        self.crf = MarginalLinearChainCRF(num_tags=N_CRF_STATES, stay_bonus=2.4)

        self.register_buffer("rel_idx", torch.tensor(REL_PITCH_INDEX, dtype=torch.long), persistent=False)
        self.register_buffer("skel_type_mask", torch.tensor(SKEL_TYPE_MASK, dtype=torch.bool), persistent=False)

    def _clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _register_hooks(self) -> None:
        self._clear_hooks()
        # Unwrap peft base model if wrapped so hooks attach cleanly once
        base = getattr(self.mert, "base_model", self.mert)
        base = getattr(base, "model", base)

        def make_hook():
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if isinstance(h, torch.Tensor):
                    self._captured.append(h.detach() if not h.requires_grad else h)
            return hook

        fp = getattr(base, "feature_projection", None)
        if fp is not None:
            self._hooks.append(fp.register_forward_hook(make_hook()))

        encoder = getattr(base, "encoder", None)
        layers = getattr(encoder, "layers", []) if encoder is not None else []
        for layer in layers:
            self._hooks.append(layer.register_forward_hook(make_hook()))

    def _encode_single_chunk_batch(self, wav_chunk: torch.Tensor) -> torch.Tensor:
        """Run MERT on `(B_chunks, chunk_samples)` -> `(B_chunks, T_chunk, context_dim)`."""
        self._captured = []
        out = self.mert(wav_chunk)

        n = self.cfg.n_layers
        if len(self._captured) >= n:
            hidden_states = torch.stack(self._captured[:n])
        elif len(self._captured) > 0:
            pad = [self._captured[-1]] * (n - len(self._captured))
            hidden_states = torch.stack(self._captured + pad)
        else:
            lhs = out.last_hidden_state
            hidden_states = lhs.unsqueeze(0).expand(n, -1, -1, -1)

        w = F.softmax(self.layer_weights, dim=0).to(hidden_states.dtype)
        h = (w[:, None, None, None] * hidden_states).sum(0)
        return self.input_proj(h)

    def encode_chunks(self, wav: torch.Tensor) -> torch.Tensor:
        """Encode `(B, K * chunk_samples)` (or `(B, chunk_samples)`) into `(B, K * T_chunk, context_dim)`."""
        bsz, total_samples = wav.shape
        if total_samples > self.chunk_samples and total_samples % self.chunk_samples == 0:
            n_chunks = total_samples // self.chunk_samples
            flat = wav.reshape(bsz * n_chunks, self.chunk_samples)
            emb_flat = self._encode_single_chunk_batch(flat)
            return emb_flat.reshape(bsz, n_chunks * emb_flat.shape[1], emb_flat.shape[2])
        return self._encode_single_chunk_batch(wav)

    def encode_cqt(self, wav: torch.Tensor) -> torch.Tensor | None:
        """Compute frame-aligned 5-octave Constant-Q spectrum `(B, K * T_chunk, 60)` from `wav`."""
        if not self.bass_first:
            return None
        bsz, total_samples = wav.shape
        if total_samples > self.chunk_samples and total_samples % self.chunk_samples == 0:
            n_chunks = total_samples // self.chunk_samples
            flat = wav.reshape(bsz * n_chunks, self.chunk_samples)
            cqt_flat = self.cqt_bank(flat)
            return cqt_flat.reshape(bsz, n_chunks * cqt_flat.shape[1], cqt_flat.shape[2])
        return self.cqt_bank(wav)

    def decode_context(
        self,
        emb: torch.Tensor,
        frame_rms_db: torch.Tensor | None = None,
        cqt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run full musical context & bass-first root-conditioned joint heads on `emb` of shape `(B, T, context_dim)`."""
        bsz, seq_len, _ = emb.shape
        if frame_rms_db is None:
            frame_rms_db = torch.full((bsz, seq_len), -20.0, device=emb.device, dtype=torch.float32)

        low_chroma_logits = self.head_chroma_low(emb)
        full_chroma_logits = self.head_chroma_full(emb)
        c_low = torch.sigmoid(low_chroma_logits.float()).to(emb.dtype)
        c_full = torch.sigmoid(full_chroma_logits.float()).to(emb.dtype)
        sustain_env = self.sustain_layer(full_chroma_logits)

        phys_bass_logits: torch.Tensor | None = None
        phys_bass_prob: torch.Tensor | None = None
        bass_hold: torch.Tensor | None = None

        if self.bass_first:
            if cqt is None:
                cqt_in = torch.zeros((bsz, seq_len, self.cqt_bins), device=emb.device, dtype=emb.dtype)
            else:
                cqt_in = cqt.to(device=emb.device, dtype=emb.dtype)
            phys_bass_logits, phys_bass_prob, bass_hold, cqt_ctx = self.oct_bass(cqt_in, self.bass_hold_layer)
            conf_in = self.conf_fusion(torch.cat([emb, cqt_ctx, c_low, c_full, sustain_env], dim=-1))
        else:
            conf_in = self.conf_fusion(torch.cat([emb, c_low, c_full, sustain_env], dim=-1))

        z = self.conformer(conf_in)

        # ~4.0s local pooling (301 frames @ 75 Hz) for smooth time-varying Local Key estimation
        k_win = 301
        k_pad = k_win // 2
        z_key = F.avg_pool1d(
            z.float().transpose(1, 2), kernel_size=k_win, stride=1, padding=k_pad, count_include_pad=False
        ).transpose(1, 2).to(z.dtype)
        key_logits = self.head_key(z_key)
        key_prob = F.softmax(key_logits.float(), dim=-1).to(z.dtype)

        norm_rms = ((frame_rms_db.float().clamp(-80.0, 0.0) + 40.0) / 20.0).unsqueeze(-1).to(z.dtype)
        idx = self.rel_idx  # (12, 12)
        rel_c_low = c_low[:, :, idx]
        rel_c_full = c_full[:, :, idx]
        rel_sustain = sustain_env[:, :, idx]
        rel_k_maj = key_prob[:, :, :12][:, :, idx]
        rel_k_min = key_prob[:, :, 12:][:, :, idx]
        z_r = self.root_ctx_proj(z).unsqueeze(2) + self.root_emb.weight[None, None, :, :]

        true_bass_logits: torch.Tensor | None = None
        if self.bass_first and phys_bass_logits is not None and phys_bass_prob is not None and bass_hold is not None:
            # 1. Bottom-Up TrueBassHead: predicts actual physical sounding bass (0..11, 12 = N) FIRST
            tb_in = torch.cat([z, phys_bass_prob, bass_hold, c_low, norm_rms], dim=-1)
            tb_raw = self.head_true_bass(tb_in)
            tb_res = F.pad((self.phys_bass_res_scale.float() * phys_bass_logits.float()).to(tb_raw.dtype), (0, 1))
            true_bass_logits = tb_raw + tb_res
            true_bass_prob = F.softmax(true_bass_logits.float(), dim=-1).to(z.dtype)

            # 2. Pitch-Equivariant Rotation including TrueBass & BassPedalHold relative to candidate root `r`
            rel_tb = true_bass_prob[:, :, :12][:, :, idx]
            rel_bh = bass_hold[:, :, idx]
            u = torch.cat([z_r, rel_tb, rel_bh, rel_c_low, rel_c_full, rel_sustain, rel_k_maj, rel_k_min], dim=-1)

            # 3. Bass-Conditioned RootHead + Pitch-Equivariant Candidate-Root Score
            root_in = torch.cat([z, true_bass_prob, bass_hold, c_low, c_full, sustain_env, key_prob, norm_rms], dim=-1)
            r_rel = F.pad(self.head_root_rel(u).squeeze(-1), (0, 1))
            root_logits = self.head_root(root_in) + r_rel
            type_logits_by_root = self.head_type(u)
        else:
            root_in = torch.cat([z, c_low, c_full, sustain_env, key_prob, norm_rms], dim=-1)
            root_logits = self.head_root(root_in)
            u = torch.cat([z_r, rel_c_low, rel_c_full, rel_sustain, rel_k_maj, rel_k_min], dim=-1)
            type_logits_by_root = self.head_type(u)

        # Exact float32 joint log-probabilities & Skeleton marginalization
        root_logp = F.log_softmax(root_logits.float(), dim=-1)             # (B, T, 13)
        type_logp = F.log_softmax(type_logits_by_root.float(), dim=-1)     # (B, T, 12, 41)
        joint_chord_logp = root_logp[:, :, :12, None] + type_logp          # (B, T, 12, 41)

        # Marginalize 41 types -> 12 Skeleton Families per candidate root:
        # skel_type_mask has shape (12_skel, 41_type)
        masked_joint = joint_chord_logp.unsqueeze(3).masked_fill(
            ~self.skel_type_mask[None, None, None, :, :], -1e9
        )  # (B, T, 12_root, 12_skel, 41_type)
        skel_logp_by_root = torch.logsumexp(masked_joint, dim=-1)          # (B, T, 12_root, 12_skel)
        frame_skel_emissions = torch.cat(
            [root_logp[:, :, 12:13], skel_logp_by_root.reshape(bsz, seq_len, 144)],
            dim=-1,
        )  # (B, T, 145)

        stride = max(1, int(self.cfg.crf_stride))
        if stride > 1:
            crf_emissions = F.avg_pool1d(
                frame_skel_emissions.transpose(1, 2),
                kernel_size=stride,
                stride=stride,
                ceil_mode=True,
            ).transpose(1, 2)
        else:
            crf_emissions = frame_skel_emissions

        # Marginal skeleton distribution over 12 families (summed across roots) for Slash BassHead conditioning
        root_prob_13 = root_logp.exp().to(z.dtype)
        skel_prob_12 = torch.logsumexp(skel_logp_by_root, dim=2).exp().to(z.dtype)
        if self.bass_first and true_bass_logits is not None and bass_hold is not None:
            tb_prob_13 = F.softmax(true_bass_logits.float(), dim=-1).to(z.dtype)
            bass_in = torch.cat([z, tb_prob_13, bass_hold, c_low, root_prob_13, skel_prob_12], dim=-1)
        else:
            bass_in = torch.cat([z, c_low, root_prob_13, skel_prob_12], dim=-1)
        bass_logits = self.head_bass(bass_in)

        out_dict: dict[str, torch.Tensor] = {
            "root_logits": root_logits,
            "type_logits_by_root": type_logits_by_root,
            "joint_chord_logp": joint_chord_logp,
            "frame_skel_emissions": frame_skel_emissions,
            "crf_emissions": crf_emissions,
            "bass_logits": bass_logits,
            "key_logits": key_logits,
            "low_chroma_logits": low_chroma_logits,
            "full_chroma_logits": full_chroma_logits,
            "sustain_env": sustain_env,
        }
        if true_bass_logits is not None:
            out_dict["true_bass_logits"] = true_bass_logits
        if phys_bass_logits is not None:
            out_dict["phys_bass_logits"] = phys_bass_logits
        return out_dict

    def forward(
        self,
        wav: torch.Tensor,
        frame_rms_db: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emb = self.encode_chunks(wav)
        cqt = self.encode_cqt(wav)
        if frame_rms_db is not None and frame_rms_db.shape[1] != emb.shape[1]:
            T = emb.shape[1]
            if frame_rms_db.shape[1] > T:
                frame_rms_db = frame_rms_db[:, :T]
            else:
                frame_rms_db = F.pad(frame_rms_db, (0, T - frame_rms_db.shape[1]), value=-80.0)
        return self.decode_context(emb, frame_rms_db=frame_rms_db, cqt=cqt)

    def apply_lora(self, lora_cfg) -> None:
        from peft import LoraConfig, get_peft_model
        self._clear_hooks()
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            bias="none",
        )
        self.mert = get_peft_model(self.mert, peft_config)
        for name, p in self.mert.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
        self._register_hooks()

    def merge_lora(self) -> None:
        """Fuse LoRA adapters directly into MERT linear weights for zero-overhead ONNX export."""
        if hasattr(self.mert, "merge_and_unload"):
            self._clear_hooks()
            self.mert = self.mert.merge_and_unload()
            for p in self.mert.parameters():
                p.requires_grad = False
            self._register_hooks()

    def get_trainable_params(self) -> list[dict]:
        head_params = [
            p for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith("mert.")
        ]
        lora_params = [
            p for name, p in self.named_parameters()
            if p.requires_grad and name.startswith("mert.")
        ]
        groups = [{"params": head_params, "lr_key": "lr_head"}]
        if lora_params:
            groups.append({"params": lora_params, "lr_key": "lr_lora"})
        return groups


def build_model(cfg: ModelCfg, sample_rate: int = 24000) -> nn.Module:
    arch = getattr(cfg, "arch_version", 4)
    if arch >= 6:
        return ChordModelV6(cfg, sample_rate=sample_rate)
    if arch >= 5:
        return ChordModelV5(cfg, sample_rate=sample_rate)
    return ChordModel(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Model v6 Architecture: 96-Bin Harmonic CQT, 4.0s Onset-Reset BassPedalHold,
# Parallel-Kernel Conformer + Full-Song Cross-Section Attention, Dual-Scale
# Passing Sustain, 6-D Tonnetz, Boundary Head, 14-Skeleton × 7-Tension ×
# 5-Inversion Factorized Heads & 169-State DynamicBoundaryCRF
# ─────────────────────────────────────────────────────────────────────────────


class HarmonicCQTFilterbankV6(nn.Module):
    """High-resolution 96-bin Constant-Q Wavelet Filterbank sample-center-aligned with MERT (`75 Hz`).

    - Low 3 octaves (`C1..B3`, MIDI `24.0..59.5`): `72` bins at `24 bins/oct` (`0.5`-semitone resolution, `q=32`)
      sharply separating adjacent bass fundamentals (`F#2 = 92.5 Hz` vs `G2 = 98.0 Hz`) and slight live detuning.
    - Upper 2 octaves (`C4..B5`, MIDI `60..83`): `24` bins at `12 bins/oct` (`q=24`).
    100% ONNX opset-17 native (`F.pad`, `F.conv1d`, `sqrt`, `amax`, `log1p`).
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = MERT_HOP_SAMPLES,
        receptive_samples: int = MERT_RECEPTIVE_SAMPLES,
        kernel_size: int = 6144,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.kernel_size = kernel_size
        self.n_bins = 96
        self.pad_left = (kernel_size - receptive_samples) // 2
        self.pad_right = kernel_size - receptive_samples - self.pad_left

        midi_low = 24.0 + 0.5 * np.arange(72, dtype=np.float64)   # C1..B3 (24 bins/oct)
        midi_high = 60.0 + np.arange(24, dtype=np.float64)        # C4..B5 (12 bins/oct)
        midi = np.concatenate([midi_low, midi_high], axis=0)
        q_factors = np.concatenate([np.full(72, 32.0), np.full(24, 24.0)], axis=0)
        freqs = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

        kernels_cos = np.zeros((self.n_bins, 1, kernel_size), dtype=np.float32)
        kernels_sin = np.zeros((self.n_bins, 1, kernel_size), dtype=np.float32)
        center = (kernel_size - 1) / 2.0
        t_idx = np.arange(kernel_size, dtype=np.float64) - center

        for i, (f_hz, q_val) in enumerate(zip(freqs, q_factors)):
            win_len = int(min(kernel_size, max(256, round(q_val * sample_rate / f_hz))))
            half = win_len / 2.0
            mask = np.abs(t_idx) <= half
            local_n = (t_idx[mask] + half) / max(win_len, 1)
            win = 0.5 - 0.5 * np.cos(2.0 * np.pi * local_n)
            win = win / (np.sum(win) + 1e-8) * 2.0
            phase = 2.0 * np.pi * f_hz * t_idx[mask] / sample_rate
            kernels_cos[i, 0, mask] = (win * np.cos(phase)).astype(np.float32)
            kernels_sin[i, 0, mask] = (win * np.sin(phase)).astype(np.float32)

        weight = np.concatenate([kernels_cos, kernels_sin], axis=0)  # (192, 1, K)
        self.register_buffer("weight", torch.from_numpy(weight), persistent=False)
        self._log_norm = float(np.log1p(15.0))

    def forward(self, wav_chunk: torch.Tensor) -> torch.Tensor:
        """Transform `(B_chunks, chunk_samples)` -> `(B_chunks, T_chunk, 96)` in `[0, 1]`."""
        x = F.pad(wav_chunk.float().unsqueeze(1), (self.pad_left, self.pad_right))
        spec = F.conv1d(x, self.weight, stride=self.hop_length)  # (B, 192, T_chunk)
        real = spec[:, : self.n_bins, :]
        imag = spec[:, self.n_bins :, :]
        mag = torch.sqrt(real * real + imag * imag + 1e-12).transpose(1, 2)
        ref = mag.amax(dim=-1, keepdim=True).clamp(min=2e-3)
        cqt_log = torch.log1p((mag / ref) * 15.0) / self._log_norm
        return cqt_log.clamp(0.0, 1.0)


class BassPedalHoldLayerV6(nn.Module):
    """4.00-second Onset-Reset Causal Leaky Peak-Hold (`block=12, n_taps=25, tau=2.60s`).

    - Holds Beat-1 bass fundamentals cleanly across slow 65–72 BPM ballad bars (`3.4–4.0s`)
      and ascending piano arpeggios (`E2 -> B2 -> E3`).
    - Immediately resets old held bass within 1 frame whenever `bass_reset_prob` fires
      (e.g. descending stepwise bass `G2 -> F#2 -> E2` in `G -> D/F# -> Em7`).
    100% parallel ONNX opset-17 native (exact running max of reset gate over `(t - shift, t]`).
    """

    def __init__(self, block_frames: int = 12, n_taps: int = 25, tau_frames: float = 195.0):
        super().__init__()
        self.block_frames = block_frames
        self.n_taps = n_taps
        self.decays = tuple(float(np.exp(-(block_frames * i) / float(tau_frames))) for i in range(n_taps))

    def forward(
        self,
        phys_bass_logits: torch.Tensor,
        bass_reset_prob: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prob = torch.sigmoid(phys_bass_logits.float())  # (B, T, 12)
        x = prob.transpose(1, 2)                        # (B, 12, T)
        p_blk = F.max_pool1d(F.pad(x, (self.block_frames - 1, 0)), kernel_size=self.block_frames, stride=1)

        r_blk: torch.Tensor | None = None
        if bass_reset_prob is not None:
            r = bass_reset_prob.float().clamp(0.0, 1.0).transpose(1, 2)  # (B, 1, T)
            r_blk = F.max_pool1d(F.pad(r, (self.block_frames - 1, 0)), kernel_size=self.block_frames, stride=1)

        taps = [x]
        running_reset = r_blk
        for i in range(1, self.n_taps):
            shift = self.block_frames * i
            shifted = F.pad(p_blk, (shift, 0))[:, :, :-shift] * self.decays[i]
            if r_blk is not None and running_reset is not None:
                if i > 1:
                    prev_shift = self.block_frames * (i - 1)
                    r_shifted = F.pad(r_blk, (prev_shift, 0))[:, :, :-prev_shift]
                    running_reset = torch.maximum(running_reset, r_shifted)
                # Suppress past tap whenever a new bass onset occurred in (t - shift, t]
                shifted = shifted * (1.0 - 0.92 * running_reset)
            taps.append(shifted)

        hold = torch.stack(taps, dim=0).amax(dim=0).transpose(1, 2)
        return hold.clamp(0.0, 1.0).to(phys_bass_logits.dtype)


class OctavePitchBassModuleV6(nn.Module):
    """High-resolution 96-bin CQT bass & harmonic-onset module with Cumulative Energy Below (`E_below`).

    Solves both:
    - High-register Solo Piano (Octave 3–4 intro where Octave 1–2 is silent): `E_below` identifies
      the true lowest sounding pitch class regardless of octave!
    - Fast passing chords & stepwise slash bass: harmonic-suppressed onset flux separates real
      simultaneous chord attacks (`>= 2` pitches in `C2..G4`) and new low bass notes from
      piano 2nd/3rd overtones and ascending arpeggios.
    """

    def __init__(self, cqt_ctx_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        # Per candidate pitch class: 60 rel_cqt + 60 rel_e_below + 5 f0_harm_flux = 125
        in_per_pc = 60 + 60 + 5
        self.phys_mlp = nn.Sequential(
            nn.Linear(in_per_pc, 80),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(80, 1),
        )
        # Bass-floor reset detector on lowest-register harmonic-suppressed onsets weighted by (1 - E_below)
        self.bass_reset_mlp = nn.Sequential(
            nn.Linear(48, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.bass_reset_mlp[-1].bias, -2.2)

        self.cqt_proj = nn.Sequential(
            nn.Linear(96 + 12 + 12 + 1, cqt_ctx_dim),
            nn.GELU(),
            nn.LayerNorm(cqt_ctx_dim),
        )
        self.register_buffer("rel_idx", torch.tensor(REL_PITCH_INDEX, dtype=torch.long), persistent=False)
        inv_norm = 1.0 / torch.arange(1, 61, dtype=torch.float32).clamp(min=1.0)
        self.register_buffer("eb_norm", inv_norm, persistent=False)

    def forward(
        self, cqt96: torch.Tensor, hold_layer: BassPedalHoldLayerV6
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = cqt96.shape
        c96 = cqt96.float()

        # Fold 72 quarter-tone low bins (C1..B3) -> 36 sharp semitone peaks + 24 upper bins (C4..B5) = 60 bins
        low_36 = c96[:, :, :72].reshape(bsz, seq_len, 36, 2).amax(dim=-1)
        c60 = torch.cat([low_36, c96[:, :, 72:]], dim=-1)  # (B, T, 60)

        # Cumulative Energy strictly below each semitone bin k (normalized to [0, 1])
        e_below = ((torch.cumsum(c60, dim=-1) - c60) * self.eb_norm.view(1, 1, 60) * 3.0).clamp(0.0, 1.0)

        # Harmonic-suppressed onset flux (removes 2nd harmonic +12 and 3rd harmonic +19 of held bass notes)
        prev_c60 = F.pad(c60[:, :-1, :], (0, 0, 1, 0))
        raw_flux = F.relu(c60 - prev_c60)
        ov_oct = F.pad(prev_c60[:, :, :-12], (12, 0))
        ov_5th = F.pad(prev_c60[:, :, :-19], (19, 0))
        harm_flux = F.relu(raw_flux - 0.55 * ov_oct - 0.35 * ov_5th)

        # Differentiable count of simultaneous new chord pitches in chord register C2..G4 (bins 12..43)
        chord_reg_flux = harm_flux[:, :, 12:44]
        harm_onset_count = torch.sigmoid(14.0 * (chord_reg_flux - 0.14)).sum(dim=-1, keepdim=True)  # (B, T, 1)

        # Floor-weighted low-register onset flux (C1..B4 = bins 0..47):
        # High ONLY when a new fundamental strikes at/below the active bass floor (NOT ascending arpeggios!)
        floor_flux = harm_flux[:, :, :48] * (1.0 - e_below[:, :, :48]).clamp(0.0, 1.0)
        bass_reset_logit = self.bass_reset_mlp(floor_flux)  # (B, T, 1)
        # Blend learned reset logit with direct physical floor-onset peak for zero-lag response from epoch 0
        phys_reset_peak = floor_flux.amax(dim=-1, keepdim=True)
        bass_reset_prob = torch.sigmoid(bass_reset_logit + 4.0 * F.relu(phys_reset_peak - 0.22))

        cqt_5x12 = c60.reshape(bsz, seq_len, 5, 12)
        eb_5x12 = e_below.reshape(bsz, seq_len, 5, 12)
        flux_5x12 = harm_flux.reshape(bsz, seq_len, 5, 12)

        rel_cqt = cqt_5x12[:, :, :, self.rel_idx].permute(0, 1, 3, 2, 4).reshape(bsz, seq_len, 12, 60)
        rel_eb = eb_5x12[:, :, :, self.rel_idx].permute(0, 1, 3, 2, 4).reshape(bsz, seq_len, 12, 60)
        flux_f0 = flux_5x12.permute(0, 1, 3, 2)  # (B, T, 12, 5)

        phys_in = torch.cat([rel_cqt, rel_eb, flux_f0], dim=-1).to(cqt96.dtype)
        phys_bass_logits = self.phys_mlp(phys_in).squeeze(-1)  # (B, T, 12)
        phys_bass_prob = torch.sigmoid(phys_bass_logits.float()).to(cqt96.dtype)
        bass_hold = hold_layer(phys_bass_logits, bass_reset_prob=bass_reset_prob)

        cqt_ctx = self.cqt_proj(
            torch.cat([cqt96, phys_bass_prob, bass_hold, bass_reset_prob.to(cqt96.dtype)], dim=-1)
        )
        return (
            phys_bass_logits,
            phys_bass_prob,
            bass_hold,
            bass_reset_logit.to(cqt96.dtype),
            harm_onset_count.to(cqt96.dtype),
            cqt_ctx,
        )


class ChordModelV6(nn.Module):
    """Model v6 — Boundary-Aware, Bass-First, Full-Song Harmonic Architecture:
    - Chunked 5s MERT (`25`-layer weighted sum -> `256d`) + 96-bin `HarmonicCQTFilterbankV6`
    - `OctavePitchBassModuleV6` (`E_below` for high-register solo piano + harmonic-suppressed onset flux)
      + 4.0s Onset-Reset `BassPedalHoldLayerV6`
    - Dual-Register Chroma + `DualScalePassingSustainLayer` (`120ms` gated passing env vs `520ms` sustain)
      + parameter-free `6-D Tonnetz` tonal centroid projection
    - `BiConformerEncoderV6` (parallel `k=3 || k=15`, `DropPath=0.15`) + `FullSongCrossSectionAttention (stride=15)`
    - Explicit `BoundaryHead` (`[p_chord_chg, p_bass_chg]`) + Confidence-Gated Global/Local `KeyHead (24)`
    - Factorized Pitch-Equivariant Heads: `TrueBass (13)` -> `Root (13)` + `Skel14 (12x14)` + `Inversion (12x5)`
      + `Tension7D (12x7)` + `SlashBass (13)` -> 169-state `DynamicBoundaryCRF (crf_stride=2)`
    """

    def __init__(self, cfg: ModelCfg, sample_rate: int = 24000):
        super().__init__()
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.chunk_samples = cfg.chunk_s * sample_rate
        self.cqt_bins = 96

        self.mert = AutoModel.from_pretrained(cfg.name, trust_remote_code=True)
        for p in self.mert.parameters():
            p.requires_grad = False

        self._captured: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

        self.layer_weights = nn.Parameter(torch.zeros(cfg.n_layers))

        c_dim = cfg.context_dim
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, c_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(c_dim),
        )

        # Dual-register pitch chroma + dual-scale passing sustain + 6-D Tonnetz (0 trainable params)
        self.head_chroma_low = nn.Sequential(nn.Linear(c_dim, 128), nn.SiLU(), nn.Linear(128, 12))
        self.head_chroma_full = nn.Sequential(nn.Linear(c_dim, 128), nn.SiLU(), nn.Linear(128, 12))
        self.passing_sustain = DualScalePassingSustainLayer(short_frames=9, sustain_frames=39)
        self.register_buffer("tonnetz_mat", torch.tensor(TONNETZ_MATRIX, dtype=torch.float32), persistent=False)

        cqt_ctx_dim = 64
        self.cqt_bank = HarmonicCQTFilterbankV6(sample_rate=sample_rate, hop_length=MERT_HOP_SAMPLES)
        self.bass_hold_layer = BassPedalHoldLayerV6(block_frames=12, n_taps=25, tau_frames=195.0)
        self.oct_bass = OctavePitchBassModuleV6(cqt_ctx_dim=cqt_ctx_dim, dropout=min(cfg.dropout, 0.15))

        # Input fusion: emb(256) + cqt_ctx(64) + c_low(12) + c_full(12) + sustain(12) + fast_env(12) + tonnetz(12) + passing_gate(1)
        conf_in_dim = c_dim + cqt_ctx_dim + 12 * 5 + 1
        self.conf_fusion = nn.Sequential(
            nn.Linear(conf_in_dim, c_dim),
            nn.LayerNorm(c_dim),
        )

        drop_path = float(getattr(cfg, "drop_path", 0.15))
        self.conformer = BiConformerEncoderV6(
            dim=c_dim,
            n_layers=cfg.conformer_layers,
            n_heads=cfg.conformer_heads,
            k_short=3,
            k_long=cfg.conformer_conv_kernel,
            dropout=min(cfg.dropout, 0.2),
            drop_path=drop_path,
        )
        self.global_ctx = FullSongCrossSectionAttention(
            dim=c_dim,
            stride=int(getattr(cfg, "global_ctx_stride", 15)),
            n_heads=cfg.conformer_heads,
            n_layers=2,
            dropout=min(cfg.dropout, 0.2),
            drop_path=drop_path * 0.67,
        )

        # Explicit Boundary Head: predicts [chord_change_logit, bass_change_logit]
        bnd_in_dim = c_dim + c_dim + cqt_ctx_dim + 1 + 1 + 1  # z + |z_t - z_{t-1}| + cqt_ctx + bass_reset + harm_cnt + passing_gate
        self.head_boundary = nn.Sequential(
            nn.Linear(bnd_in_dim, 128),
            nn.SiLU(),
            nn.Dropout(min(cfg.dropout, 0.2)),
            nn.Linear(128, 2),
        )
        nn.init.constant_(self.head_boundary[-1].bias, -2.5)

        # Confidence-Gated Global + Local Key Head (24 keys)
        self.head_key = nn.Sequential(
            nn.Linear(c_dim, 128),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, N_KEYS),
        )
        self.key_fusion_gate = nn.Sequential(
            nn.Linear(2 * c_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        root_ctx_dim = 128
        self.root_ctx_proj = nn.Linear(c_dim, root_ctx_dim)
        self.root_emb = nn.Embedding(N_ROOTS, root_ctx_dim)

        # Stage 1 (Bottom-Up): TrueBassHead (0..11 physical sounding bass + 12 = N)
        tb_in_dim = c_dim + 12 + 12 + 12 + 1 + 1  # z + phys_bass_prob + bass_hold + c_low + bass_reset + norm_rms
        self.head_true_bass = nn.Sequential(
            nn.Linear(tb_in_dim, cfg.bass_head_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bass_head_hidden, N_ROOT_CLASSES_V5),
        )
        self.phys_bass_res_scale = nn.Parameter(torch.tensor(0.5))

        # Stage 2: Bass-Conditioned RootHead + Pitch-Equivariant Factorized Heads
        root_in_dim = c_dim + N_ROOT_CLASSES_V5 + 12 + 12 + 12 + 12 + 12 + N_KEYS + 1
        self.head_root = nn.Sequential(
            nn.Linear(root_in_dim, cfg.head_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, N_ROOT_CLASSES_V5),
        )

        # Shared Pitch-Equivariant Relative Tensor `u`:
        # z_r(128) + rel(tb_12, bh_12, c_low, c_full, sustain, fast_env, k_maj, k_min) [8*12=96] + tonnetz(12) = 236
        rel_dim = root_ctx_dim + 8 * 12 + 12
        self.head_root_rel = nn.Sequential(
            nn.Linear(rel_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.head_root_rel[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head_root_rel[-1].bias)

        self.head_skel14 = nn.Sequential(
            nn.Linear(rel_dim, cfg.type_head_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.type_head_hidden, 96),
            nn.SiLU(),
            nn.Linear(96, N_SKELETONS_V6),
        )
        self.head_inversion = nn.Sequential(
            nn.Linear(rel_dim, 96),
            nn.SiLU(),
            nn.Dropout(min(cfg.dropout, 0.25)),
            nn.Linear(96, N_INVERSIONS_V6),
        )
        # Multi-label 7-D Tension Head conditioned on relative tensor `u` + predicted skel14 distribution
        self.head_tension = nn.Sequential(
            nn.Linear(rel_dim + N_SKELETONS_V6, 128),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, N_TENSIONS_V6),
        )

        # Stage 3: Slash-Inversion BassHead conditioned on TrueBass, BassHold, Root, Skel14 & Inversion probs
        bass_in_dim = c_dim + N_ROOT_CLASSES_V5 + 12 + 12 + N_ROOT_CLASSES_V5 + N_SKELETONS_V6 + N_INVERSIONS_V6
        self.head_bass = nn.Sequential(
            nn.Linear(bass_in_dim, cfg.bass_head_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bass_head_hidden, N_BASS_CLASSES),
        )

        # 169-State Dynamic Boundary-Conditioned Linear-Chain CRF with Music-Theory Functional Prior
        prior_mat = build_functional_crf_prior_v6(stay_bonus=2.4)
        self.crf = DynamicBoundaryCRF(num_tags=N_CRF_STATES_V6, stay_bonus=2.4, prior_matrix=prior_mat)
        self.register_buffer("rel_idx", torch.tensor(REL_PITCH_INDEX, dtype=torch.long), persistent=False)

    def _clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _register_hooks(self) -> None:
        self._clear_hooks()
        base = getattr(self.mert, "base_model", self.mert)
        base = getattr(base, "model", base)

        def make_hook():
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if isinstance(h, torch.Tensor):
                    self._captured.append(h.detach() if not h.requires_grad else h)
            return hook

        fp = getattr(base, "feature_projection", None)
        if fp is not None:
            self._hooks.append(fp.register_forward_hook(make_hook()))

        encoder = getattr(base, "encoder", None)
        layers = getattr(encoder, "layers", []) if encoder is not None else []
        for layer in layers:
            self._hooks.append(layer.register_forward_hook(make_hook()))

    def _encode_single_chunk_batch(self, wav_chunk: torch.Tensor) -> torch.Tensor:
        self._captured = []
        out = self.mert(wav_chunk)
        n = self.cfg.n_layers
        if len(self._captured) >= n:
            hidden_states = torch.stack(self._captured[:n])
        elif len(self._captured) > 0:
            pad = [self._captured[-1]] * (n - len(self._captured))
            hidden_states = torch.stack(self._captured + pad)
        else:
            lhs = out.last_hidden_state
            hidden_states = lhs.unsqueeze(0).expand(n, -1, -1, -1)

        w = F.softmax(self.layer_weights, dim=0).to(hidden_states.dtype)
        h = (w[:, None, None, None] * hidden_states).sum(0)
        return self.input_proj(h)

    def encode_chunks(self, wav: torch.Tensor) -> torch.Tensor:
        bsz, total_samples = wav.shape
        if total_samples > self.chunk_samples and total_samples % self.chunk_samples == 0:
            n_chunks = total_samples // self.chunk_samples
            flat = wav.reshape(bsz * n_chunks, self.chunk_samples)
            emb_flat = self._encode_single_chunk_batch(flat)
            return emb_flat.reshape(bsz, n_chunks * emb_flat.shape[1], emb_flat.shape[2])
        return self._encode_single_chunk_batch(wav)

    def encode_cqt(self, wav: torch.Tensor) -> torch.Tensor:
        bsz, total_samples = wav.shape
        if total_samples > self.chunk_samples and total_samples % self.chunk_samples == 0:
            n_chunks = total_samples // self.chunk_samples
            flat = wav.reshape(bsz * n_chunks, self.chunk_samples)
            cqt_flat = self.cqt_bank(flat)
            return cqt_flat.reshape(bsz, n_chunks * cqt_flat.shape[1], cqt_flat.shape[2])
        return self.cqt_bank(wav)

    def decode_context(
        self,
        emb: torch.Tensor,
        frame_rms_db: torch.Tensor | None = None,
        cqt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len, _ = emb.shape
        if frame_rms_db is None:
            frame_rms_db = torch.full((bsz, seq_len), -20.0, device=emb.device, dtype=torch.float32)

        low_chroma_logits = self.head_chroma_low(emb)
        full_chroma_logits = self.head_chroma_full(emb)
        c_low = torch.sigmoid(low_chroma_logits.float()).to(emb.dtype)
        c_full = torch.sigmoid(full_chroma_logits.float()).to(emb.dtype)

        if cqt is None:
            cqt_in = torch.zeros((bsz, seq_len, self.cqt_bins), device=emb.device, dtype=emb.dtype)
        else:
            cqt_in = cqt.to(device=emb.device, dtype=emb.dtype)

        (
            phys_bass_logits,
            phys_bass_prob,
            bass_hold,
            bass_reset_logit,
            harm_onset_count,
            cqt_ctx,
        ) = self.oct_bass(cqt_in, self.bass_hold_layer)

        bass_reset_prob = torch.sigmoid(bass_reset_logit.float()).to(emb.dtype)
        sustain_env, fast_chord_env, passing_gate = self.passing_sustain(
            full_chroma_logits,
            harm_onset_count=harm_onset_count,
            p_bass_chg=bass_reset_prob,
        )

        # Parameter-free 6-D Tonnetz tonal centroids from full chroma & sustain envelope -> 12d
        t_mat = self.tonnetz_mat.to(dtype=emb.dtype)
        ton_full = torch.matmul(c_full, t_mat)
        ton_sus = torch.matmul(sustain_env, t_mat)
        tonnetz_12 = torch.cat([ton_full, ton_sus], dim=-1)  # (B, T, 12)

        conf_in = self.conf_fusion(
            torch.cat(
                [emb, cqt_ctx, c_low, c_full, sustain_env, fast_chord_env, tonnetz_12, passing_gate],
                dim=-1,
            )
        )
        z_local = self.conformer(conf_in)
        z = self.global_ctx(z_local)

        # Explicit Boundary Head ([p_chord_chg, p_bass_chg])
        z_delta = torch.abs(z - F.pad(z[:, :-1, :], (0, 0, 1, 0)))
        bnd_in = torch.cat([z, z_delta, cqt_ctx, bass_reset_prob, harm_onset_count, passing_gate], dim=-1)
        boundary_logits = self.head_boundary(bnd_in)  # (B, T, 2)

        # Confidence-Gated Local (4.0s) + Global Song Key Estimation (24 keys)
        norm_rms = ((frame_rms_db.float().clamp(-80.0, 0.0) + 40.0) / 20.0).unsqueeze(-1).to(z.dtype)
        k_win = 301
        z_key_local = F.avg_pool1d(
            z.float().transpose(1, 2), kernel_size=k_win, stride=1, padding=k_win // 2, count_include_pad=False
        ).transpose(1, 2).to(z.dtype)
        clarity_w = (
            torch.sigmoid(norm_rms.float()) * (0.25 + c_full.float().amax(dim=-1, keepdim=True))
        ).clamp(min=1e-4)
        z_key_global_vec = (z.float() * clarity_w).sum(dim=1, keepdim=True) / clarity_w.sum(dim=1, keepdim=True).clamp(min=1e-4)
        z_key_global = z_key_global_vec.expand(-1, seq_len, -1).to(z.dtype)

        key_logits_local = self.head_key(z_key_local)
        key_logits_global = self.head_key(z_key_global)
        g_key = torch.sigmoid(
            self.key_fusion_gate(torch.cat([z_key_local, z_key_global, norm_rms], dim=-1)).float()
        ).to(z.dtype)
        key_logits = (1.0 - g_key) * key_logits_local + g_key * key_logits_global
        key_prob = F.softmax(key_logits.float(), dim=-1).to(z.dtype)

        # Bottom-Up Stage 1: TrueBassHead (0..11 actual sounding bass, 12 = N)
        tb_in = torch.cat([z, phys_bass_prob, bass_hold, c_low, bass_reset_prob, norm_rms], dim=-1)
        tb_raw = self.head_true_bass(tb_in)
        tb_res = F.pad((self.phys_bass_res_scale.float() * phys_bass_logits.float()).to(tb_raw.dtype), (0, 1))
        true_bass_logits = tb_raw + tb_res
        true_bass_prob = F.softmax(true_bass_logits.float(), dim=-1).to(z.dtype)

        # Shared Pitch-Equivariant Rotation `Roll(-r)` over all 12 candidate roots
        idx = self.rel_idx  # (12, 12)
        rel_tb = true_bass_prob[:, :, :12][:, :, idx]
        rel_bh = bass_hold[:, :, idx]
        rel_c_low = c_low[:, :, idx]
        rel_c_full = c_full[:, :, idx]
        rel_sus = sustain_env[:, :, idx]
        rel_fast = fast_chord_env[:, :, idx]
        rel_k_maj = key_prob[:, :, :12][:, :, idx]
        rel_k_min = key_prob[:, :, 12:][:, :, idx]
        z_r = self.root_ctx_proj(z).unsqueeze(2) + self.root_emb.weight[None, None, :, :]
        ton_exp = tonnetz_12.unsqueeze(2).expand(-1, -1, N_ROOTS, -1)

        u = torch.cat(
            [z_r, rel_tb, rel_bh, rel_c_low, rel_c_full, rel_sus, rel_fast, rel_k_maj, rel_k_min, ton_exp],
            dim=-1,
        )

        # Stage 2: Root + 14-Skeleton + 5-Inversion + 7-Tension Factorized Heads
        root_in = torch.cat(
            [z, true_bass_prob, bass_hold, c_low, c_full, sustain_env, fast_chord_env, key_prob, norm_rms],
            dim=-1,
        )
        r_rel = F.pad(self.head_root_rel(u).squeeze(-1), (0, 1))
        root_logits = self.head_root(root_in) + r_rel                        # (B, T, 13)
        skel14_logits_by_root = self.head_skel14(u)                          # (B, T, 12, 14)
        inv_logits_by_root = self.head_inversion(u)                          # (B, T, 12, 5)

        skel14_prob_by_root = F.softmax(skel14_logits_by_root.float(), dim=-1).to(u.dtype)
        tension_logits_by_root = self.head_tension(torch.cat([u, skel14_prob_by_root], dim=-1))  # (B, T, 12, 7)

        # Exact float32 169-State Joint Emissions: state 0 = N, states 1..168 = 12 roots x 14 skeletons
        root_logp = F.log_softmax(root_logits.float(), dim=-1)               # (B, T, 13)
        skel14_logp = F.log_softmax(skel14_logits_by_root.float(), dim=-1)   # (B, T, 12, 14)
        joint_skel_logp = root_logp[:, :, :12, None] + skel14_logp           # (B, T, 12, 14)
        frame_skel_emissions = torch.cat(
            [root_logp[:, :, 12:13], joint_skel_logp.reshape(bsz, seq_len, N_ROOTS * N_SKELETONS_V6)],
            dim=-1,
        )  # (B, T, 169)

        stride = max(1, int(self.cfg.crf_stride))
        chord_bnd_prob = torch.sigmoid(boundary_logits[:, :, 0].float())     # (B, T)
        if stride > 1:
            crf_emissions = F.avg_pool1d(
                frame_skel_emissions.transpose(1, 2),
                kernel_size=stride,
                stride=stride,
                ceil_mode=True,
            ).transpose(1, 2)
            boundary_prob_crf = F.max_pool1d(
                chord_bnd_prob.unsqueeze(1),
                kernel_size=stride,
                stride=stride,
                ceil_mode=True,
            ).squeeze(1)
        else:
            crf_emissions = frame_skel_emissions
            boundary_prob_crf = chord_bnd_prob

        # Stage 3: Slash-Inversion BassHead (0..11 slash bass, 12 = root position)
        root_prob_13 = root_logp.exp().to(z.dtype)
        skel_prob_14 = torch.logsumexp(joint_skel_logp, dim=2).exp().to(z.dtype)
        inv_prob_by_root = F.softmax(inv_logits_by_root.float(), dim=-1)
        inv_prob_5 = (root_logp[:, :, :12].exp().unsqueeze(-1) * inv_prob_by_root).sum(dim=2).to(z.dtype)
        bass_in = torch.cat([z, true_bass_prob, bass_hold, c_low, root_prob_13, skel_prob_14, inv_prob_5], dim=-1)
        bass_logits = self.head_bass(bass_in)

        return {
            "root_logits": root_logits,
            "skel14_logits_by_root": skel14_logits_by_root,
            "tension_logits_by_root": tension_logits_by_root,
            "inv_logits_by_root": inv_logits_by_root,
            "joint_skel_logp": joint_skel_logp,
            "frame_skel_emissions": frame_skel_emissions,
            "crf_emissions": crf_emissions,
            "boundary_logits": boundary_logits,
            "boundary_prob_crf": boundary_prob_crf,
            "bass_logits": bass_logits,
            "true_bass_logits": true_bass_logits,
            "phys_bass_logits": phys_bass_logits,
            "key_logits": key_logits,
            "low_chroma_logits": low_chroma_logits,
            "full_chroma_logits": full_chroma_logits,
            "sustain_env": sustain_env,
            "fast_chord_env": fast_chord_env,
            "passing_gate": passing_gate,
            "bass_reset_logit": bass_reset_logit,
        }

    def forward(
        self,
        wav: torch.Tensor,
        frame_rms_db: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emb = self.encode_chunks(wav)
        cqt = self.encode_cqt(wav)
        if frame_rms_db is not None and frame_rms_db.shape[1] != emb.shape[1]:
            T = emb.shape[1]
            if frame_rms_db.shape[1] > T:
                frame_rms_db = frame_rms_db[:, :T]
            else:
                frame_rms_db = F.pad(frame_rms_db, (0, T - frame_rms_db.shape[1]), value=-80.0)
        return self.decode_context(emb, frame_rms_db=frame_rms_db, cqt=cqt)

    def apply_lora(self, lora_cfg) -> None:
        """Apply controlled LoRA restricted to `lora_cfg.layer_range` (default layers 8..21) to prevent overfitting."""
        from peft import LoraConfig, get_peft_model

        self._clear_hooks()
        layer_range = getattr(lora_cfg, "layer_range", None)
        lora_kwargs: dict = dict(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            bias="none",
        )
        if layer_range and len(layer_range) == 2:
            lo, hi = int(layer_range[0]), int(layer_range[1])
            lora_kwargs["layers_to_transform"] = list(range(lo, hi + 1))
            lora_kwargs["layers_pattern"] = "layers"

        try:
            peft_config = LoraConfig(**lora_kwargs)
            self.mert = get_peft_model(self.mert, peft_config)
        except Exception:
            # Safe fallback if custom remote MERT module names differ from `layers_pattern`
            lora_kwargs.pop("layers_to_transform", None)
            lora_kwargs.pop("layers_pattern", None)
            peft_config = LoraConfig(**lora_kwargs)
            self.mert = get_peft_model(self.mert, peft_config)

        # Strictly enforce layer_range freeze guardrail even on fallback
        import re as _re

        for name, p in self.mert.named_parameters():
            if "lora_" in name:
                if layer_range and len(layer_range) == 2:
                    m = _re.search(r"layers\.(\d+)\.", name)
                    if m is not None:
                        l_idx = int(m.group(1))
                        p.requires_grad = int(layer_range[0]) <= l_idx <= int(layer_range[1])
                    else:
                        p.requires_grad = True
                else:
                    p.requires_grad = True
        self._register_hooks()

    def merge_lora(self) -> None:
        if hasattr(self.mert, "merge_and_unload"):
            self._clear_hooks()
            self.mert = self.mert.merge_and_unload()
            for p in self.mert.parameters():
                p.requires_grad = False
            self._register_hooks()

    def get_trainable_params(self) -> list[dict]:
        head_params = [
            p for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith("mert.")
        ]
        lora_params = [
            p for name, p in self.named_parameters()
            if p.requires_grad and name.startswith("mert.")
        ]
        groups = [{"params": head_params, "lr_key": "lr_head"}]
        if lora_params:
            groups.append({"params": lora_params, "lr_key": "lr_lora"})
        return groups
