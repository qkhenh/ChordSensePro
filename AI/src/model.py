from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.config import ModelCfg
from src.conformer import BiConformerEncoder, MultiScaleSustainLayer
from src.crf import MarginalLinearChainCRF
from src.music_theory import (
    N_BASS_CLASSES,
    N_CRF_STATES,
    N_KEYS,
    N_ROOT_CLASSES_V5,
    N_ROOTS,
    N_SKELETONS,
    N_TYPES,
    REL_PITCH_INDEX,
    SKEL_TYPE_MASK,
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


class ChordModelV5(nn.Module):
    """Jazz-Ready Model v5:
    - Chunked 5s MERT Acoustic Front-End -> 256d frame sequence
    - Dual-Register Pitch Chroma (`c_low`, `c_full`) + Multi-Scale Sustain Filter (`80ms` vs `520ms`)
    - 4-Layer RoPE Bi-Conformer (`20s` train / Full-Song serve) + Time-Varying `LocalKeyHead` (24 keys)
    - Pitch-Equivariant Root-Conditioned Joint Head (`1 + 12 * 41 = 493` valid states)
    - 145-state Guide-Tone Skeleton Marginal Linear-Chain CRF (`crf_stride=4`, ~53.3ms)
    """

    def __init__(self, cfg: ModelCfg, sample_rate: int = 24000):
        super().__init__()
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.chunk_samples = cfg.chunk_s * sample_rate

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

        # Long-range musical context encoder
        self.conf_fusion = nn.Sequential(
            nn.Linear(c_dim + 36, c_dim),
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

        # Root + Silence (`N` = 12) Head: receives z + c_low + c_full + sustain + key_prob + norm_rms
        root_in_dim = c_dim + 12 + 12 + 12 + N_KEYS + 1
        self.head_root = nn.Sequential(
            nn.Linear(root_in_dim, cfg.head_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, N_ROOT_CLASSES_V5),
        )

        # Pitch-Equivariant Root-Conditioned Type Head (41 valid chord types)
        root_ctx_dim = 128
        self.root_ctx_proj = nn.Linear(c_dim, root_ctx_dim)
        self.root_emb = nn.Embedding(N_ROOTS, root_ctx_dim)
        type_in_dim = root_ctx_dim + 12 + 12 + 12 + 12 + 12  # z_r + rel(c_low, c_full, sustain, k_maj, k_min) = 188
        self.head_type = nn.Sequential(
            nn.Linear(type_in_dim, cfg.type_head_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.type_head_hidden, 96),
            nn.SiLU(),
            nn.Linear(96, N_TYPES),
        )

        # Conditional Bass Head (0..11 + 12=none)
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

    def decode_context(
        self,
        emb: torch.Tensor,
        frame_rms_db: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run full musical context & root-conditioned joint heads on `emb` of shape `(B, T, context_dim)`."""
        bsz, seq_len, _ = emb.shape
        if frame_rms_db is None:
            frame_rms_db = torch.full((bsz, seq_len), -20.0, device=emb.device, dtype=torch.float32)

        low_chroma_logits = self.head_chroma_low(emb)
        full_chroma_logits = self.head_chroma_full(emb)
        c_low = torch.sigmoid(low_chroma_logits.float()).to(emb.dtype)
        c_full = torch.sigmoid(full_chroma_logits.float()).to(emb.dtype)
        sustain_env = self.sustain_layer(full_chroma_logits)

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
        root_in = torch.cat([z, c_low, c_full, sustain_env, key_prob, norm_rms], dim=-1)
        root_logits = self.head_root(root_in)  # (B, T, 13)

        # Pitch-Equivariant Rotation across all 12 candidate roots: (B, T, 12_root, 12_rel_interval)
        idx = self.rel_idx  # (12, 12)
        rel_c_low = c_low[:, :, idx]
        rel_c_full = c_full[:, :, idx]
        rel_sustain = sustain_env[:, :, idx]
        rel_k_maj = key_prob[:, :, :12][:, :, idx]
        rel_k_min = key_prob[:, :, 12:][:, :, idx]

        z_r = self.root_ctx_proj(z).unsqueeze(2) + self.root_emb.weight[None, None, :, :]
        u = torch.cat([z_r, rel_c_low, rel_c_full, rel_sustain, rel_k_maj, rel_k_min], dim=-1)  # (B, T, 12, 188)
        type_logits_by_root = self.head_type(u)  # (B, T, 12, 41)

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

        # Marginal skeleton distribution over 12 families (summed across roots) for BassHead conditioning
        root_prob_13 = root_logp.exp().to(z.dtype)
        skel_prob_12 = torch.logsumexp(skel_logp_by_root, dim=2).exp().to(z.dtype)
        bass_in = torch.cat([z, c_low, root_prob_13, skel_prob_12], dim=-1)
        bass_logits = self.head_bass(bass_in)

        return {
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

    def forward(
        self,
        wav: torch.Tensor,
        frame_rms_db: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emb = self.encode_chunks(wav)
        if frame_rms_db is not None and frame_rms_db.shape[1] != emb.shape[1]:
            T = emb.shape[1]
            if frame_rms_db.shape[1] > T:
                frame_rms_db = frame_rms_db[:, :T]
            else:
                frame_rms_db = F.pad(frame_rms_db, (0, T - frame_rms_db.shape[1]), value=-80.0)
        return self.decode_context(emb, frame_rms_db=frame_rms_db)

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
    if getattr(cfg, "arch_version", 4) >= 5:
        return ChordModelV5(cfg, sample_rate=sample_rate)
    return ChordModel(cfg)
