from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.config import ModelCfg

# Exact MERT-v1-330M / Wav2Vec2 7-layer conv feature extractor geometry
MERT_CONV_KERNELS: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
MERT_CONV_STRIDES: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
MERT_HOP_SAMPLES: int = 320        # product of MERT_CONV_STRIDES
MERT_RECEPTIVE_SAMPLES: int = 400  # receptive field of the 7 conv layers


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


class TemporalContextBlock(nn.Module):
    """Lightweight depthwise-grouped Conv1d + channel mixing residual block for frame-level context."""

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


def build_model(cfg: ModelCfg) -> ChordModel:
    return ChordModel(cfg)
