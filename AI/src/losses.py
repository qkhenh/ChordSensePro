from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_smoothed_class_weights(
    parquet_path: str | Path,
    column: str,
    n_classes: int,
    power: float = 0.3,
    clip_range: tuple[float, float] = (0.5, 4.0),
) -> torch.Tensor:
    """Compute inverse-frequency smoothed class weights `1 / (count ** power)` normalized to mean 1."""
    df = pd.read_parquet(parquet_path, columns=[column])
    vals = df[column].dropna().astype(int).values
    counts = np.bincount(vals[vals >= 0], minlength=n_classes)[:n_classes].astype(np.float64)
    counts = np.maximum(counts, 1.0)
    raw_w = 1.0 / (counts ** power)
    raw_w = raw_w / raw_w.mean()
    clipped = np.clip(raw_w, clip_range[0], clip_range[1])
    return torch.tensor(clipped, dtype=torch.float32)


def default_bass_class_weight(n_bass: int = 13, none_weight: float = 0.25) -> torch.Tensor:
    """Downweight dominant `none` class (index 12) so slash-chord inversions are actively learned."""
    w = torch.ones(n_bass, dtype=torch.float32)
    if n_bass >= 13:
        w[12] = none_weight
    return w


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_index
        if not mask.any():
            return logits.sum() * 0.0

        logits_m = logits[mask]
        targets_m = targets[mask]

        log_p = F.log_softmax(logits_m.float(), dim=-1)
        p = log_p.exp()

        w = self.weight.to(device=log_p.device, dtype=log_p.dtype) if self.weight is not None else None
        ce = F.nll_loss(log_p, targets_m, weight=w, reduction="none")
        p_t = p.gather(1, targets_m.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma

        return (focal_weight * ce).mean()


class ChordLoss(nn.Module):

    def __init__(
        self,
        weights: list[float] | None = None,
        focal_gamma: float = 2.0,
        quality_class_weight: torch.Tensor | None = None,
        ext_class_weight: torch.Tensor | None = None,
        bass_class_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        w = list(weights) if weights is not None else [0.3, 0.25, 0.25, 0.2]
        if len(w) == 3:
            w.append(0.0)
        self.w = w
        self.ce_root = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_quality = nn.CrossEntropyLoss(
            weight=quality_class_weight.float() if quality_class_weight is not None else None,
            ignore_index=-100,
        )
        self.focal_ext = FocalLoss(gamma=focal_gamma, weight=ext_class_weight, ignore_index=-100)
        eff_bass_w = bass_class_weight if bass_class_weight is not None else default_bass_class_weight()
        self.focal_bass = FocalLoss(gamma=focal_gamma, weight=eff_bass_w, ignore_index=-100)

    def forward(
        self,
        root_logits: torch.Tensor,
        quality_logits: torch.Tensor,
        ext_logits: torch.Tensor,
        bass_logits: torch.Tensor,
        root_targets: torch.Tensor,
        quality_targets: torch.Tensor,
        ext_targets: torch.Tensor,
        bass_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if root_logits.dim() == 3:
            T = min(root_logits.shape[1], root_targets.shape[1])
            B = root_logits.shape[0]
            root_logits = root_logits[:, :T, :].reshape(B * T, -1)
            quality_logits = quality_logits[:, :T, :].reshape(B * T, -1)
            ext_logits = ext_logits[:, :T, :].reshape(B * T, -1)
            bass_logits = bass_logits[:, :T, :].reshape(B * T, -1)
            root_targets = root_targets[:, :T].reshape(B * T)
            quality_targets = quality_targets[:, :T].reshape(B * T)
            ext_targets = ext_targets[:, :T].reshape(B * T)
            bass_targets = bass_targets[:, :T].reshape(B * T)

        l_root = self.ce_root(root_logits.float(), root_targets)
        l_qual = self.ce_quality(quality_logits.float(), quality_targets)
        l_ext = self.focal_ext(ext_logits, ext_targets)
        l_bass = self.focal_bass(bass_logits, bass_targets)

        if torch.isnan(l_root):
            l_root = root_logits.sum() * 0.0
        if torch.isnan(l_qual):
            l_qual = quality_logits.sum() * 0.0
        if torch.isnan(l_ext):
            l_ext = ext_logits.sum() * 0.0
        if torch.isnan(l_bass):
            l_bass = bass_logits.sum() * 0.0

        total = (
            self.w[0] * l_root
            + self.w[1] * l_qual
            + self.w[2] * l_ext
            + self.w[3] * l_bass
        )

        details = {
            "loss_root": l_root.item(),
            "loss_quality": l_qual.item(),
            "loss_ext": l_ext.item(),
            "loss_bass": l_bass.item(),
            "loss_total": total.item(),
        }
        return total, details
