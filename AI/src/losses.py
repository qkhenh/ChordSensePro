from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.music_theory import (
    N_KEYS,
    N_TYPES,
    QE_TO_TYPE,
    SKEL_TYPE_MASK,
    TYPE_TO_QE,
    TYPE_TO_SKEL,
)


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


def compute_type_class_weights(
    parquet_path: str | Path,
    power: float = 0.28,
    clip_range: tuple[float, float] = (0.5, 4.0),
) -> torch.Tensor:
    """Compute smoothed class weights across the 41 valid `(quality_id, ext_id)` chord types."""
    df = pd.read_parquet(parquet_path, columns=["quality_id", "ext_id"])
    valid = df["quality_id"].notna() & df["ext_id"].notna()
    q = df.loc[valid, "quality_id"].astype(int).to_numpy()
    e = df.loc[valid, "ext_id"].astype(int).to_numpy()
    in_bounds = (q >= 0) & (q < 6) & (e >= 0) & (e < 15)
    tids = QE_TO_TYPE[q[in_bounds], e[in_bounds]]
    tids = tids[tids >= 0]
    counts = np.bincount(tids, minlength=N_TYPES)[:N_TYPES].astype(np.float64)
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


def _build_relative_key_targets() -> torch.Tensor:
    """Build `(24, 24)` soft target table giving `0.85` to primary key and `0.15` to relative major/minor."""
    mat = torch.zeros((N_KEYS, N_KEYS), dtype=torch.float32)
    for k in range(12):
        # Major key k -> relative minor is 12 + (k + 9) % 12
        rel_min = 12 + (k + 9) % 12
        mat[k, k] = 0.85
        mat[k, rel_min] = 0.15
        # Minor key (12 + k) -> relative major is (k + 3) % 12
        rel_maj = (k + 3) % 12
        mat[12 + k, 12 + k] = 0.85
        mat[12 + k, rel_maj] = 0.15
    return mat


def _build_asymmetric_type_targets(triad_smooth: float = 0.10) -> torch.Tensor:
    """Build `(41, 41)` target distribution: bare triads (`ext=0`) share small mass with same-skeleton tensions."""
    mat = torch.eye(N_TYPES, dtype=torch.float32)
    for tid in range(N_TYPES):
        _, ext_id = TYPE_TO_QE[tid]
        if ext_id == 0:
            sid = int(TYPE_TO_SKEL[tid])
            siblings = [j for j in range(N_TYPES) if SKEL_TYPE_MASK[sid, j] and j != tid]
            if siblings:
                mat[tid, tid] = 1.0 - triad_smooth
                share = triad_smooth / len(siblings)
                for j in siblings:
                    mat[tid, j] = share
    return mat


class ChordLossV5(nn.Module):
    """Hierarchical Joint Loss for Model v5:
    `[w_crf, w_root, w_type, w_bass, w_chroma, w_key]` default `[0.35, 0.10, 0.25, 0.12, 0.10, 0.08]`.
    """

    def __init__(
        self,
        weights: list[float] | None = None,
        focal_gamma: float = 2.0,
        type_class_weight: torch.Tensor | None = None,
        bass_class_weight: torch.Tensor | None = None,
        crf_stride: int = 4,
    ):
        super().__init__()
        default_w = [0.35, 0.10, 0.25, 0.12, 0.10, 0.08]
        w = list(weights) if weights is not None and len(weights) == 6 else default_w
        self.w = w
        self.gamma = focal_gamma
        self.crf_stride = max(1, int(crf_stride))

        self.ce_root = nn.CrossEntropyLoss(ignore_index=-100)
        eff_bass_w = bass_class_weight if bass_class_weight is not None else default_bass_class_weight()
        self.focal_bass = FocalLoss(gamma=focal_gamma, weight=eff_bass_w, ignore_index=-100)
        self.bce_chroma = nn.BCEWithLogitsLoss(reduction="none")

        if type_class_weight is not None:
            self.register_buffer("type_weight", type_class_weight.float())
        else:
            self.register_buffer("type_weight", torch.ones(N_TYPES, dtype=torch.float32))

        self.register_buffer("key_soft_targets", _build_relative_key_targets(), persistent=False)
        self.register_buffer("type_soft_targets", _build_asymmetric_type_targets(0.10), persistent=False)

    @staticmethod
    def downsample_crf_targets(crf_targets: torch.Tensor, stride: int, target_len: int) -> torch.Tensor:
        """Downsample frame-level `(B, T)` CRF state targets to `(B, T_crf)` via center/majority step."""
        if stride <= 1:
            return crf_targets[:, :target_len]
        bsz, seq_len = crf_targets.shape
        out = torch.full((bsz, target_len), -100, dtype=torch.long, device=crf_targets.device)
        for i in range(target_len):
            s = i * stride
            e = min(seq_len, s + stride)
            if s >= seq_len:
                break
            mid = s + (e - s) // 2
            val = crf_targets[:, mid]
            # Fallback to first valid frame inside the stride window if midpoint is -100
            if (val == -100).any():
                win = crf_targets[:, s:e]
                for offset in range(e - s):
                    val = torch.where(val == -100, win[:, offset], val)
            out[:, i] = val
        return out

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        crf_module: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        root_logits = outputs["root_logits"]              # (B, T, 13)
        type_logits_by_root = outputs["type_logits_by_root"]  # (B, T, 12, 41)
        crf_emissions = outputs["crf_emissions"]          # (B, T_crf, 145)
        bass_logits = outputs["bass_logits"]              # (B, T, 13)
        key_logits = outputs["key_logits"]                # (B, T, 24)
        low_chroma_logits = outputs["low_chroma_logits"]  # (B, T, 12)
        full_chroma_logits = outputs["full_chroma_logits"]  # (B, T, 12)

        T = min(root_logits.shape[1], batch["root"].shape[1])
        B = root_logits.shape[0]

        root_t = batch["root"][:, :T]
        type_t = batch["type_id"][:, :T]
        crf_state_t = batch["crf_state"][:, :T]
        bass_t = batch["bass"][:, :T]
        key_t = batch["local_key"][:, :T]
        shell_c_t = batch["shell_chroma"][:, :T]
        full_c_t = batch["full_chroma"][:, :T]

        # 1. 145-state Skeleton Linear-Chain CRF Loss
        crf_tgt_ds = self.downsample_crf_targets(crf_state_t, self.crf_stride, crf_emissions.shape[1])
        l_crf = crf_module(crf_emissions, crf_tgt_ds)

        # 2. Frame-level sharp Root + Silence (`N`=12) CrossEntropy
        l_root = self.ce_root(root_logits[:, :T].float().reshape(B * T, -1), root_t.reshape(B * T))
        if torch.isnan(l_root):
            l_root = root_logits.sum() * 0.0

        # 3. Root-Conditioned ChordType Focal Loss (with asymmetric sibling smoothing on bare triads)
        valid_type = (root_t >= 0) & (root_t < 12) & (type_t >= 0) & (type_t < N_TYPES)
        if valid_type.any():
            r_idx = root_t[valid_type]  # (M,)
            t_idx = type_t[valid_type]  # (M,)
            all_type_logits = type_logits_by_root[:, :T][valid_type]  # (M, 12, 41)
            cond_logits = all_type_logits[torch.arange(len(r_idx), device=r_idx.device), r_idx].float()  # (M, 41)

            log_p = F.log_softmax(cond_logits, dim=-1)
            p = log_p.exp()
            soft_y = self.type_soft_targets[t_idx]  # (M, 41)
            ce_per_sample = -(soft_y * log_p).sum(dim=-1)
            p_true = p.gather(1, t_idx.unsqueeze(1)).squeeze(1)
            w_cls = self.type_weight[t_idx]
            l_type = (((1.0 - p_true) ** self.gamma) * w_cls * ce_per_sample).mean()
        else:
            l_type = type_logits_by_root.sum() * 0.0

        # 4. Slash Bass Focal Loss (only on active chord frames where bass != -100)
        l_bass = self.focal_bass(bass_logits[:, :T].reshape(B * T, -1), bass_t.reshape(B * T))
        if torch.isnan(l_bass):
            l_bass = bass_logits.sum() * 0.0

        # 5. Dual-Register Chroma BCE Loss (active chord frames + explicit zero target on silence `root==12`)
        valid_chroma = root_t >= 0  # includes both chords (0..11) and silence (12 -> all zeros)
        if valid_chroma.any():
            bce_low = self.bce_chroma(low_chroma_logits[:, :T].float()[valid_chroma], shell_c_t.float()[valid_chroma]).mean()
            bce_full = self.bce_chroma(full_chroma_logits[:, :T].float()[valid_chroma], full_c_t.float()[valid_chroma]).mean()
            l_chroma = 0.5 * bce_low + 0.5 * bce_full
        else:
            l_chroma = low_chroma_logits.sum() * 0.0

        # 6. Time-Varying Local Key Loss (with relative Major/Minor soft targets)
        valid_key = (key_t >= 0) & (key_t < N_KEYS)
        if valid_key.any():
            k_log_p = F.log_softmax(key_logits[:, :T].float()[valid_key], dim=-1)
            k_soft = self.key_soft_targets[key_t[valid_key]]
            l_key = -(k_soft * k_log_p).sum(dim=-1).mean()
        else:
            l_key = key_logits.sum() * 0.0

        total = (
            self.w[0] * l_crf
            + self.w[1] * l_root
            + self.w[2] * l_type
            + self.w[3] * l_bass
            + self.w[4] * l_chroma
            + self.w[5] * l_key
        )

        details = {
            "loss_crf": float(l_crf.item()),
            "loss_root": float(l_root.item()),
            "loss_type": float(l_type.item()),
            "loss_bass": float(l_bass.item()),
            "loss_chroma": float(l_chroma.item()),
            "loss_key": float(l_key.item()),
            "loss_total": float(total.item()),
        }
        return total, details
