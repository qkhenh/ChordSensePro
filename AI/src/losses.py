from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_m = logits[mask]
        targets_m = targets[mask]

        log_p = F.log_softmax(logits_m, dim=-1)
        p = log_p.exp()

        ce = F.nll_loss(log_p, targets_m, weight=self.weight, reduction="none")
        p_t = p.gather(1, targets_m.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma

        return (focal_weight * ce).mean()


class ChordLoss(nn.Module):

    def __init__(
        self,
        weights: list[float] | None = None,
        focal_gamma: float = 2.0,
        ext_class_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.w = weights or [0.3, 0.3, 0.4]
        self.ce_root = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_quality = nn.CrossEntropyLoss(ignore_index=-100)
        self.focal_ext = FocalLoss(gamma=focal_gamma, weight=ext_class_weight, ignore_index=-100)

    def forward(
        self,
        root_logits: torch.Tensor,
        quality_logits: torch.Tensor,
        ext_logits: torch.Tensor,
        root_targets: torch.Tensor,
        quality_targets: torch.Tensor,
        ext_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_root = self.ce_root(root_logits, root_targets)
        l_qual = self.ce_quality(quality_logits, quality_targets)
        l_ext = self.focal_ext(ext_logits, ext_targets)

        if torch.isnan(l_root):
            l_root = torch.tensor(0.0, device=root_logits.device)
        if torch.isnan(l_qual):
            l_qual = torch.tensor(0.0, device=root_logits.device)

        total = self.w[0] * l_root + self.w[1] * l_qual + self.w[2] * l_ext

        details = {
            "loss_root": l_root.item(),
            "loss_quality": l_qual.item(),
            "loss_ext": l_ext.item(),
            "loss_total": total.item(),
        }
        return total, details

