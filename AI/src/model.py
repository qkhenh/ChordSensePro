from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.config import ModelCfg


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

        mert_config = AutoConfig.from_pretrained(cfg.name, trust_remote_code=True)
        mert_config.output_hidden_states = True
        self.mert = AutoModel.from_pretrained(cfg.name, config=mert_config, trust_remote_code=True)
        for p in self.mert.parameters():
            p.requires_grad = False

        self.layer_weights = nn.Parameter(torch.zeros(cfg.n_layers))

        d = cfg.hidden_dim
        self.head_root = ChordHead(d, cfg.head_hidden, cfg.n_root)
        self.head_quality = ChordHead(d + cfg.n_root, cfg.head_hidden, cfg.n_quality)
        self.head_ext = ExtensionHead(
            d + cfg.n_root + cfg.n_quality,
            cfg.ext_head_hidden,
            cfg.ext_head_mid,
            cfg.n_ext,
            cfg.dropout,
        )

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.mert(wav, return_dict=True)
        hidden_states = torch.stack(out.hidden_states)

        w = F.softmax(self.layer_weights, dim=0)
        h = (w[:, None, None, None] * hidden_states).sum(0)

        emb = self._pool(h)

        root_logits = self.head_root(emb)

        if self.cfg.detach_logits:
            root_input = root_logits.detach()
        else:
            root_input = root_logits
        quality_logits = self.head_quality(torch.cat([emb, root_input], dim=-1))

        if self.cfg.detach_logits:
            quality_input = quality_logits.detach()
        else:
            quality_input = quality_logits
        ext_logits = self.head_ext(torch.cat([emb, root_input, quality_input], dim=-1))

        return root_logits, quality_logits, ext_logits

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

