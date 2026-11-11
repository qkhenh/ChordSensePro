from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelCfg
from src.model import ChordModel, ChordHead, ExtensionHead


@dataclass
class FakeMERTOutput:
    hidden_states: tuple
    last_hidden_state: torch.Tensor = None


class FakeMERT(torch.nn.Module):

    def __init__(self, n_layers=25, hidden_dim=1024):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dummy = torch.nn.Linear(1, 1)

    def forward(self, wav, output_hidden_states=False):
        B = wav.shape[0]
        T = 375
        hs = tuple(torch.randn(B, T, self.hidden_dim) for _ in range(self.n_layers))
        return FakeMERTOutput(hidden_states=hs, last_hidden_state=hs[-1])


def test_chord_head_shape():
    head = ChordHead(1024, 128, 12)
    x = torch.randn(4, 1024)
    out = head(x)
    assert out.shape == (4, 12)


def test_extension_head_shape():
    head = ExtensionHead(1024 + 12 + 6, 256, 64, 15, 0.3)
    x = torch.randn(4, 1024 + 12 + 6)
    out = head(x)
    assert out.shape == (4, 15)


def test_bass_head_shape():
    head = ChordHead(1024 + 12 + 6 + 15, 128, 13)
    x = torch.randn(4, 1024 + 12 + 6 + 15)
    out = head(x)
    assert out.shape == (4, 13)


def test_chord_model_forward():
    cfg = ModelCfg(
        name="fake",
        n_layers=25,
        hidden_dim=1024,
        n_root=12,
        n_quality=6,
        n_ext=15,
        n_bass=13,
        head_hidden=128,
        ext_head_hidden=256,
        ext_head_mid=64,
        bass_head_hidden=128,
        dropout=0.3,
        detach_logits=True,
        pooling="mean",
    )

    with patch("src.model.AutoModel") as mock_auto:
        mock_auto.from_pretrained.return_value = FakeMERT(25, 1024)
        model = ChordModel(cfg)

    wav = torch.randn(2, 120000)
    r, q, e, b = model(wav)

    assert r.shape == (2, 12)
    assert q.shape == (2, 6)
    assert e.shape == (2, 15)
    assert b.shape == (2, 13)


def test_chord_model_center_pooling():
    cfg = ModelCfg(
        name="fake",
        n_layers=25,
        hidden_dim=1024,
        n_root=12,
        n_quality=6,
        n_ext=15,
        n_bass=13,
        head_hidden=128,
        ext_head_hidden=256,
        ext_head_mid=64,
        bass_head_hidden=128,
        dropout=0.3,
        detach_logits=False,
        pooling="center",
    )

    with patch("src.model.AutoModel") as mock_auto:
        mock_auto.from_pretrained.return_value = FakeMERT(25, 1024)
        model = ChordModel(cfg)

    wav = torch.randn(2, 120000)
    r, q, e, b = model(wav)
    assert r.shape == (2, 12)
    assert b.shape == (2, 13)


def test_chord_model_frame_level():
    cfg = ModelCfg(
        name="fake",
        n_layers=25,
        hidden_dim=1024,
        n_root=12,
        n_quality=6,
        n_ext=15,
        n_bass=13,
        head_hidden=128,
        ext_head_hidden=256,
        ext_head_mid=64,
        bass_head_hidden=128,
        dropout=0.3,
        detach_logits=True,
        pooling="mean",
        frame_level=True,
    )

    with patch("src.model.AutoModel") as mock_auto:
        mock_auto.from_pretrained.return_value = FakeMERT(25, 1024)
        model = ChordModel(cfg)

    wav = torch.randn(2, 120000)
    r, q, e, b = model(wav)

    assert r.shape == (2, 375, 12)
    assert q.shape == (2, 375, 6)
    assert e.shape == (2, 375, 15)
    assert b.shape == (2, 375, 13)


def test_trainable_params_no_lora():
    cfg = ModelCfg(name="fake")

    with patch("src.model.AutoModel") as mock_auto:
        mock_auto.from_pretrained.return_value = FakeMERT(25, 1024)
        model = ChordModel(cfg)

    groups = model.get_trainable_params()
    assert len(groups) == 1
    assert groups[0]["lr_key"] == "lr_head"

    total = sum(p.numel() for p in groups[0]["params"])
    assert total > 0

