from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses import FocalLoss, ChordLoss


def test_focal_loss_basic():
    fl = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 12)
    targets = torch.randint(0, 12, (8,))
    loss = fl(logits, targets)
    assert loss.shape == ()
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_focal_loss_all_masked():
    fl = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 12)
    targets = torch.full((8,), -100, dtype=torch.long)
    loss = fl(logits, targets)
    assert loss.item() == 0.0
    assert loss.requires_grad


def test_focal_loss_partial_mask():
    fl = FocalLoss(gamma=2.0)
    logits = torch.randn(8, 6)
    targets = torch.tensor([0, -100, 2, -100, 4, 5, -100, 1])
    loss = fl(logits, targets)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_focal_loss_with_weight():
    w = torch.ones(12)
    w[0] = 5.0
    fl = FocalLoss(gamma=2.0, weight=w)
    logits = torch.randn(16, 12)
    targets = torch.randint(0, 12, (16,))
    loss = fl(logits, targets)
    assert not torch.isnan(loss)


def test_chord_loss_shapes():
    cl = ChordLoss(weights=[0.3, 0.25, 0.25, 0.2], focal_gamma=2.0)
    r = torch.randn(4, 12)
    q = torch.randn(4, 6)
    e = torch.randn(4, 15)
    b = torch.randn(4, 13)
    rt = torch.randint(0, 12, (4,))
    qt = torch.randint(0, 6, (4,))
    et = torch.randint(0, 15, (4,))
    bt = torch.randint(0, 13, (4,))

    loss, details = cl(r, q, e, b, rt, qt, et, bt)
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert "loss_root" in details
    assert "loss_quality" in details
    assert "loss_ext" in details
    assert "loss_bass" in details
    assert "loss_total" in details


def test_chord_loss_all_masked_no_nan():
    cl = ChordLoss()
    r = torch.randn(4, 12)
    q = torch.randn(4, 6)
    e = torch.randn(4, 15)
    b = torch.randn(4, 13)
    rt = torch.full((4,), -100, dtype=torch.long)
    qt = torch.full((4,), -100, dtype=torch.long)
    et = torch.full((4,), -100, dtype=torch.long)
    bt = torch.full((4,), -100, dtype=torch.long)

    loss, details = cl(r, q, e, b, rt, qt, et, bt)
    assert not torch.isnan(loss)


def test_chord_loss_partial_labels():
    cl = ChordLoss()
    r = torch.randn(4, 12)
    q = torch.randn(4, 6)
    e = torch.randn(4, 15)
    b = torch.randn(4, 13)
    rt = torch.tensor([0, 1, -100, 3])
    qt = torch.tensor([0, -100, 2, 3])
    et = torch.tensor([-100, 1, 2, -100])
    bt = torch.tensor([12, 4, -100, 9])

    loss, details = cl(r, q, e, b, rt, qt, et, bt)
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_chord_loss_backward():
    cl = ChordLoss()
    r = torch.randn(4, 12, requires_grad=True)
    q = torch.randn(4, 6, requires_grad=True)
    e = torch.randn(4, 15, requires_grad=True)
    b = torch.randn(4, 13, requires_grad=True)
    rt = torch.randint(0, 12, (4,))
    qt = torch.randint(0, 6, (4,))
    et = torch.randint(0, 15, (4,))
    bt = torch.randint(0, 13, (4,))

    loss, _ = cl(r, q, e, b, rt, qt, et, bt)
    loss.backward()
    assert r.grad is not None
    assert q.grad is not None
    assert e.grad is not None
    assert b.grad is not None

