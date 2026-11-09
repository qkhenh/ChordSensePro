from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITY_NAMES = ["major", "minor", "dim", "aug", "sus2", "sus4"]
EXT_NAMES = [
    "none", "7", "maj7", "m7", "dim7", "aug7", "9", "maj9", "m9",
    "add9", "6", "m6", "sus4", "dom7", "b5",
]


def _masked(pred: np.ndarray, true: np.ndarray, ignore: int = -100):
    mask = true != ignore
    return pred[mask], true[mask]


def compute_metrics(
    root_pred: np.ndarray,
    root_true: np.ndarray,
    qual_pred: np.ndarray,
    qual_true: np.ndarray,
    ext_pred: np.ndarray,
    ext_true: np.ndarray,
) -> dict[str, float]:
    results: dict[str, float] = {}

    rp, rt = _masked(root_pred, root_true)
    if len(rt) > 0:
        results["root_acc"] = accuracy_score(rt, rp)
        results["root_f1_macro"] = f1_score(rt, rp, average="macro", zero_division=0)
    else:
        results["root_acc"] = 0.0
        results["root_f1_macro"] = 0.0

    qp, qt = _masked(qual_pred, qual_true)
    if len(qt) > 0:
        results["quality_acc"] = accuracy_score(qt, qp)
        results["quality_f1_macro"] = f1_score(qt, qp, average="macro", zero_division=0)
    else:
        results["quality_acc"] = 0.0
        results["quality_f1_macro"] = 0.0

    ep, et = _masked(ext_pred, ext_true)
    if len(et) > 0:
        results["ext_acc"] = accuracy_score(et, ep)
        results["ext_f1_macro"] = f1_score(et, ep, average="macro", zero_division=0)
    else:
        results["ext_acc"] = 0.0
        results["ext_f1_macro"] = 0.0

    mask_all = (root_true != -100) & (qual_true != -100) & (ext_true != -100)
    if mask_all.sum() > 0:
        chord_correct = (
            (root_pred[mask_all] == root_true[mask_all])
            & (qual_pred[mask_all] == qual_true[mask_all])
            & (ext_pred[mask_all] == ext_true[mask_all])
        )
        results["chord_acc"] = chord_correct.mean()
    else:
        results["chord_acc"] = 0.0

    return results


def per_class_recall(
    pred: np.ndarray, true: np.ndarray, n_classes: int, ignore: int = -100
) -> dict[int, float]:
    p, t = _masked(pred, true, ignore)
    recalls = {}
    for c in range(n_classes):
        mask_c = t == c
        if mask_c.sum() == 0:
            recalls[c] = float("nan")
        else:
            recalls[c] = (p[mask_c] == c).mean()
    return recalls


def build_confusion_matrix(
    pred: np.ndarray, true: np.ndarray, n_classes: int, ignore: int = -100
) -> np.ndarray:
    p, t = _masked(pred, true, ignore)
    if len(t) == 0:
        return np.zeros((n_classes, n_classes), dtype=np.int64)
    return confusion_matrix(t, p, labels=list(range(n_classes)))


def format_report(metrics: dict[str, float]) -> str:
    lines = []
    for k in sorted(metrics):
        v = metrics[k]
        lines.append(f"{k:25s}: {v:.4f}")
    return "\n".join(lines)

