from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITY_NAMES = ["major", "minor", "dominant", "diminished", "sus2", "sus4"]
EXT_NAMES = [
    "none", "b5", "#5", "maj7", "min7", "dom7", "add9", "b9", "nat9",
    "#9", "nat11", "#11", "nat13", "b13", "alt",
]
BASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "none"]

log = logging.getLogger(__name__)


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
    bass_pred: np.ndarray | None = None,
    bass_true: np.ndarray | None = None,
) -> dict[str, float]:
    results: dict[str, float] = {}

    rp, rt = _masked(root_pred, root_true)
    if len(rt) > 0:
        results["root_acc"] = float(accuracy_score(rt, rp))
        results["root_f1_macro"] = float(f1_score(rt, rp, average="macro", zero_division=0))
    else:
        results["root_acc"] = 0.0
        results["root_f1_macro"] = 0.0

    qp, qt = _masked(qual_pred, qual_true)
    if len(qt) > 0:
        results["quality_acc"] = float(accuracy_score(qt, qp))
        results["quality_f1_macro"] = float(f1_score(qt, qp, average="macro", zero_division=0))
    else:
        results["quality_acc"] = 0.0
        results["quality_f1_macro"] = 0.0

    ep, et = _masked(ext_pred, ext_true)
    if len(et) > 0:
        results["ext_acc"] = float(accuracy_score(et, ep))
        results["ext_f1_macro"] = float(f1_score(et, ep, average="macro", zero_division=0))
    else:
        results["ext_acc"] = 0.0
        results["ext_f1_macro"] = 0.0

    if bass_pred is not None and bass_true is not None:
        bp, bt = _masked(bass_pred, bass_true)
        if len(bt) > 0:
            results["bass_acc"] = float(accuracy_score(bt, bp))
            results["bass_f1_macro"] = float(f1_score(bt, bp, average="macro", zero_division=0))
        else:
            results["bass_acc"] = 0.0
            results["bass_f1_macro"] = 0.0

        slash_mask = (bass_true >= 0) & (bass_true < 12)
        if slash_mask.sum() > 0:
            sp, st = bass_pred[slash_mask], bass_true[slash_mask]
            results["bass_slash_acc"] = float(accuracy_score(st, sp))
            results["bass_slash_f1"] = float(
                f1_score(st, sp, labels=list(range(12)), average="macro", zero_division=0)
            )
        else:
            results["bass_slash_acc"] = 0.0
            results["bass_slash_f1"] = 0.0

    mask_all = (root_true != -100) & (qual_true != -100) & (ext_true != -100)
    if bass_pred is not None and bass_true is not None:
        mask_all = mask_all & (bass_true != -100)
        if mask_all.sum() > 0:
            chord_correct = (
                (root_pred[mask_all] == root_true[mask_all])
                & (qual_pred[mask_all] == qual_true[mask_all])
                & (ext_pred[mask_all] == ext_true[mask_all])
                & (bass_pred[mask_all] == bass_true[mask_all])
            )
            results["chord_acc"] = float(chord_correct.mean())
            triad_ext_correct = (
                (root_pred[mask_all] == root_true[mask_all])
                & (qual_pred[mask_all] == qual_true[mask_all])
                & (ext_pred[mask_all] == ext_true[mask_all])
            )
            results["chord_no_bass_acc"] = float(triad_ext_correct.mean())
        else:
            results["chord_acc"] = 0.0
            results["chord_no_bass_acc"] = 0.0
    else:
        if mask_all.sum() > 0:
            chord_correct = (
                (root_pred[mask_all] == root_true[mask_all])
                & (qual_pred[mask_all] == qual_true[mask_all])
                & (ext_pred[mask_all] == ext_true[mask_all])
            )
            results["chord_acc"] = float(chord_correct.mean())
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
            recalls[c] = float((p[mask_c] == c).mean())
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


def evaluate_split(
    cfg_path: str,
    checkpoint_path: str,
    split: str = "test",
    output_json: str | None = None,
) -> dict[str, float]:
    """Run full-song tiled evaluation on `val` or `test` split using a PyTorch checkpoint."""
    import torch
    from torch.amp import autocast
    from torch.utils.data import DataLoader

    from src.config import load_config
    from src.dataset import ChordDataset
    from src.losses import ChordLoss
    from src.model import build_model
    from src.train import load_checkpoint

    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ChordDataset(cfg, split=split)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any("lora_A" in k for k in state.keys()):
        model.apply_lora(cfg.lora)
    load_checkpoint(Path(checkpoint_path), model, strict=True)
    model = model.to(device).eval()

    combo_mask = None
    classes_path = Path(cfg.data_dir) / "classes.json"
    if classes_path.exists():
        meta = json.loads(classes_path.read_text(encoding="utf-8"))
        if "valid_combo_mask" in meta:
            combo_mask = torch.tensor(meta["valid_combo_mask"], dtype=torch.bool, device=device)

    criterion = ChordLoss(weights=cfg.train.loss_weights, focal_gamma=cfg.train.focal_gamma).to(device)
    use_amp = cfg.train.fp16 and device.type == "cuda"

    all_rp, all_rt = [], []
    all_qp, all_qt = [], []
    all_ep, all_et = [], []
    all_bp, all_bt = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            wav = batch["wav"].to(device)
            rt = batch["root"].to(device)
            qt = batch["quality"].to(device)
            et = batch["ext"].to(device)
            bt = batch["bass"].to(device)

            with autocast("cuda", enabled=use_amp):
                r, q, e, b = model(wav)
                loss, _ = criterion(r, q, e, b, rt, qt, et, bt)

            if r.dim() == 3:
                T = min(r.shape[1], rt.shape[1])
                r, q, e, b = r[:, :T], q[:, :T], e[:, :T], b[:, :T]
                rt, qt, et, bt = rt[:, :T], qt[:, :T], et[:, :T], bt[:, :T]

            rp = r.argmax(-1)
            qp = q.argmax(-1)
            if combo_mask is not None:
                valid_ext = combo_mask[qp]
                e_masked = e.masked_fill(~valid_ext, -1e9)
                ep = e_masked.argmax(-1)
            else:
                ep = e.argmax(-1)
            bp = b.argmax(-1)

            total_loss += loss.item()
            n_batches += 1
            all_rp.append(rp.cpu())
            all_rt.append(rt.cpu())
            all_qp.append(qp.cpu())
            all_qt.append(qt.cpu())
            all_ep.append(ep.cpu())
            all_et.append(et.cpu())
            all_bp.append(bp.cpu())
            all_bt.append(bt.cpu())

    metrics = compute_metrics(
        torch.cat(all_rp).reshape(-1).numpy(),
        torch.cat(all_rt).reshape(-1).numpy(),
        torch.cat(all_qp).reshape(-1).numpy(),
        torch.cat(all_qt).reshape(-1).numpy(),
        torch.cat(all_ep).reshape(-1).numpy(),
        torch.cat(all_et).reshape(-1).numpy(),
        torch.cat(all_bp).reshape(-1).numpy(),
        torch.cat(all_bt).reshape(-1).numpy(),
    )
    metrics[f"{split}_loss"] = total_loss / max(n_batches, 1)
    metrics["n_songs"] = len(ds._valid_songs)
    metrics["n_tiles"] = len(ds)

    print(f"\n=== Evaluation on '{split}' ({len(ds._valid_songs)} songs, {len(ds)} tiles) ===")
    print(format_report(metrics))

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        log.info(f"Saved evaluation metrics to {out_p}")

    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Evaluate ChordSense model on val/test split")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--output", default=None, help="Optional path to save JSON report")
    args = ap.parse_args()
    evaluate_split(args.config, args.checkpoint, split=args.split, output_json=args.output)
