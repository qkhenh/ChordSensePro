from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.music_theory import (
    N_SKELETONS,
    REL_PITCH_INDEX,
    ROOT_SILENCE_ID,
    SKEL_TYPE_MASK,
    TYPE_TENSION_REL_MASK,
    TYPE_TO_QE,
)


ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITY_NAMES = ["major", "minor", "dominant", "diminished", "sus2", "sus4"]
EXT_NAMES = [
    "none", "b5", "#5", "maj7", "min7", "dom7", "add9", "b9", "nat9",
    "#9", "nat11", "#11", "nat13", "b13", "alt",
]
BASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "none"]

# Diatonic (rel_root, quality_id) pairs in Major (0..11) and Minor (12..23) keys
_DIATONIC_MAJ_SET = {(0, 0), (2, 1), (4, 1), (5, 0), (7, 0), (7, 2), (7, 4), (7, 5), (9, 1), (11, 3), (0, 4), (0, 5), (5, 4)}
_DIATONIC_MIN_SET = {(0, 1), (2, 3), (3, 0), (5, 1), (7, 1), (7, 2), (7, 0), (8, 0), (10, 0), (10, 2), (11, 3)}

log = logging.getLogger(__name__)


def _masked(pred: np.ndarray, true: np.ndarray, ignore: int = -100):
    mask = true != ignore
    return pred[mask], true[mask]


@torch.no_grad()
def decode_v5_batch(
    outputs: dict[str, torch.Tensor],
    crf_module: torch.nn.Module,
    crf_stride: int = 4,
    slash_bass_threshold: float = 0.45,
) -> dict[str, torch.Tensor]:
    """Decode `ChordModelV5` outputs into frame-level `(root, quality, ext, bass, skel, key)` tensors."""
    crf_emissions = outputs["crf_emissions"]              # (B, T_crf, 145)
    type_logits_by_root = outputs["type_logits_by_root"]  # (B, T, 12, 41)
    bass_logits = outputs["bass_logits"]                  # (B, T, 13)
    key_logits = outputs["key_logits"]                    # (B, T, 24)
    sustain_env = outputs["sustain_env"]                  # (B, T, 12)

    bsz, seq_len = type_logits_by_root.shape[:2]
    dev = type_logits_by_root.device

    crf_states_ds = crf_module.decode(crf_emissions)  # (B, T_crf)
    if crf_stride > 1:
        crf_states = crf_states_ds.repeat_interleave(crf_stride, dim=1)[:, :seq_len]
        if crf_states.shape[1] < seq_len:
            crf_states = F.pad(crf_states, (0, seq_len - crf_states.shape[1]), value=0)
    else:
        crf_states = crf_states_ds[:, :seq_len]

    is_silence = crf_states <= 0
    active_idx = (crf_states - 1).clamp(min=0)
    root_pred = torch.where(
        is_silence,
        torch.full_like(crf_states, ROOT_SILENCE_ID),
        active_idx // N_SKELETONS,
    )
    skel_pred = torch.where(
        is_silence,
        torch.full_like(crf_states, -1),
        active_idx % N_SKELETONS,
    )

    # Gather conditional type logits for chosen root `r_p`
    safe_r = root_pred.clamp(0, 11)
    safe_s = skel_pred.clamp(0, N_SKELETONS - 1)
    b_idx = torch.arange(bsz, device=dev)[:, None].expand(bsz, seq_len)
    t_idx = torch.arange(seq_len, device=dev)[None, :].expand(bsz, seq_len)
    chosen_type_logits = type_logits_by_root[b_idx, t_idx, safe_r].float()  # (B, T, 41)

    # Sustain-aware tension gating: penalize tension-only extensions if upper note is not sustained
    rel_idx_t = torch.tensor(REL_PITCH_INDEX, dtype=torch.long, device=dev)
    tens_mask_t = torch.tensor(TYPE_TENSION_REL_MASK, dtype=torch.float32, device=dev)  # (41, 12)
    has_tension = tens_mask_t.sum(dim=-1) > 0.5  # (41,)
    rel_sust = sustain_env.float()[b_idx, t_idx][:, :, rel_idx_t]  # (B, T, 12_root, 12_rel)
    chosen_rel_sust = rel_sust[b_idx, t_idx, safe_r]               # (B, T, 12_rel)
    # Mean sustain over tension intervals of each chord type
    tens_sustain = torch.matmul(chosen_rel_sust, tens_mask_t.T) / tens_mask_t.sum(dim=-1).clamp(min=1.0)
    weak_tension = (tens_sustain < 0.14) & has_tension[None, None, :]
    chosen_type_logits = torch.where(weak_tension, chosen_type_logits - 0.45, chosen_type_logits)

    skel_mask_t = torch.tensor(SKEL_TYPE_MASK, dtype=torch.bool, device=dev)  # (12, 41)
    allowed_types = skel_mask_t[safe_s]  # (B, T, 41)
    masked_type_logits = chosen_type_logits.masked_fill(~allowed_types, -1e9)
    best_type = masked_type_logits.argmax(dim=-1)  # (B, T)

    type_to_qe_t = torch.tensor(TYPE_TO_QE, dtype=torch.long, device=dev)
    qual_pred = torch.where(is_silence, torch.full_like(best_type, -1), type_to_qe_t[best_type, 0])
    ext_pred = torch.where(is_silence, torch.full_like(best_type, -1), type_to_qe_t[best_type, 1])

    # Slash bass decoding folding `bass == root` into `12` (`none`)
    bass_prob = F.softmax(bass_logits.float(), dim=-1)
    root_bass_p = bass_prob.gather(-1, safe_r.unsqueeze(-1)).squeeze(-1)
    bass_prob_adj = bass_prob.clone()
    bass_prob_adj.scatter_(-1, safe_r.unsqueeze(-1), 0.0)
    bass_prob_adj[:, :, 12] = bass_prob_adj[:, :, 12] + root_bass_p

    slash_p, slash_id = bass_prob_adj[:, :, :12].max(dim=-1)
    use_slash = (~is_silence) & (slash_p >= slash_bass_threshold) & (slash_p > bass_prob_adj[:, :, 12])
    bass_pred = torch.where(use_slash, slash_id, torch.full_like(slash_id, 12))

    key_pred = key_logits.argmax(dim=-1)
    return {
        "root": root_pred,
        "quality": qual_pred,
        "ext": ext_pred,
        "bass": bass_pred,
        "skel": skel_pred,
        "key": key_pred,
    }


def compute_metrics(
    root_pred: np.ndarray,
    root_true: np.ndarray,
    qual_pred: np.ndarray,
    qual_true: np.ndarray,
    ext_pred: np.ndarray,
    ext_true: np.ndarray,
    bass_pred: np.ndarray | None = None,
    bass_true: np.ndarray | None = None,
    skel_pred: np.ndarray | None = None,
    skel_true: np.ndarray | None = None,
    key_pred: np.ndarray | None = None,
    key_true: np.ndarray | None = None,
) -> dict[str, float]:
    results: dict[str, float] = {}

    # Standard 12-root accuracy evaluated strictly on active chord frames (`0..11`) for exact v4 comparability
    chord_root_mask = (root_true >= 0) & (root_true < 12)
    if chord_root_mask.sum() > 0:
        rt_c, rp_c = root_true[chord_root_mask], root_pred[chord_root_mask]
        results["root_acc"] = float(accuracy_score(rt_c, rp_c))
        results["root_f1_macro"] = float(
            f1_score(rt_c, rp_c, labels=list(range(12)), average="macro", zero_division=0)
        )
    else:
        results["root_acc"] = 0.0
        results["root_f1_macro"] = 0.0

    # Silence (`N` = 12) accuracy if silent ground-truth frames exist
    sil_mask = root_true == ROOT_SILENCE_ID
    if sil_mask.sum() > 0:
        results["silence_acc"] = float((root_pred[sil_mask] == ROOT_SILENCE_ID).mean())

    qp, qt = _masked(qual_pred, qual_true)
    if len(qt) > 0:
        results["quality_acc"] = float(accuracy_score(qt, qp))
        results["quality_f1_macro"] = float(
            f1_score(qt, qp, labels=list(range(6)), average="macro", zero_division=0)
        )
    else:
        results["quality_acc"] = 0.0
        results["quality_f1_macro"] = 0.0

    ep, et = _masked(ext_pred, ext_true)
    if len(et) > 0:
        results["ext_acc"] = float(accuracy_score(et, ep))
        results["ext_f1_macro"] = float(
            f1_score(et, ep, labels=list(range(15)), average="macro", zero_division=0)
        )
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

    if skel_pred is not None and skel_true is not None:
        sk_mask = (root_true >= 0) & (root_true < 12) & (skel_true >= 0)
        if sk_mask.sum() > 0:
            skel_ok = (root_pred[sk_mask] == root_true[sk_mask]) & (skel_pred[sk_mask] == skel_true[sk_mask])
            results["skeleton_acc"] = float(skel_ok.mean())

    if key_pred is not None and key_true is not None:
        kp, kt = _masked(key_pred, key_true)
        if len(kt) > 0:
            exact_k = kp == kt
            # Relative major/minor equivalence (e.g. C Major == A Minor)
            rel_maj_for_min = ((kt - 12 + 3) % 12)
            rel_min_for_maj = 12 + ((kt + 9) % 12)
            rel_equiv = np.where(kt < 12, kp == rel_min_for_maj, kp == rel_maj_for_min)
            results["key_acc"] = float(exact_k.mean())
            results["key_rel_acc"] = float((exact_k | rel_equiv).mean())

        # Out-of-key quality hallucination rate on diatonic chords (e.g. predicting `Cm` when truth is diatonic `C` in C Major)
        dia_mask = (root_true >= 0) & (root_true < 12) & (qual_true >= 0) & (key_true >= 0) & (root_pred >= 0) & (root_pred < 12) & (qual_pred >= 0)
        if dia_mask.sum() > 0:
            rt_d = root_true[dia_mask]
            qt_d = qual_true[dia_mask]
            rp_d = root_pred[dia_mask]
            qp_d = qual_pred[dia_mask]
            kt_d = key_true[dia_mask]
            tonic = kt_d % 12
            is_maj = kt_d < 12
            true_dia = np.zeros(len(rt_d), dtype=bool)
            pred_nondia = np.zeros(len(rt_d), dtype=bool)
            for idx_i in range(len(rt_d)):
                d_set = _DIATONIC_MAJ_SET if is_maj[idx_i] else _DIATONIC_MIN_SET
                true_dia[idx_i] = ((int(rt_d[idx_i] - tonic[idx_i]) % 12), int(qt_d[idx_i])) in d_set
                pred_nondia[idx_i] = ((int(rp_d[idx_i] - tonic[idx_i]) % 12), int(qp_d[idx_i])) not in d_set
            if true_dia.sum() > 0:
                results["out_of_key_hallucination"] = float(pred_nondia[true_dia].mean())

    mask_all = (root_true >= 0) & (root_true < 12) & (qual_true != -100) & (ext_true != -100)
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
        lines.append(f"{k:28s}: {v:.4f}")
    return "\n".join(lines)


def evaluate_split(
    cfg_path: str,
    checkpoint_path: str,
    split: str = "test",
    output_json: str | None = None,
) -> dict[str, float]:
    """Run full-song tiled evaluation on `val` or `test` split using a PyTorch checkpoint (`v2..v5`)."""
    from torch.amp import autocast
    from torch.utils.data import DataLoader

    from src.config import load_config
    from src.dataset import ChordDataset
    from src.losses import ChordLoss, ChordLossV5
    from src.model import build_model
    from src.train import load_checkpoint

    cfg = load_config(cfg_path)
    is_v5 = getattr(cfg.model, "arch_version", 4) >= 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ChordDataset(cfg, split=split)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg.model, sample_rate=cfg.sample_rate)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any("lora_A" in k for k in state.keys()):
        model.apply_lora(cfg.lora)
    load_checkpoint(Path(checkpoint_path), model, strict=True)
    model = model.to(device).eval()

    use_amp = cfg.train.fp16 and device.type == "cuda"

    if is_v5:
        criterion_v5 = ChordLossV5(
            weights=cfg.train.loss_weights,
            focal_gamma=cfg.train.focal_gamma,
            crf_stride=cfg.model.crf_stride,
        ).to(device)
        all_rp, all_rt = [], []
        all_qp, all_qt = [], []
        all_ep, all_et = [], []
        all_bp, all_bt = [], []
        all_sp, all_st = [], []
        all_kp, all_kt = [], []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                gpu_batch = {
                    k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                with autocast("cuda", enabled=use_amp):
                    outputs = model(gpu_batch["wav"], frame_rms_db=gpu_batch["frame_rms_db"])
                    loss, _ = criterion_v5(outputs, gpu_batch, model.crf)

                preds = decode_v5_batch(outputs, model.crf, crf_stride=cfg.model.crf_stride)
                T = min(preds["root"].shape[1], gpu_batch["root"].shape[1])

                total_loss += float(loss.item())
                n_batches += 1
                all_rp.append(preds["root"][:, :T].cpu())
                all_rt.append(gpu_batch["root"][:, :T].cpu())
                all_qp.append(preds["quality"][:, :T].cpu())
                all_qt.append(gpu_batch["quality"][:, :T].cpu())
                all_ep.append(preds["ext"][:, :T].cpu())
                all_et.append(gpu_batch["ext"][:, :T].cpu())
                all_bp.append(preds["bass"][:, :T].cpu())
                all_bt.append(gpu_batch["bass"][:, :T].cpu())
                all_sp.append(preds["skel"][:, :T].cpu())
                all_st.append(gpu_batch["skel_id"][:, :T].cpu())
                all_kp.append(preds["key"][:, :T].cpu())
                all_kt.append(gpu_batch["local_key"][:, :T].cpu())

        metrics = compute_metrics(
            torch.cat(all_rp).reshape(-1).numpy(),
            torch.cat(all_rt).reshape(-1).numpy(),
            torch.cat(all_qp).reshape(-1).numpy(),
            torch.cat(all_qt).reshape(-1).numpy(),
            torch.cat(all_ep).reshape(-1).numpy(),
            torch.cat(all_et).reshape(-1).numpy(),
            torch.cat(all_bp).reshape(-1).numpy(),
            torch.cat(all_bt).reshape(-1).numpy(),
            skel_pred=torch.cat(all_sp).reshape(-1).numpy(),
            skel_true=torch.cat(all_st).reshape(-1).numpy(),
            key_pred=torch.cat(all_kp).reshape(-1).numpy(),
            key_true=torch.cat(all_kt).reshape(-1).numpy(),
        )
    else:
        combo_mask = None
        classes_path = Path(cfg.data_dir) / "classes.json"
        if classes_path.exists():
            meta = json.loads(classes_path.read_text(encoding="utf-8"))
            if "valid_combo_mask" in meta:
                combo_mask = torch.tensor(meta["valid_combo_mask"], dtype=torch.bool, device=device)

        criterion = ChordLoss(weights=cfg.train.loss_weights, focal_gamma=cfg.train.focal_gamma).to(device)
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

                r, q, e, b = r.float(), q.float(), e.float(), b.float()
                rp = r.argmax(-1)
                qp = q.argmax(-1)
                if combo_mask is not None:
                    valid_ext = combo_mask[qp]
                    e_masked = e.masked_fill(~valid_ext, -1e4)
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
