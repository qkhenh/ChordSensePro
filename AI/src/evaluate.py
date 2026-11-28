from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.music_theory import (
    N_INVERSIONS_V6,
    N_SKELETONS,
    N_SKELETONS_V6,
    N_TENSIONS_V6,
    REL_PITCH_INDEX,
    ROOT_SILENCE_ID,
    SKEL14_BASE_QE,
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

    # Slash bass decoding folding `bass == root` into `12` (`none`), reinforced by physical `true_bass_logits`
    bass_prob = F.softmax(bass_logits.float(), dim=-1)
    if "true_bass_logits" in outputs:
        tb_prob = F.softmax(outputs["true_bass_logits"].float(), dim=-1)
        true_bass_pred = torch.where(
            is_silence,
            torch.full_like(root_pred, ROOT_SILENCE_ID),
            tb_prob[:, :, :12].argmax(dim=-1),
        )
        tb_13 = F.pad(tb_prob[:, :, :12], (0, 1))
        bass_prob_mix = 0.65 * bass_prob + 0.35 * tb_13
    else:
        true_bass_pred = torch.where(
            is_silence,
            torch.full_like(root_pred, ROOT_SILENCE_ID),
            safe_r,
        )
        bass_prob_mix = bass_prob

    root_bass_p = bass_prob_mix.gather(-1, safe_r.unsqueeze(-1)).squeeze(-1)
    bass_prob_adj = bass_prob_mix.clone()
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
        "true_bass": true_bass_pred,
        "skel": skel_pred,
        "key": key_pred,
    }


@torch.no_grad()
def decode_v6_batch(
    outputs: dict[str, torch.Tensor],
    crf_module: torch.nn.Module,
    crf_stride: int = 2,
    slash_bass_threshold: float = 0.45,
    tension_threshold: float = 0.45,
) -> dict[str, torch.Tensor]:
    """Decode `ChordModelV6` outputs into frame-level `(root, skel14, tension_bits, inversion, bass, true_bass, boundary, key)`."""
    crf_emissions = outputs["crf_emissions"]                      # (B, T_crf, 169)
    boundary_prob_crf = outputs.get("boundary_prob_crf")          # (B, T_crf)
    tension_logits_by_root = outputs["tension_logits_by_root"]    # (B, T, 12, 7)
    inv_logits_by_root = outputs["inv_logits_by_root"]            # (B, T, 12, 5)
    bass_logits = outputs["bass_logits"]                          # (B, T, 13)
    true_bass_logits = outputs["true_bass_logits"]                # (B, T, 13)
    boundary_logits = outputs["boundary_logits"]                  # (B, T, 2)
    key_logits = outputs["key_logits"]                            # (B, T, 24)

    bsz, seq_len = tension_logits_by_root.shape[:2]
    dev = tension_logits_by_root.device

    crf_states_ds = crf_module.decode(crf_emissions, boundary_prob=boundary_prob_crf)
    if crf_stride > 1:
        crf_states = crf_states_ds.repeat_interleave(crf_stride, dim=1)[:, :seq_len]
        if crf_states.shape[1] < seq_len:
            crf_states = F.pad(crf_states, (0, seq_len - crf_states.shape[1]), value=0)
    else:
        crf_states = crf_states_ds[:, :seq_len]

    is_silence = crf_states <= 0
    active_idx = (crf_states - 1).clamp(min=0)
    root_pred = torch.where(is_silence, torch.full_like(crf_states, ROOT_SILENCE_ID), active_idx // N_SKELETONS_V6)
    skel14_pred = torch.where(is_silence, torch.full_like(crf_states, -1), active_idx % N_SKELETONS_V6)

    safe_r = root_pred.clamp(0, 11)
    b_idx = torch.arange(bsz, device=dev)[:, None].expand(bsz, seq_len)
    t_idx = torch.arange(seq_len, device=dev)[None, :].expand(bsz, seq_len)

    t_prob = torch.sigmoid(tension_logits_by_root[b_idx, t_idx, safe_r].float())  # (B, T, 7)
    tension_bits = (t_prob >= tension_threshold) & (~is_silence.unsqueeze(-1))

    inv_pred = torch.where(
        is_silence,
        torch.full_like(root_pred, -1),
        inv_logits_by_root[b_idx, t_idx, safe_r].argmax(dim=-1),
    )

    tb_prob = F.softmax(true_bass_logits.float(), dim=-1)
    true_bass_pred = torch.where(is_silence, torch.full_like(root_pred, ROOT_SILENCE_ID), tb_prob[:, :, :12].argmax(dim=-1))

    bass_prob = F.softmax(bass_logits.float(), dim=-1)
    bass_prob_mix = 0.65 * bass_prob + 0.35 * F.pad(tb_prob[:, :, :12], (0, 1))
    root_bp = bass_prob_mix.gather(-1, safe_r.unsqueeze(-1)).squeeze(-1)
    bass_adj = bass_prob_mix.clone()
    bass_adj.scatter_(-1, safe_r.unsqueeze(-1), 0.0)
    bass_adj[:, :, 12] = bass_adj[:, :, 12] + root_bp
    slash_p, slash_id = bass_adj[:, :, :12].max(dim=-1)
    use_slash = (~is_silence) & (slash_p >= slash_bass_threshold) & (slash_p > bass_adj[:, :, 12])
    bass_pred = torch.where(use_slash, slash_id, torch.full_like(slash_id, 12))

    # Map skel14 -> representative legacy (quality, ext) for quick cross-version tracking
    sk_qe = torch.tensor(SKEL14_BASE_QE, dtype=torch.long, device=dev)
    safe_sk = skel14_pred.clamp(0, N_SKELETONS_V6 - 1)
    qual_pred = torch.where(is_silence, torch.full_like(root_pred, -1), sk_qe[safe_sk, 0])
    ext_pred = torch.where(is_silence, torch.full_like(root_pred, -1), sk_qe[safe_sk, 1])

    return {
        "root": root_pred,
        "skel14": skel14_pred,
        "quality": qual_pred,
        "ext": ext_pred,
        "tension_bits": tension_bits.to(torch.int64),
        "inversion": inv_pred,
        "bass": bass_pred,
        "true_bass": true_bass_pred,
        "boundary_prob": torch.sigmoid(boundary_logits[:, :, 0].float()),
        "key": key_logits.argmax(dim=-1),
    }


def compute_v6_metrics(
    root_pred: np.ndarray,
    root_true: np.ndarray,
    skel_pred: np.ndarray,
    skel_true: np.ndarray,
    tension_pred: np.ndarray,       # (N, 7) int/bool
    tension_true: np.ndarray,       # (N, 7) float/int
    tension_valid: np.ndarray,      # (N,) bool
    inv_pred: np.ndarray,
    inv_true: np.ndarray,
    bass_pred: np.ndarray,
    bass_true: np.ndarray,
    true_bass_pred: np.ndarray,
    true_bass_true: np.ndarray,
    key_pred: np.ndarray | None = None,
    key_true: np.ndarray | None = None,
    bnd_prob: np.ndarray | None = None,
    bnd_true: np.ndarray | None = None,
    chord_dur_ms: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute Model v6 generalization metrics and the Composite Validation Score (`composite_val_score`)."""
    res: dict[str, float] = {}

    chord_mask = (root_true >= 0) & (root_true < 12)
    res["root_acc"] = float(accuracy_score(root_true[chord_mask], root_pred[chord_mask])) if chord_mask.any() else 0.0

    sil_mask = root_true == ROOT_SILENCE_ID
    if sil_mask.any():
        res["silence_acc"] = float((root_pred[sil_mask] == ROOT_SILENCE_ID).mean())

    sk_mask = chord_mask & (skel_true >= 0) & (skel_true < N_SKELETONS_V6)
    if sk_mask.any():
        sk_ok = (root_pred[sk_mask] == root_true[sk_mask]) & (skel_pred[sk_mask] == skel_true[sk_mask])
        res["skel14_acc"] = float(sk_ok.mean())
        res["skeleton_acc"] = res["skel14_acc"]
        # 6th-family F1 (MAJ6_SHELL=1, MIN6_SHELL=6)
        is_six_t = (skel_true[sk_mask] == 1) | (skel_true[sk_mask] == 6)
        is_six_p = (skel_pred[sk_mask] == 1) | (skel_pred[sk_mask] == 6)
        res["sixth_chord_f1"] = float(f1_score(is_six_t.astype(int), is_six_p.astype(int), zero_division=0)) if is_six_t.any() else 0.0
    else:
        res["skel14_acc"] = 0.0
        res["skeleton_acc"] = 0.0
        res["sixth_chord_f1"] = 0.0

    # Short Passing Chord F1 / Accuracy (chords <= 800ms)
    if chord_dur_ms is not None:
        short_mask = sk_mask & (chord_dur_ms > 0.0) & (chord_dur_ms <= 800.0)
        if short_mask.any():
            res["short_passing_chord_f1"] = float(
                ((root_pred[short_mask] == root_true[short_mask]) & (skel_pred[short_mask] == skel_true[short_mask])).mean()
            )
        else:
            res["short_passing_chord_f1"] = res["skel14_acc"]
    else:
        res["short_passing_chord_f1"] = res["skel14_acc"]

    # 7-D Tension Macro F1
    tv = chord_mask & tension_valid.astype(bool)
    if tv.any():
        tp = (tension_pred[tv] > 0.5).astype(int)
        tt = (tension_true[tv] > 0.5).astype(int)
        per_t = [float(f1_score(tt[:, i], tp[:, i], zero_division=0)) for i in range(N_TENSIONS_V6) if tt[:, i].any()]
        res["tension_macro_f1"] = float(np.mean(per_t)) if per_t else 0.0
    else:
        res["tension_macro_f1"] = 0.0

    # TrueBass, SlashBass & 1st-Inversion F1
    tb_mask = (true_bass_true >= 0) & (true_bass_true < 12)
    res["true_bass_acc"] = float(accuracy_score(true_bass_true[tb_mask], true_bass_pred[tb_mask])) if tb_mask.any() else 0.0

    bp, bt = _masked(bass_pred, bass_true)
    res["bass_acc"] = float(accuracy_score(bt, bp)) if len(bt) else 0.0
    slash_mask = (bass_true >= 0) & (bass_true < 12)
    if slash_mask.any():
        res["bass_slash_acc"] = float(accuracy_score(bass_true[slash_mask], bass_pred[slash_mask]))
        res["bass_slash_f1"] = float(
            f1_score(bass_true[slash_mask], bass_pred[slash_mask], labels=list(range(12)), average="macro", zero_division=0)
        )
    else:
        res["bass_slash_acc"] = 0.0
        res["bass_slash_f1"] = 0.0

    inv_mask = (inv_true >= 0) & (inv_true < N_INVERSIONS_V6)
    if inv_mask.any():
        res["slash_inv1_f1"] = float(
            f1_score((inv_true[inv_mask] == 1).astype(int), (inv_pred[inv_mask] == 1).astype(int), zero_division=0)
        )
    else:
        res["slash_inv1_f1"] = 0.0

    if bnd_prob is not None and bnd_true is not None:
        bt_bin = (bnd_true >= 0.5).astype(int)
        bp_bin = (bnd_prob >= 0.45).astype(int)
        res["boundary_f1"] = float(f1_score(bt_bin, bp_bin, zero_division=0)) if bt_bin.any() else 0.0

    if key_pred is not None and key_true is not None:
        kp, kt = _masked(key_pred, key_true)
        if len(kt):
            exact_k = kp == kt
            rel_maj = (kt - 12 + 3) % 12
            rel_min = 12 + ((kt + 9) % 12)
            rel_eq = np.where(kt < 12, kp == rel_min, kp == rel_maj)
            res["key_acc"] = float(exact_k.mean())
            res["key_rel_acc"] = float((exact_k | rel_eq).mean())

    exact_mask = sk_mask & tv & (bass_true != -100)
    if exact_mask.any():
        tp_m = (tension_pred[exact_mask] > 0.5).astype(int)
        tt_m = (tension_true[exact_mask] > 0.5).astype(int)
        exact_ok = (
            (root_pred[exact_mask] == root_true[exact_mask])
            & (skel_pred[exact_mask] == skel_true[exact_mask])
            & np.all(tp_m == tt_m, axis=-1)
            & (bass_pred[exact_mask] == bass_true[exact_mask])
        )
        res["chord_acc"] = float(exact_ok.mean())
    else:
        res["chord_acc"] = res["skel14_acc"]

    # Composite Generalization Score (Section 5.4 of 80_plan_v6_full_retrain.md)
    res["composite_val_score"] = float(
        0.30 * res["skel14_acc"]
        + 0.25 * res["short_passing_chord_f1"]
        + 0.20 * res["bass_slash_f1"]
        + 0.15 * res["tension_macro_f1"]
        + 0.10 * res["sixth_chord_f1"]
    )
    return res


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
    true_bass_pred: np.ndarray | None = None,
    true_bass_true: np.ndarray | None = None,
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

    if true_bass_pred is not None and true_bass_true is not None:
        tb_mask = (true_bass_true >= 0) & (true_bass_true < 12)
        if tb_mask.sum() > 0:
            results["true_bass_acc"] = float(accuracy_score(true_bass_true[tb_mask], true_bass_pred[tb_mask]))

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
    from src.losses import ChordLoss, ChordLossV5, ChordLossV6
    from src.model import build_model
    from src.train import load_checkpoint

    cfg = load_config(cfg_path)
    arch = getattr(cfg.model, "arch_version", 4)
    is_v6 = arch >= 6
    is_v5 = arch >= 5
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

    if is_v6:
        criterion_v6 = ChordLossV6(
            weights=cfg.train.loss_weights,
            focal_gamma=cfg.train.focal_gamma,
            label_smoothing=float(getattr(cfg.train, "label_smoothing", 0.08)),
            crf_stride=cfg.model.crf_stride,
            crf_prior_reg=float(getattr(cfg.train, "crf_prior_reg", 1e-3)),
        ).to(device)
        all_rp, all_rt = [], []
        all_sp, all_st = [], []
        all_tp, all_tt, all_tv = [], [], []
        all_ip, all_it = [], []
        all_bp, all_bt = [], []
        all_tbp, all_tbt = [], []
        all_kp, all_kt = [], []
        all_bndp, all_bndt = [], []
        all_dur = []
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
                    loss, _ = criterion_v6(outputs, gpu_batch, model.crf)

                preds = decode_v6_batch(outputs, model.crf, crf_stride=cfg.model.crf_stride)
                T = min(preds["root"].shape[1], gpu_batch["root"].shape[1])

                total_loss += float(loss.item())
                n_batches += 1
                all_rp.append(preds["root"][:, :T].cpu())
                all_rt.append(gpu_batch["root"][:, :T].cpu())
                all_sp.append(preds["skel14"][:, :T].cpu())
                all_st.append(gpu_batch["skel14"][:, :T].cpu())
                all_tp.append(preds["tension_bits"][:, :T].cpu())
                all_tt.append(gpu_batch["tension_bits"][:, :T].cpu())
                all_tv.append(gpu_batch["tension_valid"][:, :T].cpu())
                all_ip.append(preds["inversion"][:, :T].cpu())
                all_it.append(gpu_batch["inversion"][:, :T].cpu())
                all_bp.append(preds["bass"][:, :T].cpu())
                all_bt.append(gpu_batch["bass"][:, :T].cpu())
                all_tbp.append(preds["true_bass"][:, :T].cpu())
                all_tbt.append(gpu_batch["true_bass"][:, :T].cpu())
                all_kp.append(preds["key"][:, :T].cpu())
                all_kt.append(gpu_batch["local_key"][:, :T].cpu())
                all_bndp.append(preds["boundary_prob"][:, :T].cpu())
                all_bndt.append(gpu_batch["boundary_chord"][:, :T].cpu())
                all_dur.append(gpu_batch["chord_dur_ms"][:, :T].cpu())

        metrics = compute_v6_metrics(
            torch.cat(all_rp).reshape(-1).numpy(),
            torch.cat(all_rt).reshape(-1).numpy(),
            torch.cat(all_sp).reshape(-1).numpy(),
            torch.cat(all_st).reshape(-1).numpy(),
            torch.cat(all_tp).reshape(-1, N_TENSIONS_V6).numpy(),
            torch.cat(all_tt).reshape(-1, N_TENSIONS_V6).numpy(),
            torch.cat(all_tv).reshape(-1).numpy(),
            torch.cat(all_ip).reshape(-1).numpy(),
            torch.cat(all_it).reshape(-1).numpy(),
            torch.cat(all_bp).reshape(-1).numpy(),
            torch.cat(all_bt).reshape(-1).numpy(),
            torch.cat(all_tbp).reshape(-1).numpy(),
            torch.cat(all_tbt).reshape(-1).numpy(),
            key_pred=torch.cat(all_kp).reshape(-1).numpy(),
            key_true=torch.cat(all_kt).reshape(-1).numpy(),
            bnd_prob=torch.cat(all_bndp).reshape(-1).numpy(),
            bnd_true=torch.cat(all_bndt).reshape(-1).numpy(),
            chord_dur_ms=torch.cat(all_dur).reshape(-1).numpy(),
        )
    elif is_v5:
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
        all_tbp, all_tbt = [], []
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
                if "true_bass" in gpu_batch:
                    all_tbp.append(preds["true_bass"][:, :T].cpu())
                    all_tbt.append(gpu_batch["true_bass"][:, :T].cpu())

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
            true_bass_pred=torch.cat(all_tbp).reshape(-1).numpy() if all_tbp else None,
            true_bass_true=torch.cat(all_tbt).reshape(-1).numpy() if all_tbt else None,
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
