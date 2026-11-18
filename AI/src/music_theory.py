from __future__ import annotations

import re
import numpy as np

N_ROOTS: int = 12
ROOT_SILENCE_ID: int = 12
N_ROOT_CLASSES_V5: int = 13  # 0..11 = C..B, 12 = N (silence / no-sound)
BASS_NONE_ID: int = 12
N_BASS_CLASSES: int = 13
N_KEYS: int = 24  # 0..11 = C..B Major, 12..23 = C..B Minor

ROOT_NAMES: list[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOT_NAMES_FLAT: list[str] = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Exact 41 valid (quality_id, ext_id) combinations matching DE/data/export/v3/classes.json
VALID_COMBO_TUPLES: tuple[tuple[int, int], ...] = (
    # major (0) — 9 combos
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 6), (0, 8), (0, 10), (0, 11), (0, 12),
    # minor (1) — 10 combos
    (1, 0), (1, 3), (1, 4), (1, 6), (1, 7), (1, 8), (1, 10), (1, 11), (1, 12), (1, 13),
    # dominant (2) — 11 combos
    (2, 1), (2, 2), (2, 5), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14),
    # diminished (3) — 3 combos
    (3, 0), (3, 4), (3, 5),
    # sus2 (4) — 3 combos
    (4, 0), (4, 3), (4, 5),
    # sus4 (5) — 5 combos
    (5, 0), (5, 3), (5, 5), (5, 8), (5, 12),
)
N_TYPES: int = len(VALID_COMBO_TUPLES)  # 41

QE_TO_TYPE: np.ndarray = np.full((6, 15), -1, dtype=np.int64)
TYPE_TO_QE: np.ndarray = np.zeros((N_TYPES, 2), dtype=np.int64)
for _tid, (_q, _e) in enumerate(VALID_COMBO_TUPLES):
    QE_TO_TYPE[_q, _e] = _tid
    TYPE_TO_QE[_tid, 0] = _q
    TYPE_TO_QE[_tid, 1] = _e

# 12 Jazz Guide-Tone Skeleton Families partitioning all 41 valid chord types
SKELETON_NAMES: tuple[str, ...] = (
    "MAJ_SHELL",   # 0: maj, add9
    "MAJ7_SHELL",  # 1: maj7, maj9, maj11, maj7#11, maj13
    "DOM7_NAT",    # 2: 7, 9, 11, 13
    "DOM7_ALT",    # 3: 7b5, 7#5, 7b9, 7#9, 7#11, 7b13, 7alt
    "MIN_SHELL",   # 4: m, madd9
    "MIN7_SHELL",  # 5: m7, m7b9, m9, m11, m7#11, m13, m7b13
    "MIN_MAJ7",    # 6: m(maj7)
    "HDIM_SHELL",  # 7: m7b5
    "DIM_SHELL",   # 8: dim, dim7
    "AUG_SHELL",   # 9: b5, aug
    "SUS2_SHELL",  # 10: sus2, maj7sus2, 7sus2
    "SUS4_SHELL",  # 11: sus4, maj7sus4, 7sus4, 9sus4, 13sus4
)
N_SKELETONS: int = len(SKELETON_NAMES)  # 12
N_CRF_STATES: int = 1 + N_ROOTS * N_SKELETONS  # 145 (state 0 = N, states 1..144 = 12 roots * 12 skeletons)

_QE_TO_SKEL_MAP: dict[tuple[int, int], int] = {
    (0, 0): 0, (0, 6): 0,
    (0, 3): 1, (0, 8): 1, (0, 10): 1, (0, 11): 1, (0, 12): 1,
    (2, 5): 2, (2, 8): 2, (2, 10): 2, (2, 12): 2,
    (2, 1): 3, (2, 2): 3, (2, 7): 3, (2, 9): 3, (2, 11): 3, (2, 13): 3, (2, 14): 3,
    (1, 0): 4, (1, 6): 4,
    (1, 4): 5, (1, 7): 5, (1, 8): 5, (1, 10): 5, (1, 11): 5, (1, 12): 5, (1, 13): 5,
    (1, 3): 6,
    (3, 4): 7,
    (3, 0): 8, (3, 5): 8,
    (0, 1): 9, (0, 2): 9,
    (4, 0): 10, (4, 3): 10, (4, 5): 10,
    (5, 0): 11, (5, 3): 11, (5, 5): 11, (5, 8): 11, (5, 12): 11,
}

# Fallback skeleton when `ext_id` is missing (`partial` chords like C6 -> MAJ_SHELL, Cm6 -> MIN_SHELL)
QUALITY_FALLBACK_SKEL: np.ndarray = np.array([0, 4, 2, 8, 10, 11], dtype=np.int64)

TYPE_TO_SKEL: np.ndarray = np.zeros(N_TYPES, dtype=np.int64)
SKEL_TYPE_MASK: np.ndarray = np.zeros((N_SKELETONS, N_TYPES), dtype=bool)
for _tid, _qe in enumerate(VALID_COMBO_TUPLES):
    _sid = _QE_TO_SKEL_MAP[_qe]
    TYPE_TO_SKEL[_tid] = _sid
    SKEL_TYPE_MASK[_sid, _tid] = True

# Tension / color types that require sustained upper-voice evidence to upgrade over the base shell type
SKEL_BASE_TYPE: np.ndarray = np.array(
    [
        QE_TO_TYPE[0, 0],  # 0: MAJ_SHELL -> maj ("")
        QE_TO_TYPE[0, 3],  # 1: MAJ7_SHELL -> maj7
        QE_TO_TYPE[2, 5],  # 2: DOM7_NAT -> 7
        QE_TO_TYPE[2, 7],  # 3: DOM7_ALT -> 7b9 (default altered representative; resolved by max prob)
        QE_TO_TYPE[1, 0],  # 4: MIN_SHELL -> m
        QE_TO_TYPE[1, 4],  # 5: MIN7_SHELL -> m7
        QE_TO_TYPE[1, 3],  # 6: MIN_MAJ7 -> m(maj7)
        QE_TO_TYPE[3, 4],  # 7: HDIM_SHELL -> m7b5
        QE_TO_TYPE[3, 0],  # 8: DIM_SHELL -> dim
        QE_TO_TYPE[0, 2],  # 9: AUG_SHELL -> aug
        QE_TO_TYPE[4, 0],  # 10: SUS2_SHELL -> sus2
        QE_TO_TYPE[5, 0],  # 11: SUS4_SHELL -> sus4
    ],
    dtype=np.int64,
)

# Relative pitch-class index table: REL_PITCH_INDEX[r, k] = (r + k) % 12
REL_PITCH_INDEX: np.ndarray = (np.arange(12)[:, None] + np.arange(12)[None, :]) % 12


def _build_relative_interval_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(shell_mask, full_mask, tension_only_mask)` of shape `(41, 12)` in relative semitones."""
    shell = np.zeros((N_TYPES, 12), dtype=np.float32)
    full = np.zeros((N_TYPES, 12), dtype=np.float32)

    base_triads = {
        0: [0, 4, 7],   # maj
        1: [0, 3, 7],   # min
        2: [0, 4, 7, 10],  # dom (implies b7)
        3: [0, 3, 6],   # dim
        4: [0, 2, 7],   # sus2
        5: [0, 5, 7],   # sus4
    }

    for tid, (q, e) in enumerate(VALID_COMBO_TUPLES):
        sh = set(base_triads[q])
        if e == 1:  # b5
            sh.discard(7)
            sh.add(6)
        elif e == 2:  # #5
            sh.discard(7)
            sh.add(8)

        # 7th guide tones
        if e == 3:  # maj7
            sh.add(11)
        elif e == 4:  # min7
            sh.add(10)
        elif e == 5:  # dom7 (dim + dom7 = dim7 -> bb7 = 9)
            sh.add(9 if q == 3 else 10)
        elif e in (7, 8, 9, 10, 11, 12, 13, 14):
            # Extended chords (9/11/13/alt) imply a 7th in standard harmony
            if q == 0:
                sh.add(11)
            elif q in (1, 2, 5):
                sh.add(10)

        fu = set(sh)
        if e == 6:  # add9 (no 7th)
            fu.add(2)
        elif e == 7:  # b9
            fu.add(1)
        elif e == 8:  # nat9
            fu.add(2)
        elif e == 9:  # #9
            fu.add(3)
        elif e == 10:  # nat11
            fu.update((2, 5))
        elif e == 11:  # #11
            fu.update((2, 6))
        elif e == 12:  # nat13
            fu.update((2, 9))
        elif e == 13:  # b13
            fu.add(8)
        elif e == 14:  # alt (b9, #9, #11/b5, b13/#5)
            fu.update((1, 3, 6, 8))

        for iv in sh:
            shell[tid, iv % 12] = 1.0
        for iv in fu:
            full[tid, iv % 12] = 1.0

    tension_only = np.clip(full - shell, 0.0, 1.0)
    return shell, full, tension_only


TYPE_SHELL_REL_MASK, TYPE_FULL_REL_MASK, TYPE_TENSION_REL_MASK = _build_relative_interval_masks()

_DEGREE_SEMITONES: dict[str, int] = {
    "1": 0, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4, "4": 5, "#4": 6,
    "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9, "bb7": 9, "b7": 10, "7": 11,
    "b9": 1, "9": 2, "#9": 3, "11": 5, "#11": 6, "b13": 8, "13": 9,
}
_PAREN_RE = re.compile(r"\(([^()]*)\)")


def chord_chroma_targets(
    root_id: int,
    quality_id: int,
    ext_id: int,
    bass_id: int = 12,
    label_raw: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 12-dim `(shell_chroma, full_chroma)` multi-hot targets in `[0, 1]`."""
    shell = np.zeros(12, dtype=np.float32)
    full = np.zeros(12, dtype=np.float32)
    if root_id < 0 or root_id >= 12:
        return shell, full

    tid = QE_TO_TYPE[quality_id, ext_id] if (0 <= quality_id < 6 and 0 <= ext_id < 15) else -1
    if tid >= 0:
        shell = np.roll(TYPE_SHELL_REL_MASK[tid], root_id)
        full = np.roll(TYPE_FULL_REL_MASK[tid], root_id)
    elif 0 <= quality_id < 6:
        # Partial chord fallback (e.g., C6 / Cm6)
        fb_tid = int(SKEL_BASE_TYPE[QUALITY_FALLBACK_SKEL[quality_id]])
        shell = np.roll(TYPE_SHELL_REL_MASK[fb_tid], root_id)
        full = shell.copy()
    else:
        shell[root_id] = 1.0
        full[root_id] = 1.0

    if 0 <= bass_id < 12:
        shell[bass_id] = 1.0
        full[bass_id] = 1.0

    if label_raw:
        raw_head = label_raw.split("/")[0]
        if "6" in raw_head:
            full[(root_id + 9) % 12] = 1.0
        m = _PAREN_RE.search(raw_head)
        if m:
            for tok in m.group(1).split(","):
                tok = tok.strip()
                if tok in _DEGREE_SEMITONES:
                    full[(root_id + _DEGREE_SEMITONES[tok]) % 12] = 1.0

    return shell, full


def encode_crf_skeleton_state(root_id: int, quality_id: int, ext_id: int) -> tuple[int, int, int]:
    """Return `(type_id, skel_id, crf_state)` given `(root_id, quality_id, ext_id)`."""
    if root_id == ROOT_SILENCE_ID:
        return -100, -100, 0
    if root_id < 0 or root_id >= 12 or quality_id < 0 or quality_id >= 6:
        return -100, -100, -100

    if 0 <= ext_id < 15:
        tid = int(QE_TO_TYPE[quality_id, ext_id])
        if tid >= 0:
            sid = int(TYPE_TO_SKEL[tid])
            return tid, sid, 1 + root_id * N_SKELETONS + sid

    # Partial chord with valid root & quality (e.g. maj6 / min6)
    sid = int(QUALITY_FALLBACK_SKEL[quality_id])
    return -100, sid, 1 + root_id * N_SKELETONS + sid


def decode_crf_skeleton_state(crf_state: int) -> tuple[int, int]:
    """Inverse of `encode_crf_skeleton_state`: returns `(root_id, skel_id)` (`root_id=12` for `N`)."""
    if crf_state <= 0:
        return ROOT_SILENCE_ID, -1
    idx = int(crf_state) - 1
    return idx // N_SKELETONS, idx % N_SKELETONS


# Krumhansl-Kessler + Diatonic Root Bonus profiles for local key estimation (24 keys)
_KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
_KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)
_KK_MAJ = (_KK_MAJ - _KK_MAJ.mean()) / _KK_MAJ.std()
_KK_MIN = (_KK_MIN - _KK_MIN.mean()) / _KK_MIN.std()

# Diatonic chord match table: (rel_root, quality_id) -> harmonic bonus in Major vs Minor key
_DIATONIC_MAJ_BONUS: dict[tuple[int, int], float] = {
    (0, 0): 2.5,   # I
    (2, 1): 1.5,   # ii
    (4, 1): 1.2,   # iii
    (5, 0): 1.8,   # IV
    (7, 0): 2.0, (7, 2): 2.2, (7, 5): 1.5,  # V / V7 / Vsus4
    (9, 1): 1.6,   # vi
    (11, 3): 1.2,  # vii°
}
_DIATONIC_MIN_BONUS: dict[tuple[int, int], float] = {
    (0, 1): 2.5,   # i
    (2, 3): 1.6,   # iiø
    (3, 0): 1.5,   # bIII
    (5, 1): 1.7,   # iv
    (7, 1): 1.4, (7, 2): 2.2, (7, 0): 1.6,  # v / V7
    (8, 0): 1.5,   # bVI
    (10, 0): 1.4, (10, 2): 1.4,  # bVII
    (11, 3): 1.3,  # vii°7
}

FLAT_KEY_IDS: frozenset[int] = frozenset({
    # Major: F(5), Bb(10), Eb(3), Ab(8), Db(1)
    5, 10, 3, 8, 1,
    # Minor (12 + tonic): Dm(14), Gm(19), Cm(12), Fm(17), Bbm(22), Ebm(15)
    14, 19, 12, 17, 22, 15,
})


def estimate_local_keys(
    starts_ms: np.ndarray,
    ends_ms: np.ndarray,
    roots: np.ndarray,
    quals: np.ndarray,
    exts: np.ndarray,
    half_window_ms: float = 8000.0,
) -> np.ndarray:
    """Estimate `local_key_id` (`0..11` Maj, `12..23` Min) per chord event using a sliding window."""
    n = len(starts_ms)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    mids = 0.5 * (starts_ms + ends_ms)
    durs = np.maximum(100.0, ends_ms - starts_ms) / 1000.0

    event_pcs = np.zeros((n, 12), dtype=np.float64)
    for i in range(n):
        r, q, e = int(roots[i]), int(quals[i]), int(exts[i])
        if r < 0 or r >= 12:
            continue
        sh, _ = chord_chroma_targets(r, q, e)
        event_pcs[i] = sh
        event_pcs[i, r] += 0.75

    out_keys = np.zeros(n, dtype=np.int64)
    for i in range(n):
        dist = np.abs(mids - mids[i])
        w = durs * np.exp(-0.5 * (dist / max(half_window_ms, 1000.0)) ** 2)
        pc_hist = (w[:, None] * event_pcs).sum(axis=0)
        if pc_hist.sum() <= 1e-8:
            out_keys[i] = 0
            continue
        pc_norm = (pc_hist - pc_hist.mean()) / (pc_hist.std() + 1e-8)

        best_k = 0
        best_s = -1e18
        near = np.nonzero(w > 0.05 * w.max())[0]
        for tonic in range(12):
            rot = np.roll(pc_norm, -tonic)
            s_maj = float(np.dot(rot, _KK_MAJ))
            s_min = float(np.dot(rot, _KK_MIN))
            d_maj = 0.0
            d_min = 0.0
            w_sum = float(w[near].sum()) + 1e-8
            for j in near:
                rj, qj = int(roots[j]), int(quals[j])
                if 0 <= rj < 12 and 0 <= qj < 6:
                    rel_r = (rj - tonic) % 12
                    d_maj += w[j] * _DIATONIC_MAJ_BONUS.get((rel_r, qj), -0.35)
                    d_min += w[j] * _DIATONIC_MIN_BONUS.get((rel_r, qj), -0.35)
            s_maj += d_maj / w_sum
            s_min += d_min / w_sum
            if s_maj > best_s:
                best_s = s_maj
                best_k = tonic
            if s_min > best_s:
                best_s = s_min
                best_k = 12 + tonic
        out_keys[i] = best_k

    return out_keys
