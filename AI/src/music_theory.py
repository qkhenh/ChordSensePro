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

# v5.1.2 Taxonomy Cleanup (PA1):
# 7 chords eliminated from user-facing output (`dom11`, `maj11`, `m7b9`, `m7#11`, `m7b13`, `b5`, `7sus2`).
# Note: `(0, 10)` and `(2, 10)` keep their acoustic logits in `ChordTypeHead` because in v3 they were trained on
# `:maj(11)` (`add4`) and `:11` (`9sus4`), then `resolve_jazz_chord_theory` deterministically remaps them.
BANNED_COMBO_TUPLES: tuple[tuple[int, int], ...] = (
    (0, 10),  # maj11 (does not exist -> add4 / maj9#11 / maj9)
    (2, 10),  # dom11 (does not exist -> 9sus4 / 7sus4 / 9)
    (1, 7),   # m7b9
    (1, 11),  # m7#11
    (1, 13),  # m7b13
    (0, 1),   # maj b5
    (4, 5),   # 7sus2
)
DEPRECATED_TYPE_MASK: np.ndarray = np.zeros(N_TYPES, dtype=bool)
for _dq, _de in ((1, 7), (1, 11), (1, 13), (0, 1), (4, 5)):
    _dtid = int(QE_TO_TYPE[_dq, _de])
    if _dtid >= 0:
        DEPRECATED_TYPE_MASK[_dtid] = True

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
    """Return `(shell_mask, full_mask, tension_only_mask)` of shape `(41, 12)` in relative semitones.

    Music-theory rules enforced for 7 / 9 / 11 / 13:
    - Natural `11` (`ext=10` on `maj`/`dom`) omits the clashing major `3rd` (`4` semitones, minor-9th avoid interval)
      and voices `{1, 4(11), 5, 7/b7, 9}` (making `dom 11` acoustically equivalent to `9sus4`).
    - Natural `13` (`ext=12`) treats the `13th` (`9` semitones) as the primary tension in `tension_only_mask`
      so voicings with or without the optional `9th` (`1-3-b7-13` vs `1-3-b7-9-13`) both score full tension support.
    """
    shell = np.zeros((N_TYPES, 12), dtype=np.float32)
    full = np.zeros((N_TYPES, 12), dtype=np.float32)
    tension_only = np.zeros((N_TYPES, 12), dtype=np.float32)

    base_triads = {
        0: [0, 4, 7],      # maj
        1: [0, 3, 7],      # min
        2: [0, 4, 7, 10],  # dom (implies b7)
        3: [0, 3, 6],      # dim
        4: [0, 2, 7],      # sus2
        5: [0, 5, 7],      # sus4
    }

    for tid, (q, e) in enumerate(VALID_COMBO_TUPLES):
        sh = set(base_triads[q])
        if e == 1:  # b5
            sh.discard(7)
            sh.add(6)
        elif e == 2:  # #5
            sh.discard(7)
            sh.add(8)
        elif e == 10 and q in (0, 2):
            # Avoid minor-9th clash between major 3rd (4) and natural 11th (5)
            sh.discard(4)
            sh.add(5)

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
        tens: set[int] = set()
        if e == 6:  # add9 (no 7th)
            fu.add(2)
            tens.add(2)
        elif e == 7:  # b9
            fu.add(1)
            tens.add(1)
        elif e == 8:  # nat9
            fu.add(2)
            tens.add(2)
        elif e == 9:  # #9
            fu.add(3)
            tens.add(3)
        elif e == 10:  # nat11 (on min11, 5 is tension; on dom11/9sus4, 5 is in shell and 2+5 confirm suspension)
            fu.update((2, 5))
            tens.update((5,) if q == 1 else (2, 5))
        elif e == 11:  # #11
            fu.update((2, 6))
            tens.add(6)
        elif e == 12:  # nat13 (9 semitones = 6th/13th is the defining color; 9th is optional in shell voicings)
            fu.update((2, 9))
            tens.add(9)
        elif e == 13:  # b13
            fu.add(8)
            tens.add(8)
        elif e == 14:  # alt (b9, #9, #11/b5, b13/#5)
            fu.update((1, 3, 6, 8))
            tens.update((1, 3, 6, 8))

        for iv in sh:
            shell[tid, iv % 12] = 1.0
        for iv in fu:
            full[tid, iv % 12] = 1.0
        for iv in tens:
            tension_only[tid, iv % 12] = 1.0

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


def resolve_power_chord_quality_by_key(
    root_id: int,
    key_id: int | None,
    default_qual: int = 0,
) -> int:
    """Resolve Major (`0`) vs Minor (`1`) quality using song/local `key_id` (`0..11` Maj, `12..23` Min)
    when a chord voices only `1 + 5` (missing 3rd, e.g. `C5` in Key of C Major -> `C`, never `Cm`).
    """
    if key_id is None or key_id < 0 or key_id >= 24 or root_id < 0 or root_id >= 12:
        return int(default_qual)
    if key_id < 12:
        deg = (int(root_id) - int(key_id)) % 12
        if deg in (0, 5, 7, 3, 8, 10):  # I, IV, V (+ modal bIII, bVI, bVII)
            return 0
        if deg in (2, 4, 9):            # ii, iii, vi
            return 1
        return int(default_qual)
    tonic_min = int(key_id) - 12
    deg = (int(root_id) - tonic_min) % 12
    if deg in (0, 5, 7):                # i, iv, v (natural minor when 3rd is absent)
        return 1
    if deg in (3, 8, 10):               # bIII, bVI, bVII (e.g. C, F, G in A minor)
        return 0
    return int(default_qual)


def resolve_jazz_chord_theory(
    root_id: int,
    quality_id: int,
    ext_id: int,
    bass_id: int,
    rel_sustain: np.ndarray,
    tb_prob: np.ndarray | None = None,
    key_id: int | None = None,
) -> tuple[int, int, int, int, str | None]:
    """Apply Jazz/Pop music-theory equivalence, v5.1.2 PA1 taxonomy rules & Diatonic Key Fallback.

    Args:
        root_id, quality_id, ext_id, bass_id: Base decoded chord IDs.
        rel_sustain: `(12,)` mean sustained pitch-class activation rotated relative to `root_id`
            (`0=1, 1=b9, 2=9, 3=b3/#9, 4=3, 5=4/11, 6=b5/#11, 7=5, 8=#5/b13, 9=6/13, 10=b7, 11=maj7`).
        tb_prob: Optional `(13,)` mean physical `TrueBass` probability (`0..11`, `12=N`) in absolute pitch class.
        key_id: Optional `0..23` (`0..11` Maj, `12..23` Min) song/local key for missing-3rd (`1+5`) resolution.

    Returns:
        `(new_root, new_qual, new_ext, new_bass, suffix_override)`.
    """
    r, q, e, b = int(root_id), int(quality_id), int(ext_id), int(bass_id)
    if r < 0 or r >= 12:
        return r, q, e, b, None

    s_1 = float(rel_sustain[0])
    s_b9 = float(rel_sustain[1])
    s_9 = float(rel_sustain[2])
    s_b3 = float(rel_sustain[3])
    s_3 = float(rel_sustain[4])
    s_4 = float(rel_sustain[5])
    s_b5 = float(rel_sustain[6])
    s_5 = float(rel_sustain[7])
    s_sharp5 = float(rel_sustain[8])
    s_13 = float(rel_sustain[9])
    s_b7 = float(rel_sustain[10])
    s_maj7 = float(rel_sustain[11])

    # 0. Eliminate all 7 banned combos (`b5`, `m7b9`, `m7#11`, `m7b13`, `7sus2`, `maj11`, `dom11`)
    if q == 0 and e == 1:
        e = 2 if s_sharp5 >= 0.35 else 0
    elif q == 1 and e in (7, 11, 13):
        e = 4
    elif q == 4 and e == 5:
        if s_b7 >= 0.35 and s_4 >= 0.35:
            return r, 5, 8, b, None
        e = 0
    elif q == 0 and e == 10:  # `maj11` does not exist -> `maj9#11` / `maj9` / `maj7` or `add4`
        if s_maj7 >= 0.30:
            if s_b5 >= 0.40 and s_9 >= 0.36:
                return r, 0, 11, b, "maj9#11"
            return r, 0, (8 if s_9 >= 0.36 else 3), b, None
        if s_4 >= 0.42 and s_3 >= 0.30:
            return r, 0, 0, b, "add4"
        return r, 0, (6 if s_9 >= 0.42 else 0), b, None
    elif q == 2 and e == 10:  # `dom11` does not exist -> `m11` / `9sus4` / `7sus4` / `9` / `7`
        if s_b3 >= 0.48 and s_3 < 0.14 and s_b5 < 0.35:
            return r, 1, 10, b, None
        if s_3 < 0.30 and s_b3 < 0.28:
            return r, 5, (8 if s_9 >= 0.28 else 5), b, None
        e = 8 if s_9 >= 0.36 else 5

    # 1. Diatonic Key Fallback for missing-3rd (`1 + 5` power chords, where both `3` and `b3` are absent)
    if q in (0, 1) and e == 0 and (b == 12 or b == r):
        if s_3 < 0.18 and s_b3 < 0.18 and abs(s_3 - s_b3) < 0.10 and (s_5 >= 0.28 or s_1 >= 0.35):
            q_diatonic = resolve_power_chord_quality_by_key(r, key_id, default_qual=q)
            return r, q_diatonic, 0, b, None

    # 2. Half-diminished `Xm7b5` over its minor 3rd `Y = (X + 3) % 12` === root-position `Ym6`
    y_m6 = (r + 3) % 12
    if q == 3 and e == 4:
        if b == y_m6:
            return y_m6, 1, 0, 12, "m6"
        if b == 12 and tb_prob is not None:
            if float(tb_prob[r]) < 0.12 and float(tb_prob[y_m6]) >= 0.22 and s_b3 >= 0.50:
                return y_m6, 1, 0, 12, "m6"

    # 3. `SUS4_SHELL` resolution & `9sus4`
    if q == 5 and e in (0, 5):
        if e == 0 and s_3 > s_4 + 0.12 and s_3 >= 0.38:
            return r, 0, 0, b, None
        if s_b7 >= 0.34 and s_9 >= 0.36 and s_4 >= 0.40:
            return r, 5, 8, b, None

    # 4. `MAJ_SHELL` (`add4`, `6`, `6/9`)
    if q == 0 and e in (0, 6):
        if s_3 >= 0.38 and s_4 >= 0.54 and s_maj7 < 0.18 and s_b7 < 0.20 and s_4 > s_9 + 0.12:
            return r, 0, 0, b, "add4"
        if (
            (b == 12 or b == r)
            and (tb_prob is None or float(tb_prob[r]) >= 0.58)
            and s_3 >= 0.45
            and s_5 >= 0.38
            and s_13 >= 0.60
            and s_maj7 < 0.14
            and s_b7 < 0.16
        ):
            if s_9 >= 0.50:
                return r, 0, 6, 12, "6/9"
            return r, 0, 0, 12, "6"

    # 5. `MAJ7_SHELL` (`maj9#11`)
    if q == 0 and e in (3, 8, 11, 12):
        if s_3 >= 0.32 and s_maj7 >= 0.28 and s_9 >= 0.44 and s_b5 >= 0.44 and s_4 < 0.28:
            return r, 0, 11, b, "maj9#11"

    # 6. `MIN_SHELL` (`madd4`, `m6`)
    if q == 1 and e in (0, 6):
        if s_b3 >= 0.42 and s_4 >= 0.55 and s_b7 < 0.20 and s_4 > s_9 + 0.12 and s_13 < 0.45:
            return r, 1, 0, b, "madd4"
        if (
            (b == 12 or b == r)
            and (tb_prob is None or float(tb_prob[r]) >= 0.50)
            and s_b3 >= 0.45
            and s_13 >= 0.56
            and s_b7 < 0.22
        ):
            return r, 1, 0, 12, "m6"

    # 7. `DOM7_NAT` / `DOM7_ALT` (`dim7` rootless, `b3` guard, `9sus4`, `13b9`, `13#11`, `9#11`, `7#5`, `9(13)`)
    if q == 2:
        # Rootless `R7b9` over its major 3rd `(R + 4) % 12` === `(R + 4)dim7` (when `#5` is absent)
        if e == 7 and s_sharp5 < 0.35:
            third_pc = (r + 4) % 12
            if b == third_pc:
                return third_pc, 3, 5, 12, None
            if tb_prob is not None and float(tb_prob[third_pc]) > float(tb_prob[r]) + 0.12 and float(tb_prob[r]) < 0.35:
                return third_pc, 3, 5, 12, None

        # Minor-3rd guard: without a Major 3rd (`s_3 < 0.14`), strong `b3` (`>= 0.48`) is `m7`/`m11`, never `dom7`/`9sus4`
        if s_b3 >= 0.48 and s_3 < 0.14 and s_b5 < 0.35:
            return r, 1, (10 if s_4 >= 0.55 else 4), b, None

        if e in (5, 8) and s_4 >= 0.48 and s_3 < 0.20 and s_b3 < 0.24 and s_4 > s_3 + 0.25:
            return r, 5, (8 if s_9 >= 0.34 else 5), b, None

        if s_3 >= 0.30:
            if s_b9 >= 0.38 and s_13 >= 0.48 and s_sharp5 < 0.35:
                return r, 2, 7, b, "13b9"
            if s_b5 >= 0.44 and s_13 >= 0.48 and s_b9 < 0.22 and s_4 < 0.30:
                return r, 2, 11, b, "13#11"
            if s_9 >= 0.44 and s_b5 >= 0.44 and s_13 < 0.45 and s_b9 < 0.22 and s_4 < 0.30:
                return r, 2, 11, b, "9#11"
            if e in (5, 7, 13, 14) and s_sharp5 >= 0.46 and s_5 < 0.22 and s_b9 < 0.20:
                return r, 2, 2, b, None
            if e in (5, 8, 12) and s_9 >= 0.46 and s_13 >= 0.46 and s_sharp5 < 0.35:
                return r, 2, 12, b, "9(13)"

    # 8. `MIN7_SHELL` (`m6` when `b7` is absent and `6th` is sustained; `m11` / `m13` upgrades)
    if q == 1 and e == 4:
        if (b == 12 or b == r) and s_b3 >= 0.45 and s_13 >= 0.58 and s_b7 < 0.24:
            return r, 1, 0, 12, "m6"
        if s_4 >= 0.58 and s_4 > s_9 + 0.15:
            return r, 1, 10, b, None
        if s_13 >= 0.62 and s_b7 >= 0.40:
            return r, 1, 12, b, None

    return r, q, e, b, None
