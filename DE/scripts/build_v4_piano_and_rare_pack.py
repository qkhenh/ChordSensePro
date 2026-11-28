from __future__ import annotations

"""Build the Dataset v4 Solo-Piano (POP909) & Controlled Rare-Chord Progression Pack.

Produces a v4-compatible directory with:
  - `train/`, `val/`, `test_piano_pop/` from song-level 80/10/10 split of `DE/data/datasets/POP909-Dataset/POP909/`
    (dual views: `piano_accompaniment` = PIANO+BRIDGE and `piano_with_melody` = PIANO+BRIDGE+MELODY)
  - Controlled multi-voicing 4-bar progressions in `train/` flagged `is_synthetic_rare = True` covering
    the 9 ultra-scarce Jazz/Pop chords (`m6/9, 7b5b9, 7b5#9, 7#5#9, 7#5b9, m11b5, 7b9sus4, 13b9sus4, 7#5`).

Audio synthesis uses `pretty_midi` + `pyfluidsynth` with `GeneralUser_GS` SoundFont when available,
and automatically falls back to a clean polyphonic additive electric/acoustic piano synthesizer
so the pipeline works deterministically on any machine without external native binaries.
"""

import argparse
import hashlib
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data_processing.application.mappers.harte_mapper import (
    BASS_NONE_ID,
    LabelStatus,
    map_harte,
    parse_v6_chord_targets,
)
from src.shared.domain.value_objects import RootNote

SR = 24000
ROOT_NAMES = [r.value for r in RootNote]

# 9 ultra-scarce backlog chords rendered in realistic 4-bar ii-V-I / modal-interchange progressions
RARE_CHORD_SPECS: list[tuple[str, list[int]]] = [
    ("min6(9)",    [0, 3, 7, 9, 14]),         # m6/9
    ("7(b5,b9)",   [0, 4, 6, 10, 13]),        # 7b5b9
    ("7(b5,#9)",   [0, 4, 6, 10, 15]),        # 7b5#9
    ("7(#5,#9)",   [0, 4, 8, 10, 15]),        # 7#5#9
    ("7(#5,b9)",   [0, 4, 8, 10, 13]),        # 7#5b9
    ("min11(b5)",  [0, 3, 6, 10, 14, 17]),    # m11b5
    ("7sus4(b9)",  [0, 5, 7, 10, 13]),        # 7b9sus4
    ("7sus4(b9,13)", [0, 5, 7, 10, 13, 21]),  # 13b9sus4
    ("aug7",       [0, 4, 8, 10]),            # 7#5
]


def _midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _synthesize_piano_note(
    midi_note: int,
    dur_s: float,
    velocity: float,
    sr: int = SR,
    brightness: float = 1.0,
) -> np.ndarray:
    """Lightweight harmonic piano note synthesizer with realistic attack and exponential overtone decay."""
    n = max(1, int(round(dur_s * sr)))
    t = np.arange(n, dtype=np.float32) / sr
    f0 = _midi_to_hz(midi_note)
    sig = np.zeros(n, dtype=np.float32)

    # Harmonic weights tuned for warm acoustic/electric piano timbre
    harmonics = (
        (1.0, 1.00, 1.4),
        (2.0, 0.42 * brightness, 2.3),
        (3.0, 0.22 * brightness, 3.6),
        (4.0, 0.11 * brightness, 5.2),
        (5.0, 0.05 * brightness, 7.0),
    )
    attack = np.minimum(1.0, t / 0.008)
    release_len = min(n, int(0.06 * sr))
    release = np.ones(n, dtype=np.float32)
    if release_len > 1:
        release[-release_len:] = np.linspace(1.0, 0.0, release_len, dtype=np.float32)

    for mult, amp, decay in harmonics:
        if f0 * mult >= 0.45 * sr:
            continue
        env = np.exp(-decay * t * (1.0 + 0.015 * max(0, midi_note - 48)))
        sig += (amp * env * np.sin(2.0 * math.pi * f0 * mult * t)).astype(np.float32)

    return sig * attack * release * float(velocity)


def _render_notes_numpy(
    notes: list[tuple[float, float, int, int]],
    total_dur_s: float,
    sr: int = SR,
    brightness: float = 1.0,
) -> np.ndarray:
    """Render `(start_s, end_s, pitch, velocity)` tuples into normalized mono `int16` waveform."""
    n_total = max(sr, int(math.ceil(total_dur_s * sr)))
    buf = np.zeros(n_total, dtype=np.float32)
    for start_s, end_s, pitch, vel in notes:
        i0 = max(0, int(round(start_s * sr)))
        dur_s = max(0.06, min(end_s - start_s, 6.0))
        if i0 >= n_total:
            continue
        wave = _synthesize_piano_note(pitch, dur_s, velocity=max(0.15, vel / 127.0), sr=sr, brightness=brightness)
        i1 = min(n_total, i0 + len(wave))
        buf[i0:i1] += wave[: i1 - i0]

    peak = float(np.max(np.abs(buf))) if buf.size else 0.0
    if peak > 1e-6:
        buf = (buf / peak) * 0.85
    return np.clip(buf * 32767.0, -32768, 32767).astype(np.int16)


def _render_pop909_midi(
    midi_path: Path,
    include_melody: bool,
    soundfont_path: Path | None,
    sr: int = SR,
) -> tuple[np.ndarray, float] | None:
    """Render a POP909 MIDI file using FluidSynth (when available) or NumPy piano fallback."""
    try:
        import pretty_midi
    except ImportError:
        return None

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    filtered = pretty_midi.PrettyMIDI()
    keep_names = {"PIANO", "BRIDGE", "MELODY"} if include_melody else {"PIANO", "BRIDGE"}
    notes_flat: list[tuple[float, float, int, int]] = []
    max_end = 0.0

    for inst in pm.instruments:
        name = (inst.name or "").strip().upper()
        if name not in keep_names and len(pm.instruments) > 1:
            continue
        filtered.instruments.append(inst)
        for n in inst.notes:
            notes_flat.append((float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
            if n.end > max_end:
                max_end = float(n.end)

    if not notes_flat or max_end < 5.0:
        return None

    if soundfont_path is not None and soundfont_path.exists():
        try:
            audio = filtered.fluidsynth(fs=sr, sf2_path=str(soundfont_path))
            if audio is not None and len(audio) > sr:
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                peak = float(np.max(np.abs(audio)))
                if peak > 1e-6:
                    audio = (audio / peak) * 0.85
                return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16), max_end
        except Exception:
            pass

    audio_i16 = _render_notes_numpy(notes_flat, max_end + 0.5, sr=sr, brightness=1.0 if include_melody else 0.85)
    return audio_i16, max_end


def _parse_pop909_chords(chord_txt: Path, song_id: str, source_domain: str) -> list[dict]:
    rows: list[dict] = []
    for line in chord_txt.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        start_ms = int(round(float(parts[0]) * 1000.0))
        end_ms = int(round(float(parts[1]) * 1000.0))
        label_raw = parts[2]
        if end_ms <= start_ms:
            continue

        mapped = map_harte(label_raw)
        if mapped.status == LabelStatus.UNMAPPED:
            continue

        t = parse_v6_chord_targets(
            label_raw,
            mapped.root_id if mapped.status != LabelStatus.NO_CHORD else 12,
            mapped.quality_id,
            mapped.ext_id,
            mapped.bass_id,
        )
        rows.append(
            {
                "song_id": song_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "label_raw": label_raw,
                "root_id": 12 if mapped.status == LabelStatus.NO_CHORD else mapped.root_id,
                "quality_id": mapped.quality_id,
                "ext_id": mapped.ext_id,
                "bass_id": t.bass_id,
                "skel14_id": t.skel14_id,
                "tension_mask": t.tension_mask,
                "tension_valid": t.tension_valid,
                "inversion_id": t.inversion_id,
                "missing_third": t.missing_third,
                "is_synthetic_rare": False,
                "source_domain": source_domain,
            }
        )
    return rows


def build_pop909_pack(
    pop909_dir: Path,
    out_dir: Path,
    soundfont_path: Path | None = None,
    max_songs: int | None = None,
    include_both_views: bool = True,
) -> dict[str, int]:
    """Render song-split POP909 (`80% train / 10% val / 10% test_piano_pop`) to v4 parquet + `.npy`."""
    song_dirs = sorted(p for p in pop909_dir.iterdir() if p.is_dir() and (p / f"{p.name}.mid").exists())
    if max_songs is not None:
        song_dirs = song_dirs[:max_songs]

    split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test_piano_pop": []}
    split_counts = {"train": 0, "val": 0, "test_piano_pop": 0}

    for sdir in song_dirs:
        h = int(hashlib.md5(sdir.name.encode("utf-8")).hexdigest(), 16) % 100
        split = "train" if h < 80 else ("val" if h < 90 else "test_piano_pop")
        chord_txt = sdir / "chord_midi.txt"
        midi_path = sdir / f"{sdir.name}.mid"
        if not chord_txt.exists():
            continue

        views = [("piano_accompaniment", False)]
        if include_both_views and split == "train":
            views.append(("piano_with_melody", True))

        audio_dir = out_dir / split / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        for view_name, inc_mel in views:
            song_id = f"pop909_{sdir.name}_{'mel' if inc_mel else 'acc'}"
            rendered = _render_pop909_midi(midi_path, include_melody=inc_mel, soundfont_path=soundfont_path)
            if rendered is None:
                continue
            audio_i16, _ = rendered
            rows = _parse_pop909_chords(chord_txt, song_id=song_id, source_domain=view_name)
            if not rows:
                continue
            np.save(audio_dir / f"{song_id}.npy", audio_i16)
            split_rows[split].extend(rows)
            split_counts[split] += 1

    for split, rows in split_rows.items():
        if not rows:
            continue
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(split_out / "chords.parquet", index=False)
        print(f"[POP909:{split}] rendered {split_counts[split]} songs ({len(rows):,} chords)")

    return split_counts


def build_rare_chord_pack(
    out_dir: Path,
    songs_per_type: int = 8,
    seed: int = 42,
) -> int:
    """Render controlled multi-voicing 4-bar progressions for 9 scarce Jazz/Pop chords (`is_synthetic_rare=True`).

    Always written into `train/` only and flagged `is_synthetic_rare=True` so:
      - Phase 2 (LoRA) automatically drops 100% of synthetic rows (`0%` synthetic).
      - Phase 1/3 caps synthetic share at `<= 4%` via `max_synthetic_ratio`.
    """
    rng = np.random.default_rng(seed)
    train_audio = out_dir / "train" / "audio"
    train_audio.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_songs = 0

    for spec_idx, (suffix, intervals) in enumerate(RARE_CHORD_SPECS):
        for rep in range(songs_per_type):
            tonic = int((spec_idx * 5 + rep * 7) % 12)
            bpm = float(rng.uniform(68.0, 118.0))
            bar_s = 4.0 * 60.0 / bpm
            brightness = float(rng.uniform(0.7, 1.25))
            octave_shift = int(rng.choice([0, 12])) if rep % 3 == 0 else 0

            # 4-bar realistic context progression:
            # Bar 0: ii7 (or IVmaj7) -> Bar 1: target rare chord -> Bar 2: I (or Imaj7) -> Bar 3: vi7 / walking bass
            ii_root = (tonic + 2) % 12
            v_root = (tonic + 7) % 12
            iv_root = (tonic + 5) % 12

            if suffix.startswith("min6"):
                prog = [
                    (f"{ROOT_NAMES[iv_root]}:maj7", iv_root, [0, 4, 7, 11]),
                    (f"{ROOT_NAMES[iv_root]}:{suffix}", iv_root, intervals),
                    (f"{ROOT_NAMES[tonic]}:maj7", tonic, [0, 4, 7, 11]),
                    (f"{ROOT_NAMES[tonic]}:maj6", tonic, [0, 4, 7, 9]),
                ]
            elif suffix.startswith("min11"):
                prog = [
                    (f"{ROOT_NAMES[ii_root]}:{suffix}", ii_root, intervals),
                    (f"{ROOT_NAMES[v_root]}:7(b9)", v_root, [0, 4, 7, 10, 13]),
                    (f"{ROOT_NAMES[tonic]}:min9", tonic, [0, 3, 7, 10, 14]),
                    (f"{ROOT_NAMES[tonic]}:min6", tonic, [0, 3, 7, 9]),
                ]
            else:
                prog = [
                    (f"{ROOT_NAMES[ii_root]}:min7", ii_root, [0, 3, 7, 10]),
                    (f"{ROOT_NAMES[v_root]}:{suffix}", v_root, intervals),
                    (f"{ROOT_NAMES[tonic]}:maj7", tonic, [0, 4, 7, 11]),
                    (f"{ROOT_NAMES[tonic]}:maj", tonic, [0, 4, 7]),
                ]

            song_id = f"syn_rare_{spec_idx:02d}_{rep:02d}"
            notes: list[tuple[float, float, int, int]] = []
            t_cur = 0.2
            for bar_i in range(2):  # 2 cycles = 8 bars (~16-25s)
                for label_raw, chord_r, ivs in prog:
                    start_s = t_cur
                    end_s = t_cur + bar_s
                    bass_midi = 36 + chord_r + octave_shift
                    vel = int(rng.integers(72, 108))
                    # Left-hand root/bass + right-hand voicing (varied arpeggio/block style)
                    notes.append((start_s, end_s - 0.05, bass_midi, vel))
                    for k, iv in enumerate(ivs):
                        pitch = 48 + chord_r + iv + (octave_shift // 2)
                        onset_jit = float(k * 0.025 if rep % 2 == 0 else rng.uniform(0.0, 0.018))
                        notes.append((start_s + onset_jit, end_s - 0.04, pitch, max(58, vel - 8)))

                    mapped = map_harte(label_raw)
                    t = parse_v6_chord_targets(label_raw, chord_r, mapped.quality_id, mapped.ext_id, BASS_NONE_ID)
                    rows.append(
                        {
                            "song_id": song_id,
                            "start_ms": int(round(start_s * 1000.0)),
                            "end_ms": int(round(end_s * 1000.0)),
                            "label_raw": label_raw,
                            "root_id": chord_r,
                            "quality_id": mapped.quality_id,
                            "ext_id": mapped.ext_id,
                            "bass_id": t.bass_id,
                            "skel14_id": t.skel14_id,
                            "tension_mask": t.tension_mask,
                            "tension_valid": t.tension_valid,
                            "inversion_id": t.inversion_id,
                            "missing_third": t.missing_third,
                            "is_synthetic_rare": True,
                            "source_domain": "synthetic_rare_piano",
                        }
                    )
                    t_cur = end_s

            audio_i16 = _render_notes_numpy(notes, t_cur + 0.5, sr=SR, brightness=brightness)
            np.save(train_audio / f"{song_id}.npy", audio_i16)
            n_songs += 1

    train_pq = out_dir / "train" / "chords.parquet"
    df_new = pd.DataFrame(rows)
    if train_pq.exists():
        df_old = pd.read_parquet(train_pq)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_parquet(train_pq, index=False)
    print(f"[RarePack] generated {n_songs} controlled rare-chord songs ({len(rows):,} chords) -> {train_pq}")
    return n_songs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Dataset v4 POP909 Solo-Piano & Rare-Chord Pack")
    ap.add_argument("--pop909-dir", default="DE/data/datasets/POP909-Dataset/POP909")
    ap.add_argument(
        "--soundfont",
        default="DE/data/datasets/soundfonts/GeneralUser_GS_v2.0.3--doc_r6/GeneralUser-GS.sf2",
    )
    ap.add_argument("--out", default="DE/data/export/v4_piano_pack")
    ap.add_argument("--max-pop909-songs", type=int, default=None)
    ap.add_argument("--rare-songs-per-type", type=int, default=8)
    ap.add_argument("--skip-pop909", action="store_true")
    ap.add_argument("--skip-rare", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    sf = Path(args.soundfont) if args.soundfont else None
    if sf is not None and not sf.exists():
        # Auto-discover any .sf2 inside the soundfonts directory
        sf_root = Path("DE/data/datasets/soundfonts")
        candidates = list(sf_root.rglob("*.sf2")) if sf_root.exists() else []
        sf = candidates[0] if candidates else None

    if not args.skip_pop909 and Path(args.pop909_dir).exists():
        build_pop909_pack(
            Path(args.pop909_dir),
            out_dir,
            soundfont_path=sf,
            max_songs=args.max_pop909_songs,
        )
    if not args.skip_rare:
        build_rare_chord_pack(out_dir, songs_per_type=args.rare_songs_per_type)


if __name__ == "__main__":
    main()
