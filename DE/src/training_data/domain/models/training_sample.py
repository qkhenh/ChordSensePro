"""TrainingSample — one labeled 2-second audio segment, the unit of the training dataset.

Metadata + labels live in PostgreSQL (table ``training_samples``); the audio itself is a
row of an int16 ``.npy`` file on the training volume, pointed to by ``audio_path`` + ``audio_idx``.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TrainingSample:
    song_id:      str
    source_type:  str
    source_url:   str
    segment_idx:  int
    start_ms:     int
    end_ms:       int
    audio_path:   str            # relative to TRAINING_DIR, e.g. "audio/<song_id>.npy"
    audio_idx:    int            # row index inside that .npy
    sample_rate:  int            # always 24000 (MERT input)
    label_raw:    str            # Harte label as annotated, e.g. "C:maj7"
    label_status: str            # ok | partial | no_chord | unmapped
    root_id:      int | None
    quality_id:   int | None
    ext_id:       int | None
    split:        str            # train | val | test
    align_status: str = "n/a"    # ok | n/a  (duration_mismatch songs are rejected earlier)
