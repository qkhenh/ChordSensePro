from __future__ import annotations
import os

from src.data_processing.application.annotation_parsers.parsers import (
    get_parser, read_annotation_duration, read_annotation_section,
)
from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio

DEFAULT_TOLERANCE_SEC = 3.0
DURATION_CHECKED = ("jaah", "choco")
SECTION_CHECKED = "hooktheory"

CHROMA_SR = 24_000
CHROMA_HOP = 2048
CHROMA_FRAME_SEC = CHROMA_HOP / CHROMA_SR
CHROMA_SHIFT_RANGE = 2.5
CHROMA_SHIFT_STEP = 0.1
CHROMA_MIN_FRAMES = 30
CHROMA_OK_THRESHOLD_MS = 200
CHROMA_REJECT_THRESHOLD_MS = 2000
CHROMA_REJECT_SCORE = 0.35

TRIAD = {0: (0, 4, 7), 1: (0, 3, 7), 2: (0, 4, 7, 10), 3: (0, 3, 6), 4: (0, 2, 7), 5: (0, 5, 7)}
SEVENTH = {3: 11, 4: 10, 5: 10}


def _chord_template(root_id: int, quality_id: int, ext_id: int | None):
    import numpy as np
    pcs = set(TRIAD[quality_id])
    if ext_id in SEVENTH and quality_id != 2:
        pcs.add(9 if (quality_id == 3 and ext_id == 5) else SEVENTH[ext_id])
    v = np.zeros(12, dtype=np.float32)
    for p in pcs:
        v[(root_id + p) % 12] = 1.0
    return v / np.linalg.norm(v)


def _compute_chroma_offset(wav_path, annotation_path):
    import numpy as np
    import librosa
    import soundfile as sf

    from src.data_processing.application.mappers.harte_mapper import map_harte

    audio, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != CHROMA_SR:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=CHROMA_SR)

    if len(audio) < CHROMA_SR * 3:
        return 0, 0.0

    parser = get_parser(annotation_path)
    timeline = parser.parse(annotation_path)
    if not timeline:
        return 0, 0.0

    starts, ends, temps = [], [], []
    for start, end, label in timeline:
        m = map_harte(label)
        if m.root_id is None or m.quality_id is None:
            continue
        starts.append(start)
        ends.append(end)
        temps.append(_chord_template(m.root_id, m.quality_id, m.ext_id))
    if not temps:
        return 0, 0.0

    starts_arr = np.asarray(starts)
    ends_arr = np.asarray(ends)
    temps_arr = np.stack(temps)

    chroma = librosa.feature.chroma_cqt(y=audio, sr=CHROMA_SR, hop_length=CHROMA_HOP)
    n = chroma.shape[1]
    norm = np.linalg.norm(chroma, axis=0)
    t = np.arange(n) * CHROMA_FRAME_SEC
    frame_ok = norm > 1e-3
    if frame_ok.sum() < CHROMA_MIN_FRAMES:
        return 0, 0.0
    chroma = chroma / np.maximum(norm, 1e-6)
    sims = temps_arr @ chroma

    shifts = np.round(np.arange(-CHROMA_SHIFT_RANGE, CHROMA_SHIFT_RANGE + 0.001, CHROMA_SHIFT_STEP), 2)
    scores = []
    for s in shifts:
        idx = np.searchsorted(starts_arr, t + s, side="right") - 1
        valid = (idx >= 0) & frame_ok
        idx_c = np.clip(idx, 0, len(starts_arr) - 1)
        valid &= (t + s) < ends_arr[idx_c]
        if valid.sum() < CHROMA_MIN_FRAMES:
            scores.append(np.nan)
            continue
        scores.append(float(sims[idx_c[valid], np.nonzero(valid)[0]].mean()))
    scores_arr = np.asarray(scores)
    if np.all(np.isnan(scores_arr)):
        return 0, 0.0

    best_idx = int(np.nanargmax(scores_arr))
    best_shift = float(shifts[best_idx])
    best_score = float(scores_arr[best_idx])

    return round(best_shift * 1000), best_score


class AlignmentCheckHandler(BaseProcessingHandler):

    def __init__(self, tolerance_sec: float | None = None) -> None:
        super().__init__()
        self.tolerance = (
            tolerance_sec if tolerance_sec is not None
            else float(os.getenv("ALIGN_TOLERANCE_SEC", DEFAULT_TOLERANCE_SEC))
        )

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        source = getattr(data.audio_file.source_type, "value", str(data.audio_file.source_type))
        checked = source in DURATION_CHECKED or source == SECTION_CHECKED
        if not checked or data.annotation_path is None or data.wav_path is None:
            data.align_status = "n/a"
            return self._call_next(data)

        if source == SECTION_CHECKED:
            section = read_annotation_section(data.annotation_path)
            expected = (section[1] - section[0]) if section else None
        else:
            expected = read_annotation_duration(data.annotation_path)
        if expected is None:
            data.align_status = "n/a"
            return self._call_next(data)

        try:
            import soundfile as sf
            actual = float(sf.info(str(data.wav_path)).duration)
        except Exception as e:
            data.error = f"AlignmentCheck: cannot read WAV duration: {e}"
            return data

        diff = abs(actual - expected)
        if diff > self.tolerance:
            data.align_status = "duration_mismatch"
            data.error = (
                f"duration_mismatch: audio {actual:.1f}s vs annotation {expected:.1f}s "
                f"(Δ{diff:.1f}s > {self.tolerance:.1f}s) — wrong YouTube version"
            )
            return data

        try:
            offset_ms, score = _compute_chroma_offset(data.wav_path, data.annotation_path)
        except Exception:
            offset_ms, score = 0, 0.0

        data.align_offset_ms = offset_ms
        data.align_score = score

        abs_offset = abs(offset_ms)
        if abs_offset <= CHROMA_OK_THRESHOLD_MS:
            data.align_status = "ok"
        elif abs_offset <= CHROMA_REJECT_THRESHOLD_MS and score >= CHROMA_REJECT_SCORE:
            data.align_status = "shifted"
        else:
            data.align_status = "rejected"
            data.error = (
                f"alignment rejected: offset={offset_ms}ms, score={score:.3f} "
                f"(threshold: {CHROMA_REJECT_THRESHOLD_MS}ms / {CHROMA_REJECT_SCORE})"
            )
            return data

        return self._call_next(data)
