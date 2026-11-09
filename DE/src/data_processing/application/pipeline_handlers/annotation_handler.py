from __future__ import annotations

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio
from src.data_processing.application.annotation_parsers.parsers import get_parser, read_annotation_range


class AnnotationHandler(BaseProcessingHandler):

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data

        if data.annotation_path is None:
            label = self._label_from_filename(data.audio_file.source_url)
            if label:
                for segment in data.segments:
                    segment.chord_label = label
            return self._call_next(data)

        if not data.annotation_path.exists():
            data.error = f"Annotation file not found: {data.annotation_path}"
            return data

        try:
            parser   = get_parser(data.annotation_path)
            timeline = parser.parse(data.annotation_path)
        except Exception as e:
            data.error = f"Annotation parse failed: {e}"
            return data

        if data.segments:
            labelled = read_annotation_range(data.annotation_path)
            for segment in data.segments:
                if labelled is not None and not self._inside(segment, labelled):
                    continue
                mid_sec = (segment.start_ms + segment.end_ms) / 2 / 1000.0
                segment.chord_label = self._find_chord(mid_sec, timeline)
        else:
            offset_ms = getattr(data, "align_offset_ms", 0) or 0
            offset_sec = offset_ms / 1000.0
            data.chord_timeline = []
            for start, end, label in timeline:
                adjusted_start = start - offset_sec
                adjusted_end = end - offset_sec
                if adjusted_end <= 0:
                    continue
                adjusted_start = max(0.0, adjusted_start)
                data.chord_timeline.append({
                    "start_ms": round(adjusted_start * 1000),
                    "end_ms": round(adjusted_end * 1000),
                    "label_raw": label,
                })

        return self._call_next(data)

    @staticmethod
    def _inside(segment, labelled: tuple[float, float], tolerance: float = 0.25) -> bool:
        start, end = segment.start_ms / 1000.0, segment.end_ms / 1000.0
        return start >= labelled[0] - tolerance and end <= labelled[1] + tolerance

    @staticmethod
    def _find_chord(mid_sec: float, timeline: list[tuple[float, float, str]]) -> str:
        for start, end, label in timeline:
            if start <= mid_sec < end:
                return label
        return "N"

    @staticmethod
    def _label_from_filename(source_url: str) -> str | None:
        from pathlib import Path
        name = Path(source_url).stem
        parts = name.split("_")
        if len(parts) < 3:
            return None

        qual_map = {"maj": "maj", "min": "min", "dim": "dim"}
        for i, p in enumerate(parts):
            if p in qual_map:
                note = "_".join(parts[:i])
                if not note:
                    return None
                note = note.replace("s", "#").replace("b", "b")
                return f"{note}:{qual_map[p]}"
        return None
