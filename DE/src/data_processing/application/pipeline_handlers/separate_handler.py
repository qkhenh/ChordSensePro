"""Handler: runs Demucs htdemucs_6s source separation and returns the harmonic stem (piano → guitar → other)."""
from __future__ import annotations
import subprocess
from pathlib import Path

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio


class SeparateHandler(BaseProcessingHandler):
    """Handler: WAV → Demucs htdemucs_6s → harmonic stem.

    Separates 6 stems: drums | bass | vocals | other | guitar | piano
    Returns the first available stem in priority order: piano → guitar → other
    """

    MODEL = "htdemucs_6s"

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data
        if data.wav_path is None:
            data.error = "SeparateHandler: wav_path is None"
            return data

        result = self._run(data.wav_path, data.wav_path.parent / "separated")
        if result.is_err:
            data.error = result.error
            return data

        data.stem_path = result.unwrap()
        return self._call_next(data)

    def _run(self, wav_path: Path, output_dir: Path | None = None):
        from src.shared.domain.processing_result import ProcessingResult
        out_dir = output_dir or wav_path.parent / "separated"

        cmd = [
            "python", "-m", "demucs",
            "-n", self.MODEL,
            "--out", str(out_dir),
            str(wav_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return ProcessingResult.err(f"Demucs failed: {result.stderr[:300]}")

            stem_dir = out_dir / self.MODEL / wav_path.stem

            for stem in ("piano", "guitar", "other"):
                stem_path = stem_dir / f"{stem}.wav"
                if stem_path.exists():
                    return ProcessingResult.ok(stem_path)

            return ProcessingResult.err(f"Can not find harmonic stem: {stem_dir}")

        except subprocess.TimeoutExpired:
            return ProcessingResult.err("Demucs timeout (>5mins)")

        except Exception as e:
            return ProcessingResult.err(f"Demucs error: {e}")
