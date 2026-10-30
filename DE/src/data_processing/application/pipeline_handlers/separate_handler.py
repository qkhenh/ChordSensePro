from __future__ import annotations
import os
import subprocess
from pathlib import Path

from src.data_processing.domain.models.base_handler import BaseProcessingHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio

CLEAN_SOURCES = ("kaggle", "local", "pianoteq", "guitarset")


def pick_device() -> str:
    forced = os.getenv("DEMUCS_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class SeparateHandler(BaseProcessingHandler):

    HARMONIC_STEMS = ("other", "bass")

    MODEL = "htdemucs_ft"
    TIMEOUT_GPU = 900
    TIMEOUT_CPU = 1800

    def handle(self, data: ProcessedAudio) -> ProcessedAudio:
        if data.is_failed:
            return data
        if data.wav_path is None:
            data.error = "SeparateHandler: wav_path is None"
            return data

        source = getattr(data.audio_file.source_type, "value", str(data.audio_file.source_type))
        if source in CLEAN_SOURCES:
            data.stem_path = data.wav_path
            return self._call_next(data)

        result = self._run(data.wav_path, data.wav_path.parent / "separated")
        if result.is_err:
            data.error = result.error
            return data

        data.stem_path = result.unwrap()
        return self._call_next(data)

    def _run(self, wav_path: Path, output_dir: Path | None = None):
        from src.shared.domain.processing_result import ProcessingResult
        out_dir = output_dir or wav_path.parent / "separated"

        device = pick_device()
        outcome = self._demucs(wav_path, out_dir, device)
        if device == "cuda" and outcome[0] == "error" and "out of memory" in outcome[1].lower():
            outcome = self._demucs(wav_path, out_dir, "cpu")

        kind, payload = outcome
        if kind == "error":
            return ProcessingResult.err(payload)

        stem_dir = out_dir / self.MODEL / wav_path.stem
        return self._mix_harmonic(stem_dir)

    def _mix_harmonic(self, stem_dir: Path):
        from src.shared.domain.processing_result import ProcessingResult
        found = [stem_dir / f"{s}.wav" for s in self.HARMONIC_STEMS if (stem_dir / f"{s}.wav").exists()]
        if not found:
            return ProcessingResult.err(f"Can not find harmonic stems: {stem_dir}")
        try:
            import numpy as np
            import soundfile as sf
            mix, sr = None, None
            for path in found:
                audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
                if mix is None:
                    mix, sr = audio, file_sr
                else:
                    n = min(len(mix), len(audio))
                    mix = mix[:n] + audio[:n]
            if mix is None or sr is None:
                return ProcessingResult.err(f"Harmonic mix failed: no readable stems in {stem_dir}")
            peak = float(np.abs(mix).max()) if mix.size else 0.0
            if peak > 0.99:
                mix = mix * (0.99 / peak)
            out_path = stem_dir / "harmonic.wav"
            sf.write(str(out_path), mix, sr)
            return ProcessingResult.ok(out_path)
        except Exception as e:
            return ProcessingResult.err(f"Harmonic mix failed: {e}")

    def _demucs(self, wav_path: Path, out_dir: Path, device: str) -> tuple[str, str]:
        cmd = [
            "python", "-m", "demucs",
            "-n", self.MODEL,
            "-d", device,
            "--segment", os.getenv("DEMUCS_SEGMENT", "6"),
            "--out", str(out_dir),
            str(wav_path),
        ]
        timeout = self.TIMEOUT_GPU if device == "cuda" else self.TIMEOUT_CPU
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                return "error", f"Demucs failed ({device}): {result.stderr[-600:]}"
            return "ok", ""
        except subprocess.TimeoutExpired:
            return "error", f"Demucs timeout (>{timeout // 60}mins, {device})"
        except Exception as e:
            return "error", f"Demucs error: {e}"
