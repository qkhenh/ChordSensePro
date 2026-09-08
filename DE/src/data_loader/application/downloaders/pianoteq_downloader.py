"""PianoteqDownloader — renders extended chord audio via Pianoteq CLI.

Pianoteq Synthetic Data:
  - Physical modeling synthesizer with highly accurate acoustic model
  - Used to generate extended chords missing from real datasets (b5, #5, add9, b9, #9, nat11, #11, nat13, b13, alt)
  - License: Internal — requires Pianoteq license (proprietary)
  - Purpose: Head 3 (Extension) training + pre-training data augmentation

source_url format: chord label in Harte notation (e.g. "C:maj7", "D:min9", "F#:7b9")
Output: WAV file rendered at 44100Hz mono

Note: Requires Pianoteq to be installed in the Docker container or locally.
      Set PIANOTEQ_PATH env var to the Pianoteq CLI executable path.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse

# Harte notation → MIDI note number mapping for root notes
_ROOT_MIDI = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
    "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71,
}


class PianoteqDownloader:
    """Renders extended chord audio via Pianoteq CLI.

    source_url = Harte notation chord label: "C:maj7", "F#:7b9", etc.
    Renders a 3-second sustain of the chord at 44100Hz mono.
    """

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Render chord label → WAV via Pianoteq CLI."""
        pianoteq_bin = Path(os.getenv("PIANOTEQ_PATH", "/opt/pianoteq/Pianoteq"))
        if not pianoteq_bin.exists():
            return DownloadResponse.err(
                audio_file,
                f"Pianoteq binary not found at {pianoteq_bin}. "
                "Set PIANOTEQ_PATH env var or install Pianoteq."
            )

        chord_label = audio_file.source_url   # e.g. "C:maj7"
        out_dir     = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{chord_label.replace(':', '_').replace('#', 's')}.wav"

        cmd = [
            str(pianoteq_bin),
            "--headless",
            "--midi-file", self._make_midi(chord_label, out_dir),
            "--wav", str(out_path),
            "--sample-rate", "44100",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                return DownloadResponse.err(audio_file, f"Pianoteq render failed: {result.stderr[:200]}")
            return DownloadResponse.ok(audio_file, out_path)
        except subprocess.TimeoutExpired:
            return DownloadResponse.err(audio_file, "Pianoteq render timeout (>60s)")
        except Exception as e:
            return DownloadResponse.err(audio_file, f"Pianoteq error: {e}")

    def _make_midi(self, chord_label: str, out_dir: Path) -> str:
        """Generate a minimal MIDI file for the chord label.

        Placeholder — actual MIDI generation requires a MIDI library.
        TODO: implement using midiutil or music21 to build chord MIDI.
        """
        # Stub: return path where caller should place the MIDI file
        midi_path = out_dir / f"{chord_label.replace(':', '_')}.mid"
        return str(midi_path)
