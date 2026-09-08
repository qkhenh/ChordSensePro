"""KaggleDownloader — downloads Kaggle Guitar Chords dataset.

Kaggle Guitar Chords v3:
  - 7,000+ isolated chord recordings (guitar)
  - License: CC0 (public domain)
  - URL: https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3
  - Purpose: Guitar optional — Head 1+2 supplemental
  - Requires: KAGGLE_USERNAME + KAGGLE_KEY env vars (Kaggle API credentials)

Download via Kaggle API:
    kaggle datasets download -d fabianavinci/guitar-chords-v3
"""
from __future__ import annotations
import os
import subprocess
import zipfile
from pathlib import Path

from src.data_loader.domain.models.audio_file import AudioFile
from src.data_loader.domain.models.download_response import DownloadResponse


class KaggleDownloader:
    """Downloader for Kaggle Guitar Chords v3 dataset.

    Downloads the entire dataset zip via Kaggle CLI, then extracts.
    source_url = Kaggle dataset identifier: "fabianavinci/guitar-chords-v3"
    """

    def download(self, audio_file: AudioFile) -> DownloadResponse:
        """Download Kaggle dataset via kaggle CLI."""
        kaggle_user = os.getenv("KAGGLE_USERNAME", "")
        kaggle_key  = os.getenv("KAGGLE_KEY", "")
        if not kaggle_user or not kaggle_key:
            return DownloadResponse.err(
                audio_file,
                "KAGGLE_USERNAME and KAGGLE_KEY env vars required. "
                "Get credentials from https://www.kaggle.com/settings → API"
            )

        out_dir = audio_file.dest_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # kaggle datasets download -d <dataset_id> -p <output_dir> --unzip
        cmd = [
            "kaggle", "datasets", "download",
            "-d", audio_file.source_url,
            "-p", str(out_dir),
            "--unzip",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                env={**os.environ, "KAGGLE_USERNAME": kaggle_user, "KAGGLE_KEY": kaggle_key}
            )
            if result.returncode != 0:
                return DownloadResponse.err(audio_file, f"Kaggle download failed: {result.stderr[:300]}")

            # Return a sentinel path — actual files are in subdirs of out_dir
            return DownloadResponse.ok(audio_file, out_dir / "download_complete")

        except subprocess.TimeoutExpired:
            return DownloadResponse.err(audio_file, "Kaggle download timeout (>10min)")
        except Exception as e:
            return DownloadResponse.err(audio_file, f"Kaggle error: {e}")
