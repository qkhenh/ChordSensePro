from __future__ import annotations
from contextlib import contextmanager
import tempfile
from pathlib import Path
import yt_dlp

from src.shared.domain.processing_result import ProcessingResult

class DownloadHandler:
    """Handler: YouTube URL / local file → WAV 44100Hz mono.
    Output with 44100Hz
    """
    
    # ffmpeg args: -ar 44100 (resample), -ac 1 (mono)
    _YDL_OPTS = {
        "format": "bestaudio/best",
        "outtmpl": "%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": {
            "ffmpeg": ["-ar", "44100", "-ac", "1"]
        },
        "quiet": True,
        "no_warnings": True,
    }
    
    def run(self, source: str, output_dir: Path | None = None) -> ProcessingResult[Path]:
        """
        
        Download and convert to wav 44100Hz mono

        Args:
            source (str): Ytb URL or local file
            output_dir (Path | None, optional): Folder save the output. Defaults to None.

        Returns:
            ProcessingResult[Path]: path to file.wav
        """
        
        out_dir = output_dir or Path(tempfile.mkdtemp())
        out_dir.mkdir(parents=True, exist_ok=True)
        
        #If localfile - resample
        if not source.startswith("http"):
            return self._convert_local(Path(source), out_dir)
        
        #Ytb download
        opts = {**self._YDL_OPTS, "outtmpl": str(out_dir / "%(id)s.%(ext)s")}
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(source, download=True)
                video_id = info["id"]
                wav_path = out_dir / f"{video_id}.wav"
                
                if not wav_path.exists():return ProcessingResult.err(f"WAV không tồn tại sau download: {wav_path}")
                
                return ProcessingResult.ok(wav_path)
            
        except Exception as e:
            return ProcessingResult.err(f"Download thất bại: {e}")
        
    def _convert_local(self, src: Path, out_dir: Path) -> ProcessingResult[Path]:
        """Local file → WAV 44100Hz mono bằng soundfile + resampling."""
        if not src.exists(): return ProcessingResult.err(f"File không tồn tại: {src}")
        
        try:
            import soundfile as sf
            import librosa
            audio, sr = librosa.load(str(src), sr=44100, mono=True)
            out_path = out_dir / (src.stem + ".wav")
            sf.write(str(out_path), audio, 44100)
            return ProcessingResult.ok(out_path)
        
        except Exception as e:
            return ProcessingResult.err(f"Convert thất bại: {e}")
        
    @contextmanager
    def download_temp(self, source: str):
        """Context manager: download → dùng → tự xóa."""
        result = self.run(source)
        if result.is_err:
            yield result
            return
        wav_path = result.unwrap()
        try:
            yield ProcessingResult.ok(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)   # xóa file sau khi block kết thúc
            wav_path.parent.rmdir()             # xóa temp dir

        
        