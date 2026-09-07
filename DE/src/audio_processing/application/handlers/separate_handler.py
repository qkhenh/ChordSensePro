from __future__ import annotations
import subprocess
from pathlib import Path

from src.shared.domain.processing_result import ProcessingResult

class SeparateHandler:
    """Handler: WAV -> Demucs htdemucs -> 'other' stem
    
    Split drums | bass | vocals | other -> only keep 'other'
    'other' = piano, keyboard, guitar 
    """
    
    MODEL = "htdemucs_6s"
    
    def run (self, wav_path: Path, output_dir: Path | None = None) -> ProcessingResult[Path]:
        out_dir = output_dir or wav_path.parent / "separated"
        
        cmd = [
            "python", "-m", "demucs",
            "-n", self.MODEL,
            "--out", str(out_dir),
            str(wav_path),
        ]
        
        try: 
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0: return ProcessingResult.err(f"Demucs thất bại: {result.stderr[:300]}")
            
            stem_dir = out_dir / self.MODEL / wav_path.stem
            
            for stem in ("piano", "guitar", "other"):
                stem_path = stem_dir / f"{stem}.wav"
                if stem_path.exists(): return ProcessingResult.ok(stem_path)
                
            return ProcessingResult.err(f"Can not find harmonic stem: {stem_dir}")
        
        except subprocess.TimeoutExpired:
            return ProcessingResult.err("Demucs timeout (>5mins)")
        
        except Exception as e:
            return ProcessingResult.err(f"Demucs error: {e}")