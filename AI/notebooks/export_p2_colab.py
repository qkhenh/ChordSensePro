"""
ChordSense Pro — ONNX Export Phase 2 (chạy cell này trên Colab)
Paste từng cell vào notebook hoặc chạy as script.
"""

# ============================================================
# Cell 1: Paths
# ============================================================
import os
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/ChordSensePro")
REPO = Path("/content/ChordSensePro")
AI = REPO / "AI"

P2_CKPT = DRIVE_ROOT / "AI/results/checkpoints_p2/best.pt"
OUT_DIR  = DRIVE_ROOT / "AI/results"
OUT_ONNX = OUT_DIR / "model_p2.onnx"
CONFIG   = AI / "configs/phase2_lora.yaml"

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("P2 checkpoint:", P2_CKPT.exists())

# ============================================================
# Cell 2: Export + Benchmark
# ============================================================
os.chdir(AI)

cmd = (
    f"python -m src.export_onnx"
    f" --config {CONFIG}"
    f" --checkpoint {P2_CKPT}"
    f" --output {OUT_ONNX}"
    f" --benchmark"
    f" --warmup 5"
    f" --runs 20"
)
print(cmd)
os.system(cmd)

# ============================================================
# Cell 3: Verify output
# ============================================================
print("ONNX exists:", OUT_ONNX.exists())
if OUT_ONNX.exists():
    size_mb = OUT_ONNX.stat().st_size / 1e6
    print(f"Size: {size_mb:.1f} MB")

