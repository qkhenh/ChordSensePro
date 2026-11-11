import sys
import os

sys.path.insert(0, ".")

from src.serve import ONNXPredictor, infer_audio, load_audio_bytes

AUDIO_PATH   = r"D:\full.mp3"
MODEL_PATH   = "results/model_v3_5s_p2.onnx"
CLASSES_JSON = "../DE/data/export/v3/classes.json"

if not os.path.exists(AUDIO_PATH):
    alt = r"D:\test.wav"
    if os.path.exists(alt):
        AUDIO_PATH = alt
    else:
        print("File not found:", AUDIO_PATH)
        sys.exit(1)

print("Loading model:", MODEL_PATH)
pred = ONNXPredictor(MODEL_PATH, CLASSES_JSON)

print("Loading audio:", AUDIO_PATH)
with open(AUDIO_PATH, "rb") as f:
    data = f.read()
audio = load_audio_bytes(data, 24000, AUDIO_PATH)
dur_s = len(audio) / 24000
print(f"Audio duration: {dur_s:.1f}s\n")

results = infer_audio(pred, audio)

print(f"{'Start':>7}  {'End':>7}  {'Chord':<14} {'Conf':>5}")
print("-" * 42)
for r in results:
    start = r["start_ms"] / 1000
    end   = r["end_ms"]   / 1000
    print(f"{start:6.1f}s  {end:6.1f}s  {r['chord']:<14} {r['confidence']:.2f}")

print(f"\nTotal segments: {len(results)}")

