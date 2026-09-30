"""Quick smoke test for all pipeline fixes."""
import sys
sys.path.insert(0, "/opt/airflow")

from pathlib import Path
from src.data_processing.application.annotation_parsers.parsers import get_parser, JaahParser
from src.data_processing.application.pipeline_handlers.annotation_handler import AnnotationHandler
from src.data_processing.application.pipeline_handlers.summary_handler import SummaryHandler
from src.data_processing.domain.models.processed_audio import ProcessedAudio

# Test 1: JaahParser detection + parse
print("=== Test 1: JaahParser ===")
jaah_path = Path("/data/datasets/jaah/JAAH-v0.1/MTG-JAAH-7686b91/annotations/giant_steps.json")
p = get_parser(jaah_path)
print(f"Parser type: {type(p).__name__}")
assert type(p).__name__ == "JaahParser", f"Expected JaahParser, got {type(p).__name__}"

tl = p.parse(jaah_path)
print(f"Timeline entries: {len(tl)}")
for t in tl[:5]:
    print(f"  t={t[0]:.2f}s - {t[1]:.2f}s  chord={t[2]}")
assert len(tl) > 0, "Timeline should not be empty"
assert tl[0][2] == "B:maj7", f"First chord should be B:maj7, got {tl[0][2]}"
print("✅ JaahParser PASSED\n")

# Test 2: Kaggle filename parsing
print("=== Test 2: Kaggle filename ===")
handler = AnnotationHandler()
tests = [
    ("A_dim_2_0.wav", "A:dim"),
    ("C_maj_6_0.wav", "C:maj"),
    ("Bb_min_3_1.wav", "Bb:min"),
    ("Fs_dim_4_0.wav", "F#:dim"),
]
for fname, expected in tests:
    result = handler._label_from_filename(f"/data/datasets/kaggle/archive/piano_triads/{fname}")
    print(f"  {fname} → {result} (expected: {expected})")
    assert result == expected, f"FAIL: got {result}"
print("✅ Kaggle filename PASSED\n")

# Test 3: ProcessedAudio has new fields
print("=== Test 3: ProcessedAudio fields ===")
assert hasattr(ProcessedAudio, "__dataclass_fields__")
fields = ProcessedAudio.__dataclass_fields__
for f in ("chord_timeline", "detected_key", "chord_sheet"):
    assert f in fields, f"Missing field: {f}"
    print(f"  ✓ {f}")
print("✅ ProcessedAudio fields PASSED\n")

# Test 4: SummaryHandler
print("=== Test 4: SummaryHandler ===")
from src.data_processing.domain.models.audio_segment import AudioSegment
from src.data_loader.domain.models.audio_file import AudioFile

af = AudioFile(source_url="test", analysis_id="test123")
data = ProcessedAudio(audio_file=af)
data.segments = [
    AudioSegment(song_id="t", segment_idx=0, start_ms=0, end_ms=2000, chord_label="C:maj"),
    AudioSegment(song_id="t", segment_idx=1, start_ms=1500, end_ms=3500, chord_label="C:maj"),
    AudioSegment(song_id="t", segment_idx=2, start_ms=3000, end_ms=5000, chord_label="G:7"),
    AudioSegment(song_id="t", segment_idx=3, start_ms=4500, end_ms=6500, chord_label="A:min"),
    AudioSegment(song_id="t", segment_idx=4, start_ms=6000, end_ms=8000, chord_label="F:maj"),
]

sh = SummaryHandler()
result = sh.handle(data)
print(f"  chord_timeline: {result.chord_timeline}")
print(f"  detected_key: {result.detected_key}")
print(f"  chord_sheet: {result.chord_sheet}")
assert len(result.chord_timeline) > 0, "chord_timeline should not be empty"
assert result.detected_key != "", "detected_key should not be empty"
print("✅ SummaryHandler PASSED\n")

# Test 5: Pipeline factory includes SummaryHandler
print("=== Test 5: Pipeline factory ===")
from src.data_processing.application.pipeline_factory import build_training_pipeline
head = build_training_pipeline()
chain = []
h = head
while h is not None:
    chain.append(type(h).__name__)
    h = h._next
print(f"  Chain: {' → '.join(chain)}")
assert "SummaryHandler" in chain, "SummaryHandler missing from chain"
print("✅ Pipeline factory PASSED\n")

print("🎉 ALL TESTS PASSED")
