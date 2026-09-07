# ChordSense Pro — DE & AI Specification
> **Đề tài:** *ChordSense Pro: An AI-Powered Context-Aware System for Musician's Interactive Practice Workspace with Harmonic Analysis Support*
> **Phạm vi:** Tài liệu này chỉ bao gồm phần **Data Engineering** và **AI/ML** của đồ án.
> Phần BE/FE do team member quyết định riêng.

---

## 0. Tóm Tắt Quyết Định Đã Chốt

| Hạng mục | Quyết định |
|----------|-----------|
| **Nhạc cụ chính** | Piano (primary) — Guitar optional nếu còn thời gian |
| **Pretrained model** | MERT-v1-330M (`m-a-p/MERT-v1-330M`) |
| **Fine-tuning method** | LoRA (PEFT) — rank r=16 |
| **Model architecture** | Cascaded 3-Head (Root → Quality → Extension) |
| **Dataset chính** | RWC + McGill Billboard + JAAH + ChoCo + Pianoteq synthetic |
| **Augmentation chính** | Pitch-shift + label update (×12 keys) + noise/reverb |
| **Loss function** | Focal Loss (γ=2) cho Head 3 (Extended) |
| **Post-processing** | HMM smoothing + Chord Transition Prior |
| **ETL orchestration** | Apache Airflow |
| **Primary DB** | PostgreSQL (structured) + MongoDB (audio/raw) |

---

## 1. Kiến Trúc Tổng Thể — DE + AI

```
                    ┌────────────────────────────────────┐
                    │       CHORD RECOGNITION ENGINE      │
                    │   MERT-v1-330M + LoRA Cascaded Head │
                    └──────────────┬─────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  LEARNING FLOW  │  │  REAL-TIME PRAC  │  │  ANALYTICS       │
   │  Song import →  │  │  Mic input →     │  │  Student mastery │
   │  Full-song ACR  │  │  2s chord detect │  │  Airflow rollup  │
   └─────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 2. Data Engineering — ETL Pipeline

### 2.1 Ba Luồng Dữ Liệu

```
LUỒNG 1: Curriculum (1 lần, static)
  Nguồn: SGK GDPT 2018 Âm nhạc THCS (PDF công khai)
  Nội dung: Chord list theo từng khối lớp (6-9)
  Output: PostgreSQL tables: chords, lessons
  DAG: dag_curriculum_ingest (manual trigger)

LUỒNG 2: Training Audio (1 lần, offline)
  Nguồn: Dataset học thuật (CC license)
  Output: Parquet files → /data/training/
  DAG: dag_dataset_ingest (manual trigger, chạy ~4-6 giờ)

LUỒNG 3: Student Interaction (real-time, ongoing)
  Nguồn: WebRTC audio từ user (2s WAV)
  Output: PostgreSQL: chord_attempts, practice_sessions
  DAG: dag_analytics_rollup (23:30 daily)
```

### 2.2 Dataset Strategy

| Dataset | Nội dung | Mục đích | License |
|---------|----------|---------|---------|
| **RWC Popular Music** | 100 bài, chord annotation theo beat | Head 1+2 training | Free (register) |
| **McGill Billboard** | 1,300 bài pop/rock, JAMS format | Head 1+2 training | CC Research |
| **JAAH** | 113 jazz tracks, có dim7/maj7/aug đầy đủ | Head 2+3 training | CC (Zenodo) |
| **ChoCo Corpus** | 20,000 bài, Harte notation, 18 nguồn | Head 3 (extension) | CC (GitHub) |
| **Pianoteq Synthetic** | Render extended chords → label hoàn hảo | Head 3 + pre-train | Internal |
| **Kaggle Guitar v3** | 7,000+ isolated chord recordings | Guitar optional | CC |

> **Piano First:** Ưu tiên piano tracks trong RWC/McGill. Pianoteq render bổ sung cho extended chords hiếm.

### 2.3 Airflow DAGs

```
dag_curriculum_ingest    → schedule=None (1 lần)
  Task 1: Parse SGK PDF → chord list per grade
  Task 2: Seed PostgreSQL: chords, lessons, curriculum_path
  Task 3: Validate integrity

dag_dataset_ingest       → schedule=None (1 lần)
  Task 1: Download McGill + RWC + JAAH + ChoCo
  Task 2: Parse JAMS → (audio_segment, chord_label) pairs
  Task 3: Render Pianoteq synthetic extended chords
  Task 4: Audio preprocessing: resample 22050Hz → chroma CQT → mel spectrogram
  Task 5: Augmentation pipeline (pitch-shift x12 + noise)
  Task 6: Export chord_dataset_v1.parquet → /data/training/

dag_analytics_rollup     → schedule="30 23 * * *"
  Task 1: Aggregate chord_attempts → accuracy per (student, chord, date)
  Task 2: Tính rolling 3-day accuracy
  Task 3: Update student_chord_mastery (is_mastered flag)
  Task 4: Update lesson_completions nếu đủ điều kiện

dag_model_retrain        → schedule="0 2 1 * *" (monthly)
  Task 1: Export student recordings được review → new training samples
  Task 2: Merge với existing training data
  Task 3: Fine-tune LoRA thêm (Phase 3 on new data)
  Task 4: Evaluate WCS → deploy nếu cải thiện
```

---

## 3. Audio Processing Pipeline

### 3.1 Pipeline Xử Lý Bài Nhạc (Learning Flow)

```
YouTube URL / File Upload
    │
    ▼
[1] Download: yt-dlp → WAV (mono, 44.1kHz)
    │
    ▼
[2] Source Separation: Demucs (Meta AI, open-source)
    → Tách: drums | bass | other (piano) | vocals
    → Chỉ giữ lại "other" track
    │
    ▼
[3] Beat Tracking: madmom BeatTracker
    → beat_times = [0.0, 0.5, 1.0, 1.5, ...]
    → time_signature detection
    │
    ▼
[4] Segmentation: chia thành 2s segments căn theo beat
    │
    ▼
[5] Feature Extraction (Librosa):
    chroma_cqt      → root info (12-dim)
    chroma_cens     → noise-robust chroma
    HPSS harmonic   → pure harmonic component
    mel_high        → 64 bins, fmin=2000Hz (high extension info)
    → Multi-scale 4-channel input
    │
    ▼
[6] MERT + LoRA Cascaded → Root + Quality + Extension
    │
    ▼
[7] Post-processing:
    HMM Smoothing: loại chord flickering
    [C, C, Am, C, C] → [C, C, C, C, C]
    Chord Transition Prior từ dataset statistics
    Key-conditioned disambiguation (Db vs C#)
    │
    ▼
[8] Section Detection: grouping theo pattern repetition
    → intro / verse / chorus / bridge labels
    │
    ▼
Output: Chord Sheet (PostgreSQL: song_analyses) + Learning Plan
```

### 3.2 Pipeline Real-time (Practice Mode)

```
WebRTC audio (2s WAV, 44.1kHz)
    │
    ▼
LoadHandler → validate duration 1.5-4s
SegmentHandler → 2s segments, 0.5s overlap
ExtractHandler → Chroma CQT + Mel spectrogram
NormalizeHandler → min-max normalization
    │
    ▼
MERT + LoRA → chord prediction (<500ms total)
    │
    ▼
Response → API
```

### 3.3 Feature Extraction Chi Tiết

```python
# Multi-scale input cho extended chord detection
def extract_features(audio, sr=22050):
    chroma_cqt  = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sr)
    y_harm, _   = librosa.effects.hpss(audio)
    mel_high    = librosa.feature.melspectrogram(
        y=audio, sr=sr, fmin=2000, fmax=8000, n_mels=64
    )
    return {
        "chroma_cqt":    chroma_cqt.mean(axis=1),     # (12,) — Root info
        "chroma_cens":   chroma_cens.mean(axis=1),    # (12,) — Noise-robust
        "hpss_harmonic": y_harm,                       # waveform → MERT input
        "mel_high":      mel_high.mean(axis=1),        # (64,) — 9th/11th/13th
    }
```

---

## 4. AI Model — Kiến Trúc Đã Chốt

### 4.1 Pretrained Model: MERT-v1-330M

```
HuggingFace: "m-a-p/MERT-v1-330M"
Params:       330M (frozen hoàn toàn, không train)
Output:       1024-dim embedding vector per audio segment
Pre-train:    Multi-task SSL trên 160,000 giờ nhạc
  - RVQ Teacher → acoustic features (pitch, timbre)
  - MFCC Teacher → temporal features (rhythm, transitions)
SOTA trên MIREX chord recognition benchmark (ICLR 2024, >200 citations)
Note:         Cần Google Colab Pro (A100 40GB) để fine-tune
```

### 4.2 LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rank — experiments: 8 / 16 / 32
    lora_alpha=32,           # Scaling = 2×r
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)
# Trainable: ~1,982,464 params (0.600% of total)
# VRAM: ~20-24GB → cần A100 40GB (Google Colab Pro)
```

### 4.3 Cascaded 3-Head Architecture ⭐ Contribution chính

```
MERT (frozen) → 1024-dim embedding
    │
    ▼
┌─────────────────────────────────────────┐
│  HEAD 1 — ROOT (12 classes)             │
│  Linear(1024→128) → ReLU → Linear(128→12)│
│  → root_logits: [C=0.02, ..., A=0.91,...]
│  Accuracy target: >95%                  │
└───────────────┬─────────────────────────┘
                │ concat root_logits
                ▼
┌─────────────────────────────────────────┐
│  HEAD 2 — QUALITY (6 classes)           │
│  Input: [embedding(1024) + root(12)]    │
│  Linear(1036→128) → ReLU → Linear(128→6)│
│  Classes: major/minor/dominant/dim/aug/sus
│  Accuracy target: >85%                  │
└───────────────┬─────────────────────────┘
                │ concat root + quality logits
                ▼
┌─────────────────────────────────────────┐
│  HEAD 3 — EXTENSION (8 classes)         │
│  Input: [embedding(1024) + root(12) + quality(6)]
│  Linear(1042→256) → ReLU → Dropout(0.3)│
│  → Linear(256→64) → ReLU → Linear(64→8)│
│  Classes: none/maj7/min7/dom7/9th/11th/13th/altered
│  Loss: Focal Loss (γ=2)                 │
│  Accuracy target: >70%                  │
└─────────────────────────────────────────┘

Final output: "A" + "minor" + "min7" → Am7
              "G" + "dominant" + "altered" → G7#9
```

> **Cascaded vs Parallel:** Head 3 biết root context từ Head 1 → phân biệt Am7 vs C6 tốt hơn (cùng 4 nốt nhưng khác root).

### 4.4 Partial Voicing Recognition

```python
# Khi người chơi không đánh đủ nốt (ví dụ chỉ F+G thay vì Fadd9=F+A+C+G)
CHORD_TEMPLATES = {
    "Fadd9": {"F": 1.0, "A": 0.7, "C": 0.4, "G": 0.9},
    # Root=1.0 (bắt buộc), 3rd=0.7, 5th=0.4 (hay omit), 9th=0.9 (defining note)
}
# Tie-break bằng HMM Chord Transition Prior:
# P(Fadd9 | prev=C) > P(Fsus2 | prev=C) → chọn Fadd9
```

### 4.5 Training Pipeline

```
PHASE 1 — Foundation (5 epochs)
  Train: Head 1 + Head 2 (Head 3 frozen)
  Data:  McGill + RWC + JAAH
  Loss:  0.5 × loss_root + 0.5 × loss_quality
  Optimizer: AdamW(lr=2e-4)

PHASE 2 — Extended Chord Focus (10 epochs)
  Train: LoRA adapters + All 3 heads
  Data:  + ChoCo + Pianoteq synthetic
  Loss:  0.3 × loss_root + 0.3 × loss_quality + 0.4 × loss_ext
         Head 3 → Focal Loss (γ=2)
  Optimizer: AdamW(lr=5e-5)

PHASE 3 — Domain Fine-tune (5 epochs)
  Train: LoRA adapters + Heads only (MERT frozen)
  Data:  VN piano recordings (tự thu âm)
  Optimizer: AdamW(lr=1e-5)

Early stopping: patience=3 trên val WCS (Weighted Chord Score)
```

### 4.6 Augmentation Pipeline

```python
CHROMATIC_NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def pitch_shift_with_label(audio, chord_label, semitones, sr=22050):
    """Pitch shift + cập nhật label. Quality và Extension KHÔNG ĐỔI."""
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    idx = CHROMATIC_NOTES.index(chord_label.root)
    new_root = CHROMATIC_NOTES[(idx + semitones) % 12]
    return shifted, new_root + chord_label.quality_suffix

waveform_aug = A.Compose([
    A.AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.5),
    A.RoomSimulator(p=0.3),
    A.Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
    A.TimeStretch(min_rate=0.85, max_rate=1.15, p=0.3),
])

# Kết quả: 1 sample → 12 keys × 2 versions = ~24 samples
# Am9 (50 recordings) → 1,200 samples sau augmentation
```

**Quy tắc SpecAugment:**
- ✅ Time Masking `param=20`: an toàn
- ⚠️ Frequency Masking `param≤10`: chỉ mask tần số >4kHz
- ❌ Mixup: KHÔNG dùng (label bị mơ hồ)

---

## 5. Experiments Thesis

| ID | So sánh | Câu hỏi nghiên cứu |
|----|---------|-------------------|
| A | Chroma template vs Our Model B | LoRA beat classical method? |
| B | CNN flat 60-class vs Our Model B | Hierarchical vs flat? |
| C | MERT zero-shot vs Our Model A | LoRA cần thiết? |
| D | Conformer 2024 SOTA vs Our Model B | Architecture so sánh |
| E | Parallel heads vs Cascaded heads | Cascaded hiệu quả hơn? |
| F | LoRA rank 8 vs 16 vs 32 | Rank tối ưu cho chord? |
| G | Our Model B vs Model C (+VN data) | VN data cải thiện bao nhiêu? |
| H | Our Model C vs ChordAI | Extended chord comparison |

**Metrics:**
```
Per-head:    Accuracy, F1-macro mỗi Head 1/2/3
Chord-level: Chord Accuracy (CA), Weighted Chord Score (WCS)
Latency:     ms/request (real-time target ≤500ms)
Extended:    Riêng accuracy trên 9th/11th/13th/dim/aug
```

---

## 6. API Contracts (DE/AI → BE)

```
POST /api/v1/recognize
  Body:     { "audio": "<base64 WAV 2s>" }
  Response: {
    "chord": "Am7",
    "root": "A",         "root_conf": 0.97,
    "quality": "minor",  "quality_conf": 0.91,
    "extension": "min7", "extension_conf": 0.78,
    "alternatives": [{"chord": "C6", "conf": 0.12}],
    "latency_ms": 385
  }

POST /api/v1/song/analyze
  Body:     { "youtube_url": "..." }
  Response: { "job_id": "uuid", "status": "processing" }

GET /api/v1/song/analyze/{job_id}
  Response: {
    "status": "done",
    "key": "Am", "tempo_bpm": 75,
    "chord_sheet": {
      "sections": [
        {"name": "Intro",  "chords": ["Am","Am","F","C G"]},
        {"name": "Verse",  "chords": ["Am","Am","Dm","E"]}
      ]
    },
    "learning_plan": [
      {"chord": "Am", "status": "mastered"},
      {"chord": "F",  "status": "to_learn", "order": 1}
    ]
  }
```

---

## 7. Database Schema

```sql
-- Song analysis cache
CREATE TABLE song_analyses (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id      UUID,
    source_url      TEXT,
    song_title      VARCHAR(200),
    detected_key    VARCHAR(10),
    tempo_bpm       FLOAT,
    chord_timeline  JSONB,
    chord_sheet     JSONB,
    learning_plan   JSONB,
    status          VARCHAR(20),  -- processing | done | error
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chord_attempts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id      UUID,
    session_id      UUID,
    target_chord    VARCHAR(10),
    detected_chord  VARCHAR(10),
    root_conf       FLOAT,
    quality_conf    FLOAT,
    extension_conf  FLOAT,
    is_correct      BOOLEAN,
    timestamp       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE student_chord_mastery (
    student_id          UUID,
    chord               VARCHAR(10),
    date                DATE,
    accuracy_today      FLOAT,
    rolling_accuracy_3d FLOAT,
    is_mastered         BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (student_id, chord, date)
);
```

**MongoDB Collections:**
```
raw_audio:         Binary WAV từ student (TTL: 30 ngày)
mel_spectrograms:  (128,T) matrix per segment (TTL: 90 ngày)
training_samples:  Processed samples từ datasets
song_audio_cache:  Demucs output (TTL: 7 ngày)
```

---

## 8. Tech Stack

```
Audio Processing:
  librosa 0.10+       Chroma CQT, mel spectrogram, HPSS, pitch shift
  audiomentations     Waveform augmentation (noise, reverb, gain)
  madmom              Beat tracking
  demucs              Source separation (Meta AI, open-source)
  yt-dlp              YouTube audio download
  soundfile           WAV encode/decode

AI Framework:
  PyTorch 2.0+        Base framework
  HuggingFace MERT    "m-a-p/MERT-v1-95M"
  PEFT                LoRA implementation
  torchaudio          SpecAugment transforms
  MLflow              Experiment tracking + model registry

Data Pipeline:
  Apache Airflow 2.8  DAG orchestration
  PostgreSQL 16       Structured data
  MongoDB 7.0         Raw audio, spectrograms
  Redis 7.2           Inference cache
  pyarrow/parquet     Training data format

Optimization:
  ONNX Runtime        Export sau training → faster inference
  INT8 Quantization   CPU deploy optimization
```

---

## 9. Timeline DE & AI

### Kỳ 1 (3 tháng) — DE Focus

```
Tháng 1:
  Tuần 1-2: Setup Docker (PostgreSQL + MongoDB + Airflow + Redis)
  Tuần 3:   dag_curriculum_ingest: parse SGK → seed DB
  Tuần 4:   Audio processing pipeline (Librosa + Demucs + madmom)

Tháng 2:
  Tuần 1-2: dag_dataset_ingest: download McGill + JAAH + ChoCo
             Render Pianoteq synthetic extended chords
  Tuần 3:   Augmentation pipeline (pitch-shift×12 + noise)
             Export chord_dataset_v1.parquet
  Tuần 4:   dag_analytics_rollup + POST /recognize stub

Tháng 3:
  Tuần 1-2: Song Analysis pipeline (yt-dlp + Demucs + beat + chord)
  Tuần 3:   Thu âm VN piano dataset (extended chords)
  Tuần 4:   Demo kỳ 1: Import bài YouTube → Chord Sheet
```

### Kỳ 2 (3 tháng) — AI Focus

```
Tháng 4:
  Tuần 1-2: MERT baseline + Baseline A/B (chroma + CNN flat)
  Tuần 3-4: Baseline C/D (MERT zero-shot + Conformer reference)

Tháng 5:
  Tuần 1-2: Cascaded 3-Head + Focal Loss
             Phase 1+2 training, Experiments B/C/D/E/F
  Tuần 3-4: Partial Voicing + HMM Prior
             Experiment G (VN fine-tune)

Tháng 6:
  Tuần 1:   Experiment H (vs ChordAI trên extended chords)
  Tuần 2:   ONNX export + quantization
  Tuần 3:   Tích hợp vào API end-to-end
  Tuần 4:   Final demo + thesis writing
             Demo: Đánh Am9 → "Am9" ✅ | ChordAI → "Am7" ❌
```

---

## 10. Academic Contribution

```
1. VN Piano Extended Chord Dataset
   240 recordings: 12 extended chords × 10 takes × 2 expert players
   Vary voicings, registers, velocities → publish Zenodo

2. Cascaded 3-Head LoRA (vs Parallel heads)
   Experiment E chứng minh Cascaded > Parallel trên extended chords
   Reason: root context giúp Head 3 phân biệt Am7 vs C6

3. Partial Voicing Recognition
   Weighted Note Presence + HMM Chord Prior
   First work formalize piano partial voicing trong ACR

4. Evaluation Extended Chords vs Commercial Tool (ChordAI)
   Test set riêng cho 9th/11th/13th/dim/aug
   Gap rõ ràng để justify thesis contribution
```

---

## 11. Output Modes — Vocal vs Instrumental

ChordSense Pro hỗ trợ 2 chế độ output tùy theo loại nhạc đầu vào:

---

### Mode A — Vocal Music → ChordPro Format

**Use case:** Nhạc có lời hát — pop, ballad, folk, nhạc Việt.

**Pipeline:**
```
Audio (có lời)
  │
  ├─► Demucs v4 (source separation)
  │         │
  │         ├── harmony track → MERT-v1-330M → Chord timestamps
  │         │                   [F:0.02s] [G:1.14s] [Am:2.30s]...
  │         │
  │         └── vocal track  → OpenAI Whisper → Lyrics + word timestamps
  │                             "Em"(0.00s) "là"(0.10s) "ai"(0.21s)...
  │
  ▼
Chord-Lyric Alignment
  → map chord change timestamp → nearest lyric word boundary
  │
  ▼
ChordPro Format output
```

**Output example:**
```
Em là [F]ai từ đâu bước [G]đến nơi đây dịu [Am]dàng chân phương
Em là [F]ai tựa như ánh [G]nắng ban mai ngọt [C]ngào trong sương
```

**Export formats:** .chordpro, .txt, .pdf

---

### Mode B — Instrumental Music → Lead Sheet

**Use case:** Nhạc không lời — guitar instrumental, piano solo, jazz, film score, synthesis lead.

**Pipeline:**
```
Audio (không lời)
  │
  ▼
Demucs v4
  ├── drums  (bỏ qua)
  ├── bass   (bỏ qua / optional)
  └── other  ← guitar lead / synth lead / piano melody
         │
         ├─► MERT-v1-330M → Chord timeline
         │                   Fmaj7(0.0s) G7(2.0s) Am7(4.0s)...
         │
         └─► CREPE / Basic Pitch (Spotify)
             → Melody note events (monophonic / polyphonic)
             → [F4: 0.02s–0.18s] [G4: 0.18s–0.32s] [A4: 0.32s–0.50s]...
                    │
                    ▼
             Quantize → beat grid (snap to nearest 16th note)
                    │
                    ▼
             VexFlow (browser render) → Lead Sheet
```

**Output example:**
```
    Fmaj7           G7              Am7
╔══════════════════════════════════════════╗
║  C5─D5─E5─F5   | G5─F5─E5─D5  | C5─── ║  ← melody notes
║   ♩  ♩  ♩  ♩   |  ♩  ♩  ♩  ♩  |  𝅗𝅥    ║  ← rhythm
╚══════════════════════════════════════════╝
```

**Export formats:** .mid (MIDI), .musicxml, .pdf (sheet music), .png

---

### Bảng so sánh 2 mode

| | Mode A — Vocal | Mode B — Instrumental |
|---|---|---|
| Input | Nhạc có lời | Nhạc không lời |
| Chord detection | MERT-v1-330M ✅ | MERT-v1-330M ✅ |
| Lyrics | Whisper ASR ✅ | Không cần |
| Melody notes | Không cần | CREPE / Basic Pitch ✅ |
| Output | ChordPro format | Lead Sheet (staff notation) |
| Render | Text / PDF | VexFlow (web) / MuseScore |
| Export | .chordpro .txt .pdf | .mid .musicxml .pdf |

---

### Tech Stack bổ sung cho Output Modes

| Tool | Vai trò | License |
|------|---------|---------|
| **OpenAI Whisper** | Lyrics extraction + word-level timestamps | MIT |
| **CREPE** (Google/Marl) | Monophonic pitch/melody detection | MIT |
| **Basic Pitch** (Spotify) | Polyphonic note transcription → MIDI | Apache 2.0 |
| **VexFlow** | Render sheet music trên browser (JS) | MIT |
| **music21** | MusicXML generation, music theory ops | BSD |
| **LilyPond** (optional) | High-quality PDF sheet music render | GPL |

---

### Accuracy benchmark (Melody transcription)

| Loại nhạc | Model | Expected accuracy |
|-----------|-------|------------------|
| Guitar solo đơn note | CREPE | ~85–92% |
| Synth lead (sine/saw) | CREPE | ~90%+ |
| Piano melody | Basic Pitch | ~85%+ |
| Guitar rhythm phức tạp | Basic Pitch | ~70–80% |

> Note: Accuracy phụ thuộc nhiều vào chất lượng source separation của Demucs.
> Fine-tune CREPE trên GuitarSet dataset có thể cải thiện thêm ~5–10%.
