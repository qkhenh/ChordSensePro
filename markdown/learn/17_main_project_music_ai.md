# 17 — Main Project: MelodIQ — Hệ Thống Phân Tích Âm Nhạc & Gợi Ý Thông Minh

> **Mục tiêu:** Xây dựng hệ thống end-to-end: Thu thập metadata âm nhạc từ các nguồn mở → ETL Pipeline (DDD) → AI phân tích thể loại + gợi ý bài hát + dự đoán xu hướng viral → Web/API triển khai được. Đủ phức tạp cho đồ án tốt nghiệp, đủ thực tế để impress senior IT.

---

## 0. Tổng Quan Dự Án

### Tên dự án: **MelodIQ**
> *"Hiểu nhạc của bạn, gợi ý nhạc cho bạn, dự báo nhạc tương lai."*

### Bài toán thực tế

Người nghe nhạc hôm nay đang bị "overwhelmed" với hàng triệu bài hát trên Spotify, YouTube, TikTok. Câu hỏi luôn đặt ra:
- *"Bài này thuộc thể loại gì? Nghe tiếp bài nào?"*
- *"Bài này có viral không? Viral trên TikTok hay YouTube?"*
- *"Tôi muốn nghe nhạc giống kiểu này, tìm đâu?"*

### MelodIQ giải quyết gì?

```
Người dùng upload/link bài nhạc
       ↓
Audio Feature Extraction (Librosa + Spleeter)
       ↓
AI Phân loại thể loại (Fine-tuned với LoRA)
       ↓
Gợi ý bài hát cùng âm thanh/phong cách
       ↓
Dự đoán viral potential (TikTok / YouTube)
       ↓
Web App + REST API (deploy được)
```

### Phân chia nhân lực

| Role | Người | Trách nhiệm chính |
|------|-------|-------------------|
| **DE + AI** | Bạn | ETL Pipeline, Audio Feature Extraction, AI Model (LoRA fine-tuning), Data Warehouse |
| **BE + FE** | Bạn bè | REST API, Web App, UI/UX, Authentication, Deployment |

---

## 1. Ý Tưởng Cốt Lõi & Điểm Khác Biệt

### Tại sao chọn bài toán này?

| Yếu tố | Giải thích |
|--------|-----------|
| **Dữ liệu phong phú** | Spotify API, Last.fm, MusicBrainz — free, documented, không cần crawl phức tạp |
| **Audio analysis** | Librosa là thư viện chuẩn ngành, không cần tự build từ đầu |
| **AI ứng dụng thực tế** | Fine-tuning với LoRA = kỹ thuật hot 2024-2025, apply được vào thesis |
| **Deploy được** | API + Web App → demo cho GVHD thấy kết quả trực tiếp |
| **Cả 2 đều có đất dụng võ** | DE/AI: pipeline + model; BE/FE: API + giao diện → không bị chồng chéo |
| **Senior IT sẽ đánh giá cao** | Multi-source ETL + AI fine-tuning + production-ready deployment = stack hiện đại |

### Tính năng chính

```
Feature 1: Genre Classification (AI)
  → Upload bài nhạc → nhận diện thể loại (Pop, R&B, EDM, Jazz, Classical, Hip-hop,...)
  → Độ tin cậy (confidence) cho từng thể loại

Feature 2: Music Recommendation (AI + Data)
  → Từ bài nhạc hiện tại → gợi ý top-10 bài tương tự
  → Dựa trên audio features (không chỉ metadata)

Feature 3: Viral Potential Prediction (AI)
  → Phân tích audio features + metadata → dự đoán khả năng viral
  → Phân loại: TikTok viral / YouTube viral / Mainstream / Niche
  → Đây là điểm độc đáo nhất — chưa ai làm hệ thống như thế này mà để public

Feature 4: Music Discovery Feed (Data)
  → Dashboard trending: nhạc đang lên, thể loại đang nóng theo tuần/tháng
```

---

## 2. Kiến Trúc Tổng Thể

```
MelodIQ/
├── .env
├── pyproject.toml
├── docker-compose.yml          ← PostgreSQL + MongoDB + Airflow + Redis
│
├── config/
│   └── genre_mapping.yml       ← Map genre labels, audio thresholds
│
├── dags/                       ← Airflow DAGs (Kỳ 1)
│   ├── dag_ingest.py           ← DAG 1: Crawl metadata từ Spotify + Last.fm
│   ├── dag_audio_etl.py        ← DAG 2: Trích xuất audio features
│   ├── dag_ai_training.py      ← DAG 3: Trigger AI training/fine-tuning
│   └── dag_trend_refresh.py    ← DAG 4: Cập nhật trending data hàng tuần
│
├── src/
│   ├── shared/                         ← Nền tảng chung (DDD base)
│   │   ├── domain/
│   │   │   ├── base_model.py
│   │   │   ├── processing_result.py
│   │   │   └── value_objects.py        ← AudioFeatureVector, GenreLabel, ...
│   │   │
│   │   ├── infrastructure/
│   │   │   ├── settings/
│   │   │   │   ├── base_setting.py
│   │   │   │   ├── spotify_setting.py
│   │   │   │   ├── postgres_setting.py
│   │   │   │   └── mongo_setting.py
│   │   │   ├── postgres/
│   │   │   │   ├── client.py
│   │   │   │   └── base_repository.py
│   │   │   └── mongo/
│   │   │       ├── client.py
│   │   │       └── base_repository.py
│   │   │
│   │   └── utils/
│   │       ├── logging.py
│   │       └── audio_utils.py          ← Helper cho Librosa
│   │
│   ├── data_ingest/                    ← MODULE 1: Thu thập metadata nhạc
│   │   ├── domain/
│   │   │   ├── models/
│   │   │   │   ├── track.py            ← Track domain model
│   │   │   │   ├── artist.py
│   │   │   │   └── raw_track_data.py   ← Raw data từ API
│   │   │   └── ports/
│   │   │       └── music_source.py     ← Protocol cho các source
│   │   │
│   │   └── application/
│   │       ├── entrypoints.py
│   │       └── sources/
│   │           ├── spotify_source.py
│   │           ├── lastfm_source.py
│   │           └── musicbrainz_source.py
│   │
│   ├── audio_processing/               ← MODULE 2: Phân tích audio
│   │   ├── domain/
│   │   │   ├── models/
│   │   │   │   ├── audio_features.py   ← AudioFeatureSet model
│   │   │   │   └── mel_spectrogram.py
│   │   │   └── ports/
│   │   │       └── feature_extractor.py
│   │   │
│   │   └── application/
│   │       ├── entrypoints.py
│   │       ├── pipeline_handlers/
│   │       │   ├── load_handler.py     ← Load audio file
│   │       │   ├── extract_handler.py  ← Librosa feature extraction
│   │       │   ├── normalize_handler.py
│   │       │   └── save_handler.py
│   │       └── extractors/
│   │           ├── librosa_extractor.py
│   │           └── spectrogram_extractor.py
│   │
│   ├── ai_engine/                      ← MODULE 3: AI (Kỳ 2)
│   │   ├── domain/
│   │   │   ├── models/
│   │   │   │   ├── genre_prediction.py
│   │   │   │   ├── recommendation.py
│   │   │   │   └── viral_prediction.py
│   │   │   └── ports/
│   │   │       └── predictor.py
│   │   │
│   │   └── application/
│   │       ├── entrypoints.py
│   │       ├── genre_classifier/
│   │       │   ├── trainer.py          ← LoRA fine-tuning
│   │       │   ├── predictor.py
│   │       │   └── lora_config.py
│   │       ├── recommender/
│   │       │   ├── embedding_builder.py
│   │       │   └── similarity_engine.py
│   │       └── viral_predictor/
│   │           ├── feature_builder.py
│   │           └── predictor.py
│   │
│   └── api/                            ← MODULE 4: REST API (BE làm chính)
│       ├── routes/
│       │   ├── analyze.py              ← POST /analyze
│       │   ├── recommend.py            ← GET /recommend/{track_id}
│       │   └── trending.py             ← GET /trending
│       └── schemas/
│           ├── request.py
│           └── response.py
│
└── scripts/
    ├── init_db.py
    ├── seed_genres.py                  ← Seed genre labels
    └── download_sample_data.py         ← Tải sample audio cho dev
```

---

## 3. Nguồn Dữ Liệu

> **Tổng quan chiến lược dữ liệu:** Chia làm 3 lớp rõ ràng:
> - **Lớp 1 — Live API:** Thu thập metadata (tên, artist, genre tags, popularity) theo thời gian thực
> - **Lớp 2 — Static Dataset:** Dữ liệu training có label sẵn, dùng để train AI models
> - **Lớp 3 — Self-Extraction:** Tự trích xuất audio features từ file nhạc bằng Librosa/Essentia

---

### ⚠️ Cảnh Báo Quan Trọng: Spotify Audio Features API Đã Bị Kill

> Spotify đã **deprecated và block** endpoint `/audio-features` vào ngày **27/11/2024**.
> Mọi app đăng ký sau ngày đó → **403 Forbidden**. Không dùng được nữa.
> Chỉ còn dùng Spotify cho: track search, basic metadata, popularity score.

---

## 🔵 LỚP 1: Live API Sources (Thu Thập Metadata)

### Source 1: Deezer API ⭐ (NGUỒN THAY THẾ CHÍNH)

**URL:** `https://api.deezer.com`
**Auth:** Không cần (public GET), OAuth chỉ cần cho user data
**Rate limit:** ~50 requests / 5 giây
**Độ tin cậy:** ✅ High — đang hoạt động ổn định 2025

**Tại sao Deezer thay Spotify?**
- Deezer API **public và không cần auth** cho metadata → dễ dùng hơn
- Có **BPM** trong track object (Spotify đã kill endpoint này)
- Có **30-second preview MP3 URL** — đây là vàng: dùng Librosa phân tích trực tiếp
- Có chart endpoint theo genre và country → lấy trending VN được
- Catalog 90M+ bài, cover VN tốt

**Endpoint cần dùng:**

```
a) Track metadata + BPM
   GET https://api.deezer.com/track/{id}
   → title, artist, album, duration, bpm, rank (Deezer popularity), genre_id
   → Quan trọng: có preview (URL 30s MP3 để extract audio features)

b) Search track
   GET https://api.deezer.com/search?q={query}
   → Tìm theo tên bài/nghệ sĩ, trả về track list

c) Lookup bằng ISRC (chuẩn quốc tế)
   GET https://api.deezer.com/2.0/track/isrc:{ISRC}
   → Cross-platform matching: dùng ISRC từ MusicBrainz → tìm track trên Deezer

d) Chart theo country
   GET https://api.deezer.com/chart/0/tracks        ← Global top 10
   GET https://api.deezer.com/chart/{genre_id}/tracks ← Chart theo thể loại
   → Thu thập trending data hàng tuần → ground truth cho viral prediction

e) Genre info
   GET https://api.deezer.com/genre
   GET https://api.deezer.com/genre/{id}/artists
   → Danh sách genres và artists nổi bật

f) Artist info
   GET https://api.deezer.com/artist/{id}
   → nb_fan (followers), nb_album, tracklist
```

**Data mẫu từ Deezer Track API:**
```json
{
  "id": 3135556,
  "title": "Harder, Better, Faster, Stronger",
  "duration": 224,
  "rank": 868893,
  "bpm": 123.0,
  "gain": -12.4,
  "preview": "https://cdns-preview-d.dzcdn.net/...30s.mp3",
  "artist": { "name": "Daft Punk", "nb_fan": 5200000 },
  "genre_id": 113,
  "release_date": "2001-10-22"
}
```

> **Điểm mạnh:** `preview` URL → download 30s MP3 → chạy Librosa → lấy ra đầy đủ: MFCC, tempo, spectral features, chroma. Đây là cách tự build audio features mà không cần endpoint đặc biệt.

---

### Source 2: Spotify Web API (Chỉ Metadata)

**URL:** `https://api.spotify.com/v1`
**Auth:** OAuth 2.0 Client Credentials (free)
**Độ tin cậy:** ✅ Metadata OK — ⛔ Audio features DEAD

**Dùng Spotify cho:**

```
a) Track search + basic info
   GET /tracks/{id}
   → name, artist, album, release_date, popularity (0-100), duration_ms

b) Search
   GET /search?q={query}&type=track
   → Tìm Spotify ID để cross-reference với Deezer qua ISRC

c) Artist popularity + followers
   GET /artists/{id}
   → popularity score, followers.total → feature cho viral prediction

d) Playlist metadata (không lấy được audio features)
   GET /playlists/{id}
   → Lấy danh sách track IDs, dùng để build training dataset
```

> **Lưu ý:** Dùng Spotify popularity (0-100) như 1 feature cho viral model. Đây là metric reliable vì Spotify tính từ streams thực tế.

---

### Source 3: Last.fm API

**URL:** `https://ws.audioscrobbler.com/2.0/`
**Auth:** API Key (free, không rate limit nặng)
**Độ tin cậy:** ✅ High — stable API 20+ năm

**Endpoint:**

```
a) Track info + community tags (genre ground truth)
   track.getInfo?track={name}&artist={artist}&api_key={key}&format=json
   → playcount (total plays toàn thời gian), listeners (unique listeners)
   → toptags: community-labeled genres → GROUND TRUTH cho genre classifier

b) Similar tracks (validate recommendation engine)
   track.getSimilar?track={name}&artist={artist}&limit=20
   → Last.fm gợi ý gì → so sánh với AI mình

c) Weekly chart
   chart.getTopTracks?limit=200&format=json
   → Top tracks theo tuần → historical viral data

d) Artist tags
   artist.getTopTags?artist={name}
   → Genre của artist → enrich track metadata
```

**Tại sao Last.fm vẫn là nguồn quan trọng?**
- **playcount = ground truth tốt nhất** cho viral prediction: bài có playcount cao = đã viral
- Community tags từ hàng triệu người = reliable genre labels cho training
- 20+ năm historical data → build time-series trend analysis
- VN artists có data: Sơn Tùng M-TP (130M+ plays), Hoàng Thùy Linh,...

---

### Source 4: MusicBrainz API

**URL:** `https://musicbrainz.org/ws/2/`
**Auth:** Không cần, rate limit 1 req/sec (cần throttle)
**Độ tin cậy:** ✅ High — nonprofit, stable, open data

**Endpoint:**

```
a) Track lookup bằng ISRC
   /recording?query=isrc:{ISRC}&fmt=json
   → Tìm bài bằng ISRC code → cross-platform ID mapping

b) Recording info
   /recording/{mbid}?inc=genres+tags+releases&fmt=json
   → Official genre labels (chuẩn hóa, không phải community noise)
   → release country, release date (chính xác hơn Spotify)

c) Release / Album
   /release/{mbid}?inc=genres&fmt=json
   → Genre ở level album → enrich track
```

**Tại sao MusicBrainz?**
- ISRC là chuẩn ISO quốc tế → dùng để **link track giữa các platform** (Deezer ↔ Spotify ↔ Last.fm)
- Genre labels chuẩn hóa nghiêm túc → reliable hơn Last.fm community tags
- Open database, không bao giờ bị deprecated (nonprofit)

---

## 🟢 LỚP 2: Static Training Datasets (Dùng Cho Train AI)

Đây là nguồn quan trọng nhất cho AI — có audio thực + labels sẵn, không phụ thuộc API.

### Dataset 1: FMA (Free Music Archive) ⭐⭐⭐

**Source:** `https://github.com/mdeff/fma` | HuggingFace: `benjamin-paine/free-music-archive-full`
**License:** Creative Commons → Dùng cho research thoải mái
**Kích thước:**

```
fma_small:   8,000 tracks × 30s × 8 genres      ← Dùng để dev/prototype nhanh
             Size: ~7.2 GB MP3

fma_medium:  25,000 tracks × 30s × 16 genres     ← ⭐ KHUYẾN NGHỊ DÙNG
             Size: ~22 GB MP3

fma_large:   106,574 tracks × 30s × 161 genres   ← Nếu muốn go big
             Size: ~93 GB MP3
```

**Load bằng HuggingFace:**
```python
from datasets import load_dataset

# Load trực tiếp, không cần download thủ công
dataset = load_dataset("minhqng/fma-small")       # Pre-computed mel-spectrogram
# hoặc
dataset = load_dataset("rpmon/fma-genre-classification")  # Clean genre labels
```

**Tại sao FMA là best choice cho genre classification:**
- Audio files thực + genre labels sẵn → train genre classifier ngay
- Creative Commons → không vi phạm bản quyền khi dùng research
- Widely cited trong academic papers → thesis có thể so sánh với SOTA
- Pre-computed Mel spectrograms có sẵn trên HuggingFace → tiết kiệm GPU time

---

### Dataset 2: GTZAN Genre Collection

**Source:** `https://marsyas.info/downloads/data_sets.html` | Kaggle: `andradaolteanu/gtzan-dataset-music-genre-classification`
**Kích thước:** 1,000 tracks × 10 genres × 30s = 1.3 GB
**License:** Research only

```
10 genres: blues, classical, country, disco, hiphop,
           jazz, metal, pop, reggae, rock

→ Đây chính xác là 10 genres bạn đã chọn
→ Dataset kinh điển, mọi paper MIR đều benchmark trên đây
→ Tải về, chạy Librosa extract features, train model → 1 ngày xong
```

**Khi nào dùng GTZAN:**
- Giai đoạn đầu (prototype): nhỏ, nhanh, dễ debug
- Benchmark baseline: SVM trên MFCC features → so sánh với LoRA sau

---

### Dataset 3: Kaggle — Viral/Hit Prediction Datasets ⭐ (Cho Viral Predictor)

Các dataset này có labels về popularity/chart position → dùng train XGBoost viral model:

```
1. "Most Streamed Spotify Songs 2024"
   kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024
   → ~6,000 tracks với: total_streams, spotify_popularity, tiktok_views,
     youtube_views, danceability, energy, tempo, valence,...
   → Label cho viral = tracks với streams > threshold
   → ⭐ Có sẵn TikTok + YouTube metrics → phù hợp viral prediction

2. "Hit Predictor 5000"
   kaggle.com/datasets/thedevastator/hit-predictor-5000
   → Đã có label "hit" / "non-hit" dựa trên Billboard charts
   → Có: peak_chart_position, weeks_on_chart, tiktok_virality_score
   → Dùng trực tiếp để train binary classifier

3. "Spotify Global Streaming Data 2024"
   kaggle.com/datasets/atharva1234/spotify-global-streaming-data-2024
   → Global trend data theo tháng → feature về seasonality
```

---

### Dataset 4: AcousticBrainz Data Dump (Frozen Dataset)

**Status:** ⚠️ Project đã shutdown 2022 — nhưng data dump vẫn available
**URL:** `https://acousticbrainz.org/download`
**Kích thước:** Hàng triệu tracks với low-level audio features

```
Data bao gồm (pre-computed bởi Essentia):
  - BPM, key, mode, scale
  - MFCC (mean + variance)
  - Spectral features: centroid, rolloff, flux
  - Rhythm features: onset rate, beat loudness
  - Tonal features: chroma, tuning, HPCP
  - High-level: mood (happy/sad/aggressive), genre probability

Note: Data tính bởi Essentia từ full audio → chính xác hơn 30s preview
Dùng: join với MusicBrainz MBID → enrich training dataset
```

---

## 🔴 LỚP 3: Self-Extraction Pipeline (Audio → Features)

Đây là cách chính để tạo audio features cho bài nhạc **mới không có trong dataset sẵn**.

### Librosa (Research-grade, Python)

```
Mục đích: Extract features từ audio file/preview URL
Input: 30s MP3 từ Deezer preview URL
Output: Feature vector để inference

# Cách hoạt động:
import librosa

# Download 30s preview
audio_url = "https://cdns-preview-d.dzcdn.net/...30s.mp3"
y, sr = librosa.load(audio_url, sr=22050)

# Extract features
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)      # 13 coefficients
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)           # BPM
chroma = librosa.feature.chroma_stft(y=y, sr=sr)         # 12 chroma bins
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
rms_energy = librosa.feature.rms(y=y)
```

### Essentia (Production-grade, 200+ algorithms)

```
Dùng cho: batch processing lớn, cần thêm features phức tạp
Được Spotify dùng internally (trước khi có deep learning)

Ưu điểm so với Librosa:
  - C++ backend → nhanh hơn 5-10x với large batch
  - Có sẵn TensorFlow model inference
  - Hơn 200 algorithms: mood detection, key estimation, danceability
  - Output format chuẩn hóa → dễ integrate vào pipeline

Note: License AGPLv3 → OK cho research/thesis
```

---

### Tóm Tắt: Data Flow Cho Từng AI Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        GENRE CLASSIFIER                          │
│  Training: FMA medium (25k tracks, 16 genres) + GTZAN (1k, 10g) │
│  Features: Librosa extract từ audio → MFCC, chroma, spectral    │
│  Base model: CLAP (fine-tune với LoRA)                          │
│  Inference: Deezer 30s preview → Librosa → CLAP → genre         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     RECOMMENDATION ENGINE                        │
│  Data: Deezer API (metadata) + Librosa features + Last.fm tags  │
│  Embeddings: CLAP encoder → 512-dim vector → pgvector           │
│  Query: User's track → embedding → cosine similarity search     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      VIRAL PREDICTOR                             │
│  Training: Kaggle "Most Streamed 2024" + Hit Predictor 5000     │
│  Features: Deezer (BPM, rank) + Spotify (popularity) +         │
│            Librosa (energy via RMS, onset_strength) +           │
│            Last.fm (playcount history) + metadata               │
│  Model: XGBoost → SHAP → interpretable predictions              │
│  Label: tracks với Spotify popularity > 80 OR Last.fm plays > 10M│
└─────────────────────────────────────────────────────────────────┘
```

---

### Nguồn Dữ Liệu Cho Scope VN (60%) + Global (40%)

```
VN Data:
  - Deezer chart VN: GET /chart với country=VN
  - Last.fm: artists VN (Sơn Tùng, Mỹ Tâm, HIEUTHUHAI, tlinh,...)
  - Zing MP3 trending (scrape HTML nếu cần — no official API)
  - Spotify VN Top 50 playlist (public, lấy được qua Spotify API)

Global Data:
  - Deezer Global chart
  - Last.fm Weekly Top 200
  - Kaggle datasets (US/UK centric)
  - FMA dataset (international, Creative Commons)

Note: VN data ít hơn global nhưng đủ để bias model về VN market.
      Target: 30k VN tracks + 20k global tracks cho viral predictor.
```

---




## 4. Chi Tiết Từng Module

### Module 1: `data_ingest/` — Thu Thập Metadata Nhạc

**Nhiệm vụ:** Crawl metadata từ Deezer + Spotify + Last.fm + MusicBrainz, merge và lưu.

---

#### 📐 Định Dạng Data — JSON qua các tầng xử lý

**Tầng 1: Raw API Response (lưu thẳng vào MongoDB, không transform)**

```json
// Collection: raw_track_data — document từ Deezer
{
  "_id": "ObjectId(...)",
  "source": "deezer",
  "collected_at": "2026-08-15T00:00:00Z",
  "status": "success",
  "raw_payload": {
    "id": 3135556,
    "title": "Harder, Better, Faster, Stronger",
    "duration": 224,
    "rank": 868893,
    "bpm": 123.0,
    "gain": -12.4,
    "preview": "https://cdns-preview-d.dzcdn.net/...30s.mp3",
    "release_date": "2001-10-22",
    "artist": { "id": 27, "name": "Daft Punk", "nb_fan": 5200000 },
    "album": { "id": 302127, "title": "Discovery", "cover_xl": "..." },
    "genre_id": 113
  }
}

// Collection: raw_track_data — document từ Last.fm (cùng bài)
{
  "_id": "ObjectId(...)",
  "source": "lastfm",
  "track_key": "daft-punk::harder-better-faster-stronger",
  "collected_at": "2026-08-15T00:00:00Z",
  "status": "success",
  "raw_payload": {
    "name": "Harder, Better, Faster, Stronger",
    "artist": "Daft Punk",
    "playcount": "285000000",
    "listeners": "4200000",
    "toptags": {
      "tag": [
        { "name": "electronic", "count": 100 },
        { "name": "dance",      "count": 87  },
        { "name": "house",      "count": 65  }
      ]
    }
  }
}
```

**Tầng 2: Normalized Track (sau ETL → lưu vào PostgreSQL)**

```json
// Table: tracks — 1 row = 1 bài nhạc đã merge từ các source
{
  "id": "uuid-xxxx-xxxx",
  "deezer_id": 3135556,
  "spotify_id": "0DiWol3AO6WpXZgp0goxAV",
  "isrc": "GBDUW0000059",
  "title": "Harder, Better, Faster, Stronger",
  "artist_name": "Daft Punk",
  "artist_followers": 5200000,
  "album_title": "Discovery",
  "release_year": 2001,
  "duration_sec": 224.0,
  "deezer_bpm": 123.0,
  "deezer_rank": 868893,
  "deezer_preview_url": "https://cdns-preview-d.dzcdn.net/...30s.mp3",
  "spotify_popularity": 75,
  "lastfm_playcount": 285000000,
  "lastfm_listeners": 4200000,
  "genres": ["electronic", "dance", "house"],
  "primary_genre": "electronic",
  "created_at": "2026-08-15T00:00:00Z",
  "audio_processed": false
}
```

**Tầng 3: AudioFeatureSet (sau Librosa extraction → PostgreSQL + MongoDB raw)**

```json
// Table: audio_features — kết quả Librosa phân tích 30s preview
{
  "id": "uuid-yyyy-yyyy",
  "track_id": "uuid-xxxx-xxxx",
  "source_preview_url": "https://cdns-preview-d.dzcdn.net/...30s.mp3",

  "tempo": 123.04,
  "rms_energy": 0.182,
  "zero_crossing_rate": 0.091,

  "spectral_centroid": 3241.7,
  "spectral_bandwidth": 2104.3,
  "spectral_rolloff": 6890.2,

  "mfcc_1":  -312.4,   "mfcc_2":  98.1,  "mfcc_3": -21.3,
  "mfcc_4":   18.7,    "mfcc_5":  -8.4,  "mfcc_6":  12.1,
  "mfcc_7":   -5.2,    "mfcc_8":   3.8,  "mfcc_9":  -4.1,
  "mfcc_10":   2.9,    "mfcc_11": -1.7,  "mfcc_12":  0.9,  "mfcc_13": -0.4,

  "chroma_c": 0.42, "chroma_cs": 0.18, "chroma_d": 0.31,
  "chroma_ds": 0.09, "chroma_e": 0.27, "chroma_f": 0.21,
  "chroma_fs": 0.14, "chroma_g": 0.38, "chroma_gs": 0.11,
  "chroma_a": 0.29,  "chroma_as": 0.17, "chroma_b": 0.22,

  "onset_strength": 0.74,
  "onset_strength_intro": 0.81,

  "extracted_at": "2026-08-15T00:05:00Z"
}

// Collection MongoDB: audio_raw_features
// Lưu mel_spectrogram (matrix 128×T) — quá lớn cho PostgreSQL
{
  "track_id": "uuid-xxxx-xxxx",
  "mel_spectrogram": [[...128 bins...], [...], ...],  // shape: (128, T)
  "beat_frames": [22, 44, 66, 88, ...],
  "extracted_at": "2026-08-15T00:05:00Z"
}
```

**Tầng 4: Training Record (input cho AI model)**

```json
// File CSV/Parquet — merge track + audio_features + label
{
  "track_id": "uuid-xxxx-xxxx",
  "title": "Harder, Better, Faster, Stronger",

  // Audio features (normalized 0-1 cho model)
  "tempo_norm": 0.68,
  "rms_energy_norm": 0.45,
  "mfcc_1_norm": 0.31,
  // ... 13 mfcc + spectral + chroma

  // Label cho genre classifier
  "genre_label": "electronic",
  "genre_label_id": 4,

  // Label cho viral predictor
  "lastfm_playcount": 285000000,
  "spotify_popularity": 75,
  "is_viral": 1,              // 1 nếu playcount > 10M hoặc popularity > 80
  "viral_platform": "both"   // "tiktok", "youtube", "both", "none"
}
```

---

#### Architecture: Source Registry

```
MusicSourceDispatcher
  ├── "deezer"        → DeezerSource       ← PRIMARY: BPM + preview URL
  ├── "spotify"       → SpotifySource      ← popularity score
  ├── "lastfm"        → LastFmSource       ← playcount + genre tags
  └── "musicbrainz"   → MusicBrainzSource  ← ISRC + genre chuẩn
```

Mỗi source implement Protocol:
```python
def fetch(track_query: TrackQuery) -> RawTrackData
def fetch_batch(queries: list[TrackQuery]) -> list[RawTrackData]
```

---

### Module 2: `audio_processing/` — Trích Xuất Audio Features

**Nhiệm vụ:** Load audio file → trích xuất các đặc trưng âm thanh → lưu vào database.

#### Audio Features sẽ trích xuất (với Librosa)

```
Temporal Features:
  - tempo:                BPM (nhịp đập)
  - beat_strength:        Độ mạnh của beat
  - zero_crossing_rate:   Tần suất âm thanh đổi dấu → phân biệt vocal vs instrumental

Spectral Features:
  - spectral_centroid:    "Trọng tâm" tần số → liên quan đến brightness của âm thanh
  - spectral_bandwidth:   Độ rộng dải tần
  - spectral_rolloff:     Tần số phân cách 85% năng lượng → liên quan đến timbre
  - mfcc[0..12]:          Mel-Frequency Cepstral Coefficients (13 coefficients)
                          → Fingerprint quan trọng nhất của âm thanh

Energy Features:
  - rms_energy:           Root Mean Square energy → loudness
  - chroma_stft[0..11]:  Chroma features (12 bins, map C,C#,D,...,B)
                          → Biểu diễn hợp âm/key của bài nhạc

Rhythm Features:
  - onset_strength:       Mức độ "bùng nổ" âm thanh → liên quan đến energy cảm nhận
  - tempogram:            Biến thiên tempo theo thời gian

Mel Spectrogram:
  - mel_spectrogram:      128 x T matrix → input cho CNN model
```

#### Handler Chain (DDD Pipeline)

```
LoadHandler
  → Nhận path/URL audio file
  → Load với Librosa: y, sr = librosa.load(path, sr=22050)
  → Validate: duration đủ dài không? (tối thiểu 30s)

    ↓

ExtractHandler
  → Tính tất cả features với Librosa
  → Tạo AudioFeatureSet object
  → Validate: không có NaN, Inf

    ↓

NormalizeHandler
  → Min-max normalization cho mỗi feature
  → Tạo AudioFeatureVector (chuẩn hóa 0-1)
  → Feature này sẽ dùng làm input cho AI model

    ↓

SaveHandler
  → Lưu AudioFeatureSet vào MongoDB (raw)
  → Lưu AudioFeatureVector vào PostgreSQL (cho AI)
  → Update track record: audio_processed = True
```

#### AudioFeatureSet Model

```
class AudioFeatureSet:
    track_id:           str
    duration_sec:       float
    sample_rate:        int

    # Temporal
    tempo:              float       # BPM
    beat_frames:        list[int]   # Frame indices của beats

    # Spectral
    mfcc:               list[float] # 13 coefficients (mean của toàn bài)
    spectral_centroid:  float
    spectral_bandwidth: float
    spectral_rolloff:   float
    zero_crossing_rate: float

    # Energy
    rms_energy:         float
    chroma_stft:        list[float] # 12 values

    # Rhythm
    onset_strength:     float

    # Derived
    mel_spectrogram:    np.ndarray  # 128 × T (lưu riêng)

    extracted_at:       datetime
```

---

### Module 3: `ai_engine/` — AI Layer (Kỳ 2, điểm nhấn kỹ thuật)

#### 3.1 Multi-task Classification với LoRA Fine-tuning ⭐⭐⭐

**Bài toán:** Phân loại bài hát theo **2 chiều độc lập**:
- **Genre** (âm nhạc): blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **Region/Market** (thị trường): V-pop, K-pop, J-pop, C-pop, US-UK

> **Tại sao 2 chiều riêng?** Vì chúng không loại trừ nhau:
> BTS = K-pop + hip-hop | Sơn Tùng = V-pop + EDM/pop | Mỹ Tâm = V-pop + ballad
> Một bài có đồng thời 1 genre và 1 region → cần 2 label riêng.

---

##### 🧠 LoRA Là Gì? Tại Sao Dùng Được Cho Bài Toán Này?

LoRA (Low-Rank Adaptation) là kỹ thuật **fine-tuning** — nghĩa là bạn bắt buộc phải có **model đã được pre-train sẵn** trước, rồi mới dùng LoRA để tối ưu nó cho task cụ thể của mình.

```
  PRE-TRAINED MODEL            LoRA ADAPTERS           OUTPUT
  ─────────────────            ─────────────           ──────

  CLAP (frozen)           +    ΔW = A × B         =   Genre (10) + Region (5)
  ┌─────────────┐              ┌────┐ ┌────┐
  │ W (gốc)     │    →         │ A  │×│ B  │   →    Head 1: Genre
  │ 125M params │              │r×d │ │d×r │         (10 classes)
  │ KHÔNG train │              └────┘ └────┘   →    Head 2: Region
  └─────────────┘              ~500k params train        ( 5 classes)

  W_effective = W + α/r × (A × B)
```

**Tại sao không train từ đầu?**
- Train CLAP từ đầu cần 630k audio-text pairs + hàng tuần GPU → không khả thi
- CLAP đã "biết" âm thanh nghe thế nào rồi — LoRA chỉ dạy nó gán đúng label 10 genres của mình
- Chỉ ~500k parameters được train thay vì 125M → Google Colab free GPU là đủ

---

##### 📋 Hướng Đi Đúng: Pre-trained → LoRA Fine-tune → Genre Classifier

```
BƯỚC 0: Chọn Pre-trained Base Model
────────────────────────────────────
CLAP (Contrastive Language-Audio Pretraining)
  → Nguồn: HuggingFace "laion/clap-htsat-unfused"
  → LAION đã train trên 630,000 audio-text pairs
  → Model đã biết phân biệt: guitar vs piano, slow vs fast, quiet vs loud
  → Architecture: HTSAT (Hierarchical Token-Semantic Audio Transformer)
  → Output: 512-dim embedding vector cho mỗi đoạn audio

  Tại sao CLAP mà không phải model khác?
    - Wav2Vec2: Chỉ tốt cho speech, không phải music
    - MusicGen: Generative model, không phù hợp cho classification
    - CLAP: Được thiết kế cho music understanding → phù hợp nhất

BƯỚC 1: Chuẩn Bị Data
────────────────────────────────────
[TASK 1 — GENRE] Dữ liệu cho phân loại 10 thể loại âm nhạc:
  Dataset: FMA medium (25,000 tracks × 16 genres) + GTZAN (1,000 × 10 genres)
  Label nguồn: FMA metadata (đã có sẵn) + GTZAN labels (đã có sẵn)

  Mapping genre về 10 classes của mình:
    FMA "Instrumental" → "classical" | FMA "Electronic" → "disco" hoặc "metal"
    FMA "Hip-Hop" → "hiphop" | FMA "Folk" → "country"
    (Cần viết label consolidation script)

[TASK 2 — REGION] Dữ liệu cho phân loại 5 thị trường âm nhạc:
  ┌──────────┬──────────────────┬────────────────────────────────────────────┐
  │ Region   │ Target tracks    │ Nguồn data                                  │
  ├──────────┼──────────────────┼────────────────────────────────────────────┤
  │ V-pop    │ ~6,000 tracks    │ Spotify "V-Pop" playlist, Zing MP3 chart,  │
  │ (Nhạc VN)│                  │ Last.fm VN artists (Sơn Tùng, Mỹ Tâm,...) │
  ├──────────┼──────────────────┼────────────────────────────────────────────┤
  │ K-pop    │ ~8,000 tracks    │ Spotify K-pop playlist, Melon Chart API,   │
  │          │                  │ Last.fm tag "korean pop"                    │
  ├──────────┼──────────────────┼────────────────────────────────────────────┤
  │ J-pop    │ ~6,000 tracks    │ Spotify J-pop/Anime playlist,              │
  │          │                  │ Last.fm tag "j-pop", "anime"               │
  ├──────────┼──────────────────┼────────────────────────────────────────────┤
  │ C-pop    │ ~5,000 tracks    │ Spotify Chinese playlist,                  │
  │ (Mandopop│                  │ Last.fm tag "mandopop", "cantopop"         │
  │  + Canto)│                  │                                            │
  ├──────────┼──────────────────┼────────────────────────────────────────────┤
  │ US-UK    │ ~10,000 tracks   │ Billboard Hot 100 historical, Kaggle       │
  │ (Phương  │                  │ "Most Streamed 2024", Last.fm global chart │
  │  Tây)    │                  │                                            │
  └──────────┴──────────────────┴────────────────────────────────────────────┘

  Cách label region: Dùng artist nationality + language của bài hát
    → MusicBrainz: artist.area = "Vietnam" → V-pop
    → Last.fm tag "k-pop" threshold > 50 → K-pop
    → Spotify playlist membership → reliable label

  Lưu ý imbalance: US-UK > K-pop > V-pop/J-pop ≈ C-pop
  → Dùng weighted loss + oversampling cho V-pop

Pipeline preprocessing (chung cho cả 2 tasks):
  Audio (30s MP3) → Librosa resample 48kHz → Mel spectrogram → CLAP tokenizer

Split (stratified theo cả genre lẫn region):
  train: 70%
  val:   15%  ← Dùng để chọn best checkpoint
  test:  15%  ← Chỉ dùng 1 lần để report kết quả

Data augmentation (tránh overfitting, đặc biệt V-pop ít data):
  - Time stretch: ±10% tempo
  - Pitch shift: ±2 semitones
  - Add noise: SNR = 20dB
  - Random crop: lấy 30s ngẫu nhiên từ bài dài hơn
  - Mixup augmentation (trộn 2 bài cùng region)

BƯỚC 2: Gắn LoRA Adapters
────────────────────────────────────
Freeze toàn bộ CLAP weights.
Thêm LoRA adapter vào các attention layers:

  from peft import LoraConfig, get_peft_model

  lora_config = LoraConfig(
      r=8,                    # Rank — số chiều của low-rank decomposition
      lora_alpha=16,          # Scaling factor (thường = 2×r)
      target_modules=[        # Layer nào được thêm adapter
          "q_proj",           # Query projection
          "v_proj",           # Value projection
      ],
      lora_dropout=0.1,
      bias="none",
      task_type="FEATURE_EXTRACTION"
  )

  model = get_peft_model(clap_model, lora_config)
  # → Chỉ 0.8% parameters được train

Thêm 2 classification heads (Multi-task):

  # Head 1: Genre classifier
  genre_head = Sequential(
      Linear(512, 256), ReLU(), Dropout(0.3),
      Linear(256, 10),  Softmax()
  )  # Output: 10 genre probabilities

  # Head 2: Region classifier
  region_head = Sequential(
      Linear(512, 128), ReLU(), Dropout(0.3),
      Linear(128, 5),   Softmax()
  )  # Output: 5 region probabilities (V-pop/K-pop/J-pop/C-pop/US-UK)

  # → Cả 2 head dùng chung 1 CLAP encoder + LoRA backbone
  # → Shared representation học được feature chung cho cả 2 task

BƯỚC 3: Training
────────────────────────────────────
  optimizer: AdamW(lr=2e-4, weight_decay=0.01)
  scheduler: CosineAnnealingLR(T_max=10)
  epochs: 10
  batch_size: 32

  # Multi-task loss: cộng 2 loss lại
  loss_genre  = CrossEntropyLoss(weight=genre_weights)    # weighted vì genre imbalanced
  loss_region = CrossEntropyLoss(weight=region_weights)   # weighted vì V-pop ít hơn USUK
  loss_total  = 0.6 × loss_genre + 0.4 × loss_region     # Genre được ưu tiên hơn

  Early stopping: patience=3 trên val loss tổng

BƯỚC 4: Evaluation & Thesis Experiments
────────────────────────────────────
Baseline 1: SVM trên MFCC features (classical ML, chỉ genre)
Baseline 2: Zero-shot CLAP (không fine-tune gì cả)
Baseline 3: Single-task LoRA chỉ genre (không có region head)
Our model:  Multi-task LoRA (genre + region cùng lúc, rank=8)

Experiments cho thesis:
  Experiment A: LoRA rank 4 vs 8 vs 16 → accuracy vs training time
  Experiment B: Single-task vs Multi-task → chứng minh multi-task tốt hơn
  Experiment C: Loss weight 0.5/0.5 vs 0.6/0.4 vs 0.7/0.3 → optimal balance
  Experiment D: Data size cho region (V-pop ít data hơn) → few-shot performance

Metrics:
  Genre task:  Accuracy, F1-score macro, Confusion Matrix
  Region task: Accuracy, F1-score macro (chú ý V-pop recall đặc biệt)
  Overall:     Training time + GPU memory (chứng minh LoRA efficient)

Bất ngờ về multi-task learning:
  Multi-task thường tốt hơn single-task vì genre và region có correlation:
  K-pop thường = pop/hiphop genre → shared features giúp cả 2 tasks

BƯỚC 5: Lưu & Deploy Model
────────────────────────────────────
  # Save chỉ LoRA weights (~2MB) thay vì full model (500MB)
  model.save_pretrained("./lora_genre_classifier")

  # Inference
  audio → 30s preview → CLAP encoder → LoRA head → genre: "electronic" (0.87)
```

**Thesis contribution:**
- Bảng so sánh 4 approaches: SVM baseline vs Zero-shot vs Single-task LoRA vs Multi-task LoRA
- Ablation study: loss weight ratio ảnh hưởng thế nào đến cả 2 tasks
- Chứng minh multi-task LoRA tốt hơn single-task nhờ shared audio representation
- Điểm độc đáo: V-pop classification từ audio (không cần text/lyrics)

---

#### 3.2 Music Recommendation Engine ⭐⭐

**Bài toán:** Cho 1 bài nhạc → tìm top-10 bài giống nhất.

**Approach: Audio Embedding Similarity**

```
Bước 1: Mỗi bài nhạc → 1 embedding vector (512-dim)
  Dùng CLAP encoder (đã fine-tuned ở trên)
  embedding = clap_model.encode_audio(audio_path)   # shape: (512,)

Bước 2: Lưu tất cả embeddings vào vector database
  Dùng pgvector (PostgreSQL extension)
  → Không cần database riêng, tích hợp vào PostgreSQL hiện có

Bước 3: Query similarity
  Cosine similarity search trong N triệu bài
  similar_tracks = pgvector.search(
      query_embedding=embedding,
      top_k=10,
      metric="cosine"
  )
```

**Hybrid approach (điểm cộng):**

```
Audio Similarity (70%) + Metadata Similarity (30%)

Metadata similarity dựa trên:
  - Same genre: +0.3
  - Same decade: +0.1
  - Same tempo range (±10 BPM): +0.2
  - Same key: +0.2
  - Similar energy (±0.15): +0.2
  - Similar danceability (±0.15): +0.1
```

---

#### 3.3 Viral Potential Predictor ⭐⭐⭐ (Tính năng độc đáo nhất)

**Bài toán:** Dự đoán khả năng bài hát sẽ viral trong 3-6 tháng tới, và viral trên platform nào.

**Tại sao làm được?**

```
Nghiên cứu cho thấy các bài viral TikTok thường có đặc điểm:
  - Tempo: 110-140 BPM (dễ choreography)
  - Duration: <3 phút (TikTok friendly)
  - Energy cao (0.7+)
  - Danceability cao (0.7+)
  - Hook mạnh trong 10-15 giây đầu (onset_strength cao ở đầu bài)
  - Valence cao (bài vui vẻ, tích cực)

YouTube viral thường:
  - Có buildup + drop rõ ràng (spectral contrast thay đổi mạnh)
  - Duration 3-5 phút (đủ watch time)
  - High production quality (spectral richness)
  - Không nhất thiết phải dễ nhảy
```

**Features cho prediction:**

```
# Từ audio analysis
viral_features = {
    "tempo": 128.5,
    "energy": 0.85,
    "danceability": 0.78,           ← Spotify feature
    "valence": 0.72,
    "duration_sec": 175,
    "onset_strength_intro": 0.92,   ← Hook strength (15s đầu)
    "spectral_contrast": 0.65,

    # Từ metadata
    "artist_follower_count": 5_000_000,
    "release_month": 6,             ← Mùa hè thường viral hơn
    "has_music_video": True,
    "label_type": "major",          ← Major label vs indie

    # Từ lịch sử bài cũ của nghệ sĩ
    "artist_avg_viral_score": 0.72,
    "previous_hit_count": 3
}
```

**Model:**

```
XGBoost hoặc LightGBM (tabular data, không cần deep learning)
  → Nhanh train, dễ explain (feature importance)
  → SHAP values → giải thích tại sao bài này có tiềm năng viral

Output:
  tiktok_viral_prob:      0.78       ← % khả năng viral TikTok
  youtube_viral_prob:     0.45       ← % khả năng viral YouTube
  mainstream_prob:        0.62       ← % lên mainstream chart
  predicted_peak_streams: 5_200_000  ← Dự đoán peak streams/ngày
  viral_window:           "2-4 tháng sau release"
  key_strengths:          ["High danceability", "Perfect TikTok tempo", "Catchy hook"]
  improvement_suggestions: ["Consider shorter intro", "Add bass drop at 1:30"]
```

**Ground truth data cho training:**

```
Lấy từ Last.fm:
  - Tracks released 2019-2023 với playcount > 10M → label: "viral"
  - Tracks released cùng giai đoạn với playcount < 500k → label: "not viral"

Lấy từ Spotify:
  - Popularity score > 80 → viral indicator
  - Popularity < 40 → non-viral

→ Tạo dataset ~50,000 tracks với binary labels
→ Train XGBoost, evaluate với F1 + AUC-ROC
```

---

## 5. Database Design

### PostgreSQL (Structured Data)

```sql
-- Bảng chính: tracks
CREATE TABLE tracks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spotify_id      VARCHAR(50) UNIQUE,
    lastfm_id       VARCHAR(100),
    title           VARCHAR(500) NOT NULL,
    artist_name     VARCHAR(300) NOT NULL,
    release_year    SMALLINT,
    duration_sec    FLOAT,
    primary_genre   VARCHAR(100),
    genres          TEXT[],             -- Array of genres
    popularity      SMALLINT,           -- Spotify 0-100
    playcount       BIGINT,             -- Last.fm
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Bảng audio features (chuẩn hóa)
CREATE TABLE audio_features (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    track_id        UUID REFERENCES tracks(id),
    tempo           FLOAT,
    energy          FLOAT,
    danceability    FLOAT,
    valence         FLOAT,
    acousticness    FLOAT,
    instrumentalness FLOAT,
    speechiness     FLOAT,
    liveness        FLOAT,
    loudness        FLOAT,
    mfcc_1 FLOAT, mfcc_2 FLOAT, mfcc_3 FLOAT,
    mfcc_4 FLOAT, mfcc_5 FLOAT, mfcc_6 FLOAT,
    mfcc_7 FLOAT, mfcc_8 FLOAT, mfcc_9 FLOAT,
    mfcc_10 FLOAT, mfcc_11 FLOAT, mfcc_12 FLOAT, mfcc_13 FLOAT,
    spectral_centroid FLOAT,
    spectral_bandwidth FLOAT,
    zero_crossing_rate FLOAT,
    rms_energy      FLOAT,
    extracted_at    TIMESTAMPTZ
);

-- Bảng AI predictions
CREATE TABLE track_predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    track_id        UUID REFERENCES tracks(id),
    predicted_genre VARCHAR(100),
    genre_confidence FLOAT,
    tiktok_viral_prob FLOAT,
    youtube_viral_prob FLOAT,
    mainstream_prob FLOAT,
    model_version   VARCHAR(50),
    predicted_at    TIMESTAMPTZ
);

-- Bảng embeddings (với pgvector)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE track_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    track_id        UUID REFERENCES tracks(id),
    embedding       vector(512),        -- CLAP embedding
    model_version   VARCHAR(50)
);
CREATE INDEX ON track_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Bảng trending (weekly snapshot)
CREATE TABLE trending_weekly (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    week_date       DATE,
    genre           VARCHAR(100),
    track_id        UUID REFERENCES tracks(id),
    rank            SMALLINT,
    playcount_delta BIGINT,             -- Tăng bao nhiêu plays trong tuần
    viral_score     FLOAT
);
```

### MongoDB (Raw Data)

```
Collection: raw_track_data
  → Full API responses từ Spotify, Last.fm, MusicBrainz
  → Không normalize, giữ nguyên để debug/retrain

Collection: audio_raw_features
  → Full feature arrays (mel spectrogram, beat_frames,...) trước khi aggregate
  → Schema flexible vì mỗi bài có shape khác nhau

Collection: etl_logs
  → Log từng bước ETL: success/fail, duration, error messages
```

---

## 6. Phân Chia Công Việc Chi Tiết

### Bạn (DE + AI) — Phần Core

| Phần | Công việc cụ thể |
|------|----------------|
| **ETL Infrastructure** | Setup Airflow, Docker, PostgreSQL, MongoDB |
| **Data Ingest** | Viết SpotifySource, LastFmSource, MusicBrainzSource |
| **Audio Processing** | Pipeline với Librosa, AudioFeatureSet, handler chain (DDD) |
| **Database Design** | Schema PostgreSQL + pgvector + MongoDB collections |
| **AI - Genre Classifier** | LoRA fine-tuning trên CLAP model, evaluation |
| **AI - Recommender** | Embedding similarity với pgvector/FAISS |
| **AI - Viral Predictor** | XGBoost model, SHAP explanation |
| **Data API** | Internal Python functions/services mà BE gọi |

### Bạn bè (BE + FE) — Phần Application

| Phần | Công việc cụ thể |
|------|----------------|
| **REST API** | FastAPI/Express: `/analyze`, `/recommend`, `/trending`, `/predict-viral` |
| **Authentication** | JWT, user accounts |
| **File Upload** | Upload audio file → gọi AI pipeline |
| **Web Frontend** | React/Next.js: giao diện upload, kết quả phân tích, dashboard |
| **Music Player** | Waveform visualizer, genre badge, recommendation sidebar |
| **Deployment** | Docker Compose, Cloud hosting (Railway, Render, hoặc VPS) |

### Interface giữa 2 người (Contract)

```python
# Bạn export các functions này, bạn bè gọi:

async def analyze_track(
    audio_file: bytes | None,
    spotify_url: str | None,
) -> AnalysisResult:
    """
    Returns:
        genres: list[tuple[str, float]]   # [("pop", 0.85), ("dance", 0.72)]
        audio_features: AudioFeatureSet
        viral_prediction: ViralPrediction
    """

async def get_recommendations(
    track_id: str,
    top_k: int = 10
) -> list[RecommendedTrack]:
    """Returns similar tracks với similarity score"""

async def get_trending(
    genre: str | None,
    period: str = "week"
) -> list[TrendingTrack]:
    """Returns trending tracks"""
```

---

## 7. Timeline — 2 Kỳ × 3 Tháng

### 🗓️ KỲ 1 (Tháng 1-3): ETL + Infrastructure + Data Pipeline

**Mục tiêu kỳ 1:** Hệ thống thu thập và lưu trữ data chạy được tự động. Database có ~50,000 bài nhạc với đầy đủ metadata và audio features. Airflow chạy hàng ngày/tuần.

---

**Tháng 1 — Foundation**

```
Tuần 1-2: Setup & Design
  ✓ Thiết kế database schema (PostgreSQL + MongoDB)
  ✓ Setup Docker Compose (PG + Mongo + Airflow)
  ✓ Cấu trúc project theo DDD
  ✓ Shared module: base models, settings, logging

Tuần 3-4: Data Ingest Module
  ✓ SpotifySource: fetch track metadata + audio features
  ✓ LastFmSource: fetch tags, playcount
  ✓ MusicBrainzSource: fetch genre labels
  ✓ Lưu raw data vào MongoDB
  ✓ Unit tests cho mỗi source

Deliverable T1: Crawl được 5,000 tracks với metadata đầy đủ
```

**Tháng 2 — Audio Processing**

```
Tuần 1-2: Audio Processing Pipeline
  ✓ Setup môi trường Librosa
  ✓ LoadHandler: download + validate audio
  ✓ ExtractHandler: extract 30+ features
  ✓ NormalizeHandler: chuẩn hóa features
  ✓ SaveHandler: lưu vào database

Tuần 3-4: Airflow DAGs + Integration
  ✓ dag_ingest.py: crawl metadata hàng ngày
  ✓ dag_audio_etl.py: xử lý audio mỗi đêm
  ✓ Error handling, retry logic
  ✓ Monitoring dashboard Airflow
  ✓ Integration test end-to-end

Deliverable T2: Pipeline tự động chạy hàng ngày.
               20,000 tracks với audio features đầy đủ.
```

**Tháng 3 — Data Quality + Handoff to AI**

```
Tuần 1-2: Data Quality & EDA
  ✓ Exploratory Data Analysis (visualize features, genre distribution)
  ✓ Data quality checks: missing values, outliers
  ✓ Genre label consolidation (merge duplicates: "hip hop" = "hip-hop" = "rap")
  ✓ Build training dataset file (train.csv, val.csv, test.csv)

Tuần 3-4: API Interface + Docs
  ✓ Viết Python interface functions (cho BE gọi)
  ✓ API Documentation (OpenAPI spec)
  ✓ README, setup guide
  ✓ Demo: chạy pipeline từ đầu → kết quả

Deliverable T3 (Cuối kỳ 1):
  - 50,000+ tracks trong database
  - Audio features cho 30,000+ bài
  - Training data sẵn sàng cho AI
  - Pipeline tự động chạy ổn định
  - Báo cáo kỳ 1: kiến trúc, thiết kế, kết quả
```

---

### 🗓️ KỲ 2 (Tháng 4-6): AI Models + Deployment

**Mục tiêu kỳ 2:** 3 AI models chạy được, tích hợp vào API, deploy lên cloud. App dùng được thật sự.

---

**Tháng 4 — Genre Classifier (LoRA)**

```
Tuần 1-2: Setup & Baseline
  ✓ Cài đặt môi trường: PyTorch, HuggingFace, PEFT (LoRA library)
  ✓ Load CLAP pre-trained model
  ✓ Baseline experiment: zero-shot classification (không train)
  ✓ Baseline 2: SVM trên MFCC features (classic approach)
  ✓ Ghi kết quả baseline vào bảng so sánh

Tuần 3-4: LoRA Fine-tuning
  ✓ Cấu hình LoRA (rank 4, 8, 16 → experiment)
  ✓ Train trên FMA dataset (50,000 tracks, 10 genres)
  ✓ Evaluation: Accuracy, F1 macro, Confusion Matrix
  ✓ Hyperparameter tuning
  ✓ Export model artifact

Deliverable T4: Genre classifier accuracy > 80% trên test set
               Bảng so sánh: SVM baseline vs LoRA fine-tuned
```

**Tháng 5 — Recommender + Viral Predictor**

```
Tuần 1-2: Recommendation Engine
  ✓ Generate embeddings cho 50k tracks (CLAP encoder)
  ✓ Setup pgvector trong PostgreSQL
  ✓ Implement similarity search
  ✓ Hybrid scoring (audio + metadata)
  ✓ Evaluation: compare với Spotify recommendations (NDCG metric)

Tuần 3-4: Viral Predictor
  ✓ Feature engineering (30+ features)
  ✓ Build training labels từ historical data
  ✓ Train XGBoost model
  ✓ SHAP analysis: feature importance visualization
  ✓ Evaluation: AUC-ROC, Precision@K
  ✓ Tích hợp cả 3 models vào Python interface

Deliverable T5: 3 models chạy được. Interface Python hoàn chỉnh.
               Viral predictor AUC > 0.75
```

**Tháng 6 — Integration + Deployment + Thesis**

```
Tuần 1-2: Integration với BE/FE
  ✓ Tích hợp AI pipeline vào FastAPI (BE làm API, bạn support)
  ✓ End-to-end test: upload bài nhạc → kết quả phân tích
  ✓ Performance optimization: batch inference, caching với Redis
  ✓ Bug fixing, edge cases

Tuần 3-4: Deployment + Final Polish
  ✓ Docker image cho AI service
  ✓ Deploy lên cloud (Railway / Render / VPS)
  ✓ Load testing
  ✓ Demo video record
  ✓ Hoàn thiện báo cáo + thesis

Deliverable cuối (Kỳ 2):
  - App deploy được, truy cập qua URL
  - 3 AI features hoạt động: Genre, Recommend, Viral
  - Báo cáo thesis: 2 chương về DE (ETL) + AI (LoRA fine-tuning)
  - Demo video 5 phút
```

---

## 8. Tech Stack Summary

### DE + AI (Bạn)

```yaml
Language:       Python 3.11+

Data Collection:
  - spotipy:          Spotify API client
  - pylast:           Last.fm API client
  - musicbrainzngs:   MusicBrainz API client
  - httpx:            Async HTTP (cho các API thủ công)

Audio Processing:
  - librosa:          Audio analysis (industry standard)
  - soundfile:        Audio I/O
  - yt-dlp:           Download audio (nếu cần)

AI / ML:
  - torch:            PyTorch
  - transformers:     HuggingFace (CLAP model)
  - peft:             LoRA implementation (HuggingFace PEFT)
  - xgboost:          Viral predictor
  - shap:             Explainability
  - faiss-cpu:        Vector similarity (hoặc pgvector)
  - datasets:         HuggingFace Datasets (FMA, GTZAN)
  - evaluate:         Model evaluation metrics

ETL / Orchestration:
  - apache-airflow:   Workflow orchestration
  - sqlalchemy:       ORM
  - pymongo:          MongoDB client
  - psycopg2:         PostgreSQL client
  - pydantic:         Data validation

Infrastructure:
  - Docker + Docker Compose
  - PostgreSQL 16 + pgvector extension
  - MongoDB 7
  - Redis (caching)
```

### BE + FE (Bạn bè)

```yaml
Backend:
  - FastAPI (Python) hoặc Express.js (Node)
  - JWT authentication
  - REST API + OpenAPI docs

Frontend:
  - React + Next.js
  - Tailwind CSS
  - Wavesurfer.js (audio waveform)
  - Chart.js / Recharts (visualization)

Deployment:
  - Docker
  - Railway / Render / Vercel (frontend)
```

---

## 9. Điểm Mạnh Khi Trình Bày Với GVHD

```
✅ Kiến trúc DDD rõ ràng (kỳ 1) → Senior IT hiểu ngay
✅ LoRA fine-tuning = kỹ thuật AI tiên tiến, có thể viết paper
✅ Multi-source ETL → không chỉ là CRUD app
✅ 3 AI features khác nhau → đủ complexity cho 2 người
✅ pgvector → modern database stack, không dùng solution cũ
✅ SHAP explainability → AI không black-box
✅ Deploy được → demo live cho GVHD xem ngay
✅ Dataset có sẵn (FMA, GTZAN) → không bị block bởi data
✅ Có ground truth để evaluate → báo cáo có số liệu rõ ràng
✅ Viral prediction → use case thực tế, media company sẽ trả tiền cho cái này
```

---

## 10. Rủi Ro & Phương Án Dự Phòng

| Rủi ro | Xác suất | Phương án dự phòng |
|--------|----------|-------------------|
| Spotify API giới hạn / thay đổi | Cao | Dùng FMA dataset hoàn toàn, không cần Spotify audio |
| GPU không đủ mạnh cho training | Trung bình | Google Colab Pro ($10/tháng) hoặc Kaggle free GPU |
| LoRA accuracy thấp | Thấp | Fallback về MFCC + SVM (classic, vẫn workable) |
| Viral predictor thiếu training data | Trung bình | Thu hẹp: chỉ predict TikTok viral (more data available) |
| BE/FE chậm → app chưa xong kỳ 2 | Thấp | DE/AI tự làm simple FastAPI endpoint để demo |
| GVHD yêu cầu đổi hướng | Thấp | Core ETL + 1 AI feature vẫn đủ cho đồ án |

---

## 11. Open Questions (Cần Xác Nhận)

> **[!IMPORTANT]**
> Các câu hỏi sau cần thảo luận với GVHD hoặc bạn đồng nhóm trước khi bắt đầu:

1. **GVHD yêu cầu tự thu thập audio hay dùng dataset có sẵn?**
   - Nếu phải tự thu thập: cần clarify luật bản quyền, chỉ dùng Creative Commons
   - Khuyến nghị: Dùng FMA (free & open dataset for research)

2. ~~**Số lượng thể loại nhạc cần phân loại?**~~ ✅ **ĐÃ QUYẾT ĐỊNH: 10 thể loại**
   - Dùng GTZAN 10 genres làm nền: `blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock`
   - Đủ phủ rộng, dataset GTZAN/FMA đều support tốt, model không quá phức tạp
   - Có thể mở rộng thêm `v-pop` hoặc `electronic` ở kỳ 2 nếu có thời gian

3. **Platform nào để deploy?**
   - Railway (đơn giản, free tier có) vs VPS (phức tạp hơn, rẻ hơn dài hạn)
   - Cần biết GVHD có yêu cầu gì về infrastructure không

4. ~~**Scope của viral prediction:**~~ ✅ **ĐÃ QUYẾT ĐỊNH: Global, nghiêng về Việt Nam (60% VN / 40% Global)**
   - Training data: **60% Vietnamese tracks** (V-pop, indie VN, nhạc trẻ) + **40% Global** (US/UK charts, K-pop)
   - Thu thập data VN: Spotify VN charts, NhacCuaTui metadata, Last.fm VN artists, Zing MP3 trending
   - Thu thập data Global: Spotify Global Top 200, Billboard Hot 100 historical, Last.fm global charts
   - Lý do nghiêng VN: use case thực tế hơn cho người dùng VN, ít cạnh tranh với tool global
   - Model vẫn hoạt động tốt với bài global — chỉ là training bias về VN market hơn

---

*File này được tạo: 2026-08-15 | Version: 1.1 | Dự án: MelodIQ*
*Cập nhật: 2026-08-15 — Xác nhận 10 genres + scope Global/VN 40:60*
