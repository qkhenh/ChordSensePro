# Dataset Download Guide — ChordSensePro DE

> Thư mục gốc: `D:\qkhenh\Project\ChordSensePro\DE\data\datasets\`
>
> Folder này được mount vào container tại `/data/datasets/`
> (xem `docker-compose.yml` line 138: `${DATA_DIR:-./data}:/data`)

---

## Cấu trúc thư mục cần tạo

```
DE/
  data/
    datasets/
      jaah/
        audio/          ← MP3 files
        annotations/    ← JSON files
      choco/
        partitions/     ← Clone từ GitHub (chỉ annotations)
      mcgill/
        {id}/           ← Mỗi bài 1 folder
      kaggle/
        guitar_chords/  ← WAV files theo chord folders
      rwc/
        audio/
        annotations/
```

---

## 1. JAAH — 113 Jazz Tracks ⭐ Làm trước

**License:** CC BY 4.0 | **Size:** ~316MB (zip) → ~2GB sau convert

### Tải về

1. Vào **[zenodo.org/record/1290737](https://zenodo.org/record/1290737)**
2. Download file `MTG/JAAH-v0.1.zip` (~316MB)
3. Giải nén ra thư mục tạm

### Cấu trúc zip sau khi giải nén

```
JAAH-v0.1/
  audio/
    Alone_Together.mp3
    Autumn_Leaves.mp3
    ...
  annotations/
    Alone_Together.json
    Autumn_Leaves.json
    ...
```

### Copy vào project

```powershell
# Tạo thư mục
New-Item -ItemType Directory -Force "DE\data\datasets\jaah\audio"
New-Item -ItemType Directory -Force "DE\data\datasets\jaah\annotations"

# Copy files (thay <path_giai_nen> bằng nơi bạn giải nén)
Copy-Item "<path_giai_nen>\JAAH-v0.1\audio\*"       "DE\data\datasets\jaah\audio\"
Copy-Item "<path_giai_nen>\JAAH-v0.1\annotations\*" "DE\data\datasets\jaah\annotations\"
```

### Format annotation JSON

```json
{
  "title": "Autumn Leaves",
  "artist": "Miles Davis",
  "metre": {"numerator": 4, "denominator": 4},
  "chords": [
    {"timestamp": 0.0, "duration": 4.0, "chord": "Cm7"},
    {"timestamp": 4.0, "duration": 4.0, "chord": "F7"}
  ]
}
```

---

## 2. ChoCo — ~20,000 Tracks (Annotations only)

**License:** CC BY 4.0 | **Size:** ~1GB repo (annotations) + audio qua YouTube

> ⚠️ ChoCo không chứa audio. Seed script sẽ lấy YouTube ID từ JAMS files.

### Tải về

```bash
# Clone chỉ folder choco/partitions/ (sparse checkout)
git clone --no-checkout https://github.com/smashub/choco DE\data\datasets\choco_repo
cd DE\data\datasets\choco_repo
git sparse-checkout init --cone
git sparse-checkout set choco/partitions
git checkout main
```

Hoặc tải toàn bộ (chậm hơn):
```bash
git clone https://github.com/smashub/choco DE\data\datasets\choco_repo
```

### Copy partitions

```powershell
Copy-Item "DE\data\datasets\choco_repo\choco\partitions" "DE\data\datasets\choco\" -Recurse
```

### Cấu trúc JAMS file

```json
{
  "file_metadata": {
    "title": "Let It Be",
    "identifiers": { "youtube_id": "dQw4w9WgXcQ" }
  },
  "annotations": [{
    "namespace": "chord",
    "data": [
      {"time": 0.0, "duration": 2.5, "value": "C"},
      {"time": 2.5, "duration": 2.5, "value": "Am"}
    ]
  }]
}
```

---

## 3. McGill Billboard — 1,300 Tracks

**License:** CC0 (public domain) | **Size:** Annotations ~65MB + audio qua YouTube

> ⚠️ Không có audio trực tiếp, chỉ có chord annotations. Audio tải qua YouTube IDs.

### Cách A — Kaggle (dễ nhất)

1. Vào **[kaggle.com/datasets/landlord/mcgill-billboard](https://www.kaggle.com/datasets/landlord/mcgill-billboard)**
2. Download zip → giải nén

### Cách B — `mirdata` Python library

```bash
pip install mirdata
python -c "import mirdata; mirdata.initialize('billboard').download()"
```

### Cách C — Official DDMAL page

Vào **[ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/)**

### Copy vào project

```powershell
Copy-Item "<path_giai_nen>\McGill-Billboard\" "DE\data\datasets\mcgill\" -Recurse
```

---

## 4. Kaggle Guitar Chords — 7,000+ WAV

**License:** CC BY 4.0 | **Size:** ~3.5GB

> ✅ Dataset này có audio sẵn, không cần YouTube. Không cần annotation (folder name = chord label).

### Tải về

1. Tạo tài khoản Kaggle nếu chưa có
2. Vào **[kaggle.com/datasets/fabianavinci/guitar-chords-v3](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3)**
3. Download → giải nén

### Cấu trúc sau khi giải nén

```
guitar_chords/
  A/   Am/  B/  Bm/  C/  D/  Dm/  E/  Em/  F/  G/
    *.wav files (2-5 giây mỗi file)
```

### Copy

```powershell
Copy-Item "<path_giai_nen>\guitar_chords\" "DE\data\datasets\kaggle\guitar_chords\" -Recurse
```

---

## 5. RWC — 100 Tracks (Cần đăng ký)

**License:** Institutional | **Size:** ~1.5GB

> ⚠️ Cần đăng ký tại [staff.aist.go.jp/m.goto/RWC-MDB](https://staff.aist.go.jp/m.goto/RWC-MDB/) trước.
> Skip cái này trước, làm sau khi có đủ dữ liệu từ các nguồn khác.

---

## Sau khi tải xong

```powershell
# Chạy seed script để insert vào database
python scripts\seed_local_datasets.py

# Kiểm tra
docker exec chordsense_postgres psql -U chordsense -d chordsense -c "SELECT source_type, count(*) FROM crawl_queue GROUP BY source_type;"
```

---

## Thứ tự ưu tiên

| Thứ tự | Dataset | Lý do |
|--------|---------|-------|
| 1 | **JAAH** | Nhỏ nhất, annotation chất lượng cao nhất (jazz chords), có audio sẵn |
| 2 | **Kaggle** | Audio sẵn, không cần YouTube, dễ dùng |
| 3 | **McGill** | 1,300 bài pop/rock, cần YouTube |
| 4 | **ChoCo** | Lớn nhất, cần YouTube, làm cuối |
| 5 | **RWC** | Cần đăng ký tổ chức |
