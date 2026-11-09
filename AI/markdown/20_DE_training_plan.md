# ChordSense Pro — Kế hoạch train model nhận diện hợp âm

> Ngày lập: 2026-10-21 · **Cập nhật lần cuối: 2026-11-09 (pipeline v2, data hoàn tất).**
> Phạm vi: **phía AI** (train + đánh giá + chuẩn bị deploy) dùng data do pipeline DE v2 tạo ra.
> Tài liệu này là **kế hoạch**. Repo hiện **chưa có code train**. Mọi con số về thời gian chạy / bộ nhớ là **ước lượng**, phải đo thật rồi cập nhật lại vào đây.
> Liên quan: [data_flow.md](../../DE/markdown/data_flow.md) · [dataset_sources.md](../../DE/markdown/dataset_sources.md) · [data_dictionary.md](../../DE/markdown/21_DE_data_dictionary.md) · [change_log.md](../../DE/markdown/change_log.md).

---

## 0. Tóm tắt (đọc phần này trước)

| Mục | Quyết định |
|---|---|
| **Bài toán** | Với một đoạn audio **5 s** (random crop từ bài), đoán hợp âm đang vang dưới dạng 3 nhãn: root (12 lớp), quality (6 lớp), extension (15 lớp). Nhãn gán theo **chord timeline** (frame-level), không phải midpoint. |
| **Data** | Pipeline v2 hoàn tất: **6.222 bài** (train 4.324 / val 1.251 / test 647), **222.816 chord labels**. Audio là `.npy` 24 kHz int16 mono, **chiều dài biến thiên** (whole section). Parquet chứa chord timeline `(start_ms, end_ms, label)`. |
| **Model nền** | MERT-v1-330M (model âm nhạc đã pretrain, nhận waveform 24 kHz). |
| **Cách thích nghi** | Đi từ nhẹ đến nặng: (1) đóng băng MERT, chỉ train head → (2) thêm LoRA (chỉ train ~2 triệu tham số phụ). **Không** fine-tune toàn bộ 330M. |
| **Head** | 3 head nối tầng (root → quality → extension), mask nhãn thiếu, Focal Loss cho extension. |
| **Chống lệch lớp** | Pitch-shift kèm đổi nhãn root + sampler cân bằng + Focal Loss. So sánh bằng thí nghiệm, không đoán. |
| **Môi trường** | Google Colab (GPU T4 16 GB). Máy bạn (RTX 3050 4 GB) chỉ dùng để thử inference, **không** đủ để train LoRA 330M. |
| **Đánh giá** | Theo **bài hát/video** (không theo crop), báo cáo từng lớp + độ tin cậy; test chỉ chạm **một lần** ở cuối. |
| **Deploy** | Web app chạy trên máy tính: đo tốc độ trước, chỉ thu nhỏ model (fp16/ONNX → cắt tầng → distillation) nếu cần. |
| **Thứ tự làm** | S0 chuẩn bị → S1 baseline → S2 LoRA → S3 thí nghiệm cân bằng lớp → S4 đánh giá sâu → S5 inference + tốc độ → S6 viết báo cáo. |

> [!IMPORTANT]
> Rủi ro lớn nhất không phải là kiến trúc mà là **data lớp hiếm quá ít** (b5: 80, b13: 106, #9: 128) và **lệch domain** (data là jazz/pop thu thật, app dành cho piano). Kế hoạch dưới đây tập trung đo và giảm hai rủi ro này.

---

## 1. Bài toán, nói cho thật rõ

### 1.1. Đầu vào và đầu ra

- **Đầu vào:** 1 đoạn audio **5 giây** (random crop từ bài gốc có chiều dài biến thiên), mono, **24 kHz** → 120.000 mẫu (lưu dạng int16 trong `.npy`, mỗi bài một file riêng).
- **Đầu ra:** 3 số nguyên cho **mỗi frame** (MERT output 75 Hz → 375 frame cho 5 s):
  - `root_id` ∈ 0..11 = C, C#, D, D#, E, F, F#, G, G#, A, A#, B.
  - `quality_id` ∈ 0..5 = major, minor, dominant, dim, sus2, sus4.
  - `ext_id` ∈ 0..14 = none, b5, #5, maj7, min7, dom7, add9, b9, 9, #9, 11, #11, 13, b13, alt.
- Ghép 3 số lại ra hợp âm, ví dụ (A, minor, min7) = Am7; (G, dominant, #9) = G7#9.
- `dim7` **không có lớp riêng**: là `quality=dim` + `ext=dom7` (quy ước trong `value_objects.py`).

> [!NOTE]
> **Khác biệt v1 → v2:** v1 dùng segment cố định 2 s + nhãn tại midpoint (1 nhãn/segment). v2 dùng **whole section** + **chord timeline** (nhiều nhãn/bài, mỗi nhãn có `start_ms`/`end_ms`). AI tự crop random 5 s, gán nhãn frame-level từ timeline.

### 1.2. Nhãn của một crop được gán như thế nào (quan trọng cho cách train)

Chord timeline từ `chords.parquet` gồm chuỗi `(start_ms, end_ms, label_raw, root_id, quality_id, ext_id)` cho mỗi bài. Khi random crop 5 s từ vị trí `t_start`:

1. **Segment-level (baseline):** lấy hợp âm tại **giữa crop** (`t_start + 2500 ms`) từ timeline → 1 nhãn cho cả crop. Pooling MERT output → 1 embedding → 3 head. Đơn giản, nhanh, phù hợp baseline.
2. **Frame-level (nâng cao):** với mỗi MERT frame (75 Hz), tra chord timeline để gán nhãn → 375 nhãn cho 5 s. Head dự đoán per-frame. Chính xác hơn, tận dụng tối đa chord timeline.

Đề xuất: **bắt đầu với segment-level** (baseline), thử frame-level ở thí nghiệm sau.

Hệ quả khi segment-level:
1. Nếu hợp âm **đổi** gần midpoint → nhiễu nhãn (giống v1 nhưng ít hơn vì biết chính xác ranh giới).
2. Có thể **tránh crop tại ranh giới**: chỉ crop ở vị trí mà toàn bộ 5 s nằm trong 1 chord (hoặc chord tại midpoint chiếm ≥ 80% crop).
3. Khi phân tích lỗi phải tách crop "ổn định" và crop "gần ranh giới" (xem §8.3).

### 1.3. Nhãn thiếu (`partial`) và "không có hợp âm" (`N`)

| Trạng thái | Ý nghĩa | Xử lý khi train |
|---|---|---|
| `ok` | cả 3 nhãn có | dùng bình thường |
| `partial` | có root nhưng thiếu quality và/hoặc extension (hợp âm 6, 6/9, power chord `C5`) | **mask** head thiếu khỏi loss (mục 5.3) |
| `no_chord` (`N`) | im lặng / không hợp âm | `export_dataset_v2.py` **không xuất** (chỉ lấy ok + partial). Xem quyết định ở §11 |
| `unmapped` | nhãn lạ chưa map được | không xuất |

---

## 2. Kiến thức nền (giải thích từng khái niệm)

### 2.1. MERT là gì và vì sao dùng

- **MERT** là model học tự giám sát trên rất nhiều audio âm nhạc (không cần nhãn hợp âm). Nó biến audio thành chuỗi vector biểu diễn "âm nhạc có nghĩa" (cao độ, hòa âm, âm sắc).
- **MERT-v1-330M**: 24 tầng transformer, mỗi vector 1024 chiều, **75 vector mỗi giây** (75 Hz), nhận **24 kHz**. Đoạn 5 s → khoảng **375 vector** × 1024.
  - Output `output_hidden_states=True` cho **25 tập vector** (embedding đầu vào + 24 tầng). Mỗi tầng giữ loại thông tin khác nhau; tầng thấp thiên về âm sắc/cao độ, tầng cao thiên về ngữ cảnh. **Tầng nào tốt nhất cho hợp âm là thứ phải đo** (bước B1), không đoán.
- Lý do dùng model pretrain: ta chỉ có ~55 giờ audio (6.222 bài × ~30 s trung bình). Train từ đầu một mạng sâu trên đó sẽ quá khớp. MERT đã "nghe" hàng chục nghìn giờ nhạc nên ta chỉ cần dạy nó **bảng nhãn hợp âm**.
- Có sẵn **MERT-v1-95M** (12 tầng, 768 chiều) nhẹ hơn nhiều; dùng làm đối chứng (§9.4).

### 2.2. Ba mức "dạy" MERT (từ nhẹ đến nặng)

| Mức | Cập nhật gì | Tham số train | Ưu | Nhược |
|---|---|---|---|---|
| **A. Đóng băng + head** | chỉ 3 head | ~vài trăm nghìn | rất nhanh, an toàn, có thể **cache embedding** | MERT không tự thích nghi với piano/nhãn của ta |
| **B. LoRA** | head + các ma trận nhỏ chèn vào attention của MERT | head + ~1,6–2,4 triệu | thích nghi được mà không phá model gốc; rẻ | cần GPU Colab, chậm hơn mức A |
| **C. Full fine-tune** | toàn bộ 330M | 330 triệu | mạnh nhất (lý thuyết) | quá khớp với ít data, tốn VRAM, **không khuyến nghị** |

**LoRA hoạt động thế nào?** Một ma trận trọng số `W` (1024×1024) của MERT bị đóng băng. LoRA thêm hai ma trận nhỏ `A` (1024×r) và `B` (r×1024) với `r=16`; đầu ra là `W·x + (α/r)·B·A·x`. Chỉ `A`, `B` được train. Vì `r` nhỏ nên mỗi module chỉ thêm `16×(1024+1024)=32.768` tham số. Chèn vào `q_proj` và `v_proj` của 24 tầng → ~1,57 triệu tham số; thêm `out_proj` → ~2,36 triệu (khớp con số "~2 triệu" trong spec).

### 2.3. Head nối tầng (cascaded)

```
embedding (1024) ──► HEAD 1 (root, 12 lớp) ──► root_logits (12)
embedding (1024) + root_logits (12) ──► HEAD 2 (quality, 6 lớp) ──► quality_logits (6)
embedding (1024) + root_logits (12) + quality_logits (6) ──► HEAD 3 (extension, 15 lớp)
```

Ý tưởng: biết root và quality trước giúp đoán extension (ví dụ cùng 4 nốt nhưng root khác thì thành hợp âm khác). Nhược: lỗi root có thể lan sang head sau. Vì vậy **phải có thí nghiệm so với head song song** (spec thí nghiệm E).

### 2.4. Focal Loss (cho extension)

Cross-entropy thường để các lớp phổ biến (`none` 18.864, `dom7` 8.559) chi phối. **Focal Loss** nhân thêm hệ số `(1 − p_đúng)^γ` (γ=2): mẫu model đã đoán đúng chắc chắn bị giảm trọng số, mẫu khó/hiếm được giữ lại. Nó **không tạo thêm data**, chỉ chỉnh sự chú ý; vì vậy vẫn cần sampler/augment (§6).

### 2.5. Pitch-shift kèm đổi nhãn

Dịch cao độ audio lên `n` nửa cung thì **root dịch `n`** (mod 12), còn **quality và extension không đổi** (Am7 lên 2 cung → Bm7). Đây là cách hợp lệ nhất để nhân data và cân bằng root, và để một loại hợp âm hiếm xuất hiện ở đủ 12 tông. **Không** giúp với việc có quá ít *ví dụ độc lập* (ví dụ chỉ có vài bài chứa b13 thì 12 phiên bản của chúng vẫn chỉ là vài bài).

---

## 3. Hiện trạng data (chốt 2026-11-09, pipeline v2 hoàn tất)

Data đã export tại `DE/data/export/v2/`. Cấu trúc:
```
v2/
  {split}/audio/{song_id}.npy     # 24 kHz int16 mono, 1D, chiều dài biến thiên
  {split}/chords.parquet          # song_id, start_ms, end_ms, label_raw, label_status, root_id, quality_id, ext_id
  config.yaml                    # sample_rate: 24000, crop_s: 5, dtype: int16, mono: true, variable_length: true
  classes.json                   # root/quality/extension names, valid_combo_mask [6×15], counts
```

### 3.1. Quy mô

| Mục | Số |
|---|---|
| Bài (training_sections) | **6.222** (train 4.324 · val 1.251 · test 647) |
| Chord labels (training_chords) | **222.816** |
| Nguồn | Hooktheory ~78% · JAAH ~20% · GuitarSet · local piano |
| Thời lượng ước tính | ~55 giờ audio (bài ngắn ~20 s Hooktheory, bài dài ~5 phút JAAH) |
| Dung lượng | `training_v2/` 17 GB, export ~15 GB |

> [!NOTE]
> Split chia theo hash của `source_url`: các chord cùng một bài luôn nằm cùng một split. Lớp hiếm trên test vẫn chỉ vài chục mẫu → **số liệu theo từng lớp hiếm trên test sẽ rất nhiễu**.

### 3.2. Phân bố lớp (từ training_chords, 222K labels)

| Quality | Count | | Extension | Count |
|---|---|---|---|---|
| major | ~87K | | none | ~57K |
| minor | ~64K | | min7 | ~28K |
| dominant | ~25K | | dom7 | ~15K |
| dim | ~8K | | maj7 | ~14K |
| sus4 | ~6K | | nat9 | ~14K |
| sus2 | ~4K | | nat11 | ~7K |
| | | | add9 | ~5K |
| | | | nat13 | ~3K |
| | | | b9 | ~2K |
| | | | #5 | ~1.5K |
| | | | #11 | ~1.5K |
| | | | alt | ~1K |
| | | | #9 | ~500 |
| | | | b13 | ~400 |
| | | | b5 | ~350 |

> [!NOTE]
> Số liệu chính xác nằm trong `classes.json` sau export. Con số trên đây lấy từ truy vấn DB trực tiếp (tất cả label, chưa lọc). So với v1 (~48K segments), v2 có **4.6× nhiều hơn** vì mỗi bài giờ có nhiều chord labels thay vì chỉ ~16 segments.

### 3.3. Vấn đề của data (phải nhớ khi thiết kế)

1. **Lệch lớp nặng ở extension.** Tỉ lệ `none` / `b5` ≈ 160:1. Tốt hơn v1 (180:1) nhưng vẫn nghiêm trọng.
2. **Chiều dài biến thiên.** Bài Hooktheory ~20 s, JAAH có thể 5+ phút. Random crop 5 s sẽ thiên về bài dài (nhiều crop hơn). Cần sampler có trọng số theo bài.
3. **Mất cân bằng giữa các nguồn.** Hooktheory ≈ 78%, JAAH ≈ 20%. Hợp âm màu đa số đến từ JAAH (jazz). Luôn báo metric **theo từng nguồn**.
4. **Lệch domain.** Phần lớn là jazz (JAAH) và pop thu thật qua Demucs (stem harmonic: `other + bass`, htdemucs_ft). App thật nhận **piano từ micro/file** không qua Demucs. Cần đo riêng trên nguồn piano.
5. **Nhiễu nhãn:**
   - Hooktheory: nhãn do người dùng chuyển soạn, canh với video YouTube bằng alignment check (chroma offset, pipeline v2).
   - JAAH: tín hiệu nhãn–audio yếu nhất (kiểm chroma 2026-10-23).
6. **Audio chưa chuẩn hóa biên độ**: stem Demucs nhỏ (median −24 dBFS) → phải chuẩn hóa ở phía train (mục 4.1).
7. **Alignment quality (v2):** pipeline v2 có `align_offset_ms`, `align_score`, `align_status` (ok/shifted/rejected). Bài bị rejected đã loại khỏi data. Bài shifted đã được điều chỉnh offset.

---

## 4. Thiết kế pipeline train

### 4.1. Chuẩn bị dữ liệu vào model

```python
# dataset.py (ý tưởng)
import pandas as pd

chords_df = pd.read_parquet("train/chords.parquet")  # song_id, start_ms, end_ms, root_id, quality_id, ext_id
song_chords = chords_df.groupby("song_id")           # nhóm theo bài

# Mỗi lần lấy mẫu:
audio = np.load(f"train/audio/{song_id}.npy")         # 1D int16, chiều dài biến thiên
crop_start = random_int(0, len(audio) - 5*24000)      # random crop 5 s
x = audio[crop_start : crop_start + 120000].astype(np.float32) / 32768.0
x = (x - x.mean()) / (x.std() + 1e-7)                 # MERT zero-mean/unit-var

# Gán nhãn từ chord timeline (segment-level: lấy chord tại midpoint)
t_mid_ms = (crop_start / 24000 * 1000) + 2500
chord = lookup_chord_at(song_chords[song_id], t_mid_ms)
```

- `Wav2Vec2FeatureExtractor` của MERT (`do_normalize=True`) làm bước chuẩn hóa tương tự; có thể tự làm để nhanh hơn và để áp augment trước.
- Mỗi bài là **1 file .npy riêng**, không cần mmap array lớn. Đọc bằng `np.load` bình thường.
- Nhãn thiếu: `quality_id`/`ext_id` là `null` trong parquet → đổi thành `-100` và dùng `ignore_index=-100` khi tính loss (mục 5.3).
- Nếu bài < 5 s: pad zero hoặc bỏ qua (rất ít bài dưới 5 s).

### 4.2. Môi trường

| Việc | Ở đâu | Ghi chú |
|---|---|---|
| Export data | máy bạn (Docker) | `docker exec -w /opt/airflow chordsense_airflow env PYTHONPATH=/opt/airflow python scripts/export_dataset_v2.py` → `DE/data/export/v2/` (~15 GB) |
| Train, extract embedding | **Colab (T4 16 GB)** | upload export lên Google Drive; mount Drive; lưu checkpoint lên Drive mỗi epoch |
| Thử inference / đo tốc độ | **máy bạn** (RTX 3050 4 GB) | fp16 MERT-330M ~0,7 GB nên vừa; đây là nơi sẽ deploy |

- Phiên bản thư viện: PyTorch ≥ 2.5, transformers ≥ 4.47, PEFT ≥ 0.14. Ghi lại `pip freeze` của mỗi lần chạy.
- T4 **không có bf16**: dùng **fp16 + GradScaler** (AMP), không dùng bf16.
- Phiên Colab có thể bị ngắt: lưu checkpoint + trạng thái optimizer/scheduler + seed mỗi epoch; code train phải **resume** được.
- Cố định seed (`torch`, `numpy`, `random`), ghi hyperparameter vào file `config.yaml` kèm mỗi lần chạy để tái lập được.

### 4.3. Kiến trúc model chi tiết

```
waveform (B, 120000)    ← 5 s × 24 kHz
   │
   ▼  MERT-330M  (output_hidden_states=True → 25 × (B, 375, 1024))
   │
   ▼  Gộp tầng:  h = Σ_k softmax(w)_k · hidden_k      (w học được; hoặc chọn tầng tốt nhất sau B1)
   │
   ▼  Pooling theo thời gian (xem dưới)  → (B, 1024)
   │
   ├─► HEAD 1  Linear(1024→128) → ReLU → Linear(128→12)           → root_logits
   ├─► HEAD 2  [emb ‖ root_logits](1036) → 128 → 6                → quality_logits
   └─► HEAD 3  [emb ‖ root_logits ‖ quality_logits](1042) → 256 → Dropout(0.3) → 64 → 15   → ext_logits
```

**Gộp tầng có trọng số học được** (weighted-sum, như SUPERB) thay vì chỉ lấy tầng cuối: tầng cuối của model tự giám sát không phải tầng tốt nhất cho tác vụ hòa âm; để model tự học trọng số `w` (25 số) là rẻ và hiệu quả.

**Pooling theo thời gian** (chọn bằng thí nghiệm, 3 phương án):

| Phương án | Cách làm | Khi nào hợp |
|---|---|---|
| P1 Trung bình đều | mean 375 vector | baseline đơn giản |
| P2 **Trung bình quanh trung tâm** | mean các vector ở khoảng 1,5–3,5 s (khung ~112–262 của 375) | giảm nhiễu ranh giới chord khi label = midpoint |
| P3 Attention pooling | `softmax(MLP(h_t))` làm trọng số | model tự chọn vùng quan trọng; thêm ít tham số |

Mặc định đề xuất: **P1 làm baseline** (đơn giản), thử P2/P3 nếu cần.

**Truyền logit giữa các head khi train:** spec nối `logits` của head trước vào head sau. Có 2 cách:
- (i) nối logit **chưa detach** (gradient của head sau chảy ngược về head trước);
- (ii) nối logit **detach** hoặc dùng one-hot nhãn thật (teacher forcing) để head sau không bị lỗi head trước kéo theo lúc đầu.
Đề xuất: thử (ii) **detach logit dự đoán** làm mặc định, và so với (i) và với **head song song** trong §9.

### 4.4. Pseudo-code phần lõi

```python
class ChordModel(nn.Module):
    def __init__(self, mert, n_layers=25, d=1024):
        super().__init__()
        self.mert = mert
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.h1 = mlp(d, 128, 12)
        self.h2 = mlp(d + 12, 128, 6)
        self.h3 = nn.Sequential(nn.Linear(d + 18, 256), nn.ReLU(), nn.Dropout(0.3),
                                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 15))

    def forward(self, wav):
        hs = torch.stack(self.mert(wav, output_hidden_states=True).hidden_states)  # (25,B,T,D)
        h = (F.softmax(self.layer_w, 0)[:, None, None, None] * hs).sum(0)          # (B,T,D)
        emb = h.mean(1)                                                             # pooling P1
        r = self.h1(emb)
        q = self.h2(torch.cat([emb, r.detach()], -1))
        e = self.h3(torch.cat([emb, r.detach(), q.detach()], -1))
        return r, q, e

def loss_fn(r, q, e, y, w=(0.3, 0.3, 0.4), gamma=2.0, class_w_ext=None):
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    l_root = ce(r, y["root"])
    l_qual = ce(q, y["quality"])                               # nhãn -100 (partial) tự bị bỏ qua
    l_ext  = focal_loss(e, y["ext"], gamma, class_w_ext, ignore_index=-100)
    return w[0]*l_root + w[1]*l_qual + w[2]*l_ext
```

Lưu ý khi cài: nếu **cả batch** không có nhãn hợp lệ cho một head thì `CrossEntropyLoss(ignore_index)` trả `nan` → phải kiểm tra và đặt loss của head đó bằng 0.

---

## 5. Hàm mất mát và xử lý nhãn thiếu

### 5.1. Công thức

`L = w_root·CE(root) + w_qual·CE(quality) + w_ext·Focal(ext)`

- Pha chỉ train head root + quality: `w = (0,5; 0,5; 0)` (theo spec).
- Pha đầy đủ: `w = (0,3; 0,3; 0,4)`.

### 5.2. Focal Loss

`FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)` với `γ = 2`. `α_t` là trọng số lớp (mục 6.2). Cài đặt phải hỗ trợ `ignore_index`.

### 5.3. Mask nhãn thiếu

| Nhãn | quality_id | ext_id | Loss root | Loss quality | Loss ext |
|---|---|---|---|---|---|
| ok | có | có | ✓ | ✓ | ✓ |
| partial (6th, 6/9) | có | null | ✓ | ✓ | ✗ |
| partial (power chord `C5`) | null | null | ✓ | ✗ | ✗ |

Hiện có 1.230 segment `partial` (≈ 4,6%): 6th/6-9 chiếm phần lớn.

---

## 6. Chống lệch lớp và augmentation

### 6.1. Augmentation chỉ áp cho `train` (không bao giờ cho val/test)

| Loại | Tham số đề xuất | Đổi nhãn? |
|---|---|---|
| **Pitch-shift** | ngẫu nhiên −5…+6 nửa cung mỗi lần lấy mẫu (online, mỗi epoch một giá trị khác) | **root dịch `n` (mod 12)**, quality/ext giữ nguyên |
| Gain | ±6 dB | không |
| Nhiễu Gaussian | SNR 10–30 dB, xác suất 0,5 | không |
| Reverb nhẹ | xác suất 0,3 | không |
| Time-stretch | 0,85–1,15 (cẩn thận: v2 chord timeline tính bằng ms nên stretch phải điều chỉnh boundary), xác suất 0,3 | không |
| ❌ Mixup | **không dùng** (nhãn mơ hồ) | — |

- Pitch-shift **online trên GPU** (ví dụ `torchaudio.functional.pitch_shift`, hoặc resample + time-stretch) để không phải lưu ×12 data. Pitch-shift bằng `librosa` trên CPU mỗi batch sẽ **rất chậm** — nếu cần dùng thì precompute trước.
- Sau pitch-shift, **kiểm tra chéo**: nhãn root mới phải bằng `(root_id + n) % 12`. Viết **unit test** cho hàm này (một lỗi lệch nhãn sẽ phá toàn bộ pha train mà không báo lỗi).
- Pitch-shift lớn (|n| > 6) làm méo âm sắc; vì dịch −5…+6 đã phủ đủ 12 tông nên không cần lớn hơn.

### 6.2. Cân bằng lớp (so sánh 3 cách trong §9)

| Cách | Mô tả |
|---|---|
| **S0** | Không làm gì (baseline đối chứng) |
| **S1 Sampler theo căn bậc hai** | xác suất lấy mẫu ∝ `1/√(tần số lớp extension)`; làm mềm hơn "đều tuyệt đối" để không lặp lại quá mức vài ví dụ hiếm |
| **S2 Class-balanced weight** | `α_c = (1 − β) / (1 − β^{n_c})`, `β = 0,999` (effective number of samples), chuẩn hóa về trung bình 1, dùng làm `α` trong Focal Loss |
| **S3 = S1 + S2** | thường mạnh nhất nhưng dễ quá khớp lớp hiếm |

Thêm **trọng số theo bài**: mỗi crop có trọng số `∝ 1/√(duration_s của bài)` để bài JAAH dài (5+ phút) không chi phối so với clip Hooktheory (~20 s).

> [!WARNING]
> Với lớp có vài trăm chord labels, dù sampler lấy lại nhiều lần thì model vẫn chỉ thấy **cùng vài ví dụ**. Dấu hiệu quá khớp: train recall của lớp đó ≈ 100% nhưng val recall ≈ 0%. Phải theo dõi riêng recall từng lớp hiếm trên val mỗi epoch.

---

## 7. Các giai đoạn thực hiện (có đầu ra và tiêu chí đi tiếp)

> Thời gian chỉ là ước lượng; "ngày" = ngày làm việc cho người đã quen Colab.

### S0 — Chuẩn bị (~0,5 ngày)

1. ~~Chạy export~~ **Đã export** tại `DE/data/export/v2/` (2026-11-04). **`classes.json` đã có kèm.**
2. **Kiểm tra sức khỏe data:**
   - số file `.npy` = số bài trong `chords.parquet` cho từng split;
   - nghe thử 5–10 bài ngẫu nhiên mỗi nguồn + đối chiếu chord timeline (đảm bảo không lệch canh);
   - biên độ không bị clipping, không có file toàn 0;
   - không có `song_id` xuất hiện ở hai split (viết assert).
3. Tạo môi trường Colab, upload data lên Drive (~15 GB), cài thư viện, tải `m-a-p/MERT-v1-330M` (cần `trust_remote_code=True`).
4. **Chạy thử một batch** MERT lấy đủ `(B, 375, 1024)` × 25 tầng; xác nhận tốc độ khung 75 Hz.
- **Đầu ra:** notebook môi trường chạy được + data verified. **Tiêu chí đi tiếp:** các assert đều qua.

### S1 — Baseline (~1–2 ngày)

| Mã | Mô tả | Mục đích |
|---|---|---|
| **B0** | Dự đoán lớp phổ biến nhất; và **chroma + hồi quy logistic** (librosa `chroma_cqt`, trung bình 5 s) | chặn dưới; trả lời "LoRA có đáng không" (thí nghiệm A của spec) |
| **B1** | **Thăm dò từng tầng:** cho MERT đóng băng, lấy vector pooling P1 của từng tầng 0..24, huấn luyện một Linear cho root, quality, extension riêng | **chọn tầng/cách gộp tốt nhất** trước khi tốn công LoRA; thí nghiệm C (MERT zero-shot) |
| **B2** | 3 head nối tầng trên embedding đóng băng, **cache sẵn** | baseline đầy đủ nhanh |

**Cache embedding (ý tưởng quan trọng để lặp nhanh):** vì MERT đóng băng nên có thể chạy MERT **một lần** cho mỗi crop cố định (không augment), lưu vector pooling cho cả 25 tầng. Với v2 (variable-length audio), cần chọn vị trí crop cố định cho mỗi bài (ví dụ lấy N crop không chồng lấn mỗi 5 s).
- Dung lượng ước tính: `4.324 bài × 3 crop trung bình × 25 tầng × 1024 × 2 byte (fp16) ≈ 0,66 GB` — Colab chịu được.
- Sau đó thử hàng trăm cấu hình head trong vài phút, **không cần GPU nặng**.
- Hạn chế: cache **không** có augment (không pitch-shift được), nên baseline này không phản ánh hết khả năng; augment chỉ có ở S2.

- **Đầu ra:** bảng độ chính xác B0/B1/B2 theo từng head + biểu đồ độ chính xác theo tầng. **Tiêu chí đi tiếp:** B2 phải hơn B0 rõ rệt ở root và quality; nếu không → kiểm tra lại data/nhãn trước khi làm tiếp.

### S2 — LoRA (mô hình chính) (~2–3 ngày)

1. Cấu hình LoRA (đề xuất):

   | Tham số | Giá trị | Giải thích |
   |---|---|---|
   | `r` | 16 | hạng của ma trận phụ; so thêm 8 và 32 (thí nghiệm F) |
   | `lora_alpha` | 32 | hệ số co giãn = 2×r |
   | `lora_dropout` | 0,1 | chống quá khớp |
   | `target_modules` | `q_proj`, `v_proj` (+ `out_proj` nếu cần) | **kiểm tra tên thật** bằng `model.named_modules()` của MERT trước |
   | Tầng áp | cả 24 tầng; thử chỉ nửa trên | |

2. Hyperparameter khởi điểm (cần tinh chỉnh):

   | Mục | Giá trị |
   |---|---|
   | Optimizer | AdamW, weight decay 0,01 |
   | LR head | 2e-4 |
   | LR LoRA | 5e-5 đến 1e-4 (LoRA cần LR thấp hơn head) |
   | Scheduler | warmup 5% bước + cosine decay |
   | Batch | 16–32 (T4 16 GB, fp16, bật gradient checkpointing nếu thiếu VRAM) |
   | Epoch | 10–15 + early stopping (patience 3) theo điểm val |
   | Grad clip | 1,0 |

3. Chạy theo pha (rút gọn từ spec, **bỏ phần Pianoteq/VN piano** vì chưa có data):
   - **Pha 1** (5 epoch): MERT đóng băng, chỉ head root + quality (loss 0,5/0,5).
   - **Pha 2** (10 epoch): mở LoRA + cả 3 head (loss 0,3/0,3/0,4), Focal cho extension, augment bật.
   - **Pha 3** *(chưa làm được ngay)*: fine-tune trên **piano thật** — chỉ khi có data piano (xem §10).
4. **Theo dõi:** loss từng head, accuracy từng head, **recall từng lớp hiếm**, trọng số tầng `w`, LR — ghi vào `results.csv` (TensorBoard chỉ là tùy chọn).
5. Lưu **checkpoint tốt nhất theo điểm val** (không phải epoch cuối) + file `config.yaml`.

- **Đầu ra:** model LoRA tốt nhất + báo cáo val. **Tiêu chí đi tiếp:** hơn B2 (nếu không hơn, LoRA không đáng — vẫn là kết quả hợp lệ cho báo cáo).

### S3 — Thí nghiệm cân bằng lớp, pooling, kiến trúc (~2 ngày)

Chạy theo bảng ở §9, **thay đổi một yếu tố mỗi lần**, giữ nguyên seed và split. Lặp ít nhất **3 seed** cho các cấu hình quan trọng để biết độ dao động (với data nhỏ, chênh lệch dưới ~1–2% có thể chỉ là nhiễu).

### S4 — Đánh giá sâu và phân tích lỗi (~1–2 ngày)

Xem §8. Test chỉ chạy **một lần** sau khi đã chốt mô hình bằng val.

### S5 — Pipeline suy luận (inference) và tốc độ (~2–3 ngày)

1. **Cửa sổ trượt:** web app nhận audio liên tục → cắt đoạn 5 s, bước 0,5–1 s (tùy yêu cầu độ trễ) → mỗi đoạn qua model → chuỗi xác suất theo thời gian.
2. **Làm mượt theo thời gian:** thuật toán Viterbi/HMM trên chuỗi xác suất với **ma trận chuyển hợp âm** (học từ chuỗi hợp âm trong data hoặc đặt "ở lại hợp âm cũ" có xác suất cao) để bớt nhấp nháy.
3. **Ngưỡng "không có hợp âm":** vì model **không có lớp N** (data export không có `no_chord`), dùng **cổng năng lượng (RMS)** và **ngưỡng độ tin cậy tối đa** để trả "không rõ".
4. **Hiệu chỉnh độ tin cậy:** temperature scaling trên val để xác suất phản ánh đúng độ chắc chắn.
5. **Đo tốc độ trên máy deploy** (RTX 3050 4 GB): thời gian cho một đoạn 5 s ở fp32/fp16/ONNX, đo cả bước tiền xử lý. Mục tiêu spec: **< 500 ms**.
6. **Chỉ khi chưa đạt** mới thu nhỏ theo thứ tự rẻ → đắt:
   1. fp16 + xuất ONNX (+ int8 nếu cần);
   2. cắt tầng (giữ N tầng đầu theo kết quả B1) hoặc dùng MERT-95M;
   3. **distillation** sang model nhỏ (hoặc chạy trong trình duyệt bằng `onnxruntime-web` nếu cần chạy phía client).
   Mỗi bước phải **đo lại accuracy** để biết cái giá phải trả.

### S6 — Báo cáo (~2 ngày)

Gom bảng thí nghiệm (§9), biểu đồ, phân tích lỗi, hạn chế (data hiếm, domain) vào báo cáo.

---

## 8. Cách đánh giá (chi tiết)

### 8.1. Quy tắc để đánh giá không bị lừa

1. **Chia theo bài/video, không theo crop** (đã đúng: split theo `source_url`). Không bao giờ trộn lại các bài.
2. **Test chỉ chạm một lần.** Mọi chọn lựa (siêu tham số, epoch, ngưỡng) dùng val. Nếu cần tinh chỉnh nhiều, dùng **cross-validation theo nhóm bài** (GroupKFold) thay vì "nhìn test".
3. **Khoảng tin cậy bootstrap theo bài** (lấy lại mẫu các bài, không phải crop) vì crop cùng bài tương quan mạnh; nếu bootstrap theo crop thì khoảng tin cậy hẹp giả tạo.
4. **Báo cáo riêng theo nguồn** (JAAH / Hooktheory / piano local / GuitarSet). Một con số trung bình che mất lệch domain.

### 8.2. Chỉ số

| Chỉ số | Định nghĩa |
|---|---|
| Root accuracy | % crop đoán đúng root (chỉ tính crop có nhãn root) |
| Quality accuracy | % đúng quality (bỏ crop mask) |
| Extension accuracy | % đúng extension (bỏ crop mask) |
| **Chord accuracy** | % đúng **cả 3** (chỉ tính crop có đủ 3 nhãn) |
| Macro-F1 (quality, extension) | trung bình F1 các lớp — **không** bị lớp lớn lấn át, đây là chỉ số quan trọng nhất cho lớp hiếm |
| Recall từng lớp | bảng 15 lớp extension, 6 lớp quality |
| Ma trận nhầm lẫn | để thấy nhầm kiểu gì (vd. min7 ↔ dom7, 9 ↔ add9) |
| Điểm theo mức (kiểu mir_eval) | root-only, major/minor, sevenths — xấp xỉ vì taxonomy của ta khác bảng chuẩn |

Mục tiêu trong spec: root > 95%, quality > 85%, extension > 70% (lớp 0–5) và > 55% (lớp 6–14). **Với data hiện tại, mục tiêu cho các lớp hiếm có thể không đạt được**; kết quả thực tế mới là con số để báo cáo (và nêu rõ nguyên nhân data).

### 8.3. Phân tích lỗi bắt buộc

1. **Tách crop "ổn định" và "gần ranh giới"** — nếu crop 5 s chứa 2+ chord khác nhau trong timeline thì coi là gần ranh giới. Nếu lỗi tập trung ở ranh giới, vấn đề nằm ở **nhiễu nhãn**, không phải ở model.
2. **Nhầm lẫn quen thuộc:** min7 ↔ maj6, dom7 ↔ 9, sus2 ↔ sus4 (cùng nốt, khác root: Csus2 = Gsus4) và sus ↔ add9, dim ↔ hdim (cùng một số nốt).
3. **Theo tông:** accuracy từng root; nếu một số tông yếu → tăng pitch-shift hoặc kiểm tra thiên lệch data.
4. **Theo nguồn:** so JAAH vs Hooktheory vs piano; nếu piano kém xa → ưu tiên thu/seed thêm piano (§10).
5. **Xem 50 lỗi cao tin cậy nhất** bằng tay (model chắc chắn nhưng sai) — thường là **nhãn sai** chứ không phải model sai; ghi lại để sửa/lọc data.

---

## 9. Bảng thí nghiệm (cho báo cáo và để chọn thiết kế)

> Mỗi hàng là một lần chạy riêng; ghi vào bảng kết quả `results.csv` kèm `config.yaml`.

| # | Câu hỏi | So sánh | Đo bằng |
|---|---|---|---|
| E1 | Có cần model sâu không? | chroma + logreg (B0) vs MERT đóng băng (B2) vs MERT + LoRA | accuracy 3 head |
| E2 | Tầng nào tốt? | linear probe từng tầng 0..24 vs tổng có trọng số | accuracy root/quality/ext |
| E3 | Pooling | P1 đều vs P2 trung tâm vs P3 attention | accuracy, lỗi ở ranh giới |
| E4 | Head nối tầng hay song song? | cascaded (detach) vs cascaded (không detach) vs parallel | extension accuracy, Chord accuracy |
| E5 | Cân bằng lớp | S0 vs S1 vs S2 vs S3 | macro-F1 extension, recall lớp hiếm |
| E6 | Augment | không / chỉ pitch-shift / đầy đủ | accuracy root + lớp hiếm |
| E7 | LoRA rank | 8 vs 16 vs 32 (và thêm `out_proj`) | accuracy so với số tham số |
| E8 | Cỡ model | MERT-330M vs MERT-95M vs cắt tầng | accuracy so với tốc độ |
| E9 | Tác động nhãn `partial` | train có/không có partial | accuracy quality |
| E10 | Frame-level vs segment-level | segment-level (midpoint label) vs frame-level (per-MERT-frame label từ timeline) | accuracy + chord accuracy |

Hướng tiếp cận E1–E3 trước (rẻ, dùng cache embedding), E4–E7 sau.

---

## 10. Data bổ sung (không bắt buộc trước khi train)

| Việc | Lợi ích | Trạng thái |
|---|---|---|
| ~~Crawl hết Hooktheory~~ | ~~tăng hợp âm hiếm~~ | **✅ Hoàn tất** (2026-11-04). 6.222 bài, 1.434 failed (video unavailable/alignment rejected) |
| Seed `kaggle/archive1` (1.760 file piano triad) | domain piano cho root/quality, làm tập kiểm tra piano | chưa seed; đã có trên ổ |
| Thu piano thật (có nhãn) để fine-tune cuối | **giải lệch domain** (quan trọng nhất cho app piano) | cao; nằm ngoài DE hiện tại |
| Synth piano (Pianoteq/FluidSynth) từ chuỗi hợp âm | thêm piano với nhãn hoàn hảo, cân bằng lớp hiếm | trung bình |

---

## 11. Quyết định đang mở (cần bạn chọn)

| # | Vấn đề | Lựa chọn | Đề xuất |
|---|---|---|---|
| 1 | Model không có lớp "không hợp âm" (`N`) | (a) cổng RMS + ngưỡng tin cậy khi inference · (b) thêm lớp `N` hoặc head nhị phân "có hợp âm" (1.787 segment N sẵn có) | **Đã chốt 2026-10-23: (a).** (b) chỉ khi (a) cho nhiều báo động giả. Chi tiết: `AI/markdown/ai_checklist.md` mục E.3 |
| 2 | `dim7` gộp vào `dim + dom7` | giữ quy ước hiện tại · tách lớp riêng | giữ (đổi taxonomy ảnh hưởng toàn bộ tài liệu/DB) |
| 3 | Dùng `partial` để train | có (mask) · không | có (mask) |
| 4 | Nơi đặt code train | **thư mục `AI/` riêng ở gốc repo** (đã chốt) | `AI/` + 1 notebook Colab gọi code đó |
| 5 | Công cụ theo dõi thí nghiệm | không cần công cụ ngoài: ghi `results.csv` + `config.yaml` mỗi lần chạy; TensorBoard/W&B chỉ là tùy chọn | `results.csv` là đủ cho dự án này |

---

## 12. Rủi ro và cách giảm

| Rủi ro | Dấu hiệu | Giảm |
|---|---|---|
| Lớp hiếm quá ít | recall val ≈ 0 ở b13/b5/#9 | chờ thêm data v2; báo cáo trung thực; có thể gộp lớp (vd. b5 vào alt) nếu thật sự không học nổi |
| Quá khớp bài | train ≫ val | giảm LR LoRA, tăng dropout, early stopping, weight decay, augment |
| Lệch domain (piano) | piano kém xa JAAH | seed Kaggle piano, thu piano thật, fine-tune pha 3 |
| Nhãn lệch canh (Hooktheory) | lỗi cao tin cậy nhiều ở một nguồn | kiểm tra tay, lọc clip, cân nhắc kiểm tra căn chỉnh theo beat |
| Rò rỉ data | val cao bất thường | assert không trùng `song_id`; chia theo nhóm bài |
| Colab ngắt phiên | mất tiến độ | checkpoint mỗi epoch lên Drive + resume |
| Hết VRAM | OOM | batch nhỏ hơn, gradient checkpointing, fp16, bỏ augment nặng |
| Lỗi nhãn do pitch-shift | accuracy root tụt khi bật augment | unit test hàm đổi nhãn; kiểm tra từng ví dụ |
| Deploy chậm | > 500 ms | thang giảm kích thước ở S5 |

---

## 13. Cấu trúc code đề xuất (chưa tạo, đặt trong thư mục `AI/`)

```
AI/
├── config.yaml                # toàn bộ siêu tham số của một lần chạy
├── dataset.py                 # đọc per-song .npy + chords.parquet, random crop 5s, gán nhãn từ timeline
├── augment.py                 # pitch-shift + đổi nhãn root (có unit test), gain, nhiễu
├── model.py                   # MERT + gộp tầng + pooling + 3 head nối tầng
├── losses.py                  # CE có ignore_index, Focal Loss, trọng số lớp
├── extract_embeddings.py      # cache embedding đóng băng cho B1/B2
├── train.py                   # vòng train, checkpoint/resume, ghi log
├── evaluate.py                # chỉ số, ma trận nhầm lẫn, bootstrap theo bài, lỗi ở ranh giới
├── infer.py                   # cửa sổ trượt + Viterbi + ngưỡng
├── export_onnx.py             # xuất ONNX + đo tốc độ
└── notebooks/colab_train.ipynb
tests/unit/training/           # test: đổi nhãn pitch-shift, mask loss, shape model, crop + label lookup
```

---

## 14. Danh sách kiểm tra trước khi bắt đầu train

> Checklist đầy đủ hơn (chuẩn hóa âm lượng, lọc nhiễu, mask tổ hợp, giao diện với BE/FE): [`AI/markdown/ai_checklist.md`](ai_checklist.md).

- [x] Export data mới — **đã export** tại `DE/data/export/v2/` (2026-11-04). `classes.json` khớp bảng lớp.
- [ ] Assert: không `song_id` ở hai split; số file .npy = số bài trong parquet; không file toàn 0.
- [ ] Nghe thử và đối chiếu chord timeline một số bài mỗi nguồn.
- [ ] Hàm đổi nhãn pitch-shift có unit test.
- [ ] Loss không `nan` khi cả batch thiếu nhãn một head.
- [ ] Chạy một batch MERT qua model, shape `(B, 375, 1024)` và 25 tầng.
- [ ] Có cơ chế resume checkpoint, lưu config + seed.
- [ ] Đã quyết định các mục §11.

---

## 15. Thuật ngữ

| Từ | Nghĩa |
|---|---|
| Section / bài | đoạn audio whole section (chiều dài biến thiên), đơn vị cấp cao nhất |
| Crop | đoạn audio 5 s random crop từ section, đơn vị mẫu khi train |
| Chord timeline | chuỗi (start_ms, end_ms, label) cho mỗi bài — nhãn ground truth |
| Head | lớp phân loại nhỏ gắn trên vector của MERT |
| Embedding | vector số biểu diễn đoạn audio |
| LoRA | kỹ thuật train thêm vài ma trận nhỏ thay vì cả model |
| Focal Loss | loss nhấn mạnh mẫu khó/hiếm |
| Mask / ignore_index | bỏ một nhãn thiếu ra khỏi tính loss |
| Pitch-shift | dịch cao độ audio (kèm đổi nhãn root) |
| Macro-F1 | trung bình F1 của các lớp, không bị lớp đông lấn át |
| Bootstrap theo bài | lấy mẫu lại theo bài để ước khoảng tin cậy |
| HMM / Viterbi | làm mượt chuỗi dự đoán theo thời gian |
| Distillation | dạy model nhỏ bắt chước model lớn |
| ONNX | định dạng model để chạy nhanh/đa nền tảng |
