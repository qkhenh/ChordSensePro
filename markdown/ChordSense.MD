Dưới đây là bản mô tả chi tiết dự án được cấu trúc lại hoàn toàn theo định dạng của file tài liệu mẫu, tập trung chuyên sâu vào các tính năng dành cho nhạc công và phân định rõ ràng trách nhiệm BE/FE của bạn.

---

# Đề tài: ChordSense Pro: An AI-Powered Context-Aware System for Musician's Interactive Practice Workspace with Harmonic Analysis Support

> **Mục tiêu:** Xây dựng một Web App end-to-end chuyên biệt cho nhạc công, tích hợp AI nhận diện hợp âm (ACR) với Động cơ phân tích nhạc lý ngữ cảnh (Harmonic Engine) và Không gian luyện tập tương tác.
> **Bối cảnh:** Các giải pháp nhận diện hợp âm hiện tại trên thị trường (ChordAI, Chordify) đang thiếu hụt tư duy nhạc lý của một nhạc công thực thụ (không nhận diện được chức năng hòa âm, âm giai) và làm đứt gãy luồng luyện tập thực hành (thiếu công cụ lặp đoạn, đếm nhịp động, tùy biến hợp âm cá nhân).
> **Định vị:** Nhận diện hợp âm chỉ là lớp dữ liệu thô. Giá trị cốt lõi của ChordSense Pro nằm ở khả năng **biên dịch dữ liệu thô đó thành kiến thức nhạc lý** và **cung cấp môi trường thực hành khép kín**.

---

## 0. Tổng Quan Dự Án

### Tên đồ án: **ChordSense Pro**
> *"Phân tích sâu — Luyện tập chuẩn — Thành thạo toàn diện."*

### Phân chia nhân lực

| Role | Người | Trách nhiệm chính |
| --- | --- | --- |
| **DE + AI** | Bạn (Team member) | Audio Feature Extraction, AI Chord Recognition Model, Source Separation, Beat Tracking pipeline.

 |
| **BE + FE** | Bạn | Trình phát nhạc đồng bộ (Web Audio API), Client-side Harmonic Engine, UI/UX Workspace, REST API, Database (PostgreSQL/Prisma), Deployment.

 |

---

## 1. Ý Tưởng Cốt Lõi & Điểm Khác Biệt

### So Sánh: ChordSense Pro vs ChordAI / Song Master

| Tính năng | ChordAI | Song Master Pro | **ChordSense Pro** |
| --- | --- | --- | --- |
| **Hiển thị Basic/Precise** | ✅ Có | ❌ Không | **✅ Xử lý real-time tại Client** |
| **Phân tích Bậc (Roman Numeral)** | ❌ Không | ❌ Không | **✅ Tự động theo Key gốc** |
| **Gợi ý Âm giai (Scale Mapping)** | ❌ Không | ❌ Không | **✅ Có (Ionian, Mixolydian...)** |
| **Smart Grid & Sync Player** | ⚠️ Cơ bản | ✅ Có | **✅ Click-to-seek, sai số < 100ms** |
| **Chỉnh sửa / Lưu trữ cá nhân** | ❌ Không | ✅ Có (Local) | **✅ Cloud Sync + Edit Marker** |
| **Bách khoa hợp âm & Voicing** | ✅ Có | ✅ Có | **✅ Tích hợp Audio Synth phát âm** |

---

## 2. Kiến Trúc Luồng Dữ Liệu (Data Flow)

Sự phân tách rõ ràng giữa phân tích AI (Server) và logic nhạc lý (Client) để triệt tiêu độ trễ:

```text
[BƯỚC 1: NHẬP LIỆU]
User nạp YouTube URL / MP3 file 
       ↓
[BƯỚC 2: AI ENGINE - BE/DE]
Tách nền → Trích xuất Tempo/Key → Detect Chord Events
       ↓
(Trả về JSON: Metadata + Event Timeline tuyệt đối)
       ↓
[BƯỚC 3: HARMONIC ENGINE - FE]
Client tiếp nhận → Map Root với Key → Suy ra Bậc & Scale
Render ra Smart Grid View & Timeline
       ↓
[BƯỚC 4: TƯƠNG TÁC LUYỆN TẬP - FE]
User bật Metronome / Lặp đoạn (A/B Loop) / Giảm tốc độ (Speed Shift)
       ↓
[BƯỚC 5: CÁ NHÂN HÓA - FE/BE]
User sửa "Em7/C" thành "Cmaj9" trên lưới
FE tự động tính toán lại Scale/Bậc ngay lập tức
User lưu bản Custom Chart vào Playlist (BE lưu DB)

```

---

## 3. Đặc Tả Tính Năng Chi Tiết (Nhiệm vụ BE & FE)

### Module 1: Xử lý Nguồn Audio & Phân tích Đa lớp (Audio Ingestion)

* **1.1 Nhập liệu Đa nguồn:**
* **FE:** Xây dựng giao diện upload file (MP3, WAV, M4A) và input URL YouTube. Quản lý state thanh tiến trình thông qua cơ chế Polling hoặc WebSockets.


* **BE:** Tiếp nhận request, tạo job chạy ngầm xử lý tải luồng âm thanh YouTube (chống timeout/rate-limit).


* **1.2 Chế độ hiển thị Basic / Precise:**
* **FE:** Triển khai hàm Down-mapping trực tiếp tại máy khách. Khi người dùng gạt switch sang "Basic", FE tự động cắt các hậu tố mở rộng của hợp âm (VD: `G9sus4` → `G`) để cập nhật UI ngay lập tức mà không gọi API.



### Module 2: Không gian Tương tác & Trình phát (Interactive Workspace)

* **2.1 Lưới Hợp Âm Thông Minh (Smart Grid View):**
* **FE:** Đọc `tempo` và `first_beat_offset_ms` từ DB để nội suy và render giao diện lưới theo từng ô nhịp (Bar). Sử dụng `requestAnimationFrame` để highlight ô nhịp hiện tại đồng bộ với thời gian thực của Audio Player, triệt tiêu tình trạng nghẽn Virtual DOM.


* **2.2 Đồng bộ Tương tác (Click-to-Seek & A/B Looping):**
* **FE:** Gắn event listener trên Grid. Khi người dùng nhấp vào ô nhạc, gọi API của Player để tua đến chính xác mili-giây của hợp âm đó. Cho phép quét chọn nhóm ô nhịp để tạo vòng lặp (Loop) phục vụ luyện tập ngón đàn.


* **2.3 Metronome & Speed Shifting:**
* **FE:** Khởi tạo Web Audio API để phát âm click metronome động dựa trên dữ liệu BPM. Hỗ trợ điều chỉnh tốc độ phát (0.5x, 0.75x) mà không làm méo cao độ.



### Module 3: Động cơ Nhạc lý & Tra cứu (Harmonic Engine & Dictionary)

* **3.1 Client-side Harmonic Engine (Phân tích Ngữ cảnh):**
* **FE:** Xây dựng Rule-based Engine tĩnh. Khi render hợp âm, tự động lấy Root của hợp âm trừ đi Key của bài hát để xác định chức năng hòa âm (Roman numerals: I, ii, V, v.v.). Map kết quả để hiển thị Âm giai ngẫu hứng (Scale: Ionian, Dorian, Mixolydian...).


* **3.2 Ultimate Chord Dictionary:**
* **FE:** Render giao diện mặt đàn Guitar (Fretboard) và phím Piano. Tích hợp Soundfont/MIDI Synth trực tiếp trên trình duyệt để khi người dùng click vào một Voicing, hệ thống sẽ phát ra âm thanh của thế bấm đó.
* **BE:** Cung cấp endpoint GET truy xuất dữ liệu tĩnh về cấu trúc hàng ngàn thế bấm ngón.



### Module 4: Cá nhân hóa & Lưu trữ (User UGC)

* **4.1 Chỉnh sửa Hợp âm (Edit Marker):**
* **FE:** Xử lý logic ghi đè (Override). Mở Modal cho phép người dùng thay đổi Root, Bass, Quality, hoặc Voicing tại một thời điểm. Cập nhật State nội bộ để Harmonic Engine (Module 3.1) tính lại Bậc và Scale ngay lập tức.
* **BE:** Cung cấp API `PATCH` chỉ nhận các "điểm chạm" (những hợp âm bị thay đổi), gắn cờ `is_user_edited` để phân biệt với kết quả gốc của AI.


* **4.2 Quản lý Thư viện (Playlist Management):**
* **BE:** Xây dựng logic CRUD quản lý Playlist và các bản Custom Chart của từng tài khoản.



---

## 4. Thiết Kế Cơ Sở Dữ Liệu (PostgreSQL / Prisma)

> **Rủi ro kỹ thuật:** Tuyệt đối không lưu cấu trúc bài hát theo dạng "Lưới (Grid)" hoặc "Ô nhịp (Bar)" cứng, vì khi sửa đổi Tempo, cấu trúc này sẽ vỡ.
> **Giải pháp:** Lưu trữ theo dạng sự kiện dòng thời gian tuyệt đối (Event-based Timeline) sử dụng kiểu dữ liệu `JSONB` trong PostgreSQL để tối ưu tốc độ truy xuất Read-heavy.

```sql
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(200),
    email           VARCHAR(200) UNIQUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Bảng lưu trữ Bài hát gốc (AI Generate) và Bản độ của User
CREATE TABLE song_charts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id),
    source_url      TEXT,                  -- YouTube URL hoặc null
    is_customized   BOOLEAN DEFAULT FALSE, -- Phân biệt bản gốc AI và bản User sửa
    metadata        JSONB,                 -- Lưu Key, Tempo, Time Signature, Offset
    chord_events    JSONB,                 -- Mảng sự kiện: [{timestamp_ms, root, bass, quality...}]
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Quản lý thư viện cá nhân
CREATE TABLE playlists (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id),
    name            VARCHAR(200) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE playlist_items (
    playlist_id     UUID REFERENCES playlists(id),
    song_chart_id   UUID REFERENCES song_charts(id),
    order_index     INT,
    PRIMARY KEY (playlist_id, song_chart_id)
);

-- Kho dữ liệu tra cứu tĩnh (Phục vụ Module 3.2)
CREATE TABLE chord_dictionary (
    id              VARCHAR(50) PRIMARY KEY, -- VD: 'cmaj9_piano_root'
    root            VARCHAR(5),
    quality         VARCHAR(20),
    instrument      VARCHAR(20),
    fingering       JSONB,
    audio_midi      JSONB
);

```

---

## 5. Cấu Trúc API Endpoints (BE Thực Hiện)

```text
# --- LUỒNG AUDIO & PHÂN TÍCH ---
POST   /api/v1/analyze/url         # Gửi link YouTube → Trả về Job ID xử lý ngầm
POST   /api/v1/analyze/file        # Upload file local → Trả về Job ID
GET    /api/v1/analyze/status/{id} # Polling trạng thái xử lý của AI

# --- LUỒNG DỮ LIỆU BÀI HÁT (CHARTS) ---
GET    /api/v1/charts/{id}         # Lấy toàn bộ JSONB của một bài hát để render Grid
PATCH  /api/v1/charts/{id}/events  # Sửa đổi/Thêm/Xóa hợp âm (Update vào mảng chord_events)

# --- LUỒNG PLAYLIST & UGC ---
POST   /api/v1/playlists           # Tạo playlist mới
GET    /api/v1/playlists           # Lấy danh sách thư viện của User
POST   /api/v1/playlists/{id}/add  # Thêm một song_chart vào playlist

# --- KHO TRA CỨU ---
GET    /api/v1/dictionary/search?root=C&quality=maj9

```