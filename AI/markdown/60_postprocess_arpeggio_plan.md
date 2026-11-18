# 60 — Post-Processing & Musical Temporal Decoding Plan (Fix Arpeggio & Micro-Fragmentation)

**Ngày lập:** 2026-11-11  
**Trạng thái:** 🟢 Đã triển khai trong `AI/src/serve.py` & `AI/test_v3.py` (**52/52 unit tests pass**, 0 giờ retrain)

---

## 1. Hiện Tượng Thực Tế (Case Study: `D:\Untitled.wav`)

Khi chạy `model_v4_5s_p2.onnx` + `htdemucs_ft` trên `D:\Untitled.wav` (54.5s, giọng Rê thứ `D minor`):

- **Điểm mạnh (Model core đã tốt):** Các hợp âm ở **đầu ô nhịp / đầu phách** được nhận diện đúng vòng hòa thanh Jazz/Ballad qua cả 2 chu kỳ lặp (`1.4s–18.8s` và `18.8s–39.1s`):
  `Dm7 → Cdim7 → F7/C → F7 → Bbmaj7 (A#maj7) → Am → Abdim (G#dim) → Gm7 → Edim7 / E7 → A7#5 (Aaug) → Dm7` (confidence `0.75–0.94`).
- **Điểm lỗi nghiêm trọng lúc xuất kết quả:**
  1. **Băm vụn lý nhí tới `0.01s` (1 frame = 13.3ms):** Xuất hiện hàng chục dòng dài `0.01s–0.08s` (`2.88s–2.89s Dm7`, `4.83s–4.84s Adim7/C`, `16.75s–16.80s Adim → A7 → Aaug → A`).
  2. **Cùng một hợp âm bị cắt làm 2–3 dòng liên tiếp:** Ví dụ `Am` (`8.37–8.59`, `8.59–9.00`, `9.00–9.19`) và `Gm` (`10.52–10.61`, `10.61–12.19`).
  3. **Nốt đánh tỉa / rải dây (arpeggio & melodic ornaments) làm nhảy hợp âm:**
     - Nhảy Extension liên tục trên cùng hợp âm nền: `Dm → Dm7 → Dm9 → Dm → Dm7 → Dm9 → Dm` (`1.41s–4.08s`), `A#maj9 → A#maj7 → A#maj9 → A#maj13 → A#maj7` (`6.24s–8.36s`).
     - Lật gốc trên hợp âm đối xứng `dim7`: `Cdim7 → Adim7/C → Fdim7/C → Cdim7` (`4.08s–5.20s`).
     - Giật sang hợp âm lạ có confidence thấp (`0.41–0.55`) khi chỉ có 1–2 nốt rải đơn lẻ vang lên cuối phách (`Dm7b13/A`, `Am7b13`).

---

## 2. Định Nghĩa 4 Nguyên Nhân Gốc Rễ (Root Causes)

### 🔴 RC1 — Bug thuật toán trong `_smooth_jitter()` (`AI/src/serve.py` L798)
- **Code hiện tại:** `_chord_key(out[i - 1]) == _chord_key(out[run_end + 1])` chỉ xóa nhiễu nếu hợp âm trước và sau **giống hệt nhau** (`A → B → A`).
- **Hệ quả:** Mọi chuỗi chuyển tiếp bậc thang (`A → B(1 frame) → C(1 frame) → D`) đều lọt lưới 100%, khiến các đoạn `0.01s` xuất thẳng ra kết quả cuối.

### 🔴 RC2 — Bug chưa chuẩn hóa `bass_id == root_id` về `12` (`none`)
- **Code hiện tại:** Khi người chơi gảy dây bass trùng nốt gốc (ví dụ nốt `A` trong `Am`, `root_id = 9`), có frame `head_bass` đoán `12` (`none`), có frame đoán `9` (`A`).
- **Hệ quả:** `chord_name()` đều trả về `"Am"`, nhưng `_chord_key()` và `_merge_consecutive()` lại so sánh `bass_id` thô (`9 != 12`), làm hợp âm `Am` và `Gm` bị xẻ thành nhiều đoạn rời nhau.

### 🟠 RC3 — Giải mã Greedy Argmax độc lập từng 13.3ms (Thiếu quán tính hòa thanh)
- Ở **đầu phách (onset)**, nhạc công chập đủ Bass + Triad nên MERT nhìn thấy toàn bộ hợp âm (`conf` rất cao).
- Ở **giữa/cuối phách**, tiếng chập đầu phách suy giảm, nhạc công tỉa từng dây đơn lẻ (bậc 9, bậc 13, nốt lướt). Vì `decode_frame_probs()` lấy `argmax` độc lập cho từng frame 13.3ms mà **không có chi phí phạt chuyển trạng thái (transition penalty)**, chỉ 1 nốt tỉa vang lên cũng lật được nhãn của vài mươi mili-giây đó.
- Ngoài ra, hợp âm `dim7` gồm 4 nốt cách đều 3 bán âm (`C - Eb - F# - A`), nên `Cdim7`, `Ebdim7/C`, `F#dim7/C`, `Adim7/C` có cấu trúc âm học trùng khít — tỉa trúng nốt nào to hơn là `argmax` đổi Root sang nốt đó.

### 🟡 RC4 — Không bám lưới Phách/Nhịp (Beat Grid) & Chưa chuẩn hóa Enharmonic theo Giọng
- Trong âm nhạc thực tế, hợp âm thay đổi theo **lưới phách (beat / half-beat, ~300ms–700ms)** chứ không đổi ngẫu nhiên ở mili-giây lẻ.
- Bảng `ROOT_NAMES` hiện cố định dấu thăng (`#`), nên bài giọng Rê thứ (`Dm`) bị hiển thị thành `A#maj7`, `G#dim` thay vì `Bbmaj7`, `Abdim`.

---

## 3. Kế Hoạch Sửa 4 Tầng Trong `AI/src/serve.py` (Zero-Retrain)

### Phase 1 — Chuẩn hóa Nhạc lý & Lọc nhiễu Xác suất ở mức Frame (`decode_frame_probs`)
1. **Gộp xác suất `bass == root` vào `none` (`bass_id = 12`):**
   - Với mỗi frame có `r = root_id`, cộng `bass_prob[t, r]` vào `bass_prob[t, 12]` rồi đặt `bass_prob[t, r] = 0.0`. Triệt tiêu hoàn toàn lỗi tách đôi `Am` / `Gm`.
2. **Cổng xác suất cho Slash Chord (`SLASH_BASS_THRESHOLD = 0.50`):**
   - Chỉ gán `bass_id != 12` khi `max_slash_prob >= 0.50` (và lớn hơn `bass_prob[12]`); ngược lại giữ hợp âm gốc `bass_id = 12`.
3. **Cổng xác suất cho Ultra-Rare Extensions & Chuẩn hóa `dim7`:**
   - Các đuôi biến đổi hiếm (`b5, #5, b9, #9, 11, #11, 13, b13, alt`) chỉ được chọn khi xác suất sau mask `>= 0.50`; nếu thấp hơn, ưu tiên rơi về nhóm nền (`none, maj7, m7, 7, add9`).
   - Với hợp âm `dim7` có slash bass (`quality == dim` và `ext == dom7/dim7`), nếu nốt bass thuộc cùng trục đối xứng 3 bán âm `{(root + 3k) % 12}`, chuẩn hóa `root_id = bass_id, bass_id = 12` (ví dụ `Adim7/C` và `F#dim7/C` tự động quy về `Cdim7`).

### Phase 2 — Giải mã Sticky Viterbi (HMM) thay thế Greedy Argmax
1. **Sticky Transition Matrix trên không gian `(root, quality)` (72 trạng thái) bằng NumPy vectorization (< 5ms/bài):**
   - Xác suất giữ nguyên trạng thái `p_stay = 0.97` (tương đương phạt `log_penalty` khi nhảy hợp âm). Muốn đổi sang hợp âm mới, tín hiệu hòa thanh mới phải đủ mạnh và duy trì ổn định qua nhiều frames liên tiếp (chống tuyệt đối tiếng tỉa 1–2 nốt ngắn).
2. **Ổn định Extension trong cùng 1 đoạn Hòa thanh (`Harmonic Core Consolidation`):**
   - Sau khi xác định biên `(root_id, quality_id, bass_id)` ổn định, các dao động tỉa màu bên trong cùng một hợp âm (`Dm ↔ Dm7 ↔ Dm9` hoặc `Bbmaj7 ↔ Bbmaj9 ↔ Bbmaj13`) được gộp theo trọng số `sum(duration * ext_prob)` để chọn ra **1 nhãn đại diện sạch nhất** cho cả đoạn giữ tay đó (có cờ `stabilize_extensions: bool = True` để bật/tắt linh hoạt).

### Phase 3 — Beat-Synchronized Aggregation (Bám lưới Phách bằng `librosa`)
1. Thêm chế độ `beat_sync: bool = True` (mặc định bật khi chạy bài nhạc thật, tự động fallback nếu clip quá ngắn `< 4s` hoặc không bắt được beat):
   - Dùng `librosa.beat.beat_track` lấy danh sách mốc beat `beat_times`, chia đôi thành lưới **half-beat (nốt móc đơn / nửa phách)** cho phép bắt kịp cả passing chords.
   - Trong mỗi ô phách, tính trung bình xác suất có **trọng số ưu tiên đầu phách (exponential onset decay)** — nơi hợp âm được đánh đầy đủ nhất — rồi mới đưa qua Sticky Viterbi.

### Phase 4 — Cắt tỉa Đoạn Ngắn Hội Tụ (`_prune_short_segments`) & Enharmonic theo Giọng
1. **Thay thế `_smooth_jitter` bằng `_prune_short_segments(segments, min_duration_ms=320.0)`:**
   - Lặp cho tới khi **không còn bất kỳ đoạn nào ngắn hơn `min_duration_ms`** (mặc định `320ms`): tìm đoạn ngắn nhất `< 320ms`, so sánh với đoạn kề trái và kề phải (ưu tiên gộp vào láng giềng cùng `root_id`, nếu khác `root_id` thì gộp vào bên có `duration * confidence` lớn hơn), sau đó `_merge_consecutive`.
   - Xử lý triệt để cả mẫu `A → B → A` lẫn bậc thang `A → B → C`.
2. **Key-Aware Enharmonic Spelling (Tùy chọn hiển thị):**
   - Ước lượng giọng bài hát từ tổng thời lượng 12 Root/Triad (Krumhansl profile). Nếu bài thuộc họ giọng giáng (`F, Dm, Bb, Gm, Eb, Cm, Ab, Fm`), đổi bảng hiển thị sang `["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]` → hiển thị chuẩn `Bbmaj7`, `Abdim`.

---

## 4. Kế Hoạch Kiểm Chứng (Verification)

1. **Unit Tests (`AI/tests/test_serve.py`):**
   - Test `bass_id == root_id` tự động gộp thành 1 segment duy nhất (`Am` không bị xẻ 3 khúc).
   - Test nhiễu bậc thang `A (1.0s) → B (0.04s) → C (0.04s) → D (1.0s)` bị loại bỏ 100%, không còn bất kỳ segment nào `< min_duration_ms`.
   - Test dao động tỉa nốt `Dm → Dm7 → Dm9 → Dm` trong 2 giây hội tụ về 1 hợp âm duy nhất.
   - Test `Adim7/C` chuẩn hóa về `Cdim7`.
2. **A/B Test thực tế trên `D:\Untitled.wav` và `D:\full.mp3`:**
   - Kỳ vọng trên `D:\Untitled.wav`: Từ ~100 dòng lý nhí rút gọn xuống còn **~22–30 hợp âm sạch sẽ** đúng nhịp phách (`Dm7 → Cdim7 → F7/C → F7 → Bbmaj7 → Am → Abdim → Gm7 → Edim7 → E7 → A7#5 → Dm7`), giữ nguyên độ chính xác ở đầu ô nhịp và không bị nhảy loạn khi tỉa dây.

