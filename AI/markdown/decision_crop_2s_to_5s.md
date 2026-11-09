# Quyết định: Đổi crop từ 2s sang 5s

> **Ngày:** 2026-11-09  
> **Người:** Đặng Quốc Khánh  
> **Trạng thái:** Đã áp dụng (pipeline v2)

---

## Bối cảnh

Pipeline v1 cắt audio thành **segment cố định 2s**, mỗi segment gán đúng 1 nhãn chord. Khi chuyển sang v2 (lưu whole-song audio + chord timeline), cần chọn lại độ dài crop.

## Lý do đổi sang 5s

### 1. MERT cần đủ context để biểu diễn tốt

MERT-v1-330M xử lý ở tốc độ **75 frames/giây**:
- 2s → 150 frames
- 5s → 375 frames

MERT được pre-train trên đoạn audio dài (5–30s). Crop 2s quá ngắn, các layer sâu của transformer **không đủ context** để phân biệt chord — ví dụ phân biệt `Cmaj7` và `C` cần nghe sustain của nốt B, thường kéo dài 1–2s sau khi đánh.

### 2. Crop 2s dễ rơi vào ranh giới chuyển chord

Một bài trung bình có chord dài 1–4s. Với crop 2s:
- Xác suất crop **trùng ranh giới** 2 chord rất cao
- Midpoint (1s) nằm gần edge → nhãn không ổn định
- Model nhận input lẫn 2 chord nhưng chỉ được gán 1 nhãn → **noise trong label**

Với crop 5s:
- Midpoint ở 2.5s → nằm sâu trong chord chính
- Dùng **center pooling** (25%–75% = frame 94–281) → chỉ lấy phần giữa, bỏ transition ở 2 đầu

### 3. Ngữ cảnh hòa âm giúp phân biệt chord

Nhiều chord có âm thanh tương tự nếu chỉ nghe 2s (ví dụ `Am` vs `C/E`). Khi nghe 5s, model có thêm thông tin từ **chord trước và sau** (harmonic context) để phân biệt.

### 4. Phù hợp với cách MERT xử lý feature

MERT dùng CNN feature extractor (kernel 128 samples, tương đương ~5ms ở 24kHz) rồi qua 24 transformer layers. Các transformer layer dùng **attention toàn bộ sequence** → sequence dài hơn cho phép attention pattern phong phú hơn.

Crop 2s (150 frames) cho mỗi head chỉ attend ~150 positions — quá ít cho 24-layer transformer. Crop 5s (375 frames) tận dụng tốt hơn capacity của model.

### 5. Không tăng VRAM đáng kể

| Crop | Frames | VRAM (batch=32, fp16) |
|------|--------|----------------------|
| 2s   | 150    | ~8 GB                |
| 5s   | 375    | ~12 GB               |

L4 (24GB) và T4 (16GB) đều chạy thoải mái với 5s.

## Trade-off

| | 2s | 5s |
|--|----|----|
| Tốc độ inference | Nhanh hơn ~2.5x | Chậm hơn nhưng vẫn real-time (26ms ONNX) |
| Độ phân giải thời gian | Cao (xác định chord chính xác ±1s) | Thấp hơn (±2.5s) — bù bằng sliding window |
| Chất lượng nhãn | Noisy (dễ trùng ranh giới) | Sạch hơn (midpoint xa ranh giới) |
| MERT representation | Yếu (ít context) | Mạnh (đủ context) |

## Kết luận

5s là **sweet spot** cho bài toán chord recognition với MERT: đủ dài để model có context, đủ ngắn để inference real-time, và khớp với cách MERT được pre-train. Khi deploy, dùng **sliding window 5s với step 1s** để đạt độ phân giải thời gian ~1s.

