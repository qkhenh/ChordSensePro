# TODO: Cập nhật báo cáo sau khi có kết quả training

> **Ngày tạo:** 2026-11-09  
> **Cập nhật khi:** Phase 2 LoRA xong

---

## Các thay đổi cần sửa trong `20_DE_training_plan.md`

### §0 — Tóm tắt (dòng ~15–25)
- [ ] Cập nhật "Pitch-shift kèm đổi nhãn root" → ghi rõ: đã phân tích data, root balanced (2.3x), **quyết định không dùng pitch-shift**
- [ ] Thêm kết quả Phase 1 + Phase 2 vào tóm tắt

### §6 — Chống lệch lớp và augmentation (dòng ~303–320)
- [ ] Thêm bảng phân bố data thực tế:
  - Root: balanced (max/min = 2.3x) → **không cần pitch-shift**
  - Quality: imbalance 37x (major vs aug) → xử lý bằng Focal Loss
  - Extension: imbalance 1533x (none vs b5) → xử lý bằng Focal Loss
- [ ] Dòng 309: pitch-shift online → ghi rõ: **đã tắt**, lý do: root đã balanced + tốn thời gian
- [ ] Dòng 316: "precompute trước" → thêm kết luận: **không cần precompute** vì root balanced
- [ ] Giữ nguyên Focal Loss + weighted sampler (đang dùng đúng)

### §8 — Các pha huấn luyện (dòng ~385–400)
- [ ] Dòng 393: "augment bật" → sửa thành: **augment tắt**, LoRA + Focal Loss đủ
- [ ] Thêm kết quả thực tế Phase 1 + Phase 2 (loss, accuracy từng head)

### §9 — Đánh giá (dòng ~430–460)
- [ ] Thêm bảng kết quả thực tế so với baseline
- [ ] Dòng 455: "tăng pitch-shift" → sửa: root đã balanced, không cần

### §10 — Thí nghiệm (dòng ~465–480)
- [ ] E6 (Augment): ghi kết luận: **không dùng augment**, lý do data analysis
- [ ] Thêm kết quả thí nghiệm thực tế (E1 baseline, E2 pooling, ...)

### §13 — Troubleshooting (dòng ~510–520)
- [ ] Dòng 516: "Lỗi nhãn do pitch-shift" → đánh dấu: **không áp dụng** (đã tắt augment)

---

## Các thay đổi cần sửa trong `ai_checklist.md`

- [ ] Cập nhật trạng thái các task Phase 1 → ✅
- [ ] Cập nhật trạng thái các task Phase 2 → ✅ (khi xong)
- [ ] Task augmentation → đánh dấu: **skip** (đã phân tích, không cần)
- [ ] Thêm kết quả metric vào checklist

---

## Cần thêm mới

- [ ] Bảng so sánh Phase 1 vs Phase 2 (loss, root_acc, qual_acc, ext_acc, thời gian train)
- [ ] Thêm link đến `decision_crop_2s_to_5s.md` vào report chính

---

## Lý do chưa sửa ngay

Đợi Phase 2 xong để có **kết quả thực tế đầy đủ**, update 1 lần cho chính xác — tránh sửa đi sửa lại.
