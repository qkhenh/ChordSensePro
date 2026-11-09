# TODO: Cập nhật báo cáo sau khi có kết quả training

> **Ngày tạo:** 2026-11-09  
> **Cập nhật:** 2026-11-09 — Phase 2 LoRA xong, đã update reports

---

## Các thay đổi đã sửa trong `20_DE_training_plan.md`

### §0 — Tóm tắt (dòng ~15–25)
- [x] Cập nhật "Pitch-shift kèm đổi nhãn root" → **không dùng pitch-shift**, data analysis cho thấy root balanced 2.3×
- [x] Thêm kết quả Phase 1 + Phase 2 vào tóm tắt (header dòng 5)
- [x] Cập nhật môi trường: T4 → L4 (Colab Pro)

### §2.5 — Pitch-shift kèm đổi nhãn
- [x] Thêm NOTE box: kết quả thực tế, root balanced, pitch-shift tắt, CPU quá chậm

### §6 — Chống lệch lớp và augmentation (dòng ~303–320)
- [x] Thêm bảng phân bố data thực tế (root 2.3×, quality 37×, ext 1533×)
- [x] Sửa bảng augmentation: kế hoạch vs thực tế, tất cả ❌ Tắt
- [x] Thêm NOTE: code augment vẫn có, có thể bật lại
- [x] Giữ nguyên Focal Loss + weighted sampler (đang dùng đúng)

### §7/S2 — Các pha huấn luyện (dòng ~391–398)
- [x] "augment bật" → **augment tắt**
- [x] Thêm bảng kết quả thực tế Phase 1 + Phase 2

### §8.3 — Phân tích lỗi
- [x] "tăng pitch-shift" → sửa: root balanced, kiểm tra nguyên nhân khác

### §9 — Thí nghiệm
- [x] E6 (Augment): thêm kết luận — **không dùng**, root balanced, CPU chậm

### §12 — Rủi ro
- [x] "Lỗi nhãn do pitch-shift" → **không áp dụng** (augment tắt)

### §13 — Cấu trúc code
- [x] "chưa tạo" → **đã tạo**, cập nhật cấu trúc thực tế

### §14 — Checklist
- [x] Tick các mục đã hoàn thành

---

## Các thay đổi đã sửa trong `ai_checklist.md`

- [x] §C Train: Pitch-shift → **SKIP** với lý do
- [x] §C Train: Tick ✅ các task đã xong (MERT batch, resume, results.csv, eval theo bài)

---

## Đã thêm mới

- [x] `change_log.md`: Entry Phase 1 + Phase 2 results, ONNX export, augment decision
- [ ] Bảng so sánh Phase 1 vs Phase 2 chi tiết (đã có trong training_plan §7/S2)
- [x] Link đến `decision_crop_2s_to_5s.md` (đã có trong §2.5)

---

## Còn lại (sau deploy)

- [ ] Update `DE/markdown/18_DE_AI_chordsense.md` nếu có nhắc AI augmentation
- [ ] Update `markdown/18_main_project_edtech_music.md` nếu có section AI
- [ ] Thêm kết quả evaluate trên test set (chưa chạy test)
- [ ] Thêm ONNX Phase 2 benchmark
