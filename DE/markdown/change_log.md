# ChordSense Pro — DE Change Log

> Nhật ký thay đổi phần **DE** (crawl + xử lý data để train AI). Mới nhất ở trên.
> Phạm vi: chỉ `DE/`. Không đụng BE/FE, không sửa báo cáo (`latex_last_project/`).

---

## 2026-10-23 — Sửa `display_name`, công cụ nghe audio, dọn repo

### `ChordLabel.display_name` bị trùng tên (đã sửa)
- Lỗi (BE/FE phát hiện): `Cmaj9` và `C9` cùng hiện `"C9"`. Rà đủ 5 × 15 tổ hợp: còn trùng ở maj/dominant với 9/11/13, `C7` (4 tổ hợp), `Cm7` (min7 vs dom7), `Csus7`, `C7b9/C7#9/C7b13/C7alt/Cadd9/Cmaj7/Caug/Cb5` (major vs dominant). **ID nhãn trong data luôn đúng** — chỉ chuỗi hiển thị sai.
- `value_objects.py`: thay logic ghép chuỗi bằng bảng tường minh `CHORD_SUFFIX[(quality, ext)]` (**41 tổ hợp hợp lệ / 90** sau khi tách sus2/sus4; lúc đầu là 38 / 75). Quy ước mới: major+9/11/13 → `maj9/maj11/maj13`; dominant giữ `9/11/13`; minor+maj7 → `m(maj7)`; sus xem mục "Tách sus2/sus4" bên dưới.
- Thêm `VALID_COMBOS`, `valid_combo_mask()` (mask `[quality_id][ext_id]` theo thứ tự enum — dùng khi inference để chặn tổ hợp head vô lý, vd major+min7) và `ChordLabel.is_valid`. Tổ hợp không hợp lệ hiển thị dạng `C[major+min7]` (nhìn thấy rõ, không giả làm hợp âm khác).
- Test: thêm 6+ test (`test_value_objects.py`): maj9≠9, bảng tên mẫu, **display_name đơn ánh trên mọi tổ hợp hợp lệ**, tổ hợp sai không giống tên hợp lệ, mask khớp `VALID_COMBOS`. Toàn bộ suite: 198 pass, 5 skip.
- Đối chiếu data: 214 / 47.283 hàng (0,45 %) có (quality, ext) ngoài `VALID_COMBOS` (lúc đó chưa tách sus). Đã xử lý: export tự loại (mục dưới).

### Tách `sus` thành `sus2` + `sus4` (quality 5 → 6 lớp)
- Lý do: một lớp `sus` gộp hai hợp âm khác nhau (Csus2 = C-D-G, Csus4 = C-F-G) và không hiển thị đúng tên. Người dùng chọn phương án tách (B); BE sẽ cập nhật theo sau khi commit.
- `ChordQuality`: major 0, minor 1, dominant 2, dim 3, **sus2 4, sus4 5**. Tên: sus2 → `sus2`, `maj7sus2`, `7sus2`; sus4 → `sus4`, `maj7sus4`, `7sus4`, `9sus4`, `13sus4`. `VALID_COMBOS` = 41 tổ hợp. Lưu ý Csus2 và Gsus4 cùng nốt (khác root) nên model dễ nhầm.
- `harte_mapper.py`: `sus` trần → sus4 (quy ước phổ biến); `sus2` → sus2; `sus4` → sus4; thêm shorthand `7sus4/7sus2`; danh sách quãng: có 4 → sus4, chỉ có 2 → sus2 (cả hai → sus4).
- Sửa tương ứng: `seed_local_datasets.py` (phạm vi hợp âm hiếm), `data_report.sql`, `listen_samples.py` (`--quality sus2|sus4`), tests (`test_harte_mapper.py`, `test_hooktheory.py`, `test_value_objects.py`). Suite: **208 pass, 5 skip**.
- **Đã remap DB** bằng `remap_labels.py`: 1.615 hàng đổi (không cần crawl lại vì `label_raw` vẫn giữ đủ). Sau remap: sus2 1.043, sus4 1.615 (ok + partial).
- **BE/FE cần cập nhật:** quality có 6 lớp (id 4 = sus2, 5 = sus4), tên hợp âm mới (`Csus2`, `Csus4`, `C7sus4`, `C9sus4`…).

### Export loại tổ hợp (quality, ext) không hợp lệ
- `export_dataset.py`: tự loại hàng ngoài `VALID_COMBOS` (hàng thiếu một head thì giữ). Cờ `--keep-invalid-combos`; `classes.json` thêm `valid_combo_mask` (6 × 15) và `dropped_invalid_combos`.
- Kết quả `--stats` thật (2026-10-23): loại **233 / 49.840 (0,47 %)** (dim+b9 61, sus4+b5 22, sus2+nat13 19, minor+#5 15…), loại thêm **1.149 segment im lặng (2,32 %)** → **48.458 segment có nhãn**: train 34.392 · val 10.458 · test 3.608; nguồn Hooktheory 34.887 · JAAH 13.189 · GuitarSet 334 · local 48; quality major 19.656 · minor 14.688 · dominant 9.833 · dim 1.715 · sus2 975 · sus4 1.551.

### Dọn `DE/scripts/`
- Xóa 10 file một lần dùng/trùng lặp: `check_annotations.py`, `check_choco.py`, `check_choco_ids.py`, `check_jaah_beats.py`, `check_jaah_parts.py`, `check_mongo.py`, `test_fixes.py`, `test_dag_imports.py` (debug lúc viết parser, không được tài liệu/code nào dùng), `reset_mongo.py` (đã nằm trong `reset_for_recrawl.py`), `check_data.sql` (thay bằng `data_report.sql`). Xóa `scripts/__pycache__`.
- Còn lại 6 script: `seed_local_datasets.py`, `export_dataset.py`, `remap_labels.py`, `reset_for_recrawl.py`, `data_report.sql`, `listen_samples.py`.

### Kiểm tra tự động nhãn có khớp audio không (chroma, không dùng model, không nghe)
- Chạy 2 script tạm trong container (~3.600 segment ngẫu nhiên + ~3.000 chroma cho phép thử lệch pha), **đã xóa sau khi chạy**. Chroma CQT trên 2 s, so với mẫu tam hợp âm của nhãn.
- **Root top-1** (đoán root bằng mẫu tam hợp âm; ngẫu nhiên = 8,3 %): local piano 91,7 % · GuitarSet 43,8 % · Hooktheory 33,5 % · JAAH 21,8 %. Thấp hơn nhiều so với piano sạch vì: stem Demucs đã bỏ bass, cửa sổ 2 s có thể chứa đổi hợp âm, và mẫu tam hợp âm quá thô cho hợp âm jazz. Nhưng **cao hơn ngẫu nhiên ở mọi nguồn** → nhãn mang tín hiệu thật, không phải nhiễu ngẫu nhiên.
- **Nốt màu có nằm trong 5 nốt mạnh nhất của chroma không** (ngẫu nhiên 41,7 %): sus2 86 %, sus4 82 %, maj7 83 %, add9 82 %, nat11 77 %, b5 78 %, min7 74 %, #5 71 %, nat9 71 %, b13 67 %, #11 67 %, b9 65 %, nat13 63 %, #9 56 %, dom7 56 %. Theo nguồn: Hooktheory 77 %, **JAAH 52 %**, GuitarSet 44 % (n=36, ít). Phép thử này thiên lệch về phía "dễ đạt" (nốt trong giọng thường mạnh sẵn) nên chỉ đọc là mức sàn.
- **Thử lệch pha** (so chroma với nhãn của segment lân cận ±6, cùng bài): điểm cosine đạt đỉnh ở offset 0 ở cả 3 nguồn. Hooktheory 0,671 (nền ≈ 0,54), 59/92 bài đỉnh đúng ở 0; GuitarSet 0,619 (nền ≈ 0,50); **JAAH chỉ 0,561 (nền ≈ 0,515)**, 19/40 bài đỉnh ở 0 (31/40 trong ±1). Kết luận: căn nhãn nhìn chung đúng, nhưng **JAAH tín hiệu yếu** — chưa phân biệt được do lệch pha hay do đặc thù jazz (hợp âm đổi nhanh, voicing thiếu root, bass đã bị bỏ).
- **Chưa kiểm:** extension chính xác từng lớp (b13/#9/alt…) có đúng không, chất lượng Demucs, và bất kỳ thứ gì cần tai người → cần nghe mẫu bằng `listen_samples.py` (đặc biệt JAAH và Hooktheory #9/dom7/b13/alt).

### Công cụ nghe audio kèm nhãn
- Thêm `scripts/listen_samples.py` (chạy trong container) → trang HTML tĩnh `DE/data/export/listen/<name>/index.html`: player + nhãn Harte + hợp âm giải mã + nút ✅/❌/❓ + ghi chú + xuất CSV; có `--context` (segment trước/sau). Cách dùng: `setup.md` mục 7.

### Dọn repo
- Xóa `DE/pyproject.toml`, `DE/uv.lock`. Cấu hình pytest chuyển sang `DE/pytest.ini`; thêm `DE/requirements-dev.txt` (pytest, ruff…). `requirements.txt` là nguồn dependency duy nhất cho Docker.
- Xóa `DE/markdown/20_ETL_roadmap.md` (thiết kế cũ, đã thay bằng `19_DE_implementation_notes.md` + `data_flow.md`), `dataset_download_guide.md` (lỗi thời, thay bằng `dataset_sources.md` + `setup.md` §2), `issue_log.md` (đã xử lý hết; phần còn mở chuyển sang `19_DE_implementation_notes.md` §6.5).
- `.gitignore`: thêm `markdown/learn/` (đã `git rm --cached`, file local còn nguyên), `paper/`, `*.zip`, `.pytest_cache/`.
- Gộp diagram vào `latex_last_project/diagrams/`: `*.drawio` (class, pipeline, `data_flow.drawio`), `uml/` (puml/svg/usecase png/index.html), `build/` (`build_all_diagrams.py`, `generate_diagrams.py`). Xóa `build_diagram/diagrams/` (bản cũ) và 2 PNG trùng với `latex_last_project/image/`; xóa `latex_last_project/image/draft_logo2.jpg` (trùng `landing_page/logo/`, `main.tex` không dùng). `main.tex` không đổi.
- Hai script trong `build/` giờ luôn ghi vào `latex_last_project/diagrams/uml/` (đường dẫn tính từ vị trí file, không phụ thuộc thư mục đang đứng); gồm cả PNG và `index.html`. Đã kiểm `py_compile`, chưa chạy lại vì cần mạng và sẽ ghi đè bản hiện có.
- Còn lại cần người dùng tự xóa/quyết: nhóm legacy hệ thống (`song_analysis`, `user_analytics`, `ingest_service` cũ…) chờ xác nhận BE.

### Checklist bàn giao sang AI
- Thêm `AI/markdown/ai_checklist.md` (quyết định đã chốt, chuẩn bị data, chuẩn hóa âm lượng, lọc nhiễu, train, đánh giá, inference, giao diện với BE/FE). Quyết định **không thêm lớp N**: dùng cổng RMS + ngưỡng tự tin ở inference, BE/FE tự quyết cách hiển thị. `20_DE_training_plan.md` §11/§14 đã trỏ tới checklist.
- `scripts/export_dataset.py`: tự **bỏ segment gần như im lặng** (RMS < −50 dBFS; `--min-rms-dbfs`, `--keep-silent`), thêm cột `rms_dbfs` vào meta (đo trước chuẩn hóa, để chỉnh ngưỡng cổng âm lượng ở inference), ghi `min_rms_dbfs` + `dropped_silent` vào `classes.json`. Doc: `setup.md` mục 6.
- ~~Còn chờ quyết: 214 hàng ngoài `VALID_COMBOS` và tách `sus2`/`sus4`~~ → đã quyết và làm xong (xem các mục trên).
- `20_DE_training_plan.md` đã chuyển sang `AI/markdown/` (người dùng di chuyển); đã sửa lại các link tương đối trong file đó và trong `ai_checklist.md`.

---

## 2026-10-20 — Sơ đồ luồng data draw.io

- Thêm `DE/markdown/data_flow.drawio`: luồng đầu–cuối (6 dataset → seed → `dag_ingest_raw` → Mongo staging → `dag_process_audio` (7 handler) → Postgres `training_samples` + `.npy` → export → Colab MERT+LoRA).
- Vẽ theo code hiện tại (librosa, Demucs 6 stem, taxonomy 12/5/15). Sơ đồ cũ `latex_last_project/diagrams/pipeline.drawio` giữ nguyên (đã lỗi thời).
- `docker-compose.yml`: `INGEST_BATCH_SIZE` 50 → **100** (người dùng đang dùng wifi; 5.797 clip Hooktheory còn lại). Container `airflow` được recreate khi không có DAG nào đang chạy.
- Thêm `scripts/data_report.sql` (báo cáo hàng đợi + segment + phân bố lớp có tên, chỉ đọc) và ghi cách chạy vào `setup.md` mục 7. (`scripts/check_data.sql` cũ thiên về phía hệ thống `song_analyses`, giữ nguyên.)
- Thêm `markdown/20_DE_training_plan.md`: kế hoạch train chi tiết (MERT-330M đóng băng → LoRA, 3 head nối tầng, mask nhãn thiếu, Focal Loss, chống lệch lớp, đánh giá theo bài, bảng thí nghiệm E1–E10, inference/HMM, thu nhỏ model khi deploy). Chưa có code train trong repo; đây chỉ là kế hoạch. Code train sẽ đặt ở thư mục `AI/` riêng (người dùng chốt).
- Thêm `markdown/21_DE_data_dictionary.md`: từ điển dữ liệu (`crawl_queue`, `training_samples`, Mongo `raw_audio_jobs`, thư mục `data/`, file export, cách đọc nhãn Harte) kèm ví dụ dòng thật.

### HarteMapper: bậc 6 trong dạng interval list
- Nguyên nhân 131 segment `unmapped`: JAAH ghi hợp âm 6/9 là interval list (`Db:(3,6,9)/5`), bậc `6` không nằm trong bảng nên cả nhãn bị bỏ.
- Quy tắc mới trong `_from_interval_list` (`harte_mapper.py`):
  - có `6` **và** có 7 (`7`/`b7`/`bb7`) → `6` coi là bậc 13 (`F:(3,5,6,7,9)` → major / nat13, OK);
  - có `6` **không** có 7 → hợp âm 6 / 6-9: lấy root + quality, `extension=None` → **PARTIAL** (giống `maj6`, vì taxonomy không có lớp "6th", không ép sai vào add9).
- Test: thêm `TestSixthDegreeInIntervalForm` (71 test mapper pass).
- Chạy `remap_labels.py`: **125 hàng** `unmapped` → partial (120) / ok (5 `F:(3,5,6,7,9)`). Còn lại ~6 hàng nhãn lạ (`D:(b2,b5,b7)`...) vẫn `unmapped`.

### Hooktheory: `ffmpeg exited with code 1`
- Chạy lại clip lỗi (`7zA9Agpz8fA`) với yt-dlp verbose: **tải thành công** → lỗi chỉ là tạm thời (rớt kết nối giữa chừng). Tải theo đoạn (`download_ranges`) do chính ffmpeg stream nên `retries` của yt-dlp không áp dụng.
- `hooktheory_downloader.py`: thử lại **1 lần** khi lỗi chứa `ffmpeg exited`.
- Requeue 11 hàng `failed` do lỗi này về `pending` (SQL một lần).

---

## 2026-10-19 (e) — Thêm nguồn Hooktheory (Sheet Sage) + batch 20 → 50

### Mục tiêu
Nguồn ưu tiên cao nhất cho các lớp hiếm (sus, add9, 9/11/13, dim/hdim, aug). Có audio YouTube thật (clip ~20 s) và hợp âm giàu.

### Phát hiện
- **HLSD (lead-sheet-dataset)** không dùng được: link Google Drive hết hạn, `theorytab_crawler.py` hỏng (endpoint hooktheory.com trả 404) → thay bằng `Hooktheory.json.gz` của Sheet Sage (20 MB, SHA256 khớp bản công bố).
- 26,175 entry; **17,855 dùng được** (có audio + harmony + alignment refined, không đổi tempo). ~6,084 clip chứa ≥ 1 hợp âm hiếm (scope `rare`).
- POP909 / POP909-CL: gần như chỉ maj/min (+ sus2/sus4 ~6%, 7th), không có 9/11/13/add9 → module synth MIDI **hoãn**, chưa làm.
- Giấy phép Hooktheory: **CC BY-NC-SA 3.0** (phi thương mại).

### Thay đổi code
| File | Thay đổi |
|------|----------|
| `annotation_parsers/hooktheory.py` (mới) | Đổi interval chồng sang nhãn Harte interval-list (`G:(3,5,9)` = add9, sus2/sus4, 6/13, dim7 `bb7`), `beat_to_time`, `build_annotation` (section = vùng nhãn ± 3 s). |
| `annotation_parsers/parsers.py` | `HooktheoryParser`, `read_annotation_section`, `read_annotation_range`; `get_parser` nhận diện `"format": "hooktheory"`. |
| `audio_file.py` | `AudioSource.HOOKTHEORY`, trường `AudioFile.section`. |
| `hooktheory_downloader.py` (mới) | yt-dlp `download_ranges` + `force_keyframes_at_cuts`, bỏ fragment `#ht=`, lỗi `duration_mismatch` nếu lệch thời lượng; đăng ký trong `entrypoints.py`. |
| `ingest_raw_service.py` | Truyền `section` vào `AudioFile.from_url`. |
| `alignment_check_handler.py` | `SECTION_CHECKED = "hooktheory"`: so độ dài WAV với độ dài section. |
| `annotation_handler.py` | Segment không nằm trọn trong vùng đã gán nhãn (±0.25 s) giữ `chord_label=None` → SampleWriter bỏ, không thành 'N'. |
| `shared/domain/dataset_split.py` | Bỏ `#ht=` trước khi hash → mọi clip cùng video chung split. |
| `scripts/seed_local_datasets.py` | `--source hooktheory`, `--hooktheory-scope rare\|all` (mặc định `rare`); priority 7 (≥3 hợp âm hiếm), 6 (1–2). |
| `docker-compose.yml` | `INGEST_BATCH_SIZE` 20 → **50** (pipeline tuần tự, chỉ làm lượt chạy dài hơn; RAM Docker 7.6 GiB). |
| `tests/unit/data_processing/test_hooktheory.py` (mới) | 40 test. |

### Kiểm chứng
- `uv run pytest -q`: 174 passed, 5 skipped.
- E2E trong container (4 clip thật): WAV đúng độ dài section, `align_status=ok`, nhãn sus (`A:(2,5)`) được ghi; ~250–410 KB tải/clip.
- Seed: 6,084 clip vào `crawl_queue` (xử lý ~50 clip / lượt, 2 giờ / lượt).

### Lưu ý
- Nếu gặp YouTube 403 / "not a bot": thêm `config/cookies.txt` (xem `setup.md`) hoặc giảm batch.
- Không làm chế độ offline / script requeue (không cần).

### Xử lý lại JAAH (stem harmonic mới) — đã được người dùng đồng ý
- `scripts/reset_for_recrawl.py` thêm `--source TYPE` (chỉ reset một nguồn; với `--truncate-samples` chỉ xóa `training_samples` + file `.npy` của nguồn đó) và `--priority N` (đặt priority cho các dòng được xếp lại).
- Đã chạy `--source jaah --truncate-samples --priority 8`: xóa 3,771 mẫu + 27 file `.npy`, 27 doc Mongo; 32 dòng JAAH (27 done + 5 failed) về `pending` priority 8 (chạy trước Hooktheory). 81 dòng JAAH pending cũ giữ priority 0.
- Cần tải lại 27–32 bài từ YouTube (~0.2 GB).

### Đẩy ưu tiên lớp hiếm (2026-10-19 tối)
- Sau ~170 clip Hooktheory đầu: 0 clip có 9/11/13/aug (không phải lỗi mapper — thứ tự queue `priority, created_at`; 300 clip kế tiếp chỉ 1 clip có loại này).
- Đặt `priority = 8` cho: 81 bài JAAH còn pending (mỗi bài ~144 segment, giàu 9/11/13/alt/dim) và **2.384** clip Hooktheory pending chứa 11 / #11 / 13 / b13 / #5 / b9 / #9 hoặc 9 đi cùng 7. add9 và sus không tính (đã nhiều). Chỉ đổi cột `priority`, không xóa dữ liệu, đảo ngược được.
- Thứ tự mới: priority 8 (JAAH, rồi Hooktheory hiếm) → hooktheory 7/6 → guitarset 5 → swd 4 → choco 3/1/0.

### Đồng bộ tài liệu
- `markdown/18_main_project_edtech_music.md` (file tổng): viết lại **mục 6** (cây thư mục, luồng ETL 2 DAG, design patterns, hạ tầng, parser, bảng công nghệ & quyết định xử lý), thêm bảng nguồn dữ liệu thực tế ở mục 7 (Luồng 2), cập nhật mục 9 (Mongo), mục 11 (Tech Stack: librosa thay madmom, Airflow 2.10.3, batch 50, `training_samples` + `.npy`) và timeline kỳ 1.
- `DE/markdown/18_DE_AI_chordsense.md`: bảng "spec vs thực tế" thêm nguồn hiện có, stem harmonic, librosa, 2 DAG, Hooktheory.
- File tổng, lượt 2 — **khớp công nghệ theo code** (`value_objects.py`, `pyproject.toml`, `requirements.txt`, `docker-compose.yml`): madmom → librosa; Demucs 6 stem + harmonic mix (thay "other + bass"); đầu vào MERT là waveform 24 kHz (không phải mel); Head 2 = 5 quality, Head 3 = 15 extension, MERT embedding 1024-d, kiến trúc cascaded (thay 6/8 lớp, 768-d, parallel); LoRA ~1.98M param (0.6%), ~20–24 GB VRAM trên Colab A100; phiên bản PyTorch 2.5+, transformers 4.47+, PEFT 0.14+, librosa 0.10.2+, Airflow LocalExecutor; thêm Python 3.11/uv, pytest; ghi chú ETL không trích feature; trạng thái McGill/RWC (chưa có data), Kaggle thực tế là piano_triads + GuitarSet; tên dataset cho Model B/timeline.
- `DE/markdown/19_DE_implementation_notes.md` (mục 6), `setup.md`, `data_flow.md`, `dataset_sources.md`: cập nhật theo Hooktheory, batch 50, `reset_for_recrawl.py --source`.

---

## 2026-10-19 (d) — Demucs: trộn stem harmonic (piano + guitar + other)

### Vấn đề
`htdemucs_6s` luôn ghi `piano.wav`, nên chuỗi fallback piano → guitar → other không bao giờ chạy. Bài pop/rock không có piano bị gán nhãn hợp âm trên stem piano gần như chỉ là tiếng lọt (issue #11).

### Thay đổi
| File | Thay đổi |
|------|----------|
| `pipeline_handlers/separate_handler.py` | Thêm `HARMONIC_STEMS = (piano, guitar, other)` và `_mix_harmonic()`: cộng các stem hiện có (cắt theo stem ngắn nhất), giới hạn peak ≤ 0.99, ghi `harmonic.wav` và dùng làm `stem_path`. Drums / bass / vocals bị loại. Lỗi `Can not find harmonic stems` nếu không có stem nào. |
| `tests/unit/data_processing/test_separate_harmonic_mix.py` | Mới: tổng đúng, bỏ drums/vocals, giới hạn peak, thiếu stem, không có stem (tự skip nếu máy không có numpy/soundfile). |

### Kiểm chứng
- `uv run pytest`: 135 passed (4 skipped — nhóm test mới chỉ chạy được khi có numpy/soundfile).
- Chạy trong container với stem giả: tổng 0.1+0.2+0.3 = 0.6, peak cap 0.99, thư mục rỗng → báo lỗi đúng.

### Lưu ý
- Không cần dừng Docker: `src/` được mount trực tiếp, mỗi task Airflow là một process mới nên áp dụng từ task kế tiếp.
- ~27 bài JAAH đã xử lý trước đó vẫn dùng stem piano cũ cho tới khi xử lý lại (chưa làm, cần xác nhận vì phải xóa dữ liệu).
- Kaggle, GuitarSet (clean source) không bị ảnh hưởng.

---

## 2026-10-19 (c) — Thêm nguồn: GuitarSet, SWD, ChoCo release (audio partition)

### Mục tiêu
Tăng đa dạng lớp hợp âm (suspended, extension) và thể loại. Chỉ nguồn **có audio dùng được** mới được tích hợp.

### Thay đổi code

| File | Thay đổi |
|------|----------|
| `data_loader/domain/models/audio_file.py` | `AudioSource` + `GUITARSET`, `SWD` (dùng `LocalFileDownloader`, đăng ký trong `entrypoints.py`). |
| `data_loader/.../choco_downloader.py` | `source_url` dạng `ytsearch1:<performer> <title>` → ủy quyền cho `JaahDownloader` (search nhiều kết quả, chọn bản khớp thời lượng JAMS). |
| `data_processing/.../parsers.py` | `JamsParser` chọn chord track *performed* (`data_source` chứa "transcription") khi có nhiều track; **`SwdParser`** mới (CSV `start;end;shorthand`), `get_parser` nhận diện header SWD (RWC CSV giữ nguyên). |
| `alignment_check_handler.py` | Kiểm tra thời lượng áp dụng cho **`jaah` và `choco`** (JAMS có `file_metadata.duration`). |
| `separate_handler.py`, `beat_handler.py` | `guitarset` là *clean source* (không Demucs; guitar solo không được tách bằng `htdemucs_6s`). SWD vẫn qua Demucs. |
| `harte_mapper.py` | Interval-list không có 3rd: `F:(4,5,b7)`, `A:(1,2,4,5)` → **sus**; `F#:(1,5)`, `C:(1)` → `partial` (chỉ root); `maj(2)` → `add9`. |
| `scripts/seed_local_datasets.py` | Scanner `guitarset`, `swd`, `choco-release`; `priority` theo từng entry (guitarset 5, swd 4, choco pop/rock 3, billboard 1). `scan_choco` (clone) sửa ở mục (b). |

### Quyết định
- **GuitarSet:** dùng chord track *performed* (suy ra từ nốt, có kiểm tay) thay vì *instructed* (lead sheet) vì phản ánh đúng audio; chứa `sus2(7)`, `maj6(*5)`.
- **SWD:** chỉ 2/9 bản thu có audio (HU33, SC06 → 48 file); 168 CSV còn lại không dùng.
- **ChoCo release:** chỉ seed `isophonics`, `robbie-williams`, `rwc-pop`, `uspop2002`, `billboard` (1.309 bài). Bỏ `jaah` (trùng), `casd` (trùng chordify), `schubert-winterreise-audio` (dùng SWD local), `weimar` (chỉ là đoạn solo, thời lượng không khớp bản đầy đủ), và mọi partition symbolic.
- **AAM không tích hợp:** quét 1.000 file `*_beatinfo.arff` → chord chỉ có maj/min/N.C.; không lấp được lớp trống, không đáng 14 GB.

### Kiểm chứng
- Unit test: 135 passed (thêm `test_dataset_expansion.py`, mở rộng `test_harte_mapper.py`).
- E2E trong container (`TRAINING_DIR` tạm, dữ liệu test đã xóa): GuitarSet → 14 segment (`D#:sus2(7)/1` → sus + maj7); SWD (44 s) → 29 segment qua Demucs; ChoCo `ytsearch1:The Beatles Julia` → chọn video 174,9 s so với 174,2 s trong JAMS.
- Seed thật: +360 guitarset, +48 swd, +1.309 choco-release vào `crawl_queue`.

### Lưu ý
- Với `INGEST_BATCH_SIZE=20` mỗi 2 giờ, hàng đợi mới (≈1.700 mục) cần nhiều ngày; các mục ưu tiên cao (guitarset, swd) chạy trước.
- Tỉ lệ `failed` của ChoCo release có thể cao hơn JAAH (không tìm được bản YouTube khớp thời lượng); đó là cơ chế bảo vệ nhãn, không phải lỗi.

---

## 2026-10-19 (b) — Sửa scanner ChoCo + danh sách nguồn dữ liệu

- **Lỗi**: `scan_choco` trong `scripts/seed_local_datasets.py` tìm `ann_audio_chord.jams` (không tồn tại trong bản clone) → seed 0 bài.
- **Sửa**: đọc `partitions/<partition>/choco/meta.csv`, cột `youtube_url`; `annotation_url` trỏ tới `choco/jams/<id>.jams`. Bỏ giới hạn 500.
  Partition không có `youtube_url` (billboard, ireal-pro, biab-internet-corpus) được bỏ qua vì không có audio.
- **Kết quả**: seed thêm 50 bài (`source_type='choco'`, partition `chordify`) vào `crawl_queue`. Dry-run và seed thật trong container đều thành công.
- **Phát hiện**: bản clone `smashub/choco` chỉ chứa 4 partition. Bộ đầy đủ (20.080 JAMS, 2.283 audio) nằm ở GitHub Release `v1.0.0.zip` (178 MB).
- **Tài liệu mới**: `dataset_sources.md` — checklist nguồn dữ liệu bổ sung, đánh dấu mức kiểm chứng.

---

## 2026-10-19 — Refactor training ETL: output = clean labeled segments

### Bối cảnh

Sau lần chạy trước, pipeline có 3 vấn đề:

1. **JAAH không có nhãn** — `process_audio_wrapper.py` làm rơi `annotation_path` → 21/21 job JAAH có `chord_timeline = []`.
2. **Output không dùng được để train** — pipeline ghi `song_analyses` (timeline cấp bài hát, thuộc phía hệ thống), còn segment/feature
   không được lưu. Feature `(N,152)` trong spec cũng không khớp thứ MERT cần (waveform 24 kHz).
3. **Crawl kém hiệu quả** — Demucs chạy CPU (timeout 5 phút), YouTube lấy sai bản (lệch thời lượng so với annotation),
   yt-dlp thiếu JS runtime, không retry.

### Quyết định thiết kế (đã chốt với chủ đồ án)

| # | Quyết định | Lý do |
|---|-----------|-------|
| 1 | **Postgres = metadata + nhãn** (bảng `training_samples`), **audio = file `.npy` trên volume**. Mongo chỉ là staging. | Không thêm công nghệ DB mới; audio không nhét vào DB. Export cho Colab được sinh từ Postgres. |
| 2 | Lưu **segment sạch** (không augment, không feature). Split 70/20/10 theo `hash(source_url)` lúc ingest. | Augment chỉ áp cho train, làm ở phía train → không leakage, không phình disk. |
| 3 | JAAH: **kiểm tra thời lượng** audio vs annotation, lệch > 3 s → reject. Beat-alignment (option C) hoãn. | Rẻ, chặn được phần lớn trường hợp YouTube sai bản. |

### Thay đổi code

**Data model & lưu trữ**

| File | Thay đổi |
|------|----------|
| `src/training_data/` *(mới)* | Bounded context mới: `TrainingSample` + `TrainingSampleRepository` (`ensure_table`, `source_exists`, `replace_song`, `stats`). |
| `init/init.sql` | Thêm bảng `training_samples` (+ index). Repository cũng tự `CREATE TABLE IF NOT EXISTS` vì `init.sql` chỉ chạy trên volume rỗng. |
| `src/shared/domain/dataset_split.py` *(mới)* | `split_for(key)` → train/val/test (sha1, 70/20/10, deterministic). |
| `src/data_processing/domain/models/processed_audio.py` | + `align_status`, `sample_rows`. |
| `src/data_processing/domain/models/audio_segment.py` | + `label_status`, `root_id`, `quality_id`, `ext_id`. |

**Pipeline xử lý** (`build_training_pipeline`)

```
trước: Separate → Beat → Segment → Annotation → Summary → Augment → Feature
sau  : AlignmentCheck → Separate → Beat → Segment → Annotation → Label → SampleWriter
```

| File | Thay đổi |
|------|----------|
| `application/mappers/harte_mapper.py` *(mới)* | `map_harte(label)` → `(root_id, quality_id, ext_id, status)`. Status: `ok` / `partial` (6th, power chord…) / `no_chord` / `unmapped`. Xử lý enharmonic, bass, shorthand, `alt`, và **nhãn dạng liệt kê interval** như `F:(3,5,b7,13)` (JAAH dùng cho dominant biến đổi — phát hiện sau lượt chạy đầu: 37 segment bị `unmapped`). |
| `pipeline_handlers/alignment_check_handler.py` *(mới)* | JAAH only: so `soundfile.info().duration` với `duration` trong annotation (env `ALIGN_TOLERANCE_SEC`, mặc định 3 s). |
| `pipeline_handlers/label_handler.py` *(mới)* | Gọi `map_harte` cho từng segment. |
| `pipeline_handlers/sample_writer_handler.py` *(mới)* | Decode → resample **24 kHz** → int16 → `(N, 48000)` → ghi `{TRAINING_DIR}/audio/{song_id}.npy` (atomic). Điền `sample_rows`. Fail nếu 0 segment dùng được. |
| `pipeline_handlers/separate_handler.py` | Bỏ qua Demucs với nguồn sạch (`kaggle`/`local`/`pianoteq`); chạy **GPU** nếu có (`-d cuda`), `--segment` (env `DEMUCS_SEGMENT`, mặc định 6); timeout GPU 900 s / CPU 1800 s; OOM → retry CPU; lỗi lấy `stderr[-600:]`. |
| `pipeline_handlers/beat_handler.py` | Nguồn sạch không bắt được beat → không coi là lỗi. |
| `pipeline_handlers/segment_handler.py` | `_snap_to_beat` an toàn khi không có beat. |
| `annotation_parsers/parsers.py` | + `read_annotation_duration()`. |
| `AugmentHandler`, `FeatureHandler`, `SummaryHandler` | **Giữ nguyên file**, chỉ không còn nằm trong chain training. Inference pipeline (`build_pipeline`) không đổi. |

**Orchestration**

| File | Thay đổi |
|------|----------|
| `src/data_ingest/application/process_audio_service.py` | Viết lại: chạy chain mới → `TrainingSampleRepository.replace_song()` (idempotent) → Mongo `done` → cleanup. Không còn ghi `song_analyses`. |
| `src/data_ingest/application/ingest_raw_service.py` | Tải **annotation trước** (lấy duration) rồi mới tải audio; thiếu annotation khi source khai báo annotation → fail (không stage audio không nhãn); xóa thư mục tạm khi tải lỗi. |
| `dags/wrapper/process_audio_wrapper.py` | **Fix bug `annotation_path` bị rơi** (XCom dict + `RawAudioJob`). `BATCH_SIZE` đọc env `INGEST_BATCH_SIZE` (mặc định 20, trước là 10). |
| `dags/wrapper/ingest_raw_wrapper.py` | `BATCH_SIZE` đọc env như trên. |
| `dags/dag_ingest_raw.py`, `dag_process_audio.py` | Sửa comment/docstring sai ("6 giờ", "+30 phút") — thực tế cron `0 */2 * * *`, DAG 2 do DAG 1 trigger. |

**Crawl hiệu quả hơn**

| File | Thay đổi |
|------|----------|
| `downloaders/_ytdlp.py` *(mới)* | Option dùng chung: `retries=5`, `fragment_retries=5`, `socket_timeout=30`, sleep 1–3 s, `cookiefile` tùy chọn (env `YTDLP_COOKIES`). `pick_best_candidate()` chọn kết quả có thời lượng khớp annotation. |
| `downloaders/jaah_downloader.py` | Tìm 10 kết quả YouTube (flat, không tải) → chọn bản khớp thời lượng → mới tải. Không có kết quả khớp → fail sớm, trước khi tốn công Demucs. Hết lỗi `list index out of range`. |
| `downloaders/choco_downloader.py` | Dùng option chung. |
| `src/data_loader/domain/models/audio_file.py` | + `expected_duration`. |
| `requirements.txt` | `yt-dlp[default]` (kéo `yt-dlp-ejs`) + **`deno`** — yt-dlp bản mới cần JS runtime để lấy format YouTube; image cũ không có → nguyên nhân khả dĩ của nhiều lỗi 403/format. Thêm `pyarrow`. |

**Hạ tầng & script**

| File | Thay đổi |
|------|----------|
| `docker-compose.yml` | GPU NVIDIA cho `airflow`; bind-mount `./data/training`, `./data/export`, `./config` (→ `/etc/chordsense`); env mới `TRAINING_DIR`, `INGEST_BATCH_SIZE`, `DEMUCS_SEGMENT`, `ALIGN_TOLERANCE_SEC`, `YTDLP_COOKIES`. |
| `.gitignore` | Bỏ qua `data/training/`, `data/export/`, `config/cookies.txt`. |
| `scripts/reset_for_recrawl.py` *(mới)* | Xóa Mongo staging + đưa `crawl_queue` về `pending` (`--dry-run`, `--only-failed`, `--truncate-samples`). Không đụng `song_analyses`. |
| `scripts/export_dataset.py` *(mới)* | Xuất `{train,val,test}_audio.npy` + `_meta.parquet` + `classes.json`, in phân bố lớp. |
| `scripts/remap_labels.py` *(mới)* | Map lại `label_status/root/quality/ext` từ `label_raw` sau khi sửa `HarteMapper` — không phải crawl lại (`--dry-run`). |

**Tests** — `tests/unit/data_processing/` (mapper, handlers, yt-dlp helper), `tests/unit/shared/test_dataset_split.py`.

### Hành vi thay đổi cần biết

- `song_analyses` **không còn được ghi** bởi training ETL (89 dòng cũ vẫn nằm đó, không xóa).
- Kaggle `piano_triads` giờ **không qua Demucs** (nguồn đã sạch) — nhanh hơn ~100×, và không còn artefact do tách stem.
- Segment lưu ở **24 kHz** (đầu vào MERT), không phải 44.1 kHz.
- Chuỗi nhãn không map được **vẫn được lưu** (`label_status='unmapped'`/`'partial'`) để re-map sau mà không phải crawl lại; export mặc định chỉ lấy `ok` + `partial`.

### Chưa làm / ghi nhận

- JAAH beat-alignment (option C): hoãn.
- `htdemucs_6s` luôn sinh `piano.wav` nên fallback piano → guitar → other không bao giờ kích hoạt; với nhạc jazz không có piano, stem piano có thể là rác. Cần nghe thử vài bài rồi quyết định.
- ChoCo: chưa seed vào `crawl_queue` (hiện chỉ có JAAH + Kaggle). Số bài ChoCo có audio khớp chưa được đếm.
- McGill, RWC: chưa có dữ liệu (cần đăng ký / tải tay). Pianoteq: cần license.
