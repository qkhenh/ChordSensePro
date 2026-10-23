# ChordSense Pro — DE: Nguồn dữ liệu bổ sung (download checklist)

> Cập nhật: 2026-10-19. Mục tiêu: lấp các lớp hợp âm đang trống (sus, b5, add9, #9, nat11, b13…)
> và tăng đa dạng thể loại. Đặt mọi thứ dưới `DE/data/datasets/<tên>/` (bind-mount vào container `/data/datasets`).
>
> Cột **Kiểm chứng**: ✅ = đã gọi API Zenodo/GitHub và xác nhận tên, license, kích thước; ⚠️ = lấy từ kết quả search, chưa mở trực tiếp.

## A. Ưu tiên cao — tải ngay

| # | Nguồn | Có audio? | Nhãn | License | Tải gì | Kích thước | Kiểm chứng |
|---|-------|-----------|------|---------|--------|-----------|-----------|
| 1 | **ChoCo release v1.0.0** | Không (chỉ nhãn; 2.283 file JAMS thuộc audio partition) | Harte (JAMS) | CC BY 4.0 | https://github.com/smashub/choco/releases/download/v1.0.0/v1.0.0.zip | 178 MB | ✅ |
| 2 | **GuitarSet** | Có (360 clip × 30 s, guitar acoustic) | Chord JAMS (instructed + performed) | CC BY 4.0 | https://zenodo.org/records/3371780 — `annotation.zip` + `audio_mono-mic.zip` | 37 MB + 626 MB | ✅ |
| 3 | **Schubert Winterreise (SWD) v2.0** | Có (chỉ 2/9 bản thu có audio) | `ann_audio_chord` (CSV) | CC BY 3.0 | https://zenodo.org/records/5139893 — `Schubert_Winterreise_Dataset_v2-0.zip` | 483 MB | ✅ |
| 4 | ~~AAM~~ (hạ cấp, xem mục C) | — | — | — | — | — | — |
| 5 | **Isophonics** (Beatles, Queen, Zweieck) | Không | `.lab` Harte | Tự do học thuật | http://isophonics.net/content/reference-annotations-beatles | nhỏ | ⚠️ (cũng nằm trong ChoCo release) |
| 6 | **tmc323/Chord-Annotations** (USpop2002 195 bài + RWC-Pop) | Không | `.lab` Harte | (repo không ghi) | https://github.com/tmc323/Chord-Annotations | nhỏ | ✅ repo |
| 7 | **POP909** | Không (MIDI, cần render) | Chord + beat + key | MIT | https://github.com/music-x-lab/POP909-Dataset | ~ vài chục MB | ✅ |
| 8 | **POP909-CL** (nhãn chord đã sửa tay) | Không (MIDI) | Chord sạch hơn POP909 | MIT | https://github.com/AndyWeasley2004/POP909-CL-Dataset | nhỏ | ✅ |
| 9 | ~~Hooktheory Lead-Sheet Dataset (HLSD)~~ | Không (symbolic) | — | — | https://github.com/wayne391/lead-sheet-dataset | 4,9 GB | ❌ **Link Google Drive hết hạn; crawler `theorytab_crawler.py` hỏng** (`hooktheory.com/theorytab/artists/x` và `/songs/getXmlByPk` đều 404). Thay bằng mục 18 |
| 18 | **Hooktheory (Sheet Sage)** — `Hooktheory.json.gz` | **Có** — đoạn ~20 s của video YouTube đã căn chỉnh theo beat | Harmony (root + các quãng chồng) + beat→giây | CC BY-NC-SA 3.0 (phi thương mại) | https://github.com/chrisdonahue/sheetsage-data/raw/refs/heads/main/hooktheory/Hooktheory.json.gz | 20 MB | ✅ đã tải, SHA256 khớp (`917b7cd5…`) |

## B. Tùy chọn — giá trị trung bình

| # | Nguồn | Ghi chú | License | Kiểm chứng |
|---|-------|---------|---------|-----------|
| 10 | **Weimar Jazz Database** | Solo + chord grid jazz (SQLite); audio không kèm | CC BY-NC | ⚠️ https://jazzomat.hfm-weimar.de/download.html |
| 11 | **rwc-music/rwc-annotations** | Nhãn RWC 2.0; license audio RWC 2.0 cần tự kiểm tra | NOASSERTION | ✅ repo, ⚠️ audio |
| 12 | **Chordonomicon** | Chuỗi hợp âm văn bản của rất nhiều bài (cần render); CC BY-NC 4.0 | CC BY-NC 4.0 | ✅ HuggingFace `ailsntua/Chordonomicon` |
| 13 | **DCML ABC** (Beethoven quartets) | Roman numeral → Harte; classical | NOASSERTION | ✅ https://github.com/DCMLab/ABC |
| 14 | **When in Rome** | Meta-corpus phân tích hòa âm classical | (repo không ghi) | ✅ https://github.com/MarkGotham/When-in-Rome |
| 15 | **BPS-FH** (Beethoven piano sonatas, functional harmony) | GPL-3.0; classical | GPL-3.0 | ✅ https://github.com/Tsung-Ping/functional-harmony |
| 16 | **McGill Billboard** | Nhãn 898 bài đã có trong clone ChoCo (`partitions/billboard`); **không có audio** | — | ✅ (đã có sẵn) |
| 17 | **mirdata** | Không phải data — thư viện loader (Beatles, GuitarSet, Billboard…) | BSD-3 | ✅ https://github.com/mir-dataset-loaders/mirdata |

## C. Đã xem và không khuyến nghị (lúc này)

| Nguồn | Lý do |
|-------|-------|
| Slakh2100 (Zenodo 4599666, 99 GB) | Chỉ có MIDI, **không có nhãn chord**; quá nặng |
| MAESTRO | Chỉ nhãn nốt MIDI, phải tự suy ra chord (nhiễu) |
| MedleyDB | Search không xác nhận có chord annotation → bỏ qua |
| Robbie Williams / JayChou29 / CNPop20 | Nhãn có (Robbie nằm trong ChoCo release), audio không công khai |
| **AAM** (Zenodo 5794629, mix 14 GB mỗi phần) | Đã kiểm 1.000 file `*_beatinfo.arff`: chord chỉ có **maj / min / N.C.** (≈ 288 nghìn nhãn beat). Không giúp lấp lớp sus/ext/dim, chỉ thêm triad tổng hợp nên không đáng 14 GB. Không có scanner cho AAM |

## D. Trạng thái tích hợp (cập nhật 2026-10-19)

| Nguồn | Thư mục | Scanner (`seed_local_datasets.py --source …`) | Cách lấy audio | Ưu tiên queue |
|-------|---------|-----------------------------------------------|----------------|---------------|
| JAAH | `jaah/` | `jaah` | YouTube search + khớp thời lượng | 0 |
| Kaggle piano_triads | `kaggle/archive/` | `kaggle` | file local | 0 |
| ChoCo clone — partition `chordify` | `choco/` | `choco` | link YouTube trong `meta.csv` | 0 |
| **GuitarSet** | `guitarset/` | `guitarset` | file local (`audio_mono-mic`), clean source (không Demucs) | 5 |
| **SWD** | `Schubert_Winterreise/` | `swd` | file local (chỉ 2 bản thu HU33, SC06 = 48 file), có Demucs | 4 |
| **ChoCo release** — isophonics, robbie-williams, rwc-pop, uspop2002 | `choco_release/` | `choco-release` | YouTube search `"<performer> <title>"` + khớp thời lượng JAMS | 3 |
| **ChoCo release** — billboard | `choco_release/` | `choco-release` | như trên | 1 |
| **Hooktheory** — clip có hợp âm hiếm (sus / add9 / 9 / 11 / 13 / alt) | `hooktheory/` | `hooktheory` (`--hooktheory-scope rare`, mặc định) | yt-dlp **tải đúng đoạn có nhãn** (+3 s đệm), kiểm tra thời lượng video | 6 (7 nếu ≥ 3 hợp âm hiếm) |
| **Hooktheory** — clip chỉ maj/min/7 | `hooktheory/` | `hooktheory --hooktheory-scope all` | như trên | 0 |

Không seed từ ChoCo release: `jaah` (trùng JAAH), `casd` (trùng chordify), `schubert-winterreise-audio` (dùng SWD local), `weimar` (chỉ là đoạn solo, thời lượng không khớp bản đầy đủ), và mọi partition symbolic (`ireal-pro`, `biab`, `wikifonia`, `real-book`, `nottingham`, ...).

## E. Việc code còn lại

| Nhóm | Việc cần làm |
|------|--------------|
| POP909 / POP909-CL / iReal / Chordonomicon (symbolic) | Module **synth**: MIDI → audio (FluidSynth + SoundFont `.sf2`, đã tải `GeneralUser-GS.sf2` vào `data/datasets/soundfonts/`). Thống kê POP909: gần như chỉ maj/min (+ sus2/sus4 ≈ 6 % thời lượng, 7th), **không có 9/11/13/add9**. Giá trị chủ yếu là lớp `suspended`; đã bị Hooktheory (audio thật) lấn át nên **chưa làm** |
| tmc323/Chord-Annotations, Isophonics (`.lab`) | Chỉ nhãn, không audio; đã nằm trong ChoCo release (uspop2002, rwc-pop, isophonics) nên không cần seed riêng |

