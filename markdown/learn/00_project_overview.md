# 00 — Hệ thống này làm gì?

Đọc file này trước. Nó là bức tranh tổng thể — bạn cần thấy được toàn bộ bức tranh trước khi đọc từng mảnh ghép.

---

## 1. Bài toán

Công ty có file CSV chứa **dữ liệu hải quan** (tờ khai nhập khẩu, HS Code, tên người mua...) nằm trên Google Drive. Mỗi ngày lại có file mới.

**Nếu không tự động:** Mỗi ngày ai đó phải nhớ vào GDrive tải file, chạy script làm sạch, rồi import tay vào DB. Làm một lần thì được, làm 365 ngày thì chắc chắn có ngày quên hoặc làm sai.

**Hệ thống này giải quyết:** Airflow tự chạy lúc 2AM mỗi ngày. Không cần ai làm gì.

```
File CSV trên GDrive  →  Tải về máy  →  Validate + Clean  →  Lưu DB  →  Email báo cáo
```

---

## 2. Công nghệ dùng trong project

| Việc gì | Dùng gì |
|---|---|
| Lên lịch tự chạy hàng ngày | **Apache Airflow 3.x** |
| Viết toàn bộ logic | **Python 3.11+** |
| Validate kiểu dữ liệu cho model | **Pydantic v2** |
| Lưu metadata / kết quả xử lý | **MongoDB** |
| Lưu dữ liệu chuẩn hoá | **PostgreSQL** |
| Tải file từ Google Drive | **Google Drive API** |
| Tải file từ AWS | **boto3 (S3)** |
| Tải file từ internet | **requests (HTTP API)** |
| Làm giàu dữ liệu bằng AI | **OpenAI / Google AI Studio** |
| Đóng gói và chạy toàn bộ | **Docker + Docker Compose** |

---

## 3. Thư mục trong project

Có 2 thư mục quan trọng nhất: `dags/` và `src/`.

```
STSDataIngestion/
├── dags/                     ← Airflow đọc những file này để biết phải làm gì
│   ├── dag_ingest.py         ← DAG chính: tải → xử lý → lưu → email
│   ├── wrapper/              ← Cầu nối: chuyển lệnh từ Airflow sang src/
│   └── sensors/              ← Cảm biến: chờ file mới trên GDrive
│
├── src/                      ← Toàn bộ logic nghiệp vụ (không liên quan Airflow)
│   ├── shared/               ← ⭐ Nền tảng chung — đọc cái này trước tiên
│   ├── data_loader/          ← Module 1: Tải file từ GDrive/S3/API về máy
│   ├── data_processing/      ← Module 2: Validate + Clean + Transform dữ liệu
│   └── data_ingest/          ← Module 3: Điều phối 2 module trên
│
├── docker-compose.yml        ← Cấu hình chạy toàn bộ hệ thống bằng Docker
└── pyproject.toml            ← Danh sách thư viện cần cài
```

> **Quan trọng:** `src/` không biết Airflow tồn tại. `dags/wrapper/` là cầu nối duy nhất. Nhờ vậy bạn có thể test `src/` mà không cần chạy Airflow.

---

## 4. Dữ liệu đi qua đâu?

Hành trình của một file CSV từ lúc nằm trên GDrive đến lúc vào database:

**AIRFLOW (lịch trình)**
```
Task 1: GoogleDriveSensor   → chờ cho đến khi có file mới trên GDrive
Task 2: run_ingest_pipeline → tải file + xử lý + lưu DB
Task 3: send_summary_email  → gửi email báo kết quả
```

**BÊN TRONG Task 2** (đây là phần quan trọng nhất):

```
Bước 1 — Tải file:
  FileDispatcher → GoogleDriveDownloader
  → Kết nối Google Drive API
  → Tải file .csv về /tmp/sts_data_ingestion/
  → Lưu metadata vào MongoDB ("đã tải file này rồi")
  → Trả về danh sách file đã tải

Bước 2 — Xử lý file:
  → Đọc CSV bằng pandas
  → Rename cột ("Buyer" → "buyer_name")
  → Fix kiểu dữ liệu ("25" → 25)
  → (Optional) Gọi AI để clean data

Bước 3 — Lưu kết quả:
  → Lưu summary/metadata → MongoDB
  → Lưu từng dòng data → PostgreSQL
  → Trả về IngestionRecord (tóm tắt kết quả của run)
```

---

## 5. Nguồn dữ liệu được hỗ trợ

| Nguồn | Cần gì để kết nối |
|---|---|
| **Google Drive** | File credentials JSON + folder ID |
| **AWS S3** | Access key + secret key + bucket name |
| **HTTP API** | URL + optional headers/auth |

Nếu muốn thêm nguồn mới (ví dụ: FTP server), chỉ cần implement interface `Downloader` rồi đăng ký — không cần sửa code hiện có.

---

## 6. Hai database, hai mục đích khác nhau

**MongoDB** — lưu những thứ không có hình dạng cố định:
- Metadata mỗi lần chạy pipeline ("run này tải 5 file, thành công 4")
- Mỗi lần chạy có thể có fields khác nhau → MongoDB hợp lý hơn SQL

**PostgreSQL** — lưu dữ liệu có cấu trúc rõ ràng:
- Từng dòng dữ liệu HS Code đã chuẩn hoá (luôn có cùng columns)
- Team khác cần query, join → SQL phù hợp hơn

---

## 7. Điều quan trọng nhất cần nhớ

**`src/` không biết Airflow tồn tại.** Đây là quyết định thiết kế quan trọng nhất.

```
dags/wrapper/pipeline_wrapper.py  ← Airflow gọi vào đây
         ↓ gọi
src/data_ingest/pipeline.py       ← Logic thật ở đây, không biết Airflow
```

Nhờ thiết kế này:
- Bạn test `src/` bằng Python bình thường, không cần chạy Airflow
- Nếu công ty sau này đổi sang công cụ khác (Prefect, cron...), chỉ cần viết lại `dags/`, không đụng vào `src/`

---

## 8. Đọc tiếp theo thứ tự này

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08

01: Tại sao code tổ chức theo cách đó (DDD)
02: shared/ — nền tảng chung, đọc trước các module
03: data_loader — tải file về máy
04: data_processing — xử lý, clean dữ liệu
05: data_ingest — điều phối 2 module trên
06: Airflow DAGs — lên lịch tự chạy
07: AI prompt system — tích hợp LLM
08: Hướng dẫn tự xây project mới từ đầu

Nếu cần hiểu thêm về thư viện:
09: Pydantic  |  10: Docker  |  11: MongoDB vs PostgreSQL
12: Python nâng cao  |  13: Testing  |  14: Pandas
```
