# 05 — `data_ingest` — Orchestrator kết nối mọi thứ

Module này **không tải file, không xử lý data** — nó chỉ gọi đúng người đúng lúc.

Hiểu đơn giản: `data_ingest` là người quản lý dự án. Nó nói với `data_loader`: "mày tải file đi", rồi nói với `data_processing`: "mày xử lý đi", rồi lưu kết quả vào DB.

---

## 1. Vị trí trong hệ thống

```
data_loader    →    data_ingest    →    data_processing
   (tải file)       (điều phối)          (xử lý data)
                         ↓
                   Lưu kết quả vào PostgreSQL
```

`data_ingest` là lớp duy nhất biết cả `data_loader` lẫn `data_processing`. Hai module kia không biết nhau.

---

## 2. Cấu trúc module

```
data_ingest/
├── application/
│   ├── entrypoints.py      ← run_pipeline() — cổng vào duy nhất
│   ├── ingest_service.py   ← IngestService — orchestrator chính
│   └── pipeline.py         ← Logic kết nối loader → processing
│
└── domain/
    └── models/
        └── ingest_record.py  ← IngestRecord — track trạng thái
```

---

## 3. Flow chính

```python
# entrypoints.py
def run_pipeline(source: str, execution_date: str, **kwargs):
    service = IngestService()
    return service.run(source=source, execution_date=execution_date, **kwargs)
```

```python
# ingest_service.py
class IngestService:
    def run(self, source, execution_date, **kwargs):
        record = IngestRecord(source=source, execution_date=execution_date)
        record.status = IngestStatus.LOADING

        # Bước 1: Tải file
        responses = run_data_loader(source=source, **kwargs)
        record.status = IngestStatus.PROCESSING

        # Bước 2: Xử lý data
        result = run_data_processing(responses=responses, execution_date=execution_date)
        record.status = IngestStatus.DONE

        # Bước 3: Lưu vào PostgreSQL
        postgres_repo.save(result)

        return result
```

---

## 4. `IngestRecord` — Track trạng thái

```python
class IngestRecord(CustomBaseModel):
    source: str
    execution_date: str
    status: IngestStatus = IngestStatus.PENDING

    # Thống kê
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    error_messages: list[str] = []
```

**Vòng đời:**
```
PENDING → LOADING → PROCESSING → DONE
                              ↘ FAILED  (nếu có exception)
```

`IngestRecord` chỉ tồn tại trong memory khi pipeline chạy. Nó không được lưu vào database — chỉ được truyền qua Airflow XCom dưới dạng JSON.

---

## 5. Xử lý lỗi

```python
def run(self, source, execution_date, **kwargs):
    record = IngestRecord(...)
    try:
        record.status = IngestStatus.LOADING
        responses = run_data_loader(...)

        record.status = IngestStatus.PROCESSING
        result = run_data_processing(...)

        record.status = IngestStatus.DONE
        return result

    except Exception as e:
        record.status = IngestStatus.FAILED
        record.error_messages.append(str(e))
        raise  # Airflow cần biết task bị lỗi → re-raise
```

Nếu `data_loader` bị lỗi, exception được bắt, record chuyển sang `FAILED`, rồi exception được raise lại để Airflow đánh dấu task đó là failed.

---

## 6. Tóm lại

| Bước | Ai làm | Kết quả |
|------|--------|---------|
| 1 | `data_loader` | Tải file về `/tmp/`, trả về `list[DownloadResponse]` |
| 2 | `data_processing` | Xử lý data, trả về `ProcessedData` |
| 3 | `data_ingest` | Nhận `ProcessedData`, lưu vào PostgreSQL |

`data_ingest` biết **khi nào** gọi cái gì, nhưng không biết **cách** tải file hay **cách** xử lý data. Đó là nhiệm vụ của các module kia.
