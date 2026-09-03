# 06 — Airflow DAGs — Lịch chạy tự động

Airflow là công cụ lên lịch: "Mỗi ngày 8 giờ sáng, chạy pipeline này." Nó không biết gì về business logic — chỉ biết **khi nào** chạy và **task nào** chạy trước.

---

## 1. Airflow 3.x — Những thứ khác so với bản cũ

Nếu bạn đọc tutorial cũ online, cẩn thận vì Airflow 3.x thay đổi nhiều:

| Airflow 2.x (cũ) | Airflow 3.x (dự án này) |
|------------------|------------------------|
| `webserver` | `api-server` |
| `scheduler` xử lý DAG file | `dag-processor` riêng biệt |
| `airflow.cfg` | Cấu hình qua env vars |
| `from airflow.operators.python import PythonOperator` | Vẫn dùng được, nhưng `@task` decorator phổ biến hơn |

**Quan trọng nhất:** trong `docker-compose.yml`, bạn sẽ thấy service `api-server` chứ không phải `webserver`. Đừng nhầm.

---

## 2. Cấu trúc `dags/`

```
dags/
├── dag_ingest.py          ← DAG chính: sensor → pipeline → email
├── dag_export.py          ← DAG xuất báo cáo
├── send_email_dag.py      ← DAG gửi email riêng
│
├── sensors/
│   └── google_drive_sensor.py   ← Chờ file xuất hiện trên GDrive
│
├── utils/
│   └── notifications.py         ← Hàm gửi email thông báo
│
└── wrapper/
    ├── pipeline_wrapper.py      ← Gọi src/data_ingest/
    ├── export_wrapper.py        ← Gọi src/ để xuất data
    └── google_drive_sensor.py   ← Wrap sensor logic
```

**Quy tắc quan trọng:** `dags/` không được import trực tiếp từ `src/`. Tất cả phải đi qua `dags/wrapper/`. Wrapper là cầu nối duy nhất.

---

## 3. `dag_ingest.py` — DAG chính

DAG này có 3 task chạy theo thứ tự:

```python
# dag_ingest.py (đơn giản hoá)

@dag(
    dag_id="hs_code_ingest",
    schedule="0 8 * * *",   # 8h sáng mỗi ngày
    start_date=datetime(2026, 1, 1),
)
def hs_code_ingest_dag():

    @task.sensor(poke_interval=60, timeout=3600)
    def wait_for_file(execution_date=None) -> PokeReturnValue:
        # Chờ file xuất hiện trên Google Drive
        found = google_drive_sensor.check(execution_date)
        return PokeReturnValue(is_done=found)

    @task
    def run_pipeline(execution_date=None):
        # Gọi wrapper, wrapper gọi src/
        return pipeline_wrapper.run(
            source="google_drive",
            execution_date=execution_date
        )

    @task
    def send_notification(result):
        notifications.send_email(result)

    # Kết nối 3 task
    file_found = wait_for_file()
    result     = run_pipeline()
    send_notification(result)

    file_found >> result  # sensor xong mới chạy pipeline
```

---

## 4. `dags/wrapper/` — Cầu nối

Tại sao cần wrapper thay vì import thẳng?

- DAG file phải nhẹ — Airflow đọc tất cả DAG file mỗi vài giây để detect thay đổi. Import nặng = chậm
- Tách biệt rõ ràng: DAG biết **khi nào** chạy, wrapper biết **làm gì**

```python
# dags/wrapper/pipeline_wrapper.py
from src.data_ingest.application.entrypoints import run_pipeline

def run(source: str, execution_date: str, **kwargs):
    return run_pipeline(source=source, execution_date=execution_date, **kwargs)
```

Wrapper chỉ forward call — không có logic gì thêm.

---

## 5. Sensor — Chờ điều kiện

Sensor là loại task đặc biệt: nó **lặp đi lặp lại** cho đến khi điều kiện thoả mãn.

```python
@task.sensor(
    poke_interval=60,    # Kiểm tra mỗi 60 giây
    timeout=3600,        # Sau 1 tiếng mà không có file → FAIL
    mode="poke"          # Chiếm worker trong lúc chờ
)
def wait_for_file(execution_date=None) -> PokeReturnValue:
    result = google_drive_sensor.check(execution_date)
    return PokeReturnValue(is_done=result.is_done, xcom_value=result.file_id)
```

`PokeReturnValue(is_done=True)` → Airflow biết sensor xong, chạy task tiếp theo.

---

## 6. XCom — Truyền data giữa tasks

Airflow task là hàm Python độc lập — không chia sẻ biến. Để truyền data, dùng XCom:

```python
@task
def run_pipeline(**context):
    result = pipeline_wrapper.run(...)
    # Airflow tự lưu return value vào XCom
    return result.model_dump()   # Phải serializable (dict/JSON)

@task
def send_notification(result: dict):
    # result được Airflow inject từ XCom của task trước
    notifications.send_email(result)

# Kết nối: output của run_pipeline → input của send_notification
send_notification(run_pipeline())
```

**Lưu ý:** XCom chỉ dùng cho data nhỏ (< vài MB). DataFrame lớn → lưu file, truyền path qua XCom.

---

## 7. Tóm lại

```
Mỗi ngày 8h sáng:
    1. wait_for_file (Sensor)
       └─ Poke GDrive mỗi 60s cho đến khi thấy file
    2. run_pipeline (Task)
       └─ wrapper → src/data_ingest → loader → processing → save DB
    3. send_notification (Task)
       └─ Gửi email kết quả
```

DAG chỉ định nghĩa **thứ tự** và **lịch**. Business logic nằm trong `src/`.
