# ChordSensePro DE — Setup Guide

> **Phạm vi:** Hướng dẫn cài đặt và chạy môi trường DE (Data Engineering) cho ChordSensePro.
> Bao gồm: Docker services, database setup, Airflow, và dev workflow.

---

## 1. Yêu cầu

| Công cụ | Version | Mục đích |
|---------|---------|---------|
| **Docker Desktop** | ≥ 4.x | Chạy tất cả services |
| **Python** | 3.11 / 3.12 | Local dev + unit tests |
| **Git** | any | Source control |
| **MongoDB Compass** | any | Xem MongoDB data (optional) |
| **TablePlus / DBeaver** | any | Xem PostgreSQL data (optional) |

---

## 2. Cấu trúc Services (Docker Compose)

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                       │
│                                                         │
│  postgres       :5432  ← App DB (chordsense)           │
│  postgres_meta  :5432  ← Airflow metadata DB           │
│  mongo          :27017 ← Raw audio blob storage        │
│  redis          :6379  ← Cache / broker                │
│  airflow        :8080  ← Pipeline orchestrator         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Lần đầu chạy (First-time Setup)

### Bước 1 — Clone và vào thư mục DE
```bash
git clone https://github.com/qkhenh/ChordSensePro.git
cd ChordSensePro/DE
```

### Bước 2 — Khởi động Docker services
```bash
docker compose up -d
```

> Lần đầu sẽ mất **5–10 phút** để build Docker image (cài ffmpeg, librosa, demucs...).

### Bước 3 — Kiểm tra services
```bash
docker compose ps
```

Tất cả phải ở trạng thái `healthy` trước khi tiếp tục:
```
NAME                     STATUS
chordsense_postgres      healthy
chordsense_airflow_meta  healthy
chordsense_mongo         healthy
chordsense_redis         healthy
chordsense_airflow       healthy
```

### Bước 4 — Chạy Database Migration (tạo tables)
```bash
docker exec -it chordsense_airflow bash -c "cd /opt/airflow && alembic upgrade head"
```

Tạo các tables:
- `song_analyses` — chord analysis results per song
- `chord_attempts` — user practice attempts
- `user_chord_mastery` — rolling mastery per (user, chord)
- `crawl_queue` — ETL crawl queue

### Bước 5 — Mở Airflow UI
```
URL:      http://localhost:8080
Username: admin
Password: admin
```

---

## 4. Thông tin kết nối

### PostgreSQL (App DB)
```
Host:     localhost
Port:     5432
Database: chordsense
User:     chordsense
Password: chordsense_dev
```
> Kết nối bằng TablePlus, DBeaver, hoặc psql.

### PostgreSQL (Airflow Metadata DB)
```
Host:     localhost
Port:     5433   ← chú ý port khác (nếu expose, mặc định không expose ra ngoài)
Database: airflow
User:     airflow
Password: airflow_dev
```

### MongoDB
```
Connection string: mongodb://localhost:27017
Database:          chordsense_raw
```
> Kết nối bằng MongoDB Compass: paste connection string vào ô "Connect".

### Redis
```
Host: localhost
Port: 6379
```

---

## 5. DAGs hiện có

| DAG | Schedule | Nhiệm vụ |
|-----|----------|---------|
| `dag_crawl_batch` | `0 */6 * * *` (mỗi 6 tiếng) | Lấy URLs từ `crawl_queue` → chạy ETL → log summary |
| `dag_analytics_rollup` | `30 23 * * *` (mỗi đêm) | Rollup chord_attempts → cập nhật user_chord_mastery |

> Cả 2 DAG mặc định ở trạng thái **paused**. Vào Airflow UI để bật.

---

## 6. Dev Workflow (không cần rebuild Docker)

Volumes được mount trực tiếp từ host vào container:
```yaml
./dags:/opt/airflow/dags   # thay đổi DAG → tự reload
./src:/opt/airflow/src     # thay đổi src → tự reload
```

Tức là: **sửa code → lưu → Airflow tự pick up**, không cần `docker compose restart`.

### Chạy unit tests (local, không cần Docker)
```bash
cd DE
python -m pytest tests/unit -v
```

### Xem logs Airflow
```bash
docker compose logs -f airflow
```

### Xem logs 1 service cụ thể
```bash
docker compose logs -f postgres
docker compose logs -f mongo
```

---

## 7. Thêm URL vào crawl queue (chuẩn bị ETL)

Hiện tại chưa có script tự động. Thêm thủ công qua psql:

```bash
docker exec -it chordsense_postgres psql -U chordsense -d chordsense
```

```sql
INSERT INTO crawl_queue (id, source_url, status, priority, created_at)
VALUES (gen_random_uuid(), 'https://youtu.be/...', 'pending', 0, NOW());
```

> Script `scripts/add_urls.py` sẽ được thêm sau để batch insert từ file.

---

## 8. Tear Down

### Dừng nhưng giữ data
```bash
docker compose down
```

### Dừng và xóa toàn bộ data (volumes)
```bash
docker compose down -v
```
> ⚠️ `-v` xóa sạch database, chỉ dùng khi muốn reset hoàn toàn.

---

## 9. Đổi Password (nếu cần)

Sửa trực tiếp trong `DE/docker-compose.yml`:

```yaml
# App DB — đổi cả 2 chỗ cho khớp nhau
postgres:
  environment:
    POSTGRES_PASSWORD: <new_password>

airflow:
  environment:
    POSTGRES_PASSWORD: <new_password>   # phải giống với postgres ở trên
```

Sau đó reset:
```bash
docker compose down -v   # xóa volume cũ
docker compose up -d     # khởi động với password mới
```

---

## 10. Troubleshooting

### Airflow không lên được
```bash
docker compose logs airflow | tail -50
```
Thường do `postgres_meta` chưa `healthy`. Chờ thêm hoặc restart:
```bash
docker compose restart airflow
```

### Migration fail
```bash
# Chạy trong container để xem lỗi chi tiết
docker exec -it chordsense_airflow bash
cd /opt/airflow
alembic upgrade head
```

### Port đã bị dùng
Nếu port `5432` hoặc `8080` bị conflict, sửa trong `docker-compose.yml`:
```yaml
ports:
  - "5433:5432"   # đổi port ngoài (host)
```
