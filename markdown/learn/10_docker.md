# 10 — Docker — Đóng gói và chạy ứng dụng

Docker giải quyết vấn đề "chạy được trên máy tôi nhưng không chạy được trên máy bạn". Nó đóng gói toàn bộ môi trường (Python version, thư viện, config) vào một container.

---

## 1. Image vs Container

**Image** = công thức nấu ăn (bản thiết kế, không đổi)
**Container** = tô phở được nấu ra từ công thức đó (đang chạy)

```
Dockerfile  →  docker build  →  Image  →  docker run  →  Container
 (công thức)                  (template)                  (đang chạy)
```

Từ 1 image có thể chạy nhiều containers cùng lúc.

---

## 2. `Dockerfile` — Tạo image

```dockerfile
# Dockerfile trong dự án này
FROM apache/airflow:3.0.0-python3.11   # Bắt đầu từ image có sẵn

USER root
RUN apt-get update && apt-get install -y gcc   # Cài thêm package hệ thống

USER airflow
COPY requirements.txt .
RUN pip install -r requirements.txt   # Cài Python packages

COPY . /opt/airflow/                  # Copy code vào image
```

Mỗi dòng `RUN`, `COPY` tạo ra một **layer**. Docker cache từng layer — nếu `requirements.txt` không đổi, bước `pip install` được dùng lại từ cache → build nhanh hơn.

**Mẹo:** COPY requirements trước, cài pip, rồi mới COPY code. Như vậy khi sửa code, layer `pip install` vẫn được cache.

---

## 3. `docker-compose.yml` — Chạy nhiều services

Dự án này có nhiều services cần chạy cùng nhau: Airflow, PostgreSQL, MongoDB, Redis.

```yaml
# docker-compose.yml (đơn giản hoá)
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow

  redis:
    image: redis:7

  api-server:          # Giao diện web Airflow
    build: .           # Build từ Dockerfile trong thư mục hiện tại
    command: api-server
    ports:
      - "8080:8080"
    depends_on:
      postgres:
        condition: service_healthy   # Chờ postgres healthy mới start

  scheduler:
    build: .
    command: scheduler
    depends_on:
      - api-server

  worker:
    build: .
    command: celery worker
```

---

## 4. YAML Anchors — Tránh lặp lại

Trong `docker-compose.yml` của dự án, nhiều services dùng chung cấu hình. YAML anchors (`&`, `*`, `<<:`) tránh copy-paste:

```yaml
# Định nghĩa template chung với &airflow-common
x-airflow-common: &airflow-common
  image: ${AIRFLOW_IMAGE_NAME}
  environment:
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://...
  volumes:
    - ./dags:/opt/airflow/dags
    - ./src:/opt/airflow/src
  depends_on:
    postgres:
      condition: service_healthy

# Dùng lại với <<: *airflow-common
scheduler:
  <<: *airflow-common          # Kế thừa toàn bộ cấu hình chung
  command: scheduler           # Chỉ override phần khác

worker:
  <<: *airflow-common
  command: celery worker
```

`<<: *airflow-common` = "copy tất cả từ airflow-common vào đây, nhưng những gì tôi khai báo thêm sẽ override."

---

## 5. Volumes — Chia sẻ file

```yaml
volumes:
  - ./dags:/opt/airflow/dags     # Bind mount: thư mục local ↔ container
  - ./src:/opt/airflow/src       # Sửa code local → thay đổi ngay trong container
  - postgres-data:/var/lib/postgresql/data   # Named volume: data tồn tại kể cả khi container restart
```

Hai loại volume:
- **Bind mount** (`./folder:/path`): Map thư mục local vào container. Dùng để dev — sửa code local thấy ngay
- **Named volume** (`postgres-data:/path`): Docker quản lý. Dùng để persist data (DB)

---

## 6. Healthcheck và `depends_on`

```yaml
postgres:
  image: postgres:16
  healthcheck:
    test: ["CMD", "pg_isready", "-U", "airflow"]
    interval: 5s
    retries: 5

api-server:
  depends_on:
    postgres:
      condition: service_healthy   # Chờ postgres pass healthcheck mới start
```

Nếu không có `depends_on`, api-server có thể start trước khi postgres sẵn sàng → connection error.

---

## 7. Các lệnh hay dùng

```bash
# Build và start tất cả services
docker compose up -d --build

# Xem logs của service
docker compose logs -f scheduler

# Chạy lệnh trong container đang chạy
docker compose exec api-server bash

# Dừng tất cả
docker compose down

# Dừng và xoá volumes (cẩn thận: mất data)
docker compose down -v
```

---

## 8. Tóm lại

```
Dockerfile   → định nghĩa image (môi trường)
Image        → template bất biến
Container    → instance đang chạy từ image

docker-compose.yml → khai báo nhiều containers cùng chạy
YAML anchors       → tránh lặp cấu hình
Bind mount         → sync code local ↔ container (dev)
Named volume       → persist data khi container restart
depends_on         → đảm bảo thứ tự start đúng
```
