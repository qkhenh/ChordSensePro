# 11 — MongoDB vs PostgreSQL — Hai database, hai mục đích

Dự án này dùng cả hai database. Không phải vì thích dùng nhiều — mà vì chúng phù hợp với những loại dữ liệu khác nhau.

---

## 1. So sánh nhanh

| | MongoDB | PostgreSQL |
|--|---------|-----------|
| Lưu dữ liệu dạng | JSON document (linh hoạt) | Bảng cố định (strict) |
| Schema | Không bắt buộc | Bắt buộc định nghĩa trước |
| Khi nào dùng | Metadata, tracking, data chưa chuẩn | Data đã chuẩn hoá, cần query SQL |
| Trong dự án | Track trạng thái download file | Lưu HS code records đã xử lý |

---

## 2. MongoDB — Dùng cho gì trong dự án

MongoDB lưu **metadata về quá trình tải file** — không phải data thương mại.

```python
# Ví dụ document trong MongoDB
{
    "_id": ObjectId("665a1b2c3d4e5f6a7b8c9d0e"),
    "file_id": "1O80UyeZUXugNk3QI1IASX2PWWoBfVO82",
    "name": "import_data_2026_05.csv",
    "source": "google_drive",
    "download_status": "SUCCESS",
    "downloaded_at": "2026-05-31T08:15:23Z",
    "local_path": "/tmp/data/import_data_2026_05.csv"
}
```

Tại sao MongoDB cho việc này?
- Schema linh hoạt: file từ GDrive có `drive_file_id`, file từ S3 có `bucket` + `key` — khác nhau, khó nhét vào 1 bảng SQL
- Không cần JOIN phức tạp
- Ghi nhanh khi update status liên tục

---

## 3. CRUD cơ bản với pymongo

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["sts_data"]
collection = db["google_drive_files"]

# Insert
collection.insert_one({
    "file_id": "abc123",
    "name": "import.csv",
    "status": "PENDING"
})

# Find một document
doc = collection.find_one({"file_id": "abc123"})

# Find nhiều document
docs = list(collection.find({"status": "PENDING"}))

# Update
collection.update_one(
    {"file_id": "abc123"},           # Filter
    {"$set": {"status": "SUCCESS"}}  # Update
)

# Delete
collection.delete_one({"file_id": "abc123"})
```

---

## 4. PostgreSQL — Dùng cho gì trong dự án

PostgreSQL lưu **dữ liệu HS code đã xử lý** — hàng nghìn records cần query, filter, aggregate.

```sql
-- Bảng hs_raw_data trong PostgreSQL
CREATE TABLE hs_raw_data (
    id              SERIAL PRIMARY KEY,
    declaration_number  BIGINT,
    transaction_date    DATE,
    hs_code             INTEGER,
    buyer_name          VARCHAR(255),
    importer_address_vn TEXT,
    total_amount_usd    DECIMAL(15, 2),
    execution_date      VARCHAR(20),
    created_at          TIMESTAMP DEFAULT NOW()
);
```

Tại sao PostgreSQL cho việc này?
- Schema cố định: 30 cột đều được khai báo, không có gì lạ
- Cần query phức tạp: `GROUP BY`, `JOIN`, window functions
- Cần đảm bảo data integrity (foreign keys, constraints)

---

## 5. Repository Pattern — Không gọi DB trực tiếp

Trong dự án, không ai gọi `pymongo` hay `psycopg2` trực tiếp từ business logic. Tất cả đi qua repository:

```python
# MongoDB repository
class GoogleDriveFileRepository(BaseMongoRepository[GoogleDriveFile]):
    collection_name = "google_drive_files"
    model = GoogleDriveFile

    def find_by_status(self, status: FileDownloadStatus) -> list[GoogleDriveFile]:
        docs = self.collection.find({"download_status": status.value})
        return [self.model(**doc) for doc in docs]

    def update_status(self, file_id: str, status: FileDownloadStatus):
        self.collection.update_one(
            {"file_id": file_id},
            {"$set": {"download_status": status.value}}
        )
```

```python
# PostgreSQL repository
class HsCodeRepository(BasePostgresRepository):
    def save_dataframe(self, df: pd.DataFrame, execution_date: str):
        df["execution_date"] = execution_date
        df.to_sql(
            "hs_raw_data",
            con=self.engine,
            if_exists="append",   # Thêm vào, không xoá bảng cũ
            index=False,
        )
```

---

## 6. Khi nào dùng cái nào

**Dùng MongoDB khi:**
- Data có cấu trúc khác nhau tuỳ trường hợp (polymorphic)
- Chủ yếu insert + update đơn giản
- Không cần query quan hệ phức tạp
- Tracking, logging, metadata

**Dùng PostgreSQL khi:**
- Data có schema cố định và biết trước
- Cần SQL query phức tạp (GROUP BY, JOIN, window functions)
- Cần ACID transactions
- Là data "sản phẩm cuối" của pipeline

---

## 7. Tóm lại

```
MongoDB   → Lưu metadata tải file (GoogleDriveFile, tracking status)
            Linh hoạt, không cần schema cứng

PostgreSQL → Lưu data đã xử lý (hs_raw_data rows)
             Strict schema, query SQL mạnh

Repository → Lớp trung gian giữa business logic và database
             Business logic không cần biết đang dùng DB gì
```
