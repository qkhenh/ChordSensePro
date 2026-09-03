# 02 — `shared/` — Nền tảng của toàn hệ thống

`src/shared/` là thư mục quan trọng nhất trong project. **Mọi module khác đều kế thừa từ đây.** Đọc file này trước khi đọc bất kỳ module nào.

Hình dung đơn giản: shared/ là bộ công cụ chung — ai cũng dùng, không ai tự làm lại từ đầu.

```
shared/
├── domain/              ← Models + contracts (thuần Python, không import gì ngoài pydantic)
│   ├── base_model.py        ← Tổ tiên của tất cả models
│   ├── base_file_model.py   ← Base cho file từ GDrive, S3, API
│   ├── processing_result.py ← Lưu kết quả sau mỗi lần pipeline chạy
│   ├── prompt.py            ← Data structures cho AI
│   └── prompt_template.py   ← Abstract template để build AI prompt
│
├── infrastructure/      ← Kết nối database + external services
│   ├── setting/             ← Đọc config từ .env
│   ├── mongo/               ← MongoDB client + base repository
│   ├── postgres/            ← PostgreSQL client + base repository
│   └── service/             ← GDrive, S3, HTTP, AI clients
│
└── utils/
    └── logging.py           ← Logger dùng chung
```

---

## 1. `CustomBaseModel` — Tổ tiên của mọi model

Mọi model trong project đều kế thừa từ class này. Nó bắt buộc implement 2 method:

```python
# src/shared/domain/base_model.py
from abc import ABC, abstractmethod
from pydantic import BaseModel

class CustomBaseModel(ABC, BaseModel):
    
    @classmethod
    @abstractmethod
    def _to_model(cls, doc: dict) -> "CustomBaseModel":
        """Chuyển dict từ MongoDB → Python object"""
        pass

    @abstractmethod
    def _to_doc(self) -> dict:
        """Chuyển Python object → dict để lưu vào MongoDB"""
        pass
```

**Tại sao quan trọng?** `BaseMongoRepository` gọi `._to_doc()` khi insert và `._to_model()` khi fetch về. Mọi model implement 2 method này là đảm bảo tương thích với mọi repository.

**Ví dụ implement:**
```python
class ProcessingResult(CustomBaseModel):
    result_id: str | None = None
    run_id: str
    summary: dict
    created_at: datetime

    @classmethod
    def _to_model(cls, doc: dict) -> "ProcessingResult":
        return cls(
            result_id=str(doc["_id"]) if doc.get("_id") else None,
            run_id=doc["run_id"],
            summary=doc.get("summary", {}),
            created_at=doc.get("created_at"),
        )

    def _to_doc(self) -> dict:
        return {
            "run_id": self.run_id,
            "summary": self.summary,
            "created_at": self.created_at,
        }
```

---

## 2. `BaseFileModel` — Đại diện chung cho mọi loại file

Dù file đến từ Google Drive, S3, hay HTTP API, chúng đều là subclass của `BaseFileModel`.

```python
# src/shared/domain/base_file_model.py
class FileSource(str, Enum):
    GOOGLE_DRIVE = "google_drive"
    S3 = "s3"
    API = "api"

class FileDownloadStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    SUCCESS = "success"
    FAILED = "failed"

class BaseFileModel(CustomBaseModel):
    file_id: str | None = None
    name: str
    date_create: datetime
    date_download: datetime | None = None
    dest_path: str | None = None        # Đường dẫn local sau khi tải về
    original: FileSource                # Từ đâu: google_drive | s3 | api
    download_status: FileDownloadStatus = FileDownloadStatus.PENDING

    model_config = ConfigDict(frozen=True)  # Bất biến sau khi tạo
```

Subclasses cụ thể thêm fields riêng:
- `GoogleDriveFile` → thêm `drive_file_id`, `mime_type`, `parents`
- `S3File` → thêm `bucket`, `key`, `size`, `etag`
- `ApiFile` → thêm `url`, `headers`

---

## 3. `AppBaseSetting` — Đọc config từ `.env`

Tất cả config (password DB, API key...) không hardcode trong code mà đọc từ biến môi trường.

```python
# src/shared/infrastructure/setting/base_setting.py
class AppBaseSetting(BaseSettings):
    model_config = ConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),  # Đọc từ file .env ở gốc project
        case_sensitive=False,
        extra="ignore"
    )

# Ví dụ setting cụ thể
class MongoSetting(AppBaseSetting):
    mongo_uri: str        # biến MONGO_URI trong .env
    mongo_db_name: str    # biến MONGO_DB_NAME trong .env

# Cách dùng
setting = MongoSetting()
client = MongoClient(setting.mongo_uri)
```

Nếu thiếu biến môi trường bắt buộc → Pydantic raise lỗi ngay khi khởi động, không âm thầm dùng giá trị sai.

---

## 4. `BaseMongoRepository` — Repository cho MongoDB

Đây là lớp quan trọng nhất của infrastructure layer. Nó implement toàn bộ CRUD cho MongoDB một lần, tất cả concrete repositories chỉ cần kế thừa.

```
BaseRepository[T] (ABC)        ← contract: định nghĩa các method CRUD
    └── BaseMongoRepository[T] ← implement tất cả CRUD cho MongoDB
            └── ProcessingResultRepository  ← chỉ khai báo model + collection_name
            └── GoogleDriveFileRepository   ← chỉ khai báo model + collection_name
```

**BaseRepository — contract:**
```python
class BaseRepository(ABC, Generic[T]):
    @abstractmethod
    def find_one(self, **kwargs) -> T | None: ...
    @abstractmethod
    def insert_one(self, model: T) -> str | None: ...
    @abstractmethod
    def find_many(self, **kwargs) -> list[T]: ...
    @abstractmethod
    def update_one(self, model: T) -> bool: ...
    @abstractmethod
    def delete_one(self, **kwargs) -> bool: ...
```

**BaseMongoRepository — tất cả logic ở đây:**
```python
class BaseMongoRepository(BaseRepository[T]):
    def find_one(self, **kwargs) -> T | None:
        raw = self._get_collection().find_one(kwargs)
        return self.model._to_model(raw) if raw else None   # Gọi _to_model()

    def insert_one(self, model: T) -> str | None:
        doc = model._to_doc()                               # Gọi _to_doc()
        result = self._get_collection().insert_one(doc)
        return str(result.inserted_id)
```

**Concrete repository — chỉ 4 dòng:**
```python
class ProcessingResultRepository(BaseMongoRepository[ProcessingResult]):
    def __init__(self):
        super().__init__(
            model=ProcessingResult,
            collection_name="processing_results"
        )
    # Xong! Được kế thừa toàn bộ CRUD
```

---

## 5. `BasePostgresRepository` — Repository cho PostgreSQL

Tương tự MongoDB nhưng cho PostgreSQL. Subclass chỉ cần khai báo `table_name` và `_from_row()`.

```python
class BasePostgresRepository(ABC, Generic[T]):
    table_name: ClassVar[str]  # Subclass khai báo tên bảng
    
    def insert(self, data: T) -> None:
        doc = data._to_doc()
        cols = ", ".join(doc.keys())
        vals = ", ".join(["%s"] * len(doc))
        self._execute(f"INSERT INTO {self.table_name} ({cols}) VALUES ({vals})", list(doc.values()))
    
    def upsert(self, data: T, conflict_columns: list[str]) -> None:
        # INSERT ... ON CONFLICT (col) DO UPDATE SET ...
        ...
    
    @abstractmethod
    def _from_row(self, row: tuple) -> T: ...  # Subclass convert raw row → model
```

---

## 6. `BaseRequestHandler` — AI API Caller

Abstract class cho tất cả AI calls. Flow luôn cố định: Build Prompt → Gọi API → Parse kết quả.

```python
class BaseRequestHandler(ABC, Generic[RequestT, ResponseT]):
    
    def handle(self, request: RequestT) -> ResponseT:
        built_prompt = self._build_prompt(request)   # Subclass implement
        output = self._call_api(built_prompt)         # Đã implement sẵn
        return self._to_response(output, request)    # Subclass implement
    
    def handle_batch(self, requests: list[RequestT], max_workers=4) -> list[ResponseT]:
        """Gọi AI song song với ThreadPoolExecutor, giữ nguyên thứ tự kết quả."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.handle, req): i for i, req in enumerate(requests)}
            results = [None] * len(requests)
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results
```

Subclass chỉ cần implement `_build_prompt()` và `_to_response()`. Phần gọi API thật đã có sẵn.

---

## 7. Tóm lại — Những gì bạn cần nhớ

| Class | Ở đâu | Làm gì |
|---|---|---|
| `CustomBaseModel` | `shared/domain/base_model.py` | Base cho mọi model, bắt buộc `_to_model()` + `_to_doc()` |
| `BaseFileModel` | `shared/domain/base_file_model.py` | Base cho GoogleDriveFile, S3File, ApiFile |
| `AppBaseSetting` | `shared/infrastructure/setting/` | Đọc config từ `.env` |
| `BaseMongoRepository` | `shared/infrastructure/mongo/` | CRUD cho MongoDB, subclass chỉ cần 4 dòng |
| `BasePostgresRepository` | `shared/infrastructure/postgres/` | CRUD cho PostgreSQL |
| `BaseRequestHandler` | `shared/infrastructure/service/` | Abstract AI caller với batch support |