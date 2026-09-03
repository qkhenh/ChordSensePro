# 09 — Pydantic — Validate dữ liệu tự động

Pydantic là thư viện dùng để định nghĩa cấu trúc dữ liệu và tự động validate. Thay vì tự viết code kiểm tra "trường này có phải int không?", Pydantic làm hết.

**Dự án này dùng Pydantic v2** — cú pháp khác với v1 khá nhiều.

---

## 1. `BaseModel` — Khai báo cấu trúc

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# Tạo object — Pydantic tự validate
user = User(name="An", age=25, email="an@example.com")

# Nếu sai kiểu, Pydantic báo lỗi ngay
User(name="An", age="not_a_number", email="...")
# ValidationError: age must be int
```

---

## 2. `Field` — Thêm ràng buộc

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0)               # gt = greater than (> 0)
    quantity: int = Field(ge=0, default=0)   # ge = greater or equal (>= 0)
    description: str | None = Field(default=None)
```

---

## 3. `@field_validator` — Validate tuỳ chỉnh

```python
from pydantic import BaseModel, field_validator

class HsCodeRecord(BaseModel):
    hs_code: str
    declaration_number: int

    @field_validator("hs_code")
    @classmethod
    def hs_code_must_be_numeric(cls, v: str) -> str:
        if not v.strip().isdigit():
            raise ValueError(f"hs_code phải là số, nhận được: {v}")
        return v.strip()  # Trả về giá trị đã clean
```

`@field_validator` chạy sau khi Pydantic validate kiểu dữ liệu, cho phép thêm logic custom.

---

## 4. `model_copy` — Cập nhật field (v2)

Trong Pydantic v2, model là **immutable** (không thể sửa trực tiếp). Dùng `model_copy(update=...)` để tạo bản sao với field mới:

```python
# ĐÚNG (Pydantic v2)
updated_record = record.model_copy(update={"status": "DONE"})

# SAI — Pydantic v2 sẽ báo lỗi
record.status = "DONE"  # ValidationError!
```

Trong dự án, pattern này hay dùng để cập nhật trạng thái:
```python
ingest_record = ingest_record.model_copy(update={
    "status": IngestStatus.PROCESSING,
    "total_files": len(responses),
})
```

---

## 5. `@field_serializer` — Custom JSON output

Khi convert model sang JSON (ví dụ để lưu vào XCom), đôi khi cần custom cách serialize:

```python
from pydantic import BaseModel, field_serializer
import pandas as pd

class ProcessedData(BaseModel):
    structured_data: dict

    @field_serializer("structured_data", when_used="json")
    def serialize_structured_data(self, value: dict) -> dict:
        result = {}
        for key, item in value.items():
            if isinstance(item, pd.DataFrame):
                result[key] = item.to_dict(orient="records")  # DataFrame → list[dict]
            else:
                result[key] = item
        return result

# Khi gọi .model_dump(mode="json"), field_serializer tự chạy
data.model_dump(mode="json")
```

---

## 6. Generic Models — Model tái sử dụng

`Generic[T]` cho phép tạo model dùng được với nhiều kiểu dữ liệu:

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class PaginatedResult(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int

# Dùng với bất kỳ kiểu nào
result: PaginatedResult[HsCodeRecord] = PaginatedResult(
    items=[HsCodeRecord(...)],
    total=1540,
    page=1,
    page_size=50,
)
```

---

## 7. `BaseSettings` — Config từ env vars

```python
from pydantic_settings import BaseSettings

class DatabaseSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    name: str
    user: str
    password: str

    class Config:
        env_prefix = "POSTGRES_"   # Đọc từ POSTGRES_HOST, POSTGRES_PORT, ...
        env_file = ".env"          # Hoặc từ file .env

# Pydantic tự đọc từ environment variables
settings = DatabaseSettings()
# settings.host = giá trị của POSTGRES_HOST trong env
```

Dự án dùng pattern này trong `src/shared/infrastructure/setting/`.

---

## 8. Tóm lại

| Tính năng | Dùng khi nào |
|-----------|-------------|
| `BaseModel` | Định nghĩa bất kỳ data class nào |
| `Field(gt=0, ...)` | Thêm ràng buộc giá trị |
| `@field_validator` | Validate logic phức tạp |
| `model_copy(update=...)` | Cập nhật field (không mutate trực tiếp) |
| `@field_serializer` | Tuỳ chỉnh cách serialize sang JSON |
| `Generic[T]` | Tạo model tái sử dụng với nhiều kiểu |
| `BaseSettings` | Đọc config từ env vars / .env file |
