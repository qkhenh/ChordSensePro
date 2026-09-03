# 12 — Python Nâng cao — Các tính năng dùng trong dự án

File này giải thích những tính năng Python bạn sẽ gặp khi đọc code dự án. Mỗi mục có ví dụ từ code thực.

---

## 1. `ABC` và `@abstractmethod` — Bắt buộc implement

`ABC` (Abstract Base Class) là cách nói: "Class này chỉ là khung — subclass **bắt buộc phải** implement những method này, nếu không sẽ báo lỗi khi khởi tạo."

```python
from abc import ABC, abstractmethod

class BaseProcessingHandler(ABC):
    @abstractmethod
    def handle(self, data: ProcessedData) -> ProcessedData:
        pass   # Không cần code ở đây
```

```python
# Nếu quên implement handle():
class MyHandler(BaseProcessingHandler):
    pass   # Thiếu handle()

MyHandler()  # TypeError: Can't instantiate abstract class MyHandler
             # without an implementation for abstract method 'handle'
```

Dự án dùng pattern này cho `BaseProcessingHandler`, `BaseRequestHandler`, và các repository.

---

## 2. `Protocol` — Duck typing

`Protocol` linh hoạt hơn `ABC`. Không cần kế thừa — chỉ cần có đúng method:

```python
from typing import Protocol

class Downloader(Protocol):
    def download(self, file, dest_path, **kwargs) -> list[DownloadResponse]: ...
    def get_file_info(self, **kwargs) -> BaseFileModel: ...
```

```python
class GoogleDriveDownloader:   # Không kế thừa Downloader!
    def download(self, file, dest_path, **kwargs):
        ...
    def get_file_info(self, **kwargs):
        ...

# Vẫn hợp lệ vì có đủ 2 method
dispatcher.register(FileSource.GOOGLE_DRIVE, GoogleDriveDownloader())
```

**Khi nào dùng Protocol vs ABC:**
- `ABC`: muốn bắt lỗi khi quên implement (kế thừa bắt buộc)
- `Protocol`: muốn linh hoạt hơn, không bắt kế thừa (duck typing)

---

## 3. `Generic[T]` và `TypeVar`

`TypeVar` đặt tên cho "kiểu chưa biết". `Generic[T]` tạo class có thể nhận nhiều kiểu:

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class BaseMongoRepository(Generic[T]):
    model: type[T]

    def find_by_id(self, id: str) -> T:
        doc = self.collection.find_one({"_id": ObjectId(id)})
        return self.model(**doc)   # Trả về đúng kiểu T

    def insert_one(self, item: T) -> T:
        self.collection.insert_one(item.model_dump())
        return item
```

```python
# Khi dùng, T được binding thành GoogleDriveFile
class GoogleDriveFileRepository(BaseMongoRepository[GoogleDriveFile]):
    model = GoogleDriveFile
    # find_by_id() bây giờ trả về GoogleDriveFile, không phải "T" mơ hồ
```

---

## 4. `@property` — Computed attributes

`@property` tạo attribute được tính từ các attribute khác, nhưng dùng như thuộc tính bình thường:

```python
class IngestRecord(BaseModel):
    total_files: int = 0
    processed_files: int = 0

    @property
    def progress_pct(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.processed_files / self.total_files * 100

record = IngestRecord(total_files=100, processed_files=35)
print(record.progress_pct)   # 35.0  — dùng như thuộc tính, không phải method
```

---

## 5. `ClassVar` — Biến của class, không phải instance

```python
from typing import ClassVar
from pydantic import BaseModel

class GoogleDriveFileRepository(BaseMongoRepository):
    collection_name: ClassVar[str] = "google_drive_files"
    # ClassVar = biến này thuộc về class, không phải mỗi instance

# Tất cả instances dùng chung
repo1 = GoogleDriveFileRepository()
repo2 = GoogleDriveFileRepository()
# repo1.collection_name == repo2.collection_name == "google_drive_files"
```

Trong Pydantic, `ClassVar` quan trọng vì Pydantic sẽ cố validate tất cả field — nhưng `ClassVar` được bỏ qua.

---

## 6. Singleton Pattern — Chỉ tạo 1 instance

`Singleton` đảm bảo chỉ có **1 instance** tồn tại trong toàn bộ chương trình. Dùng cho database connections, config objects.

```python
class DatabaseConnection:
    _instance: "DatabaseConnection | None" = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connect()   # Kết nối DB chỉ 1 lần
        return cls._instance

    def _connect(self):
        self.engine = create_engine(settings.database_url)

# Lần 1: tạo mới, kết nối DB
conn1 = DatabaseConnection()

# Lần 2: trả về instance cũ, KHÔNG kết nối lại
conn2 = DatabaseConnection()

assert conn1 is conn2   # True — cùng 1 object
```

---

## 7. `Lock` — Thread safety

Khi nhiều threads cùng sửa một biến, có thể xảy ra race condition. `Lock` đảm bảo chỉ 1 thread được thực thi đoạn code quan trọng tại một thời điểm:

```python
from threading import Lock

class FileTracker:
    def __init__(self):
        self._processed: list[str] = []
        self._lock = Lock()

    def mark_processed(self, file_id: str):
        with self._lock:
            # Chỉ 1 thread được vào đây cùng lúc
            self._processed.append(file_id)
```

Dự án dùng Lock khi `handle_batch()` chạy nhiều threads song song.

---

## 8. `ThreadPoolExecutor` — Chạy song song

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def handle_batch(self, items: list) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tất cả jobs
        future_to_item = {
            executor.submit(self.handle_one, item): item
            for item in items
        }

        # Lấy kết quả khi xong
        for future in as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Xử lý lỗi từng item riêng, không dừng cả batch
                item = future_to_item[future]
                print(f"Error processing {item}: {e}")

    return results
```

`as_completed()` trả về futures theo thứ tự hoàn thành — không phải thứ tự submit.

---

## 9. Tóm lại

| Tính năng | Dùng trong dự án |
|-----------|-----------------|
| `ABC + @abstractmethod` | `BaseProcessingHandler`, `BaseRequestHandler` |
| `Protocol` | `Downloader` port trong `data_loader` |
| `Generic[T]` | `BaseMongoRepository[T]`, `BaseRequestHandler[TInput, TOutput]` |
| `@property` | Computed fields trong models |
| `ClassVar` | `collection_name` trong repositories |
| Singleton (`__new__`) | Database connection classes |
| `Lock` | Thread-safe operations trong batch processing |
| `ThreadPoolExecutor` | `handle_batch()` trong AI handlers |
