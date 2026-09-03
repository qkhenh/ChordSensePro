# 01 — Tại sao code lại tổ chức theo cách đó?

File này giải thích lý do đằng sau cách tổ chức thư mục trong `src/`. Đọc xong bạn sẽ hiểu tại sao có folder `domain/`, `application/`, `infrastructure/` thay vì chỉ bỏ tất cả vào một chỗ.

---

## 1. Vấn đề với cách code bình thường

Giả sử bạn viết một hàm tải file từ Google Drive theo cách đơn giản nhất:

```python
# ❌ Viết kiểu này thì nhanh, nhưng sau này khổ
def download_google_drive_file(file_id: str):
    # Kết nối GDrive
    creds = service_account.Credentials.from_service_account_file("key.json")
    service = build("drive", "v3", credentials=creds)
    
    # Logic kiểm tra "đã tải chưa"
    if mongo.find_one({"file_id": file_id}):
        return "already downloaded"
    
    # Tải file
    request = service.files().get_media(fileId=file_id)
    ...
    
    # Lưu vào DB
    mongo.insert_one({"file_id": file_id, "status": "done"})
```

**Nhìn thì ổn, nhưng:**
- Muốn **test** logic "đã tải chưa" → phải có MongoDB thật đang chạy → phức tạp
- Muốn **đổi từ GDrive sang S3** → phải sửa tất cả mọi thứ trong hàm này
- Muốn **đổi từ MongoDB sang PostgreSQL** → cũng phải sửa hàm này
- Sau 6 tháng, hàm này có 200 dòng, 5 người cùng sửa, không ai hiểu nữa

**Gốc rễ vấn đề:** Logic kiểm tra "đã tải chưa" bị lẫn với code Google Drive API và MongoDB. Ba thứ không liên quan đến nhau lại nằm chung một chỗ.

---

## 2. Giải pháp: Tách ra 3 lớp

**DDD (Domain-Driven Design)** là cách tổ chức code theo 3 lớp, mỗi lớp có một nhiệm vụ riêng:

```
INFRASTRUCTURE LAYER  ← biết về: MongoDB, GDrive API, AWS S3, HTTP
        ↑ phụ thuộc vào
APPLICATION LAYER     ← biết về: use case, workflow, thứ tự thực hiện
        ↑ phụ thuộc vào
DOMAIN LAYER          ← chỉ biết Python thuần + Pydantic, không biết DB hay API nào
```

**Quy tắc duy nhất cần nhớ:** Lớp dưới không được biết lớp trên. Domain không được import MongoDB. Application không được import chi tiết của Infrastructure.

Nói đơn giản hơn: **logic nghiệp vụ (domain) không phụ thuộc vào công nghệ cụ thể**.

---

## 3. Mapping vào project

### data_loader/
```
data_loader/
├── domain/               ← DOMAIN: models + contracts
│   ├── models/           ← GoogleDriveFile, S3File, DownloadResponse (Pydantic)
│   ├── ports/            ← Downloader (Protocol — interface)
│   └── services/
│       └── file_dispatcher.py   ← Registry dispatcher (không biết GDrive hay S3)
│
├── application/          ← APPLICATION: implementations
│   └── downloaders/      ← GoogleDriveDownloader, S3Downloader, ApiDownloader
│
└── infrastructure/       ← INFRASTRUCTURE: kết nối bên ngoài
    └── repositories/     ← MongoDB implementation
```

### data_processing/
```
data_processing/
├── domain/
│   └── models/           ← ProcessedData, BaseProcessingHandler (abstract)
│
└── application/
    ├── pipeline_factory.py
    └── pipeline_handlers/ ← ValidationHandler, SavingHandler, ...
```

---

## 4. Các pattern được dùng trong project

### Pattern 1: Port & Adapter

**Port** = interface mà domain yêu cầu. **Adapter** = class thật implement interface đó.

```python
# PORT — domain/ports/downloader.py
# Domain chỉ nói: "tôi cần một thứ có thể download"
class Downloader(Protocol):
    def download(self, file, dest_path, **kwargs) -> list[DownloadResponse]: ...
    def get_file_info(self, **kwargs) -> BaseFileModel: ...

# ADAPTER 1 — application/downloaders/google_drive_downloader.py
class GoogleDriveDownloader:
    def download(self, file, dest_path, **kwargs):
        # gọi Google Drive API thật

# ADAPTER 2 — application/downloaders/s3_downloader.py
class S3Downloader:
    def download(self, file, dest_path, **kwargs):
        # gọi AWS S3 thật
```

`FileDispatcher` chỉ biết `Downloader` Protocol — không quan tâm bên dưới là GDrive hay S3. Muốn thêm nguồn mới → tạo Adapter mới, không sửa FileDispatcher.

---

### Pattern 2: Registry (thay cho if/elif dài)

```python
# ❌ Cách cũ — mỗi lần thêm nguồn mới phải sửa đây
def get_downloader(source: str):
    if source == "google_drive": return GoogleDriveDownloader()
    elif source == "s3": return S3Downloader()
    # Muốn thêm "ftp"? → phải sửa hàm này

# ✅ Registry — thêm mới không cần sửa FileDispatcher
class FileDispatcher:
    def __init__(self):
        self._registry: dict[FileSource, Downloader] = {}
    
    def register(self, source: FileSource, downloader: Downloader):
        self._registry[source] = downloader  # Đăng ký vào dict
    
    def download(self, file, dest_path, **kwargs):
        handler = self._registry[file.original]  # Lookup dict → gọi đúng downloader
        return handler.download(file, dest_path, **kwargs)
```

---

### Pattern 3: Repository (ẩn chi tiết database)

Business logic không nên biết đang dùng MongoDB hay PostgreSQL. Repository che đi chi tiết đó.

```python
# Contract (abstract) — base_repository.py
class BaseRepository(ABC, Generic[T]):
    @abstractmethod
    def find_one(self, **kwargs) -> T | None: ...
    @abstractmethod
    def insert_one(self, model: T) -> str | None: ...

# Implementation cụ thể — chỉ cần khai báo model và collection_name
class ProcessingResultRepository(BaseMongoRepository[ProcessingResult]):
    def __init__(self):
        super().__init__(
            model=ProcessingResult,
            collection_name="processing_results"
        )
    # Không cần viết thêm gì — kế thừa toàn bộ find_one, insert_one, ...
```

---

### Pattern 4: Chain of Responsibility (pipeline xử lý)

Thay vì một hàm khổng lồ làm tất cả mọi thứ, mỗi bước là một Handler riêng. Xử lý xong thì chuyển cho Handler tiếp theo.

```python
# Kết nối chain
validation_handler = ValidationHandler()
saving_handler = SavingHandler()

validation_handler.set_next(saving_handler)  # Sau validate thì đến save

# Chạy — tự động đi qua toàn bộ chain
result = validation_handler.handle(data)
```

Muốn thêm bước mới (ví dụ: `AIEnrichmentHandler`) → tạo class mới, chèn vào chain, không cần sửa handlers cũ.

---

## 5. Quy tắc import — cái quan trọng nhất

```
✅ OK:
  infrastructure → application → domain

❌ KHÔNG được:
  domain → infrastructure   (domain không được biết MongoDB)
  data_loader → data_processing  (modules không import chéo nhau)
```

Khi đọc file trong `domain/`, bạn sẽ không bao giờ thấy `import pymongo` hay `import boto3`. Đó là dấu hiệu code được viết đúng.

---

## 6. Tóm lại

| Lớp | Chứa gì | Biết gì |
|---|---|---|
| **Domain** | Models, interfaces (Protocol/ABC) | Chỉ Python + Pydantic |
| **Application** | Downloaders, handlers, use cases | Biết Domain, biết cách gọi Infrastructure |
| **Infrastructure** | DB clients, API clients, repos | Biết tất cả, implement contracts của Domain |

Khi cần **thêm nguồn dữ liệu mới**: tạo Adapter mới + đăng ký vào Registry.  
Khi cần **đổi database**: chỉ viết lại Infrastructure, không đụng đến Domain.  
Khi cần **test**: test Domain mà không cần database hay internet.
