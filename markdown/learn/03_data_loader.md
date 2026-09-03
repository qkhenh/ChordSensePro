# 03 — `data_loader` — Tải file từ nhiều nguồn

Module này làm **một việc duy nhất**: nhận yêu cầu "hãy tải file từ nguồn X", rồi tải về máy local.

Không validate, không transform, không làm gì khác. Chỉ tải.

---

## 1. Input và Output

```python
# Input: nói cho biết tải từ đâu
run_data_loader(
    source="google_drive",
    execution_date="2026-05-31",
    dest_path="/tmp/data/",
    file_id="1O80UyeZUXugNk3QI1IASX2PWWoBfVO82"
)

# Output: danh sách file đã tải về
# [DownloadResponse(id="1O80...", local_path="/tmp/data/import.csv", status=SUCCESS)]
```

---

## 2. Cấu trúc module

```
data_loader/
├── application/
│   ├── entrypoints.py           ← run_data_loader() — cổng vào duy nhất
│   └── downloaders/
│       ├── google_drive_downloader.py
│       ├── s3_downloader.py
│       └── api_downloader.py
│
├── domain/
│   ├── models/                  ← GoogleDriveFile, S3File, ApiFile, DownloadResponse
│   ├── ports/
│   │   └── downloader.py        ← Downloader Protocol (interface)
│   └── services/
│       └── file_dispatcher.py   ← Registry: map source → downloader
│
└── infrastructure/
    └── repositories/
        └── google_drive_file_repository.py  ← Lưu metadata vào MongoDB
```

---

## 3. `FileDispatcher` — Trung tâm điều phối

Đây là class quan trọng nhất của module. Nó dùng **Registry Pattern** — thay vì if/elif dài dòng, dùng một dict để map source → downloader:

```python
class FileDispatcher:
    def __init__(self):
        self._registry: dict[FileSource, Downloader] = {}

    def regist_all(self):
        self.register(FileSource.GOOGLE_DRIVE, GoogleDriveDownloader())
        self.register(FileSource.S3,           S3Downloader())
        self.register(FileSource.API,          ApiDownloader())

    def download(self, file, dest_path, **kwargs):
        handler = self._registry[file.original]  # Tìm đúng downloader
        return handler.download(file, dest_path, **kwargs)
```

Khi gọi `dispatcher.download(file, ...)`:
- `file.original == GOOGLE_DRIVE` → dùng `GoogleDriveDownloader`
- `file.original == S3` → dùng `S3Downloader`
- Không cần if/elif, chỉ cần lookup dict

**Thêm nguồn FTP mới:** tạo `FtpDownloader` + 1 dòng `register()`. Không sửa code khác.

---

## 4. `Downloader` Protocol — Interface

```python
# domain/ports/downloader.py
class Downloader(Protocol):
    def download(self, file, dest_path, **kwargs) -> list[DownloadResponse]: ...
    def get_file_info(self, **kwargs) -> BaseFileModel: ...
```

`Protocol` = duck typing. Bất kỳ class nào có 2 method này là hợp lệ — không cần kế thừa gì.

---

## 5. Các Downloader

### GoogleDriveDownloader
Phức tạp nhất vì GDrive có folder chứa nhiều file:

```python
# mode="skip": bỏ qua nếu file đã tồn tại → không tải lại
# mode="replace": xoá file cũ và tải lại

def download(self, file, dest_path, mode="skip", **kwargs):
    if file.mime_type == "folder":
        children = google_drive_service.list_files(file.drive_file_id)
        return [self._download_one(child, dest_path, mode) for child in children]
    else:
        return [self._download_one(file, dest_path, mode)]

def _download_one(self, file, dest_path, mode):
    local_path = os.path.join(dest_path, file.name)
    if mode == "skip" and os.path.exists(local_path):
        return DownloadResponse(status=SUCCESS)   # Bỏ qua
    google_drive_service.download_file(file.drive_file_id, local_path)
    repo.update_status(file.file_id, FileDownloadStatus.SUCCESS)
    return DownloadResponse(id=file.file_id, local_path=local_path, status=SUCCESS)
```

### S3Downloader
```python
def download(self, file: S3File, dest_path, **kwargs):
    local_path = s3_service.download_file(file.bucket, file.key, dest_path)
    return [DownloadResponse(id=file.file_id, local_path=local_path, status=SUCCESS)]
```

### ApiDownloader
```python
def download(self, file: ApiFile, dest_path, **kwargs):
    local_path = http_service.download_file(file.url, dest_path)
    return [DownloadResponse(id=file.file_id, local_path=local_path, status=SUCCESS)]
```

---

## 6. Models

### `GoogleDriveFile` (subclass của `BaseFileModel`)

```python
class GoogleDriveFile(BaseFileModel):
    original: FileSource = FileSource.GOOGLE_DRIVE
    drive_file_id: str   # ID trên Google Drive (dùng để gọi API)
    mime_type: str        # "text/csv" | "application/vnd.google-apps.folder" | ...
```

Có 2 loại ID khác nhau — dễ nhầm:
- `file_id` = MongoDB ObjectId (track trong hệ thống nội bộ)
- `drive_file_id` = Google Drive ID thực (dùng khi gọi GDrive API)

### `DownloadResponse` — Output chuẩn

```python
class DownloadResponse(CustomBaseModel):
    id: str
    local_path: str
    file_download_status: FileDownloadStatus  # SUCCESS | FAILED | PENDING
```

---

## 7. MongoDB Tracking

`GoogleDriveDownloader` lưu metadata vào MongoDB trước và sau khi tải:

```python
# Trước khi tải → PENDING
repo.insert_one(GoogleDriveFile(..., download_status=PENDING))

# Đang tải → DOWNLOADING
repo.update_status(file_id, DOWNLOADING)

# Tải xong → SUCCESS / FAILED
repo.update_status(file_id, SUCCESS)
```

Nhờ đó, nếu file đã tải trong lần chạy trước, lần này dùng `mode="skip"` để bỏ qua.

---

## 8. Tóm lại

```
run_data_loader(source="google_drive", file_id="...")
    → FileDispatcher.regist_all()
    → dispatcher.get_file_info("google_drive")     → GoogleDriveDownloader.get_file_info()
    → dispatcher.download(file, dest_path)          → GoogleDriveDownloader.download()
    → Tải file về /tmp/...
    → Lưu metadata vào MongoDB
    → Trả về list[DownloadResponse]
```
