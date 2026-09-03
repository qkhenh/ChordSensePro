# 13 — Testing Guide — Viết test cho dự án

Test giúp bạn tự tin sửa code mà không sợ phá thứ khác. Dự án này dùng `pytest`.

---

## 1. Chạy test

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy 1 file test cụ thể
pytest tests/test_file_dispatcher.py -v

# Chạy test có tên chứa từ khoá
pytest tests/ -k "google_drive" -v

# Chạy và hiện print output
pytest tests/ -v -s
```

---

## 2. Cấu trúc test cơ bản

```python
# tests/test_file_dispatcher.py
def test_dispatcher_calls_correct_downloader():
    # 1. Arrange — chuẩn bị dữ liệu
    dispatcher = FileDispatcher()
    dispatcher.regist_all()
    
    file = GoogleDriveFile(
        file_id="abc123",
        drive_file_id="1O80Uye...",
        name="import.csv",
        mime_type="text/csv",
    )
    
    # 2. Act — chạy code cần test
    responses = dispatcher.download(file, dest_path="/tmp/")
    
    # 3. Assert — kiểm tra kết quả
    assert len(responses) > 0
    assert responses[0].file_download_status == FileDownloadStatus.SUCCESS
```

---

## 3. `pytest.fixture` — Tái sử dụng setup

Nếu nhiều test cần cùng 1 setup, dùng fixture:

```python
import pytest

@pytest.fixture
def sample_google_drive_file():
    return GoogleDriveFile(
        file_id="test_id_001",
        drive_file_id="1O80UyeZUXugNk3QI1IASX2PWWoBfVO82",
        name="test_import.csv",
        mime_type="text/csv",
        original=FileSource.GOOGLE_DRIVE,
    )

# Fixtures được inject tự động vào test
def test_download_single_file(sample_google_drive_file):
    downloader = GoogleDriveDownloader()
    # sample_google_drive_file đã được tạo sẵn
    result = downloader._build_local_path(sample_google_drive_file, "/tmp/")
    assert result == "/tmp/test_import.csv"

def test_file_has_correct_source(sample_google_drive_file):
    assert sample_google_drive_file.original == FileSource.GOOGLE_DRIVE
```

---

## 4. `unittest.mock.patch` — Mock dependencies

Khi test một function nhưng không muốn nó thực sự gọi API hay DB, dùng mock:

```python
from unittest.mock import patch, MagicMock

def test_google_drive_downloader_calls_api():
    # Thay thế google_drive_service thực bằng mock
    with patch("src.data_loader.application.downloaders.google_drive_downloader.google_drive_service") as mock_service:
        # Cấu hình mock: khi gọi download_file, trả về path giả
        mock_service.download_file.return_value = "/tmp/fake_file.csv"
        mock_service.list_files.return_value = []

        downloader = GoogleDriveDownloader()
        file = GoogleDriveFile(
            file_id="test",
            drive_file_id="1O80Uye...",
            name="test.csv",
            mime_type="text/csv",
        )
        
        result = downloader.download(file, "/tmp/")
        
        # Kiểm tra API được gọi đúng
        mock_service.download_file.assert_called_once_with("1O80Uye...", "/tmp/test.csv")
        assert result[0].file_download_status == FileDownloadStatus.SUCCESS
```

---

## 5. Fake Object Pattern — Dự án này hay dùng

Thay vì mock từng method, tạo hẳn một class "fake" đơn giản:

```python
# Fake repository: lưu trong memory thay vì MongoDB
class FakeGoogleDriveFileRepository:
    def __init__(self):
        self._data: dict[str, GoogleDriveFile] = {}

    def insert_one(self, file: GoogleDriveFile):
        self._data[file.file_id] = file

    def find_by_id(self, file_id: str) -> GoogleDriveFile | None:
        return self._data.get(file_id)

    def update_status(self, file_id: str, status: FileDownloadStatus):
        if file_id in self._data:
            self._data[file_id] = self._data[file_id].model_copy(
                update={"download_status": status}
            )

# Dùng trong test
def test_downloader_updates_status_to_success():
    fake_repo = FakeGoogleDriveFileRepository()
    downloader = GoogleDriveDownloader(repo=fake_repo)
    
    # ... chạy download ...
    
    file = fake_repo.find_by_id("test_id")
    assert file.download_status == FileDownloadStatus.SUCCESS
```

---

## 6. Test các Pydantic models

```python
# tests/test_models.py
def test_google_drive_file_validates_correctly():
    file = GoogleDriveFile(
        file_id="abc",
        drive_file_id="1O80Uye",
        name="data.csv",
        mime_type="text/csv",
    )
    assert file.original == FileSource.GOOGLE_DRIVE

def test_google_drive_file_rejects_empty_name():
    with pytest.raises(ValidationError):
        GoogleDriveFile(
            file_id="abc",
            drive_file_id="1O80Uye",
            name="",   # Empty name → ValidationError
            mime_type="text/csv",
        )
```

---

## 7. Test structure trong dự án

```
tests/
├── test_models.py                  ← Test Pydantic models
├── test_file_dispatcher.py         ← Test Registry pattern
├── test_google_drive_downloader.py ← Test GDrive downloader (với mock)
├── test_google_drive_service.py    ← Test GDrive API service
├── test_data_loader_entrypoints.py ← Test entrypoint function
├── test_processors.py              ← Test processing handlers
├── test_repository.py              ← Test repositories (với fake DB)
└── test_ingest_service.py          ← Test orchestrator
```

---

## 8. Tóm lại

```
pytest tests/ -v         → Chạy tất cả
pytest.fixture           → Tạo data dùng chung cho nhiều tests
unittest.mock.patch      → Thay thế API/DB calls bằng fake
Fake Object              → Class đơn giản thay thế dependency thực
pytest.raises(Error)     → Kiểm tra code báo đúng lỗi
```

Quy tắc: **1 test chỉ test 1 việc**. Test nào fail thì biết ngay cái gì bị hỏng.
