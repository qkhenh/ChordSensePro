# 04 — `data_processing` — Xử lý dữ liệu qua Pipeline

Module này nhận danh sách file đã tải về, đọc nội dung, và xử lý qua một chuỗi bước (pipeline).

**Kết quả cuối cùng:** DataFrame đã được validate và chuẩn hoá, sẵn sàng để lưu vào database.

---

## 1. Input và Output

```python
# Input: danh sách DownloadResponse từ data_loader
run_data_processing(
    responses=[DownloadResponse(local_path="/tmp/data/import.csv", ...)],
    execution_date="2026-05-31"
)

# Output: ProcessedData chứa DataFrame đã xử lý
# ProcessedData(structured_data={"hs_code_data": DataFrame(1540 rows × 30 cols)})
```

---

## 2. Design Pattern: Chain of Responsibility

Thay vì một hàm khổng lồ làm tất cả, mỗi bước xử lý là một **Handler** độc lập, được kết nối thành chuỗi.

```
ValidationHandler  →  GroupByBuyerHandler  →  SavingHandler  →  (end)
       |                      |                      |
  Đọc CSV,             Group buyers              Đánh dấu
  rename cols,         theo địa chỉ              need_check
  cast types
```

Cách kết nối chain:
```python
# pipeline_factory.py
def build_hscode_pipeline():
    validation  = ValidationHandler()
    group_buyer = GroupByBuyerByAddressHandler()
    saving      = SavingHandler()

    validation.set_next(group_buyer).set_next(saving)
    # a.set_next(b) trả về b, nên có thể chain liên tiếp

    return validation  # Trả về đầu chain
```

Cách chạy:
```python
pipeline = build_hscode_pipeline()
result   = pipeline.handle(initial_data)  # Tự động chạy qua cả chain
```

**Muốn thêm bước mới:** tạo class mới, chèn vào chain, không cần sửa các handlers cũ.

---

## 3. Cấu trúc module

```
data_processing/
├── entrypoints.py                    ← run_data_processing() — cổng vào
│
├── application/
│   ├── pipeline_factory.py           ← Build chain từ các handlers
│   └── pipeline_handlers/
│       ├── validation_handler.py     ← Đọc CSV, rename cols, cast types
│       ├── saving_handler.py         ← Đánh dấu rows cần check
│       └── hs_code_handlers/
│           └── clean_buyer_handler.py  ← Group buyers theo địa chỉ
│
└── domain/
    └── models/
        ├── base_handler.py           ← Abstract class cho mọi handler
        └── processed_data.py         ← Data object đi qua pipeline
```

---

## 4. `BaseProcessingHandler` — Abstract Handler

Mọi handler đều kế thừa class này. Nó định nghĩa khung chung:

```python
class BaseProcessingHandler(ABC):
    _next_handler: "BaseProcessingHandler | None" = None

    def set_next(self, handler: "BaseProcessingHandler"):
        self._next_handler = handler
        return handler  # Trả về handler để chain: a.set_next(b).set_next(c)

    @abstractmethod
    def handle(self, data: ProcessedData) -> ProcessedData:
        # Subclass implement logic xử lý ở đây
        # Cuối cùng phải gọi: return self._handle_next(data)
        pass

    def _handle_next(self, data: ProcessedData) -> ProcessedData:
        if self._next_handler:
            return self._next_handler.handle(data)
        return data  # Cuối chain: trả về kết quả
```

---

## 5. `ProcessedData` — Data object đi qua pipeline

```python
class ProcessedData(BaseModel):
    execution_date: str
    source: str

    # Chứa nhiều loại data khác nhau
    structured_data: dict[str, Any]
    # Ví dụ: {"hs_code_data": DataFrame, "local_path": "/tmp/file.csv"}

    processing_status: ProcessingStatus   # PENDING | PROCESSING | SUCCESS | FAILED
    error_messages: list[str] = []

    # Khi truyền qua Airflow XCom (JSON), DataFrame được convert sang list[dict]
    @field_serializer("structured_data", when_used="json")
    def serialize_structured_data(self, value):
        result = {}
        for key, item in value.items():
            if isinstance(item, pd.DataFrame):
                result[key] = item.to_dict(orient="records")
            else:
                result[key] = item
        return result
```

---

## 6. `ValidationHandler` — Handler quan trọng nhất

Handler này đọc file CSV và chuẩn hoá tên cột + kiểu dữ liệu.

```python
# Tên cột gốc trong CSV → tên snake_case trong hệ thống
COLUMN_IMP_MAPPING = {
    "Declaration No":   "declaration_number",
    "Transaction Date": "transaction_date",
    "HS Code":          "hs_code",
    "Buyer":            "buyer_name",
    # ... 30+ cột
}

# Kiểu dữ liệu mong muốn
COLUMN_TYPES = {
    "declaration_number": "int",
    "hs_code":            "int",
    "total_amount_usd":   "float64",
    "buyer_name":         "str",
}

def handle(self, data: ProcessedData) -> ProcessedData:
    local_path = data.structured_data["local_path"]
    df = pd.read_csv(local_path)

    # Bước 1: Rename cột
    df = df.rename(columns=COLUMN_IMP_MAPPING)

    # Bước 2: Cast kiểu dữ liệu
    for col, dtype in COLUMN_TYPES.items():
        if col in df.columns:
            if dtype in ("int", "float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(dtype)

    data.structured_data["hs_code_data"] = df
    return self._handle_next(data)
```

---

## 7. `GroupByBuyerByAddressHandler` — Clean Buyer

Handler này group các buyer theo địa chỉ để nhận ra cùng một công ty có nhiều tên khác nhau.

```python
def handle(self, data: ProcessedData) -> ProcessedData:
    df = data.structured_data["hs_code_data"]

    grouped = (
        df.groupby("importer_address_vn")["buyer_name"]
        .agg(
            buyer_set=lambda s: set(b for b in s.dropna() if str(b).strip()),
            buyer_count=lambda s: s.nunique(),
        )
        .reset_index()
        .sort_values("buyer_count", ascending=False)
    )

    # Merge buyer_count về DataFrame gốc
    df = df.merge(
        grouped[["importer_address_vn", "buyer_count"]],
        on="importer_address_vn",
        how="left"
    )

    data.structured_data["hs_code_data"] = df
    data.structured_data["buyer_groups"] = grouped
    return self._handle_next(data)
```

---

## 8. Tóm lại

```
run_data_processing(responses)
    → đọc local_path từ responses
    → tạo ProcessedData ban đầu
    → pipeline = build_hscode_pipeline()
    → pipeline.handle(data)
          ValidationHandler:          đọc CSV, rename cols, cast types
       → GroupByBuyerHandler:         group buyers theo địa chỉ
       → SavingHandler:               đánh dấu rows cần check
    → Trả về ProcessedData có chứa DataFrame đã xử lý
```
