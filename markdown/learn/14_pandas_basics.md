# 14 — Pandas Basics — Xử lý dữ liệu bảng

Pandas là thư viện Python để làm việc với dữ liệu dạng bảng (như Excel). Dự án dùng pandas chủ yếu trong `ValidationHandler` và `GroupByBuyerByAddressHandler`.

---

## 1. Đọc file CSV

```python
import pandas as pd

df = pd.read_csv("/tmp/data/import_2026_05.csv")

# Xem nhanh
print(df.shape)        # (1540, 32)  → 1540 rows, 32 cột
print(df.columns)      # Danh sách tên cột
print(df.head(3))      # 3 rows đầu
print(df.dtypes)       # Kiểu dữ liệu từng cột
```

---

## 2. Rename cột

```python
# Đổi tên nhiều cột cùng lúc
COLUMN_MAPPING = {
    "Declaration No":   "declaration_number",
    "Transaction Date": "transaction_date",
    "HS Code":          "hs_code",
    "Buyer":            "buyer_name",
}

df = df.rename(columns=COLUMN_MAPPING)
# Chỉ những cột có trong mapping mới được đổi tên
# Những cột không có trong mapping → giữ nguyên tên cũ
```

---

## 3. Đổi kiểu dữ liệu (`astype` và `to_numeric`)

```python
# Cách 1: astype — dùng cho string, category
df["buyer_name"] = df["buyer_name"].astype(str)
df["status"] = df["status"].astype("category")

# Cách 2: to_numeric — dùng cho số, xử lý được giá trị lỗi
df["hs_code"] = pd.to_numeric(df["hs_code"], errors="coerce")
# errors="coerce": nếu không convert được → NaN (thay vì báo lỗi)
# errors="raise" (mặc định): báo ValueError nếu gặp giá trị không hợp lệ

# Cách 3: to_datetime
df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y-%m-%d")
```

---

## 4. `apply` — Áp dụng hàm cho từng row/cột

```python
# apply theo từng giá trị trong cột
df["buyer_name_upper"] = df["buyer_name"].apply(lambda x: x.upper() if pd.notna(x) else x)

# apply theo từng row (axis=1)
def classify_amount(row):
    if row["total_amount_usd"] > 100000:
        return "large"
    elif row["total_amount_usd"] > 10000:
        return "medium"
    else:
        return "small"

df["deal_size"] = df.apply(classify_amount, axis=1)
```

---

## 5. `groupby` + `agg` — Nhóm và tổng hợp

Dùng trong `GroupByBuyerByAddressHandler`:

```python
# Nhóm theo địa chỉ, tính toán thống kê buyer
grouped = (
    df.groupby("importer_address_vn")["buyer_name"]
    .agg(
        buyer_set=lambda s: set(b for b in s.dropna() if str(b).strip()),
        buyer_count=lambda s: s.nunique(),
    )
    .reset_index()
    .sort_values("buyer_count", ascending=False)
)

# grouped trông như này:
# importer_address_vn          buyer_set                        buyer_count
# "123 Le Loi, HCM"            {"ABC Corp", "ABC Corporation"}  2
# "456 Nguyen Hue, HCM"        {"XYZ Ltd"}                      1
```

Ví dụ khác:
```python
# Tổng giá trị theo HS code
summary = df.groupby("hs_code").agg(
    total_value=("total_amount_usd", "sum"),
    transaction_count=("declaration_number", "count"),
    avg_value=("total_amount_usd", "mean"),
).reset_index()
```

---

## 6. `merge` — Nối hai DataFrame

```python
# Merge buyer_count về DataFrame gốc
df = df.merge(
    grouped[["importer_address_vn", "buyer_count"]],
    on="importer_address_vn",   # Join theo cột này
    how="left"                   # LEFT JOIN: giữ tất cả rows của df gốc
)
# Bây giờ df có thêm cột buyer_count

# Các loại join:
# how="left"  : Giữ tất cả rows bên trái, match nếu có
# how="right" : Giữ tất cả rows bên phải
# how="inner" : Chỉ giữ rows có match ở cả hai
# how="outer" : Giữ tất cả rows từ cả hai
```

---

## 7. Xử lý giá trị null (NaN)

```python
# Kiểm tra null
print(df["buyer_name"].isna().sum())    # Đếm số NaN trong cột
print(df.isna().sum())                  # Đếm NaN theo từng cột

# Lọc bỏ null
df_clean = df.dropna(subset=["buyer_name", "hs_code"])  # Bỏ rows có null ở 2 cột này

# Điền giá trị thay thế
df["buyer_name"] = df["buyer_name"].fillna("UNKNOWN")
df["total_amount_usd"] = df["total_amount_usd"].fillna(0.0)

# Filter null trong lambda (hay dùng trong groupby)
valid_buyers = [b for b in buyers if pd.notna(b) and str(b).strip()]
```

---

## 8. `to_dict` — Chuyển sang JSON/dict

```python
# Chuyển DataFrame sang list[dict] (để lưu vào MongoDB hay XCom)
records = df.to_dict(orient="records")
# [{"declaration_number": 12345, "hs_code": 8471, "buyer_name": "ABC Corp", ...},
#  {"declaration_number": 12346, ...},
#  ...]

# Các orient khác:
# orient="dict"    : {col: {idx: val}} — dict of dicts
# orient="list"    : {col: [val1, val2]} — dict of lists
# orient="records" : [{col: val}] — list of dicts (thường dùng nhất)
# orient="index"   : {idx: {col: val}} — indexed by row number
```

---

## 9. Filter rows

```python
# Filter rows thỏa điều kiện
large_deals = df[df["total_amount_usd"] > 100000]

# Filter nhiều điều kiện
vn_large = df[
    (df["total_amount_usd"] > 100000) &
    (df["importer_address_vn"].str.contains("HCM", na=False))
]

# Filter theo list giá trị
target_hs = df[df["hs_code"].isin([8471, 8473, 8474])]
```

---

## 10. Tóm lại

| Operation | Code |
|-----------|------|
| Đọc CSV | `pd.read_csv(path)` |
| Đổi tên cột | `df.rename(columns=mapping)` |
| Đổi kiểu số | `pd.to_numeric(col, errors="coerce")` |
| Đổi kiểu khác | `df[col].astype(dtype)` |
| Áp dụng hàm | `df[col].apply(func)` |
| Group và tổng hợp | `df.groupby(col).agg(...)` |
| Nối hai bảng | `df.merge(other, on=col, how="left")` |
| Xử lý null | `fillna()`, `dropna()`, `pd.notna()` |
| Sang list[dict] | `df.to_dict(orient="records")` |
| Filter rows | `df[df[col] > value]` |
