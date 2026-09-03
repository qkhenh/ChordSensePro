# 15 — Side Project: Weather Data Pipeline

> **Mục tiêu:** Tự tay xây một ETL pipeline hoàn chỉnh từ đầu, áp dụng đúng kiến trúc DDD + Clean Architecture đã học từ project công ty. Hoàn thành trong 1–2 tuần.

---

## 1. Bối Cảnh & Bài Toán

### Bài toán thực tế

Bạn muốn theo dõi thời tiết hàng ngày của **5 thành phố lớn Việt Nam** (Hà Nội, Đà Nẵng, Hồ Chí Minh, Cần Thơ, Hải Phòng). Mỗi ngày lúc 7h sáng, hệ thống tự động:

1. Gọi API lấy dữ liệu thời tiết hiện tại + dự báo
2. Validate và chuẩn hóa dữ liệu
3. Lưu vào database
4. Gọi AI nhận xét xu hướng thời tiết 7 ngày gần nhất
5. Tạo báo cáo HTML

### Tại sao chọn đề tài này?

- **Data source đơn giản:** OpenWeatherMap free API, không cần crawl hay xin quyền
- **Schema cố định:** Nhiệt độ, độ ẩm, gió... luôn có cùng format
- **Đủ phức tạp:** Có đủ các bước của 1 pipeline thật (download → process → save → AI → report)
- **Hoàn thành nhanh:** 1–2 tuần, không sa lầy vào vấn đề crawling

### Data Source

**OpenWeatherMap API** (https://openweathermap.org/api)

- Đăng ký free: 1000 calls/ngày, đủ cho project này
- 2 endpoint dùng chính:
  - `/weather` — thời tiết hiện tại (1 call = 1 thành phố)
  - `/forecast` — dự báo 5 ngày (1 call = 1 thành phố, trả về data mỗi 3 giờ)
- Response trả về JSON chuẩn, không cần parse phức tạp

**Ví dụ response từ API:**
```
GET https://api.openweathermap.org/data/2.5/weather?q=Hanoi&appid={API_KEY}&units=metric

Response:
{
  "name": "Hanoi",
  "main": {
    "temp": 32.5,        ← °C (vì units=metric)
    "feels_like": 36.1,
    "humidity": 78,       ← %
    "pressure": 1008      ← hPa
  },
  "wind": {
    "speed": 3.5,         ← m/s
    "deg": 180            ← hướng gió (độ)
  },
  "weather": [
    {
      "main": "Clouds",
      "description": "scattered clouds"
    }
  ],
  "dt": 1717142400        ← Unix timestamp
}
```

---

## 2. Kiến Trúc Tổng Thể

### Project mới hoàn toàn, không code trong project công ty

Tạo folder riêng, ví dụ: `WeatherPipeline/`. Cấu trúc giống hệt project công ty nhưng nhỏ hơn.

```
WeatherPipeline/
├── .env                          ← API keys, DB connection
├── pyproject.toml                ← Dependencies
├── docker-compose.yml            ← PostgreSQL + (optional) Airflow
│
├── src/
│   ├── shared/                   ← Nền tảng chung (copy pattern từ project công ty)
│   │   ├── domain/
│   │   │   ├── base_model.py         ← CustomBaseModel (ABC: _to_model, _to_doc)
│   │   │   └── processing_result.py  ← ProcessingResult
│   │   │
│   │   ├── infrastructure/
│   │   │   ├── setting/
│   │   │   │   ├── base_setting.py       ← AppBaseSetting (pydantic-settings)
│   │   │   │   ├── postgres_setting.py
│   │   │   │   └── openweather_setting.py
│   │   │   │
│   │   │   ├── postgres/
│   │   │   │   ├── client.py             ← PostgresClient singleton
│   │   │   │   └── base_repository.py    ← BasePostgresRepository
│   │   │   │
│   │   │   └── service/
│   │   │       ├── ai_api_caller_base.py     ← BaseRequestHandler
│   │   │       └── prompt_builder_service.py ← PromptBuilderService
│   │   │
│   │   └── utils/
│   │       └── logging.py
│   │
│   ├── data_loader/              ← Module 1: Gọi API lấy dữ liệu thời tiết
│   │   ├── domain/
│   │   │   ├── models/
│   │   │   │   ├── weather_api_file.py      ← Model cho API request
│   │   │   │   └── download_response.py     ← Response sau khi tải
│   │   │   ├── ports/
│   │   │   │   └── downloader.py            ← Downloader Protocol
│   │   │   └── services/
│   │   │       └── file_dispatcher.py       ← Registry: map source → downloader
│   │   │
│   │   └── application/
│   │       ├── entrypoints.py               ← run_data_loader()
│   │       └── downloaders/
│   │           └── weather_api_downloader.py ← Gọi OpenWeatherMap API
│   │
│   ├── data_processing/          ← Module 2: Validate + Transform
│   │   ├── domain/
│   │   │   └── models/
│   │   │       ├── base_handler.py          ← BaseProcessingHandler
│   │   │       └── processed_data.py        ← ProcessedData
│   │   │
│   │   └── application/
│   │       ├── entrypoints.py               ← run_data_processing()
│   │       ├── pipeline_factory.py          ← Build handler chain
│   │       └── pipeline_handlers/
│   │           └── weather_handlers/
│   │               ├── validation_handler.py    ← Validate JSON, rename fields
│   │               ├── transform_handler.py     ← Tính thêm metrics
│   │               └── saving_handler.py        ← Đánh dấu outlier
│   │
│   ├── data_ingest/              ← Module 3: Orchestrator
│   │   ├── domain/
│   │   │   └── models/
│   │   │       └── ingest_record.py         ← IngestionRecord + IngestionStatus
│   │   │
│   │   └── application/
│   │       ├── pipeline.py                  ← run_ingest_pipeline()
│   │       └── entrypoints.py               ← Entry-point thuần Python
│   │
│   └── ai_analysis/              ← Module 4: AI nhận xét thời tiết
│       ├── domain/
│       │   └── models/
│       │       └── weather_analysis.py      ← Output model cho AI
│       │
│       └── application/
│           ├── templates/
│           │   └── weather_analyst_prompt.py ← PromptTemplate
│           └── handlers/
│               └── weather_analyst_handler.py ← BaseRequestHandler subclass
│
├── reports/                      ← Output: HTML reports
│   └── report_builder.py        ← Build HTML báo cáo
│
└── main.py                       ← Entry-point: python main.py
```

### Tại sao không dùng MongoDB?

Project công ty dùng cả MongoDB lẫn PostgreSQL. Nhưng ở đây data thời tiết có schema cố định (luôn có nhiệt độ, độ ẩm, gió...), nên **chỉ cần PostgreSQL** là đủ. Đơn giản hóa cho side project.

Nếu muốn luyện thêm MongoDB → có thể dùng để lưu raw JSON response từ API (giữ nguyên cấu trúc gốc).

---

## 3. Chi Tiết Từng Module

### Module 1: `data_loader/` — Gọi API Lấy Data

**Nhiệm vụ:** Gọi OpenWeatherMap API cho từng thành phố, lưu raw JSON về local.

#### Nghiệp vụ cần làm:

**a) `WeatherApiFile` (domain model)**
- Đại diện cho 1 yêu cầu gọi API
- Chứa: `city_name` (tên thành phố), `api_key`, `endpoint_type` (current / forecast)
- Kế thừa `BaseFileModel`, set `original = FileSource.API`

**b) `WeatherApiDownloader` (downloader)**
- Nhận `WeatherApiFile` → gọi OpenWeatherMap API
- Lưu response JSON vào file local, ví dụ: `/tmp/weather/2026-06-11/hanoi_current.json`
- Trả về `DownloadResponse` (local_path + status)
- Xử lý lỗi: API rate limit, network timeout, invalid API key

**c) `FileDispatcher` (registry)**
- Đăng ký: `FileSource.API → WeatherApiDownloader`
- Ở side project này chỉ có 1 source, nhưng vẫn dùng Registry pattern để luyện

**d) `run_data_loader()` (entrypoint)**
- Input: `execution_date`, `cities=["Hanoi", "Da Nang", "Ho Chi Minh", "Can Tho", "Hai Phong"]`
- Tạo `WeatherApiFile` cho mỗi thành phố
- Gọi dispatcher.download() cho từng file
- Output: `list[DownloadResponse]` — danh sách file JSON đã lưu

**Lưu ý quan trọng:**
- Mỗi thành phố = 2 API calls (current + forecast) = 2 file JSON
- 5 thành phố × 2 = 10 API calls/lần chạy → rất ít, free tier thừa sức

---

### Module 2: `data_processing/` — Validate + Transform

**Nhiệm vụ:** Đọc raw JSON, validate, tính thêm metrics, đánh dấu bất thường.

#### Handler Chain:

```
ValidationHandler → TransformHandler → SavingHandler
```

#### a) `ValidationHandler`

**Input:** `ProcessedData` chứa `local_path` tới file JSON

**Nhiệm vụ:**
1. Đọc file JSON
2. Validate schema: kiểm tra có đủ fields bắt buộc không (temp, humidity, wind_speed...)
3. Validate kiểu dữ liệu: temp phải là số, humidity phải 0–100, wind_speed ≥ 0
4. Validate giá trị hợp lý: nhiệt độ -50 đến 60°C (ngoài range → đánh dấu suspicious)
5. Rename fields: map từ API naming sang snake_case chuẩn
6. Chuyển Unix timestamp → datetime

**Mapping fields:**
```
API field               →  Tên chuẩn trong hệ thống
──────────────────────────────────────────────────
main.temp               →  temperature
main.feels_like         →  feels_like
main.humidity           →  humidity
main.pressure           →  pressure
wind.speed              →  wind_speed
wind.deg                →  wind_direction
weather[0].main         →  weather_condition
weather[0].description  →  weather_description
dt                      →  recorded_at (convert từ unix timestamp)
name                    →  city
```

**Output:** DataFrame với các cột đã chuẩn hóa, gắn vào `data.structured_data["weather_data"]`

#### b) `TransformHandler`

**Input:** DataFrame từ ValidationHandler

**Nhiệm vụ — tính thêm các metrics:**
1. **Heat Index:** Công thức tính cảm giác nóng dựa trên nhiệt độ + độ ẩm. Google "heat index formula" — có công thức sẵn.
2. **Temperature Difference:** `feels_like - temperature` (chênh lệch thực tế vs cảm nhận)
3. **Wind Chill Category:** Phân loại gió: "Calm" (< 1 m/s), "Light" (1-5), "Moderate" (5-10), "Strong" (> 10)
4. **Weather Severity:** Phân loại: "Good" (Clear/Clouds), "Moderate" (Rain/Drizzle), "Severe" (Thunderstorm/Snow)
5. **Day/Night flag:** Dựa trên giờ recorded_at → "day" (6h-18h) / "night"

**Output:** DataFrame bổ sung thêm các cột tính toán

#### c) `SavingHandler`

**Input:** DataFrame từ TransformHandler

**Nhiệm vụ:**
1. Đánh dấu outlier: `is_outlier = True` nếu nhiệt độ nằm ngoài mean ± 2σ của thành phố đó (dựa trên data lịch sử trong DB)
2. Gắn `run_id` (execution_date) vào mỗi row
3. Gắn `data_source = "openweathermap"`
4. Chuẩn bị list[dict] sẵn sàng để bulk insert vào PostgreSQL

**Output:** Gắn `data.structured_data["records_to_save"]` = list[dict], mỗi dict là 1 row

---

### Module 3: `data_ingest/` — Orchestrator

**Nhiệm vụ:** Điều phối toàn bộ pipeline từ đầu tới cuối.

#### `run_ingest_pipeline()`

**Flow:**
```
Step 1: run_data_loader()
        → Gọi API 5 thành phố × 2 endpoints = 10 file JSON
        → Output: list[DownloadResponse]

Step 2: run_data_processing()
        → Đọc 10 file JSON, chạy qua handler chain
        → Output: ProcessedData chứa records_to_save

Step 3: Save to PostgreSQL
        → WeatherRepository.bulk_insert(records)
        → Lưu summary vào ProcessingResult

Step 4: AI Analysis (optional, chạy nếu đủ 7 ngày data)
        → Query 7 ngày data từ DB cho mỗi thành phố
        → Gọi AI nhận xét xu hướng
        → Lưu AI analysis vào DB

Step 5: Generate HTML Report
        → Build report từ data hôm nay + AI analysis
        → Lưu file HTML vào reports/
```

**Return:** `IngestionRecord` (status, files_total, files_done, error_message)

---

### Module 4: `ai_analysis/` — AI Nhận Xét Thời Tiết

**Nhiệm vụ:** Dùng LLM để phân tích xu hướng thời tiết 7 ngày gần nhất.

#### a) `WeatherAnalystPromptTemplate`

**System prompt:**
```
Bạn là chuyên gia khí tượng học. Dựa trên dữ liệu thời tiết 7 ngày gần nhất,
hãy nhận xét xu hướng và đưa ra dự báo ngắn hạn.
Trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu.
```

**User prompt template:**
```
Thành phố: {city}
Dữ liệu 7 ngày gần nhất:
{weather_table}

Hãy phân tích:
1. Xu hướng nhiệt độ (tăng/giảm/ổn định)
2. Xu hướng độ ẩm
3. Có hiện tượng thời tiết bất thường không?
4. Dự báo 2-3 ngày tới
```

#### b) `WeatherAnalysis` (output model)

```
city: str
temperature_trend: str          # "increasing" | "decreasing" | "stable"
humidity_trend: str
notable_events: list[str]       # ["Nhiệt độ cao bất thường ngày 09/06"]
forecast_summary: str           # "Dự kiến 2-3 ngày tới trời nắng nóng..."
overall_assessment: str         # "Thời tiết nhìn chung ổn định..."
```

#### c) `WeatherAnalystHandler`

- Kế thừa `BaseRequestHandler`
- `_build_prompt()`: Query 7 ngày data từ DB, format thành bảng, điền vào template
- `_to_response()`: Map output AI → `WeatherAnalysis`
- Gọi cho từng thành phố (5 calls)

---

## 4. Database Schema

### Bảng `weather_records` — Data chính

```sql
CREATE TABLE weather_records (
    id              SERIAL PRIMARY KEY,
    city            VARCHAR(100)   NOT NULL,
    temperature     NUMERIC(5,2)   NOT NULL,     -- °C
    feels_like      NUMERIC(5,2),
    humidity        INTEGER,                      -- 0-100 %
    pressure        INTEGER,                      -- hPa
    wind_speed      NUMERIC(5,2),                 -- m/s
    wind_direction  INTEGER,                      -- 0-360 degrees
    weather_condition VARCHAR(50),                 -- "Clear", "Rain", "Clouds"
    weather_description VARCHAR(200),              -- "scattered clouds"
    
    -- Computed fields (từ TransformHandler)
    heat_index          NUMERIC(5,2),
    temp_difference     NUMERIC(5,2),              -- feels_like - temperature
    wind_category       VARCHAR(20),               -- "Calm", "Light", "Moderate", "Strong"
    weather_severity    VARCHAR(20),               -- "Good", "Moderate", "Severe"
    day_night           VARCHAR(10),               -- "day" / "night"
    is_outlier          BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    recorded_at     TIMESTAMPTZ    NOT NULL,       -- Thời điểm ghi nhận (từ API)
    data_source     VARCHAR(50)    DEFAULT 'openweathermap',
    run_id          VARCHAR(20),                   -- execution_date (YYYY-MM-DD)
    created_at      TIMESTAMPTZ    DEFAULT NOW()
);

-- Index cho query thường dùng
CREATE INDEX idx_weather_city_date ON weather_records (city, recorded_at);
CREATE INDEX idx_weather_run_id ON weather_records (run_id);
```

### Bảng `weather_ai_analysis` — Kết quả AI

```sql
CREATE TABLE weather_ai_analysis (
    id                  SERIAL PRIMARY KEY,
    city                VARCHAR(100)   NOT NULL,
    analysis_date       DATE           NOT NULL,
    temperature_trend   VARCHAR(20),
    humidity_trend      VARCHAR(20),
    notable_events      JSONB,                     -- ["event1", "event2"]
    forecast_summary    TEXT,
    overall_assessment  TEXT,
    ai_model            VARCHAR(50),               -- "gpt-4o-mini"
    created_at          TIMESTAMPTZ    DEFAULT NOW(),
    
    UNIQUE(city, analysis_date)                    -- Mỗi ngày chỉ 1 analysis/thành phố
);
```

### Bảng `processing_results` — Kết quả pipeline

```sql
CREATE TABLE processing_results (
    id          SERIAL PRIMARY KEY,
    run_id      VARCHAR(20)  UNIQUE NOT NULL,       -- "2026-06-11"
    summary     JSONB        NOT NULL DEFAULT '{}',  -- Tóm tắt: bao nhiêu record, thành công/thất bại
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);
```

---

## 5. Data Flow Chi Tiết

```
[OpenWeatherMap API]
        │
        │  10 API calls (5 cities × 2 endpoints)
        ▼
[WeatherApiDownloader]
        │
        │  Lưu 10 file JSON vào /tmp/weather/2026-06-11/
        │  ├── hanoi_current.json
        │  ├── hanoi_forecast.json
        │  ├── danang_current.json
        │  └── ...
        ▼
[ValidationHandler]
        │
        │  Đọc JSON → validate → rename → DataFrame
        │  Input:  raw JSON (API format)
        │  Output: DataFrame 5 rows (current) + ~200 rows (forecast, mỗi 3h × 5 ngày × 5 cities)
        ▼
[TransformHandler]
        │
        │  Tính thêm: heat_index, wind_category, weather_severity, day_night
        │  Input:  DataFrame ~205 rows × 10 cols
        │  Output: DataFrame ~205 rows × 16 cols
        ▼
[SavingHandler]
        │
        │  Đánh dấu outlier, gắn run_id
        │  Input:  DataFrame ~205 rows × 16 cols
        │  Output: list[dict] ~205 records sẵn sàng insert
        ▼
[PostgreSQL: weather_records]
        │
        │  bulk_insert() ~205 rows
        ▼
[WeatherAnalystHandler] (chạy nếu có ≥ 7 ngày data)
        │
        │  Query 7 ngày × 5 cities từ DB
        │  Gọi AI 5 lần (1 lần/city)
        ▼
[PostgreSQL: weather_ai_analysis]
        │
        │  upsert 5 records (1/city)
        ▼
[ReportBuilder]
        │
        │  Query data hôm nay + AI analysis
        │  Build HTML
        ▼
[reports/weather_report_2026-06-11.html]
```

---

## 6. Settings (.env)

```env
# OpenWeatherMap
OPENWEATHER_API_KEY=your_api_key_here
OPENWEATHER_CITIES=Hanoi,Da Nang,Ho Chi Minh City,Can Tho,Hai Phong

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=weather_user
POSTGRES_PASSWORD=weather_pass
POSTGRES_DB=weather_pipeline

# AI (OpenAI)
OPENAI_API_KEY=your_openai_key_here
AI_MODEL=gpt-4o-mini
```

---

## 7. Các Repository Cần Tạo

### `WeatherRecordRepository` (extends BasePostgresRepository)

```
table_name = "weather_records"

Methods cần có:
- bulk_insert(records: list[dict]) → int
    Giống HsRawDataPgRepository trong project công ty.
    Insert nhiều rows 1 lần bằng execute_values().
    Trả về số rows đã insert.

- find_by_city_and_date_range(city: str, start: date, end: date) → list[dict]
    Dùng cho AI analysis: query 7 ngày data gần nhất.
    SELECT * FROM weather_records WHERE city = %s AND recorded_at BETWEEN %s AND %s

- find_latest_by_city(city: str) → dict | None
    Lấy record mới nhất của 1 thành phố.
    Dùng cho report.

- get_city_stats(city: str, days: int = 30) → dict
    Tính mean và std của nhiệt độ → dùng cho outlier detection.
    SELECT AVG(temperature), STDDEV(temperature) FROM weather_records
    WHERE city = %s AND recorded_at > NOW() - INTERVAL '%s days'
```

### `WeatherAnalysisRepository` (extends BasePostgresRepository)

```
table_name = "weather_ai_analysis"

Methods cần có:
- upsert_by_city_date(analysis: WeatherAnalysis) → dict
    Insert hoặc update theo (city, analysis_date).
    Dùng ON CONFLICT DO UPDATE.

- find_latest_by_city(city: str) → WeatherAnalysis | None
    Lấy analysis mới nhất cho 1 thành phố.
```

---

## 8. HTML Report

### Nội dung report

Tạo 1 file HTML cho mỗi ngày chạy. Gồm các phần:

**Header:**
- Tiêu đề: "Weather Report — 2026-06-11"
- Thời gian tạo report

**Section 1: Current Weather (bảng)**
```
| Thành phố    | Nhiệt độ | Cảm giác | Độ ẩm | Gió    | Tình trạng |
|-------------|----------|----------|-------|--------|-----------|
| Hà Nội      | 32.5°C   | 36.1°C   | 78%   | 3.5m/s | Clouds    |
| Đà Nẵng     | 29.0°C   | 31.2°C   | 65%   | 2.1m/s | Clear     |
| ...         |          |          |       |        |           |
```

**Section 2: AI Analysis (mỗi thành phố 1 card)**
```
🏙️ Hà Nội
Xu hướng: Nhiệt độ tăng nhẹ trong 7 ngày qua.
Nhận xét: Độ ẩm cao liên tục, khả năng mưa rào chiều tối.
Dự báo: 2-3 ngày tới tiếp tục nóng ẩm, nhiệt độ 32-35°C.
```

**Section 3: Outlier Alerts (nếu có)**
```
⚠️ Cảnh báo bất thường:
- Hồ Chí Minh: Nhiệt độ 41.2°C — cao bất thường (trung bình 30 ngày: 32.5°C)
```

### Cách build

Dùng Python string formatting thuần (giống `pipeline_wrapper.py` trong project công ty build email HTML). Không cần template engine phức tạp.

---

## 9. Checklist Implementation

```
Phase 0: Setup (Ngày 1)
  [ ] Tạo project folder + pyproject.toml + .env
  [ ] Đăng ký OpenWeatherMap API key
  [ ] Setup PostgreSQL local (Docker hoặc native)
  [ ] Tạo database + tables (chạy DDL)

Phase 1: shared/ (Ngày 2)
  [ ] Copy + adapt CustomBaseModel, AppBaseSetting
  [ ] Tạo PostgresClient + BasePostgresRepository
  [ ] Tạo OpenWeatherSetting, PostgresSetting
  [ ] Tạo logging.py
  [ ] Test: connect DB thành công

Phase 2: data_loader/ (Ngày 3-4)
  [ ] Tạo WeatherApiFile model
  [ ] Tạo DownloadResponse model
  [ ] Tạo Downloader Protocol
  [ ] Tạo WeatherApiDownloader (gọi API + lưu JSON)
  [ ] Tạo FileDispatcher + đăng ký
  [ ] Tạo run_data_loader()
  [ ] Test: gọi API 1 thành phố, kiểm tra file JSON

Phase 3: data_processing/ (Ngày 5-7)
  [ ] Tạo ProcessedData model
  [ ] Tạo BaseProcessingHandler
  [ ] Tạo ValidationHandler (đọc JSON, validate, rename)
  [ ] Tạo TransformHandler (heat_index, wind_category...)
  [ ] Tạo SavingHandler (outlier detection, prepare records)
  [ ] Tạo pipeline_factory.py (wire chain)
  [ ] Tạo run_data_processing()
  [ ] Test: chạy pipeline với 1 file JSON, kiểm tra output

Phase 4: Database + Orchestrator (Ngày 8-9)
  [ ] Tạo WeatherRecordRepository
  [ ] Tạo run_ingest_pipeline()
  [ ] Test end-to-end: API → process → DB
  [ ] Kiểm tra data trong PostgreSQL

Phase 5: AI Analysis (Ngày 10-11)
  [ ] Tạo WeatherAnalysis model
  [ ] Tạo WeatherAnalystPromptTemplate
  [ ] Tạo WeatherAnalystHandler
  [ ] Tạo WeatherAnalysisRepository
  [ ] Test: chạy AI analysis cho 1 thành phố

Phase 6: Report (Ngày 12-13)
  [ ] Tạo ReportBuilder
  [ ] Build HTML report
  [ ] Test: mở file HTML trong browser

Phase 7: Polish (Ngày 14)
  [ ] Tạo main.py chạy toàn bộ pipeline
  [ ] Error handling: API fail, DB fail
  [ ] Chạy liên tục 3 ngày, kiểm tra data tích lũy
  [ ] (Optional) Thêm Airflow DAG
```

---

## 10. Dependencies

```
# pyproject.toml
[project]
dependencies = [
    "requests",           # Gọi API
    "pydantic>=2.0",      # Domain models
    "pydantic-settings",  # Đọc .env
    "psycopg2-binary",    # PostgreSQL
    "pandas",             # DataFrame processing
    "openai",             # AI analysis
    "tiktoken",           # Token counting
]
```

---

## 11. Kết Quả Mong Đợi

Sau khi hoàn thành, bạn sẽ:

1. **Có 1 project chạy được end-to-end** — `python main.py` → data vào DB + AI analysis + HTML report
2. **Hiểu sâu kiến trúc DDD** — đã tự tay tạo từ đầu, không chỉ đọc code người khác
3. **Nắm vững các patterns** — Registry, CoR, Repository, Singleton, Template Method
4. **Sẵn sàng làm project lớn** — cùng architecture nhưng phức tạp hơn nhiều
