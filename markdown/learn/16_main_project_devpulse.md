# 16 — Main Project: DevPulse — Hệ Thống Phân Tích & Dự Báo Xu Hướng Công Nghệ

> **Mục tiêu:** Xây dựng hệ thống end-to-end: Crawl dữ liệu từ nhiều nguồn mở → ETL pipeline → AI phân tích & dự báo → BI Dashboard. Đủ phức tạp cho đồ án tốt nghiệp, đủ đặc biệt để không đụng hàng.

---

## 1. Bối Cảnh & Bài Toán

### Vấn đề thực tế

Lập trình viên, CTO, tech lead phải đưa ra quyết định: **"Nên dùng công nghệ nào cho dự án tiếp theo?"**

Hiện tại, để trả lời câu hỏi này, họ phải:
- Vào GitHub xem stars → nhưng stars ≠ usage (nhiều repo stars cao mà ít người dùng thật)
- Đọc blog, HackerNews → chủ quan, thiên vị
- Xem ThoughtWorks Tech Radar → cập nhật 6 tháng/lần, bằng tay, chỉ cover 1 góc nhỏ
- Hỏi đồng nghiệp → phụ thuộc kinh nghiệm cá nhân

**Không ai kết hợp tất cả nguồn dữ liệu lại và dùng AI để phân tích tự động.**

### DevPulse giải quyết gì?

Hệ thống **tự động** thu thập dữ liệu từ 5 nguồn mở mỗi ngày, tính toán **Technology Health Index (TH-Index)** — một chỉ số đa chiều đánh giá "sức khỏe" công nghệ, và dùng AI để tự động phân loại công nghệ theo framework **Tech Radar** (Adopt / Trial / Assess / Hold).

```
Luồng tổng thể:

5 nguồn dữ liệu mở    →   ETL Pipeline    →   Database     →   AI Analysis    →   BI Dashboard
(GitHub, npm, PyPI,        (DDD + Airflow)     (PostgreSQL      (Tech Radar,       (Metabase)
 StackOverflow,                                 + MongoDB)       TH-Index,
 HackerNews)                                                    Trend Prediction)
```

### Tại sao chủ đề này đặc biệt?

| Yếu tố | Giải thích |
|---------|-----------|
| **Multi-source fusion** | Không ai kết hợp 5 nguồn cho cùng 1 technology. OSS Insight chỉ dùng GitHub. |
| **TH-Index** | Đây là thuật toán scoring do bạn tự đề xuất — đóng góp khoa học cho thesis |
| **AI Tech Radar tự động** | ThoughtWorks làm bằng tay. Bạn tự động hóa bằng AI — điểm khác biệt lớn nhất |
| **DDD + Clean Architecture** | Hệ thống enterprise-grade, không phải script đơn giản |
| **Có BI dashboard** | Kết quả trực quan, demo được |

---ii

## 2. Các Nguồn Dữ Liệu

### Source 1: GitHub REST API (v3)

**URL:** `https://api.github.com`
**Auth:** Personal Access Token (free, 5000 requests/hour)

**Endpoint cần dùng:**

```
a) Search repositories theo language/topic
   GET /search/repositories?q=language:{lang}&sort=stars
   → Lấy top repos cho mỗi technology

b) Repository detail
   GET /repos/{owner}/{repo}
   → stars, forks, open_issues, watchers, created_at, updated_at, license

c) Commit activity
   GET /repos/{owner}/{repo}/stats/commit_activity
   → Số commits theo tuần trong 1 năm gần nhất

d) Contributors
   GET /repos/{owner}/{repo}/contributors
   → Danh sách contributors + số commits mỗi người

e) Releases
   GET /repos/{owner}/{repo}/releases
   → Lịch sử release + date

f) Trending (không có API chính thức)
   → Crawl HTML từ github.com/trending hoặc dùng third-party API
```

**Data thu được cho mỗi repository:**

```
repo_full_name          "facebook/react"
stars                   225000
forks                   46000
open_issues             1200
watchers                6700
language                "JavaScript"
license                 "MIT"
created_at              "2013-05-24"
last_push_at            "2026-06-10"
contributors_count      1800
weekly_commits          [45, 52, 38, ...]    ← 52 tuần gần nhất
latest_release_date     "2026-06-01"
release_count_12m       12                   ← Số releases trong 12 tháng
topics                  ["react", "javascript", "frontend", "ui"]
```

**Rate limit:** 5000 requests/hour với token. 1 technology cần ~5-10 calls. 100 technologies = 500-1000 calls/lần chạy → thoải mái.

---

### Source 2: npm Registry API

**URL:** `https://api.npmjs.org`
**Auth:** Không cần (public API)

**Endpoint:**

```
a) Download counts
   GET https://api.npmjs.org/downloads/point/last-week/{package}
   → Downloads tuần gần nhất

b) Download range
   GET https://api.npmjs.org/downloads/range/2026-01-01:2026-06-11/{package}
   → Downloads theo ngày trong khoảng thời gian

c) Package info
   GET https://registry.npmjs.org/{package}
   → Versions, maintainers, dependencies
```

**Ví dụ:** `react` có ~25 triệu downloads/tuần → con số phản ánh real usage tốt hơn stars.

**Áp dụng cho:** Các technology JavaScript/TypeScript (React, Vue, Angular, Svelte, Bun, Deno...)

---

### Source 3: PyPI API (JSON)

**URL:** `https://pypistats.org/api/`
**Auth:** Không cần

**Endpoint:**

```
a) Downloads tổng
   GET https://pypistats.org/api/packages/{package}/recent
   → Downloads last_day, last_week, last_month

b) Downloads theo version Python
   GET https://pypistats.org/api/packages/{package}/python_minor
   → Biết tech nào chạy trên Python 3.11 vs 3.12
```

**Áp dụng cho:** Các technology Python (Django, FastAPI, Flask, pandas, PyTorch, LangChain...)

---

### Source 4: Stack Overflow API (v2.3)

**URL:** `https://api.stackexchange.com/2.3`
**Auth:** Optional (API key tăng quota từ 300 → 10000 requests/day)

**Endpoint:**

```
a) Tag info
   GET /tags/{tag}/info?site=stackoverflow
   → count (tổng số câu hỏi cho tag đó)

b) Questions gần đây
   GET /questions?tagged={tag}&sort=creation&fromdate={epoch}&site=stackoverflow
   → Số câu hỏi mới theo ngày

c) Tag trend
   GET /tags/{tag}/top-answerers/month?site=stackoverflow
   → Ai trả lời nhiều nhất → cho biết community activeness
```

**Ý nghĩa:** Số câu hỏi Stack Overflow = proxy cho "có bao nhiêu người đang học/dùng technology này". Nhiều câu hỏi = nhiều người mới bắt đầu dùng.

---

### Source 5: HackerNews API (Algolia)

**URL:** `https://hn.algolia.com/api/v1`
**Auth:** Không cần

**Endpoint:**

```
a) Search stories
   GET /search?query={tech_name}&tags=story&numericFilters=created_at_i>{epoch}
   → Số bài post mention technology trong 30 ngày gần nhất

b) Search with points
   → Lấy thêm points (upvotes) để đo mức quan tâm
```

**Ý nghĩa:** HackerNews = ý kiến của senior developers/founders. Nếu 1 technology được bàn luận nhiều trên HN → nó đang gây chú ý trong cộng đồng tech.

---

### Tổng Hợp: Technology Master List

Bạn cần 1 danh sách khoảng **50-100 technologies** để theo dõi. Chia theo nhóm:

```
Frontend:       React, Vue, Angular, Svelte, Solid, Qwik, Astro, Next.js, Nuxt
Backend:        Node.js, Django, FastAPI, Flask, Spring Boot, Go Fiber, Rust Actix
Runtime:        Node.js, Deno, Bun
Database:       PostgreSQL, MongoDB, Redis, Supabase, PlanetScale, Turso
AI/ML:          PyTorch, TensorFlow, LangChain, LlamaIndex, Ollama, vLLM
DevOps:         Docker, Kubernetes, Terraform, Pulumi, GitHub Actions
Language:       TypeScript, Rust, Go, Python, Zig, Mojo
Mobile:         React Native, Flutter, Kotlin Multiplatform, Expo
```

Mỗi technology cần mapping:
```
technology_name:   "React"
github_repos:      ["facebook/react"]            ← GitHub
npm_package:       "react"                        ← npm
pypi_package:      null                           ← Không có trên PyPI
stackoverflow_tag: "reactjs"                      ← SO dùng tag khác
hackernews_query:  "React"                        ← Search term
category:          "Frontend Framework"
```

Mapping này lưu trong **config file** (JSON/YAML), không hardcode.

---

## 3. Kiến Trúc Tổng Thể

```
DevPulse/
├── .env
├── pyproject.toml
├── docker-compose.yml            ← PostgreSQL + MongoDB + Airflow + Metabase
│
├── config/
│   └── technologies.yml          ← Master list: 100 technologies + mapping
│
├── dags/                         ← Airflow DAGs
│   ├── dag_crawl.py              ← DAG 1: Crawl daily
│   ├── dag_etl.py                ← DAG 2: ETL + TH-Index
│   ├── dag_ai_analysis.py        ← DAG 3: AI enrichment
│   └── wrapper/
│       ├── crawl_wrapper.py
│       ├── etl_wrapper.py
│       └── ai_wrapper.py
│
├── src/
│   ├── shared/                   ← Nền tảng chung
│   │   ├── domain/
│   │   │   ├── base_model.py
│   │   │   ├── processing_result.py
│   │   │   ├── prompt.py
│   │   │   └── prompt_template.py
│   │   │
│   │   ├── infrastructure/
│   │   │   ├── setting/
│   │   │   │   ├── base_setting.py
│   │   │   │   ├── github_setting.py       ← GitHub token
│   │   │   │   ├── postgres_setting.py
│   │   │   │   ├── mongo_setting.py
│   │   │   │   └── openai_setting.py
│   │   │   │
│   │   │   ├── mongo/
│   │   │   │   ├── client.py
│   │   │   │   └── base_repository.py
│   │   │   │
│   │   │   ├── postgres/
│   │   │   │   ├── client.py
│   │   │   │   └── base_repository.py
│   │   │   │
│   │   │   └── service/
│   │   │       ├── ai_api_caller_base.py
│   │   │       └── prompt_builder_service.py
│   │   │
│   │   └── utils/
│   │       ├── logging.py
│   │       └── rate_limiter.py       ← Rate limiter cho API calls
│   │
│   ├── data_crawler/             ← MODULE 1: Thu thập dữ liệu (MỚI)
│   │   ├── domain/
│   │   │   ├── models/
│   │   │   │   ├── technology.py         ← Technology config model
│   │   │   │   ├── raw_metric.py         ← Raw metric từ 1 source
│   │   │   │   └── crawl_response.py     ← Response sau crawl
│   │   │   ├── ports/
│   │   │   │   └── scraper.py            ← Scraper Protocol
│   │   │   └── services/
│   │   │       └── scraper_dispatcher.py ← Registry: map source → scraper
│   │   │
│   │   └── application/
│   │       ├── entrypoints.py            ← run_crawler()
│   │       └── scrapers/
│   │           ├── github_scraper.py
│   │           ├── npm_scraper.py
│   │           ├── pypi_scraper.py
│   │           ├── stackoverflow_scraper.py
│   │           └── hackernews_scraper.py
│   │
│   ├── data_processing/          ← MODULE 2: ETL + TH-Index
│   │   ├── domain/
│   │   │   └── models/
│   │   │       ├── base_handler.py
│   │   │       ├── processed_data.py
│   │   │       └── th_index.py           ← TH-Index model
│   │   │
│   │   └── application/
│   │       ├── entrypoints.py
│   │       ├── pipeline_factory.py
│   │       └── pipeline_handlers/
│   │           └── devpulse_handlers/
│   │               ├── validation_handler.py
│   │               ├── normalization_handler.py
│   │               ├── fusion_handler.py         ← Merge data 5 sources
│   │               ├── th_index_handler.py        ← Tính TH-Index
│   │               └── saving_handler.py
│   │
│   ├── ai_analysis/              ← MODULE 3: AI Layer
│   │   ├── domain/
│   │   │   └── models/
│   │   │       ├── tech_radar_output.py
│   │   │       └── trend_prediction_output.py
│   │   │
│   │   └── application/
│   │       ├── entrypoints.py            ← run_ai_analysis()
│   │       ├── templates/
│   │       │   ├── tech_radar_prompt.py
│   │       │   └── trend_prediction_prompt.py
│   │       └── handlers/
│   │           ├── tech_radar_handler.py
│   │           └── trend_predictor_handler.py
│   │
│   └── data_ingest/              ← MODULE 4: Orchestrator
│       ├── domain/
│       │   └── models/
│       │       └── ingest_record.py
│       │
│       └── application/
│           ├── pipeline.py
│           └── entrypoints.py
│
└── scripts/
    ├── init_db.py                ← Tạo tables
    └── seed_technologies.py      ← Import master list vào DB
```

---

## 4. Chi Tiết Từng Module

### Module 1: `data_crawler/` — Thu Thập Dữ Liệu

**Nhiệm vụ:** Gọi 5 API sources, thu thập raw metrics cho mỗi technology, lưu raw data.

#### Architecture Pattern: Registry (giống FileDispatcher)

```
ScraperDispatcher
  ├── "github"         → GitHubScraper
  ├── "npm"            → NpmScraper
  ├── "pypi"           → PyPIScraper
  ├── "stackoverflow"  → StackOverflowScraper
  └── "hackernews"     → HackerNewsScraper
```

#### Domain Models

**a) `Technology` — Config cho 1 technology**

Đọc từ `config/technologies.yml`:

```
Chứa:
  name: str                   "React"
  category: str               "Frontend Framework"
  github_repos: list[str]     ["facebook/react"]
  npm_package: str | None     "react"
  pypi_package: str | None    null
  stackoverflow_tag: str      "reactjs"
  hackernews_query: str       "React"
  
  enabled: bool               true  (để tắt tạm 1 tech nếu cần)
```

**b) `RawMetric` — 1 data point từ 1 source**

```
Chứa:
  technology_name: str        "React"
  source: str                 "github" | "npm" | "pypi" | "stackoverflow" | "hackernews"
  metric_name: str            "stars" | "weekly_downloads" | "question_count" | ...
  metric_value: float         225000
  collected_at: datetime      2026-06-11T07:00:00Z
  raw_data: dict              { ... full API response cho tracing ... }
```

**c) `CrawlResponse` — Kết quả crawl 1 technology**

```
Chứa:
  technology_name: str
  source: str
  metrics: list[RawMetric]
  status: str                 "success" | "failed" | "partial"
  error_message: str | None
```

#### Scraper Protocol

```
Mỗi scraper phải implement:

  scrape(technology: Technology) → CrawlResponse

  Nhận vào 1 Technology object, trả về CrawlResponse chứa danh sách metrics.
```

#### Từng Scraper Cụ Thể

**a) `GitHubScraper`**

Nhiệm vụ: Gọi GitHub API, thu thập metrics cho mỗi repo trong `technology.github_repos`.

```
Metrics thu thập:
  - stars:               Tổng số stars
  - forks:               Tổng số forks
  - open_issues:         Số issues đang mở
  - watchers:            Số watchers
  - contributors_count:  Số contributors
  - weekly_commits:      Danh sách commits/tuần (52 tuần)
  - last_push_days:      Số ngày kể từ push cuối cùng
  - release_count_12m:   Số releases trong 12 tháng gần nhất
  - license:             Loại license
  - star_growth_30d:     Stars tăng trong 30 ngày (tính từ data hôm qua trong DB)

Lưu ý:
  - 1 technology có thể có nhiều repos (ví dụ: "React" = facebook/react + nhiều sub-repos)
  - Nếu nhiều repos → tổng hợp (sum stars, sum contributors...)
  - Rate limit: 5000 requests/hour → cần rate_limiter.py
  - Nếu 1 repo fail → log error, continue, đánh dấu status="partial"
```

**b) `NpmScraper`**

```
Nhiệm vụ: Gọi npm API lấy download stats.

Metrics thu thập:
  - weekly_downloads:    Downloads tuần gần nhất
  - monthly_downloads:   Downloads tháng gần nhất
  - download_trend:      So sánh tuần này vs tuần trước (%)

Lưu ý:
  - Chỉ scrape nếu technology.npm_package != null
  - npm API không cần auth, không rate limit nặng
```

**c) `PyPIScraper`**

```
Nhiệm vụ: Gọi PyPI Stats API lấy download counts.

Metrics thu thập:
  - daily_downloads:     Downloads hôm qua
  - weekly_downloads:    Downloads 7 ngày
  - monthly_downloads:   Downloads 30 ngày

Lưu ý:
  - Chỉ scrape nếu technology.pypi_package != null
  - API có thể trả về null nếu package quá nhỏ → handle gracefully
```

**d) `StackOverflowScraper`**

```
Nhiệm vụ: Gọi Stack Exchange API, đếm câu hỏi.

Metrics thu thập:
  - total_questions:         Tổng câu hỏi all-time cho tag này
  - questions_last_30d:      Câu hỏi mới trong 30 ngày
  - answer_rate:             % câu hỏi có accepted answer
  - avg_answers_per_question: Trung bình số câu trả lời

Lưu ý:
  - StackOverflow tag có thể khác tên technology: "reactjs" vs "React"
  - API trả về compressed (gzip) → cần decompress
  - Quota: 300 requests/day (no key) hoặc 10000 (with key)
```

**e) `HackerNewsScraper`**

```
Nhiệm vụ: Gọi HN Algolia API, đếm bài post mentions.

Metrics thu thập:
  - mentions_30d:        Số stories mention tech trong 30 ngày
  - total_points:        Tổng points (upvotes) của các stories đó
  - avg_points:          Trung bình points/story
  - top_story_title:     Tiêu đề bài có nhiều points nhất (context)

Lưu ý:
  - Search query cần chính xác: "React" sẽ match "React Native" → cần filter
  - Algolia API free, không cần auth
```

#### Entrypoint: `run_crawler()`

```
Input:
  - execution_date: str
  - technologies: list[Technology]    ← đọc từ technologies.yml
  - sources: list[str]               ← ["github", "npm", "pypi", "stackoverflow", "hackernews"]

Flow:
  1. Load master list từ config/technologies.yml
  2. Với mỗi technology:
     a. Với mỗi source:
        - Kiểm tra technology có mapping cho source không (npm_package != null?)
        - Nếu có → gọi scraper tương ứng
        - Lưu raw metrics vào MongoDB (raw_metrics collection)
     b. Trả về CrawlResponse
  3. Tổng hợp: bao nhiêu tech thành công, bao nhiêu fail
  4. Output: list[CrawlResponse]

Lưu ý:
  - Gọi tuần tự theo source (tất cả GitHub trước, rồi npm, rồi PyPI...)
  - Rate limiter: chờ giữa các call (đặc biệt GitHub)
  - Nếu 1 source fail hoàn toàn → log, skip, tiếp tục sources khác
```

#### Raw Data Storage (MongoDB)

```
Collection: raw_metrics
Document example:
{
  "_id": ObjectId("..."),
  "technology_name": "React",
  "source": "github",
  "collected_at": "2026-06-11T07:00:00Z",
  "metrics": {
    "stars": 225000,
    "forks": 46000,
    "contributors_count": 1800,
    "weekly_commits": [45, 52, 38, ...],
    ...
  },
  "raw_response": { ... },           ← Full API response (cho debug)
  "status": "success"
}
```

Dùng MongoDB vì mỗi source có schema khác nhau. GitHub có `stars`, npm có `downloads` → không fit vào 1 SQL table.

---

### Module 2: `data_processing/` — ETL + TH-Index

**Nhiệm vụ:** Đọc raw metrics từ MongoDB, merge 5 sources cho cùng 1 technology, tính TH-Index, lưu vào PostgreSQL.

#### Handler Chain

```
ValidationHandler → NormalizationHandler → FusionHandler → THIndexHandler → SavingHandler
```

#### a) `ValidationHandler`

```
Nhiệm vụ:
  1. Đọc raw metrics từ MongoDB cho ngày hôm nay
  2. Validate: mỗi technology có ít nhất 1 source có data
  3. Loại bỏ duplicates (nếu crawl chạy 2 lần)
  4. Kiểm tra giá trị hợp lý: stars ≥ 0, downloads ≥ 0, answer_rate 0-100%
  5. Đánh dấu technologies thiếu data nhiều sources → is_partial = True

Input:  execution_date
Output: dict[technology_name → dict[source → metrics]]
        Ví dụ: {"React": {"github": {...}, "npm": {...}, "stackoverflow": {...}}}
```

#### b) `NormalizationHandler`

```
Nhiệm vụ:
  1. Chuẩn hóa tên technology (viết hoa/thường, aliases)
     Ví dụ: "react.js", "ReactJS", "React" → "React"
  2. Chuẩn hóa đơn vị: tất cả downloads chuyển về weekly
  3. Tính derived metrics nếu cần:
     - download_per_star = weekly_downloads / stars (usage efficiency)
     - issue_close_rate = closed_issues / (closed + open) (community health)

Input:  dict[technology_name → dict[source → metrics]]
Output: dict[technology_name → dict[source → normalized_metrics]]
```

#### c) `FusionHandler` ⭐ (Điểm đặc biệt)

```
Nhiệm vụ:
  Merge data từ 5 sources thành 1 unified record cho mỗi technology.

  Với mỗi technology, tạo 1 TechnologySnapshot:
    technology_name:       "React"
    category:              "Frontend Framework"
    snapshot_date:         "2026-06-11"
    
    # GitHub metrics
    stars:                 225000
    forks:                 46000
    contributors:          1800
    open_issues:           1200
    weekly_commits_avg:    45.2        ← mean của 4 tuần gần nhất
    last_push_days:        1
    release_count_12m:     12
    star_growth_30d:       1500        ← stars tăng 30 ngày
    
    # Download metrics (merge npm + pypi)
    weekly_downloads:      25000000    ← npm hoặc pypi, tùy tech
    download_trend_pct:    5.2         ← % thay đổi so tuần trước
    
    # Community metrics (Stack Overflow)
    so_questions_30d:      3500
    so_answer_rate:        72.5
    
    # Buzz metrics (HackerNews)
    hn_mentions_30d:       45
    hn_avg_points:         120
    
    # Completeness
    sources_available:     5           ← Có data từ bao nhiêu sources
    is_partial:            false

Input:  dict[technology_name → dict[source → normalized_metrics]]
Output: list[TechnologySnapshot]
```

#### d) `THIndexHandler` ⭐⭐ (Đóng góp khoa học cho thesis)

```
Nhiệm vụ:
  Tính Technology Health Index (TH-Index) cho mỗi technology.
  
  TH-Index = chỉ số 0-100 đánh giá "sức khỏe" tổng thể của 1 technology.

  Công thức (bạn sẽ tune weights dựa trên thực nghiệm):

  TH-Index = w1 × Momentum + w2 × Adoption + w3 × Community + w4 × Maintenance + w5 × Buzz

  Trong đó:

  Momentum (tốc độ phát triển):
    = normalize(star_growth_30d)  ×  0.5
    + normalize(download_trend_pct)  ×  0.5
    → Thể hiện: technology đang tăng tốc hay chậm lại?

  Adoption (mức độ sử dụng thực tế):
    = normalize(weekly_downloads) ×  0.6
    + normalize(forks)            ×  0.2
    + normalize(download_per_star) × 0.2
    → Thể hiện: có bao nhiêu người thực sự dùng? (downloads > stars)

  Community (cộng đồng):
    = normalize(contributors)     ×  0.3
    + normalize(so_questions_30d) ×  0.3
    + normalize(so_answer_rate)   ×  0.4
    → Thể hiện: cộng đồng có active không? Câu hỏi có được trả lời không?

  Maintenance (bảo trì):
    = normalize(release_count_12m) ×  0.4
    + normalize(weekly_commits_avg) × 0.3
    + (1 / normalize(last_push_days)) × 0.3
    → Thể hiện: maintainers có đang tích cực không?

  Buzz (tiếng vang):
    = normalize(hn_mentions_30d)  ×  0.5
    + normalize(hn_avg_points)    ×  0.5
    → Thể hiện: cộng đồng senior có đang bàn về tech này không?

  Weights mặc định:
    w1 = 0.25, w2 = 0.30, w3 = 0.20, w4 = 0.15, w5 = 0.10

  normalize(): Min-max normalization dựa trên tất cả technologies cùng ngày.
  Score 0-100 cho mỗi dimension, sau đó weighted average → TH-Index 0-100.

  Output cho mỗi technology:
    th_index:           78.5
    momentum_score:     82.0
    adoption_score:     95.0
    community_score:    70.0
    maintenance_score:  65.0
    buzz_score:         55.0
    rank:               3          ← rank trong tất cả technologies
    previous_rank:      5          ← rank ngày hôm trước
    rank_change:        +2         ← lên 2 bậc

Thesis note:
  Phần phân tích thesis:
  - So sánh TH-Index vs chỉ dùng stars → chứng minh multi-factor tốt hơn
  - Experiment: thay đổi weights → xem kết quả thay đổi thế nào
  - Validation: lấy ThoughtWorks Tech Radar làm ground truth, so accuracy
```

#### e) `SavingHandler`

```
Nhiệm vụ:
  1. Gắn execution_date, run_id vào mỗi record
  2. Chuẩn bị data cho bulk insert vào PostgreSQL
  3. Ghi log summary: bao nhiêu technologies processed

Output: list[dict] sẵn sàng insert
```

---

### Module 3: `ai_analysis/` — AI Enrichment

**Nhiệm vụ:** Dùng LLM phân loại technology vào Tech Radar và dự báo trend.

#### Handler 1: `TechRadarHandler` ⭐ (Feature đặc biệt nhất)

```
Nhiệm vụ:
  Phân loại mỗi technology vào 4 rings của Tech Radar:
  
  - Adopt:  Đã chứng minh giá trị, nên dùng cho production
  - Trial:  Đáng thử trong project mới, có tiềm năng
  - Assess: Cần theo dõi thêm, chưa đủ dữ liệu để quyết định
  - Hold:   Nên cẩn trọng, đang suy giảm hoặc có vấn đề

Input (cho mỗi technology):
  Tất cả metrics từ TechnologySnapshot + TH-Index scores + data lịch sử 30 ngày

System prompt:
  "You are a senior technology analyst like those at ThoughtWorks.
   Based on multi-source data, classify the technology into the Tech Radar framework.
   Be specific with data-driven reasoning. Cite actual numbers."

User prompt:
  "Technology: {name} ({category})
   
   GitHub: {stars} stars ({star_growth_30d} growth/30d), {contributors} contributors,
   {release_count_12m} releases/year, last push {last_push_days} days ago
   
   Downloads: {weekly_downloads}/week ({download_trend_pct}% vs last week)
   
   Stack Overflow: {so_questions_30d} new questions/30d, {so_answer_rate}% answer rate
   
   HackerNews: {hn_mentions_30d} mentions/30d, avg {hn_avg_points} points
   
   TH-Index: {th_index}/100 (rank #{rank}, {rank_change})
   30-day trend: {th_index_trend}
   
   Classify into: Adopt / Trial / Assess / Hold
   Explain with specific data points. Identify risks."

Output model:
  ring:           "Trial"
  confidence:     0.82
  reasoning:      "Bun shows 340% star growth but npm downloads are still 1% of Node.js..."
  key_strengths:  ["Performance", "Node.js compatibility", "Active development"]
  risk_factors:   ["Small ecosystem", "Breaking API changes between versions"]
  recommendation: "Good for side projects and CLI tools. Not ready for large-scale production."

Batch processing:
  Gọi cho ~100 technologies → dùng handle_batch() với ThreadPoolExecutor (5 workers)
  ~100 API calls × ~2s/call ÷ 5 workers = ~40 seconds
```

#### Handler 2: `TrendPredictorHandler`

```
Nhiệm vụ:
  Dựa trên 30 ngày TH-Index history, dự báo xu hướng 30 ngày tới.

Input:
  - technology_name
  - 30 ngày TH-Index scores
  - 30 ngày star_growth, download_trend
  - Tin tức/events gần đây (từ HackerNews)

Output model:
  predicted_trend:     "rising" | "stable" | "declining"
  confidence:          0.75
  predicted_th_index:  82 (±5)        ← dự báo TH-Index 30 ngày sau
  catalysts:           ["v2.0 release coming", "Major company adoption"]
  risks:               ["Key maintainer leaving", "Competing tech gaining traction"]
  summary:             "React is expected to remain stable with slight growth due to..."

Thesis note:
  Đo accuracy: sau 30 ngày, so sánh predicted vs actual → tính error rate
  Đây là phần research có thể viết 1 chương thesis
```

#### Entrypoint: `run_ai_analysis()`

```
Flow:
  1. Query latest TechnologySnapshot + TH-Index từ PostgreSQL
  2. Với mỗi technology:
     a. TechRadarHandler.handle() → radar classification
     b. TrendPredictorHandler.handle() → trend prediction
  3. Lưu kết quả vào PostgreSQL (tech_radar, trend_predictions tables)
  4. Tổng hợp: bao nhiêu tech classified, bao nhiêu fail
```

---

### Module 4: `data_ingest/` — Orchestrator

```
Nhiệm vụ: Điều phối toàn bộ pipeline.

run_ingest_pipeline():
  Step 1: run_crawler()           → Thu thập raw data từ 5 sources
  Step 2: run_data_processing()   → ETL + TH-Index
  Step 3: run_ai_analysis()       → Tech Radar + Trend Prediction
  Step 4: Save summary            → ProcessingResult vào MongoDB
  Return: IngestionRecord
```

---

## 5. Database Schema

### PostgreSQL — Structured Data

```sql
-- Bảng chính: snapshot hàng ngày cho mỗi technology
CREATE TABLE technology_snapshots (
    id                      SERIAL PRIMARY KEY,
    technology_name         VARCHAR(100)  NOT NULL,
    category                VARCHAR(100),
    snapshot_date           DATE          NOT NULL,
    
    -- GitHub
    stars                   BIGINT,
    forks                   BIGINT,
    open_issues             INTEGER,
    contributors            INTEGER,
    weekly_commits_avg      NUMERIC(8,2),
    last_push_days          INTEGER,
    release_count_12m       INTEGER,
    star_growth_30d         INTEGER,
    
    -- Downloads
    weekly_downloads        BIGINT,
    download_trend_pct      NUMERIC(8,2),
    
    -- Stack Overflow
    so_questions_30d        INTEGER,
    so_answer_rate          NUMERIC(5,2),
    
    -- HackerNews
    hn_mentions_30d         INTEGER,
    hn_avg_points           NUMERIC(8,2),
    
    -- TH-Index
    th_index                NUMERIC(5,2),
    momentum_score          NUMERIC(5,2),
    adoption_score          NUMERIC(5,2),
    community_score         NUMERIC(5,2),
    maintenance_score       NUMERIC(5,2),
    buzz_score              NUMERIC(5,2),
    rank                    INTEGER,
    previous_rank           INTEGER,
    rank_change             INTEGER,
    
    -- Meta
    sources_available       INTEGER,
    is_partial              BOOLEAN DEFAULT FALSE,
    run_id                  VARCHAR(20),
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(technology_name, snapshot_date)
);

-- Index cho dashboard queries
CREATE INDEX idx_snapshots_date ON technology_snapshots (snapshot_date);
CREATE INDEX idx_snapshots_tech ON technology_snapshots (technology_name);
CREATE INDEX idx_snapshots_rank ON technology_snapshots (snapshot_date, rank);

-- Bảng AI Tech Radar
CREATE TABLE tech_radar (
    id                  SERIAL PRIMARY KEY,
    technology_name     VARCHAR(100) NOT NULL,
    analysis_date       DATE         NOT NULL,
    ring                VARCHAR(20)  NOT NULL,      -- Adopt / Trial / Assess / Hold
    confidence          NUMERIC(3,2),
    reasoning           TEXT,
    key_strengths       JSONB,
    risk_factors        JSONB,
    recommendation      TEXT,
    ai_model            VARCHAR(50),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(technology_name, analysis_date)
);

-- Bảng AI Trend Predictions
CREATE TABLE trend_predictions (
    id                      SERIAL PRIMARY KEY,
    technology_name         VARCHAR(100) NOT NULL,
    prediction_date         DATE         NOT NULL,    -- Ngày dự báo
    target_date             DATE         NOT NULL,    -- Dự báo cho ngày nào (prediction_date + 30)
    predicted_trend         VARCHAR(20),               -- rising / stable / declining
    confidence              NUMERIC(3,2),
    predicted_th_index      NUMERIC(5,2),
    catalysts               JSONB,
    risks                   JSONB,
    summary                 TEXT,
    -- Actual (điền vào sau 30 ngày để so sánh)
    actual_th_index         NUMERIC(5,2),              -- NULL cho tới khi đến target_date
    prediction_error        NUMERIC(5,2),              -- |predicted - actual|
    ai_model                VARCHAR(50),
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(technology_name, prediction_date)
);

-- Bảng processing results
CREATE TABLE processing_results (
    id          SERIAL PRIMARY KEY,
    run_id      VARCHAR(20)  UNIQUE NOT NULL,
    summary     JSONB        NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);
```

### MongoDB — Raw Data

```
Database: devpulse_db

Collections:
  - raw_metrics:           Raw API responses từ 5 sources (schema-less)
  - crawl_logs:            Log mỗi lần crawl (status, errors, timing)
  - processing_results:    Summary mỗi lần pipeline chạy
```

---

## 6. Airflow DAGs

### DAG 1: `dag_crawl` — Thu thập dữ liệu hàng ngày

```
Schedule: 6:00 AM UTC mỗi ngày (= 1:00 PM Việt Nam)
Timeout:  2 giờ (max)

Tasks:
  1. crawl_github      → GitHubScraper: ~500 API calls, ~10 phút
  2. crawl_npm         → NpmScraper: ~50 calls, ~2 phút (chạy song song với task 3-5)
  3. crawl_pypi        → PyPIScraper: ~30 calls, ~1 phút
  4. crawl_stackoverflow → SOScraper: ~100 calls, ~5 phút
  5. crawl_hackernews  → HNScraper: ~100 calls, ~3 phút

Flow:
  crawl_github >> [crawl_npm, crawl_pypi, crawl_stackoverflow, crawl_hackernews]
  (GitHub chạy trước vì nặng nhất + rate limit, 4 sources còn lại chạy song song sau)
```

### DAG 2: `dag_etl` — ETL + TH-Index

```
Schedule: Trigger bởi DAG 1 (dùng ExternalTaskSensor hoặc TriggerDagRunOperator)

Tasks:
  1. wait_for_crawl    → ExternalTaskSensor: chờ dag_crawl xong
  2. run_etl_pipeline  → Chạy handler chain: Validate → Normalize → Fusion → THIndex → Save
  3. send_notification → Email/Slack thông báo kết quả

Flow:
  wait_for_crawl >> run_etl_pipeline >> send_notification
```

### DAG 3: `dag_ai_analysis` — AI Enrichment

```
Schedule: Trigger bởi DAG 2

Tasks:
  1. wait_for_etl              → Chờ ETL xong
  2. run_tech_radar_analysis   → Gọi AI classify ~100 technologies
  3. run_trend_predictions     → Gọi AI predict ~100 technologies
  4. backfill_prediction_accuracy → Kiểm tra predictions 30 ngày trước, tính accuracy

Flow:
  wait_for_etl >> [run_tech_radar_analysis, run_trend_predictions] >> backfill_prediction_accuracy
```

---

## 7. BI Dashboard (Metabase)

### Setup

Metabase chạy trong Docker, kết nối trực tiếp vào PostgreSQL.

```yaml
# docker-compose.yml (thêm service)
metabase:
  image: metabase/metabase:latest
  ports:
    - "3000:3000"
  environment:
    MB_DB_TYPE: postgres
    MB_DB_HOST: postgres
    MB_DB_PORT: 5432
    MB_DB_DBNAME: devpulse
  depends_on:
    - postgres
```

### Dashboard Panels Gợi Ý

**Dashboard 1: Technology Radar Overview**
```
Panel 1: Tech Radar Visualization
  → Scatter plot: 4 quadrants (Adopt/Trial/Assess/Hold)
  → X = Adoption score, Y = Momentum score
  → Color = ring, Size = TH-Index
  → Hoặc: traditional radar circle chart

Panel 2: Top 10 Technologies by TH-Index
  → Horizontal bar chart, sorted by TH-Index
  → Color coded by ring

Panel 3: Biggest Movers (rank change)
  → Table: technologies có rank_change lớn nhất (cả tăng lẫn giảm)
  → "FastAPI: rank #15 → #8 (+7)"

Panel 4: Ring Distribution
  → Pie chart: bao nhiêu % technologies ở mỗi ring
```

**Dashboard 2: Technology Deep Dive**
```
(Filter: chọn 1 technology cụ thể)

Panel 1: TH-Index Over Time
  → Line chart: TH-Index 30/60/90 ngày
  → 5 sub-scores (momentum, adoption, community, maintenance, buzz) overlaid

Panel 2: GitHub Stats
  → Stars growth line chart
  → Commits per week bar chart

Panel 3: Download Trend
  → Weekly downloads line chart (npm/pypi)

Panel 4: Community Health
  → SO questions/month + answer rate

Panel 5: AI Assessment (text)
  → Latest Tech Radar reasoning
  → Latest Trend Prediction

Panel 6: Comparison
  → So sánh technology này vs 2-3 alternatives cùng category
```

**Dashboard 3: Category Analysis**
```
Panel 1: Category Heatmap
  → Matrix: categories × dimensions (momentum, adoption, community...)
  → Color = score

Panel 2: Category Trends
  → Line chart: average TH-Index by category over time

Panel 3: Top Technology per Category
  → Table: mỗi category → top 3 technologies
```

---

## 8. Rate Limiter Utility

```
Nhiệm vụ:
  Giới hạn tốc độ gọi API, tránh bị rate limit.

Cách hoạt động:
  - Mỗi source có config riêng: GitHub = max 5000 req/hour, npm = unlimited
  - Trước mỗi API call → check xem có vượt limit không
  - Nếu gần limit → sleep chờ reset window
  - Log cảnh báo khi sử dụng > 80% quota

Cần tạo:
  - RateLimiter class trong shared/utils/rate_limiter.py
  - Config: max_requests, window_seconds, current_count, window_start
  - Method: acquire() → True/False (có được phép gọi không)
  - Method: wait_if_needed() → sleep nếu cần
```

---

## 9. Settings (.env)

```env
# GitHub
GITHUB_TOKEN=ghp_your_personal_access_token

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=devpulse_user
POSTGRES_PASSWORD=devpulse_pass
POSTGRES_DB=devpulse

# MongoDB
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB_NAME=devpulse_db
MONGO_TIMEOUT=5000

# AI
OPENAI_API_KEY=sk-your-key
AI_MODEL=gpt-4o-mini

# Config
TECHNOLOGY_CONFIG_PATH=config/technologies.yml
```

---

## 10. Checklist Implementation

```
Phase 0: Setup (Tuần 1)
  [ ] Tạo project + pyproject.toml + .env
  [ ] Setup Docker Compose: PostgreSQL + MongoDB + Airflow
  [ ] Tạo database + tables (DDL)
  [ ] Tạo config/technologies.yml (bắt đầu với 20 technologies)
  [ ] Đăng ký GitHub token + OpenAI key

Phase 1: shared/ (Tuần 1-2)
  [ ] Copy + adapt từ project công ty: CustomBaseModel, Settings, Repos, Logger
  [ ] Tạo PostgresClient + BasePostgresRepository
  [ ] Tạo MongoDBClient + BaseMongoRepository
  [ ] Tạo RateLimiter
  [ ] Tạo tất cả Settings (GitHub, Postgres, Mongo, OpenAI)
  [ ] Test: connect cả 2 DB thành công

Phase 2: data_crawler/ (Tuần 2-3)
  [ ] Tạo domain models: Technology, RawMetric, CrawlResponse
  [ ] Tạo Scraper Protocol
  [ ] Tạo ScraperDispatcher (Registry)
  [ ] Tạo GitHubScraper + test với 1 repo
  [ ] Tạo NpmScraper + test
  [ ] Tạo PyPIScraper + test
  [ ] Tạo StackOverflowScraper + test
  [ ] Tạo HackerNewsScraper + test
  [ ] Tạo run_crawler()
  [ ] Test end-to-end: crawl 5 technologies × 5 sources

Phase 3: data_processing/ (Tuần 3-4)
  [ ] Tạo ProcessedData, BaseProcessingHandler
  [ ] Tạo ValidationHandler
  [ ] Tạo NormalizationHandler
  [ ] Tạo FusionHandler
  [ ] Tạo THIndexHandler (implement công thức TH-Index)
  [ ] Tạo SavingHandler
  [ ] Wire chain trong pipeline_factory.py
  [ ] Tạo TechnologySnapshotRepository
  [ ] Test: raw data → processed → DB

Phase 4: Orchestrator + Airflow (Tuần 5)
  [ ] Tạo data_ingest/pipeline.py
  [ ] Tạo dags/dag_crawl.py
  [ ] Tạo dags/dag_etl.py
  [ ] Tạo wrapper functions
  [ ] Test: chạy full pipeline qua Airflow

Phase 5: AI Analysis (Tuần 6-7)
  [ ] Tạo TechRadarPromptTemplate
  [ ] Tạo TechRadarHandler
  [ ] Tạo TrendPredictorHandler
  [ ] Tạo dags/dag_ai_analysis.py
  [ ] Tạo TechRadarRepository + TrendPredictionRepository
  [ ] Test: AI classify 20 technologies, review kết quả

Phase 6: BI Dashboard (Tuần 7-8)
  [ ] Setup Metabase trong Docker
  [ ] Kết nối Metabase → PostgreSQL
  [ ] Tạo Dashboard 1: Technology Radar Overview
  [ ] Tạo Dashboard 2: Technology Deep Dive
  [ ] Tạo Dashboard 3: Category Analysis

Phase 7: Polish + Thesis (Tuần 8-10)
  [ ] Mở rộng technologies.yml lên 50-100 technologies
  [ ] Chạy pipeline liên tục 7-14 ngày để tích lũy data
  [ ] Backfill prediction accuracy (so predicted vs actual)
  [ ] Experiment: thay đổi TH-Index weights, đo ảnh hưởng
  [ ] So sánh TH-Index vs stars-only → chứng minh multi-factor tốt hơn
  [ ] Viết thesis chapters
  [ ] Demo dashboard
```

---

## 11. Dependencies

```
[project]
dependencies = [
    # Core
    "pydantic>=2.0",
    "pydantic-settings",
    
    # Data
    "pandas",
    "numpy",
    
    # Database
    "psycopg2-binary",
    "pymongo",
    
    # API
    "requests",
    "httpx",                  # Async HTTP (optional, cho performance)
    
    # AI
    "openai",
    "tiktoken",
    
    # Airflow (nếu chạy local, nếu dùng Docker thì không cần)
    # "apache-airflow>=3.0",
    
    # Config
    "pyyaml",                 # Đọc technologies.yml
    
    # Utils
    "tqdm",                   # Progress bars
]
```

---

## 12. Thesis Outline Gợi Ý

```
Chương 1: Giới thiệu
  - Bài toán: tại sao cần theo dõi xu hướng công nghệ tự động
  - Mục tiêu: xây hệ thống end-to-end với TH-Index + AI Tech Radar
  - Phạm vi: 50-100 technologies, 5 data sources

Chương 2: Cơ sở lý thuyết
  - ETL Pipeline architecture
  - DDD (Domain-Driven Design) + Clean Architecture
  - Design Patterns: Registry, Chain of Responsibility, Repository
  - Prompt Engineering + Structured Output
  - Technology Radar framework (ThoughtWorks)

Chương 3: Phân tích & thiết kế
  - Kiến trúc tổng thể (diagram)
  - Database schema design
  - TH-Index formula derivation (tại sao chọn 5 dimensions, tại sao weights này)
  - AI prompt design (tại sao prompt template viết như vậy)

Chương 4: Triển khai
  - Module-by-module implementation
  - Screenshots code + Airflow UI + Metabase dashboard
  - Challenges gặp phải + cách giải quyết

Chương 5: Kết quả & đánh giá
  - TH-Index ranking vs GitHub stars ranking → so sánh
  - AI Tech Radar accuracy vs ThoughtWorks (nếu có ground truth)
  - Trend prediction accuracy sau 30 ngày
  - Dashboard screenshots + user feedback

Chương 6: Kết luận & hướng phát triển
  - Tóm tắt đóng góp
  - Hạn chế (data bias, AI hallucination, rate limits)
  - Hướng mở rộng: thêm nguồn (Reddit, Twitter/X), real-time, mobile app
```
