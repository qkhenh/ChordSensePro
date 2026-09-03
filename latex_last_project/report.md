# BÁO CÁO DỰ ÁN: HỆ THỐNG KIỂM SOÁT RA VÀO THÔNG MINH
# (SMART TOLL GATE - IoT Multidisciplinary Project)

**Mã môn học:** HK252 - Đa ngành IoT  
**Cập nhật lần cuối:** Tháng 04/2026

---

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Business Requirements](#3-business-requirements)
4. [Functional Requirements & Use Cases](#4-functional-requirements--use-cases)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [Data Requirements](#6-data-requirements)
7. [IoT Gateway Implementation (Python)](#7-iot-gateway-implementation-python)
8. [Phụ lục](#8-phụ-lục)
9. [Manager Dashboard KPIs](#9-manager-dashboard-kpis)
10. [Trạng thái Implement](#10-trạng-thái-implement)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Mô tả dự án

**Smart Toll Gate** là hệ thống quản lý kiểm soát phương tiện ra vào khu dân cư/chung cư tự động, tích hợp giữa **Internet of Things (IoT)** và **Trí tuệ Nhân tạo (AI)**. Hệ thống hướng tới việc số hóa toàn bộ quy trình kiểm soát an ninh, giảm thiểu thao tác thủ công của lực lượng bảo vệ, đồng thời cung cấp trải nghiệm tiện lợi, minh bạch cho cư dân.

### 1.2 Mục tiêu

| Mục tiêu | Mô tả |
|----------|-------|
| **Tự động hóa** | Giảm 80% thời gian xử lý thủ công tại cổng |
| **An ninh** | Ghi log 100% lượt ra vào với hình ảnh minh chứng |
| **Minh bạch** | Dashboard real-time cho bảo vệ, thống kê cho Ban quản lý |
| **Linh hoạt** | Hỗ trợ xác thực tự động toàn diện: AI đọc biển số xe, Camera OCR đọc OTP/QR từ màn hình điện thoại |

### 1.3 Đối tượng sử dụng

1. **Cư dân (Citizens):** Chủ hộ/thành viên trong khu dân cư
2. **Bảo vệ (Security Guards):** Lực lượng an ninh tại các cổng
3. **Ban Quản lý (Managers):** Quản lý khu vực/zone cụ thể

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1 Tổng quan kiến trúc

Hệ thống được thiết kế theo **mô hình microservices** với các luồng giao tiếp được quy định rõ ràng giữa các thành phần như sau:

- **Frontend (React) ↔ Backend (Node.js)**: 2 chiều (Frontend gửi API Request, Backend trả Response hoặc chủ động đẩy thông báo Realtime qua WebSocket).
- **Backend (Node.js) ↔ Database (PostgreSQL)**: 2 chiều (Backend gửi câu lệnh SQL, Database trả về kết quả truy vấn).
- **Backend (Node.js) ↔ IoT Gateway (Python)**: 2 chiều *(Điểm mấu chốt)* (Gateway đẩy ảnh/thông tin lên xin phép mở cổng; Backend trả lời hợp lệ/không hợp lệ, hoặc chủ động ra lệnh OPEN/CLOSE từ xa xuống Gateway).
- **AI Service (Python) ↔ Backend (Node.js)**: 2 chiều (Hệ thống gửi ảnh cần nhận diện sang AI; AI Service trả về chuỗi JSON chứa text biển số xe hoặc nội dung OCR từ màn hình điện thoại — OTP 6 số / QR code — cùng độ tin cậy).
- **IoT Gateway (Python) ↔ Arduino**: 2 chiều (Gateway truyền lệnh xuống mạch qua cổng Serial, Arduino báo cáo ngược lại tín hiệu phát hiện xe từ cảm biến IR/Ultrasonic).
- **IP Camera → AI Service / IoT Gateway**: 1 chiều (Camera đẩy luồng Video Stream phục vụ hai tác vụ: (1) nhận diện biển số xe cư dân/khách, (2) đọc mã OTP 6 số hoặc QR code trên màn hình điện thoại của khách vãng lai).
- **Arduino → Barrier (Servo) / LED & Buzzer**: 1 chiều (Arduino xuất tín hiệu điện PWM để kéo rào chắn lên và bật đèn/còi).

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MÔ HÌNH GIAO TIẾP HỆ THỐNG                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   ┌─────────────┐ 2 chiều (HTTP/WS) ┌─────────────┐  2 chiều (HTTP)  ┌─────────────┐   │
│   │  FRONTEND   │◄─────────────────►│   BACKEND   │◄────────────────►│ AI SERVICE  │   │
│   │   (React)   │                   │  (Node.js)  │                  │  (Python)   │   │
│   └─────────────┘                   └──────┬──────┘                  └──────▲──────┘   │
│                                            │                                │          │
│                                            │ 2 chiều                        │ 1 chiều  │
│                                            │ (SQL)                          │ (Stream) │
│                                    ┌───────▼───────┐                        │          │
│                                    │  POSTGRESQL   │                        │          │
│                                    │  (Database)   │                        │          │
│                                    └───────────────┘                        │          │
│                                                                             │          │
│   ┌─────────────┐    1 chiều (HTTP POST)   │                                │          │
│   │ QR SCANNER  ├──────────────────────────┤                                │          │
│   │  (Reader)   │                          │                                │          │
│   └─────────────┘                          │ 2 chiều                        │          │
│                                            │ (HTTP)                         │          │
│   ┌─────────────┐                  ┌───────▼───────┐    1 chiều (Stream)    │          │
│   │  IP CAMERA  ├─────────────────►│  IOT GATEWAY  │◄───────────────────────┘          │
│   └─────────────┘ 1 chiều (Stream) │   (Python)    │                                   │
│                                    └───────┬───────┘                                   │
│                                            │                                           │
│                                            │ 2 chiều (Serial)                          │
│                                            │                                           │
│                                    ┌───────▼───────┐                                   │
│                                    │    ARDUINO    │                                   │
│                                    │    (C/C++)    │                                   │
│                                    └───────┬───────┘                                   │
│                                            │                                           │
│                                            │ 1 chiều (PWM/Digital)                     │
│                            ┌───────────────┴───────────────┐                           │
│                    ┌───────▼───────┐               ┌───────▼───────┐                   │
│                    │    BARRIER    │               │ LED & BUZZER  │                   │
│                    │ (Servo Motor) │               │   Indicators  │                   │
│                    └───────────────┘               └───────────────┘                   │
│                                                                                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Backend Architecture - MVC Pattern

Backend được xây dựng theo kiến trúc **Model-View-Controller (MVC)** với Express.js:

```
backend/
└── src/
    ├── server.js           # Entry point, Express app configuration
    ├── libs/
    │   └── db.js           # PostgreSQL connection pool (pg)
    ├── middlewares/
    │   └── auth.middleware.js  # JWT authentication & role-based authorization
    ├── models/             # DATA ACCESS LAYER (MODEL)
    │   ├── auth.model.js       # User authentication queries
    │   ├── citizens.model.js   # OTP, vehicles, guests management
    │   ├── guards.model.js     # OTP verification, manual actions, logs
    │   ├── gates.model.js      # AI check-in, whitelist logic
    │   └── managers.model.js   # Analytics, vehicle approval, AI performance
    ├── controllers/        # BUSINESS LOGIC LAYER (CONTROLLER)
    │   ├── auth.controller.js
    │   ├── citizens.controller.js
    │   ├── guards.controller.js
    │   ├── gates.controller.js
    │   └── managers.controller.js
    └── routes/             # API ROUTING LAYER (VIEW routing)
        ├── auth.routes.js
        ├── citizens.routes.js
        ├── guards.routes.js
        ├── gates.routes.js
        └── managers.routes.js
```

#### Phân tách trách nhiệm (Separation of Concerns):

| Layer | Trách nhiệm | Files |
|-------|-------------|-------|
| **Model** | Truy vấn database, business rules cấp data | `*.model.js` |
| **Controller** | Xử lý request, validation, gọi model, format response | `*.controller.js` |
| **Route** | Định nghĩa endpoints, gắn middleware, delegate to controller | `*.routes.js` |
| **Middleware** | Cross-cutting concerns (auth, logging) | `auth.middleware.js` |

### 2.3 Mô hình phân quyền - Disjoint Total

```
                    ┌───────────┐
                    │   USERS   │
                    │ (Base)    │
                    └─────┬─────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ CITIZENS  │   │  GUARDS   │   │ MANAGERS  │
    │ (Role 1)  │   │ (Role 2)  │   │ (Role 3)  │
    └───────────┘   └───────────┘   └───────────┘
```

- Mỗi user chỉ có **duy nhất một role**
- Sử dụng `user_role_enum` để enforce tại database level
- JWT token chứa role để authorization ở middleware

### 2.4 Công nghệ sử dụng

| Component | Technology | Version |
|-----------|------------|---------|
| **Backend** | Node.js + Express | 5.x |
| **Realtime** | Socket.IO (WebSocket) | 4.x |
| **Database** | PostgreSQL | 17 |
| **AI Service** | Python + YOLOv8 + OCR | 3.11+ |
| **IoT Gateway** | Python + pySerial + python-socketio | 3.11+ |
| **Hardware** | Arduino Uno + Servo Motor + HC-SR04 | - |
| **Authentication** | JWT (jsonwebtoken) | - |
| **Containerization** | Docker + Docker Compose | - |

---

## 3. BUSINESS REQUIREMENTS

### 3.1 Bối cảnh kinh doanh

Các khu dân cư/chung cư hiện đại đối mặt với thách thức:
- **Chi phí nhân sự cao:** Cần bảo vệ 24/7 tại mỗi cổng
- **Xác minh thủ công chậm:** Gây ùn tắc giờ cao điểm
- **Thiếu minh bạch:** Không có bằng chứng lịch sử ra vào
- **Khó quản lý khách:** Không kiểm soát được xe lạ

### 3.2 Business Goals

| ID | Business Goal | Success Metric |
|----|---------------|----------------|
| **BG-01** | Giảm thời gian check-in trung bình | < 3 giây/xe (với AI) |
| **BG-02** | Tối ưu chi phí vận hành | Giảm 50% nhân sự bảo vệ |
| **BG-03** | Tăng an ninh khu vực | 100% ghi log có hình ảnh |
| **BG-04** | Cải thiện trải nghiệm cư dân | Satisfaction > 90% |
| **BG-05** | Dữ liệu phục vụ ra quyết định | Dashboard analytics real-time |

### 3.3 Business Scenarios

#### Scenario 1: Happy Path (Fully Automated Flow — AI OCR Biển Số)
```markdown
🔍 Bước 1: AI đọc biển số → Quét Whitelist Vĩnh viễn (Bảng vehicles)
- Camera tại làn xe chụp ảnh và gửi đến AI Service.
- AI Service (YOLOv8 + OCR) thực hiện DUY NHẤT một tác vụ: ĐỌC TEXT BIỂN SỐ XE và trả về plate_text.
  (Không có cơ chế dán tem ePass hay QR tag trên xe.)
- Backend truy vấn bảng `vehicles`: tìm xe có `license_plate = plate_text` và `is_active = true`.
- Kiểm tra kết quả:
  + Nếu KHÔNG TÌM THẤY → xe không phải cư dân → chuyển sang Bước 2.
  + Nếu TÌM THẤY → kiểm tra Anti-Passback:
    * Đọc cờ `is_inside` của xe trong DB.
    * Nếu làn đang là inbound (xe vào) nhưng `is_inside = true` → Lỗi logic → Chặn ngay, cảnh báo "Xe đang ở trong bãi".
- Phê duyệt: Pass Anti-Passback → Cập nhật `is_inside = true` → Ghi access_log → Gửi lệnh OPEN. Kết thúc luồng.

🔍 Bước 2: Quét Whitelist Tạm thời (Bảng guest_registrations)
- Nếu biển số không khớp Whitelist Vĩnh viễn, Backend tìm trong danh sách khách đã hẹn trước.
- Truy vấn bảng `guest_registrations` với điều kiện:
  + `guest_license_plate = plate_text`
  + `status = 'approved'`
- Kiểm tra Thời gian (Time Window Validation):
  + Thời gian hiện tại có nằm trong khoảng `[visit_start_time, visit_end_time]` không?
- Kiểm tra kết quả:
  + Nếu TÌM THẤY và ĐÚNG GIỜ → Ghi log (lưu guest_reg_id) → Mở cổng. Kết thúc luồng.
  + Nếu KHÔNG TÌM THẤY hoặc SAI GIỜ → Từ chối → chuyển sang Bước 3.

🛑 Bước 3: Fallback (Từ chối & Cảnh báo Bảo vệ)
- Trượt cả Bước 1 và Bước 2:
- Backend ghi log từ chối và giữ cổng ĐÓNG.`
- Trả lệnh KEEP_CLOSED xuống IoT Gateway.
- Đẩy thông báo Real-time lên màn hình Bảo vệ: "Xe không hợp lệ — vui lòng xử lý!".
- Khách muốn vào có thể chuyển sang Scenario 3: yêu cầu chủ nhà tạo OTP, sau đó dùng Camera OCR để xác thực tự động.
```

#### Scenario 2: Manual Override (Remote Human Intervention)
```
1. This flow is used for special cases that require visual confirmation by the Security Guard or Manager
2. Priority vehicles (ambulances, fire trucks, police) or exceptional cases enter the lane
3. The Security Guard or Manager monitors the live camera feed on the respective dashboard
4. The Guard or Manager clicks the Emergency Gate Open button on their web interface to remotely open the gate
5. The system requires the operator to select a reason (e.g., emergency_vehicle, system_maintenance)
6. The backend stores the action log together with the operator ID and sends the gate-open command
```

#### Scenario 3: Tự động hóa Luồng Khách Vãng Lai bằng Camera OCR
```markdown
1. Khách vãng lai (xe máy, ô tô) hoặc shipper đến cổng và bị chặn do biển số chưa đăng ký.
2. Khách gọi điện thông báo cho cư dân (chủ nhà).
3. Cư dân mở Mobile App → chọn "Tạo OTP" → hệ thống sinh mã 6 số ngẫu nhiên (hạn 3 phút) và hiển thị trên app.
4. Cư dân gửi mã OTP cho khách qua tin nhắn/đọc miệng.
5. **[Tự động — Không cần Bảo vệ gõ phím]** Khách chỉ cần MỞ MÀN HÌNH ĐIỆN THOẠI chứa 6 số OTP vào trước Camera tại làn cổng.
6. AI Service (Computer Vision / OCR) phân tích khung hình từ Camera, detect vùng màn hình sáng, OCR đọc chuỗi 6 chữ số và gửi lên Backend.
7. Backend kiểm tra OTP: hợp lệ + chưa dùng + chưa hết hạn → Đánh dấu OTP `is_used = true` → Ghi access_log → Ghi lệnh OPEN → Hệ thống tự động mở cổng.
```

#### Scenario 4: Pedestrian/Bicycle Access (Cư dân đi bộ sử dụng QR trên Camera)
```markdown
1. Cư dân đi tập thể dục hoặc đi xe đạp không có biển số để AI nhận diện.
2. Cư dân mở Mobile App → xin mã QR định danh cá nhân (Dynamic QR chứa UUID, hạn 3 phút).
3. Cư dân MỞ MÀN HÌNH PHONE chứa mã QR hướng vào Camera tại làn cổng.
4. AI Service decode QR từ nội dung Camera → gửi UUID lên Backend.
5. Backend đối chiếu UUID → xác nhận cư dân hợp lệ và QR còn hạn → Mở cổng tự động.
```

---

## 4. FUNCTIONAL REQUIREMENTS & USE CASES

### 4.1 Use Case Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │           Smart Toll Gate System            │
                    │                                             │
    ┌───────┐       │  ┌─────────────────────────────────┐       │
    │Citizen│───────┼──│ UC-01: Quản lý phương tiện      │       │
    │       │       │  │ UC-02: Đăng ký khách có hẹn     │       │
    │       │───────┼──│ UC-03: Tạo mã OTP khẩn cấp      │       │
    │       │       │  │ UC-04: Xem lịch sử cá nhân      │       │
    └───────┘       │  └─────────────────────────────────┘       │
                    │                                             │
    ┌───────┐       │  ┌─────────────────────────────────┐       │
    │ Guard │───────┼──│ UC-05: Giám sát real-time       │       │
    │       │───────┼──│ UC-07: Thao tác thủ công        │       │
    │       │       │  │ UC-08: Sửa lỗi nhận dạng AI     │       │
    │       │───────┼──│ (UC-02): Đăng ký khách cho CD   │       │
    └───────┘       │  └─────────────────────────────────┘       │
                    │                                             │
    ┌───────┐       │  ┌─────────────────────────────────┐       │
    │Manager│───────┼──│ UC-09: Phê duyệt phương tiện    │       │
    │       │       │  │ UC-10: Manager Dashboard KPIs  │       │
    │       │───────┼──│ UC-11: Tra cứu nhật ký hệ thống │       │
    │       │───────┼──│ (UC-02): Đăng ký khách có hẹn   │       │
    │       │       │  │ (UC-07): Thao tác mở cổng từ xa │       │
    └───────┘       │  └─────────────────────────────────┘       │
                    │                                             │
    ┌───────┐       │  ┌─────────────────────────────────┐       │
    │  AI   │───────┼──│ UC-13/14: Nhận diện biển số    │       │
    │Service│       │  │ UC-06: OCR OTP/QR từ Camera    │       │
    └───────┘       │  └─────────────────────────────────┘       │
                    │                                             │
    └─────────────────────────────────────────────────────────────┘
```

### 4.2 Use Cases Chi Tiết

---

#### UC-01: Quản lý phương tiện cá nhân

| Attribute | Description |
|-----------|-------------|
| **Actor** | Citizen |
| **Description** | Citizen xem, thêm, sửa, xóa thông tin phương tiện cá nhân |
| **Precondition** | Citizen đã đăng nhập |
| **Postcondition** | Xe được thêm hoặc sửa sẽ được lưu ở trạng thái chờ duyệt (`pending_new`, `pending_update`). Xóa xe được thực hiện **ngay lập tức** mà không cần Manager phê duyệt. |

**Main Flow:**
1. Citizen truy cập trang "My Vehicles"
2. Hệ thống hiển thị danh sách xe hiện có
3. Citizen chọn "Thêm xe mới", "Sửa xe", hoặc "Xóa xe"
4. Citizen nhập/sửa: Biển số, Loại xe, Màu sắc
5. Hệ thống validate biển số (format, trùng lặp với xe khác)
6. Hệ thống lưu/cập nhật thông tin với trạng thái tương ứng (`pending_new` hoặc `pending_update`) và `is_active = false` chờ duyệt. Xóa xe được xử lý ngay lập tức.
7. Thông báo thành công cho Citizen

**Alternative Flow:**
- 5a. Biển số không hợp lệ → Hiển thị lỗi, yêu cầu nhập lại
- 5b. Biển số đã tồn tại → Thông báo xe đã được đăng ký

---

#### UC-02: Đăng ký khách có hẹn trước

| Attribute | Description |
|-----------|-------------|
| **Actor** | Citizen, Security Guard, Manager |
| **Description** | Đăng ký thông tin khách sẽ đến thăm. Guard và Manager cũng có quyền đăng ký thay |
| **Precondition** | Actor đã đăng nhập |
| **Postcondition** | Guest registration được tạo với status "approved" |

**Main Flow:**
1. Actor truy cập trang "Guest Management"
2. Actor chọn "Đăng ký khách mới"
3. Actor nhập: Tên khách, Biển số xe, Thời gian đến, Thời gian về, Mục đích
4. Hệ thống validate khung giờ (không overlap với registration khác)
5. Hệ thống lưu guest_registration với status = "approved" (hệ thống sẽ check-in tự động bằng AI nhận diện biển số)
6. Thông báo thành công

**Alternative Flow:**
- 4a. Khung giờ bị overlap → Hiển thị cảnh báo, yêu cầu chọn giờ khác

---

#### UC-03: Tạo mã OTP khẩn cấp

| Attribute | Description |
|-----------|-------------|
| **Actor** | Citizen |
| **Description** | Citizen tạo mã OTP cho khách đột xuất (khách vãng lai gọi điện) |
| **Precondition** | Citizen đã đăng nhập |
| **Postcondition** | OTP 6 số được tạo, hiển thị cho Citizen & màn hình Guard |

**Main Flow:**
1. Khách gọi điện cho Citizen báo đến cổng
2. Citizen mở app, chọn "Tạo mã OTP" cho khách
3. Hệ thống sinh mã 6 số ngẫu nhiên (unique trong pool active)
4. Hệ thống lưu vào access_tokens với valid_until = NOW() + 3 phút
5. Mã OTP hiển thị trên app của Citizen VÀ tự động đồng bộ lên màn hình của Security Guard
6. Citizen đọc hoặc gửi mã OTP cho khách qua điện thoại/tin nhắn

---

#### UC-04: Xem lịch sử cá nhân

| Attribute | Description |
|-----------|-------------|
| **Actor** | Citizen |
| **Description** | Citizen xem lịch sử ra vào của xe mình |
| **Precondition** | Citizen đã đăng nhập |
| **Postcondition** | Hiển thị danh sách access_logs liên quan |

**Main Flow:**
1. Citizen truy cập "Lịch sử của tôi"
2. Hệ thống query access_logs WHERE vehicle.owner_id = citizen_id
3. Hiển thị danh sách: Thời gian, Cổng, Phương thức, Trạng thái

---

#### UC-05: Giám sát real-time (Guard)

| Attribute | Description |
|-----------|-------------|
| **Actor** | Security Guard |
| **Description** | Guard theo dõi hoạt động cổng theo thời gian thực |
| **Precondition** | Guard đã đăng nhập, được assign cổng cụ thể |
| **Postcondition** | Màn hình cập nhật liên tục |

**Main Flow:**
1. Guard đăng nhập → Redirect đến Dashboard cổng được assign
2. Hệ thống hiển thị: Stream camera, 20 logs gần nhất, Stats (1h, 24h)
3. Khi có xe mới → AI xử lý → Dashboard cập nhật real-time
4. Guard có thể xem chi tiết từng log (ảnh, biển số, trạng thái)

---

#### UC-06: Xác thực OTP/QR bằng Camera (Tự động — Không cần Guard gõ phím)

| Attribute | Description |
|-----------|-------------|
| **Actor** | AI Service / Khách vãng lai |
| **Description** | Khách đưa màn hình điện thoại có chứa mã OTP 6 số hoặc QR code (cho người đi bộ) vào Camera tại cổng. AI Service OCR tự động đọc nội dung và gửi lên Backend kiểm tra — Không yêu cầu Bảo vệ can thiệp. |
| **Precondition** | Citizen đã tạo mã OTP hợp lệ (UC-03) |
| **Postcondition** | OTP được đánh dấu `is_used = true`, ghi access_log, barrier mở tự động |

**Main Flow:**
1. Khách gốc lấy mã OTP từ Citizen (qua điện thoại/tin nhắn), hiển thị lên màn hình.
2. Khách hướng màn hình điện thoại (6 số OTP hoặc QR) vào Camera tại làn cổng.
3. AI Service phân tích luft ảnh từ Camera: detect vùng màn hình sáng, OCR đọc chuỗi số/giải mã QR.
4. Gửi kết quả lên API: `POST /api/v1/gates/verify-camera-otp` với `{ lane_id, token_data, code_type }`.
5. Backend kiểm tra: `extracted_code` khớp OTP còn hiệu lực → đánh dấu `is_used = true` → ghi access_log → gửi lệnh OPEN.
6. Bảo vệ thấy thông tin log xuất hiện trên Dashboard real-time (chủ hộ, tên khách, thời gian).

**Alternative Flow:**
- 3a. OCR không thể đọc được (màn hình mờ, góc lệch) → Hệ thống hiển thị hướng dẫn trên màn hình tại cổng → Khách điều chỉnh.
- 5a. OTP không tồn tại → Báo lỗi, thông báo Bảo vệ.
- 5b. OTP đã dùng hoặc hết hạn → Từ chối, báo Bảo vệ can thiệp nếu cần.

---

#### UC-07: Thao tác thủ công (Manual Override & Hỗ trợ từ xa)

| Attribute | Description |
|-----------|-------------|
| **Actor** | Security Guard, Manager |
| **Description** | Guard hoặc Manager mở/đóng cổng thủ công (tại chỗ hoặc từ xa). **Bắt buộc chọn lý do** trước khi thực thi — lý do được lưu vào cột `action_reason` trong `access_logs`. |
| **Precondition** | Actor đã đăng nhập |
| **Postcondition** | Action được ghi log đầy đủ bao gồm `action_reason`; có thể tra cứu lại trên Dashboard lịch sử ca trực |

**Main Flow:**
1. Guard/Manager chọn Action: "Mở cổng khNn cấp" hoặc "Giữ đóng".
2. Hệ thống **bắt buộc** hiển thị Dropdown chọn lý do (không thể Submit nếu chưa chọn):
   - `emergency_vehicle` — Xe cứu thương / cứu hỏa / cảnh sát
   - `shipper_delivery` — Shipper / giao hàng
   - `ai_error` — Lỗi AI nhận dạng
   - `vip_guest` — Khách VIP / ban lãnh đạo
   - `maintenance` — Bảo trì cổng
   - `other` — Khác (yêu cầu nhập thêm ghi chú vào ô `note`)
3. Guard/Manager xác nhận hành động.
4. Hệ thống ghi `access_logs` với `access_method = 'manual_guard'` **và `action_reason = <lý do đã chọn>`**.
5. Nếu "Mở cổng" → Gửi lệnh OPEN đến IoT Gateway.
6. Lý do luôn hiển thị rõ trên cột bên phải của bảng lịch sử ca trực (Guard Dashboard) giúp giải trình minh bạch.

---

#### UC-08: Sửa lỗi nhận dạng AI (Human-in-the-loop)

| Attribute | Description |
|-----------|-------------|
| **Actor** | Security Guard |
| **Description** | Guard sửa text biển số khi AI nhận dạng sai |
| **Precondition** | Có access_log gần đây với biển số AI đọc |
| **Postcondition** | Biển số đúng được lưu vào cột `note` của `access_logs` để tra cứu |

**Main Flow:**
1. AI nhận dạng biển số là "51A-12345" với confidence thấp
2. Guard nhận thấy thực tế là "51A-12346"
3. Guard click vào log → Chọn "Sửa biển số"
4. Guard nhập biển số đúng → Submit
5. Hệ thống update `access_logs.note` với biển số được sửa

---

#### UC-09: Phê duyệt phương tiện (Manager)

| Attribute | Description |
|-----------|-------------|
| **Actor** | Manager |
| **Description** | Manager duyệt/từ chối yêu cầu đăng ký xe mới hoặc cập nhật thông tin xe |
| **Precondition** | Manager đã đăng nhập, quản lý zone cụ thể |
| **Postcondition** | Yêu cầu được phê duyệt (cập nhật thông tin, kích hoạt xe) hoặc bị từ chối (khôi phục trạng thái cũ) |

**Main Flow:**
1. Manager vào trang "Pending Vehicles"
2. Hệ thống hiển thị danh sách xe chờ duyệt trong zone (Bao gồm đăng ký mới và sửa đổi thông tin)
3. Manager click vào xe cụ thể → Xem chi tiết (thông tin cũ và mới, lý do)
4. Manager chọn "Approve" hoặc "Reject"
5. Hệ thống cập nhật trạng thái tương ứng (`approved` hoặc khôi phục/xóa rác) và ghi audit log

---

#### UC-10: Manager Dashboard KPIs

| Attribute | Description |
|-----------|-------------|
| **Actor** | Manager |
| **Description** | Manager xem 4 chỉ số KPIs cốt lõi về vận hành cổng |
| **Precondition** | Manager đã đăng nhập |
| **Postcondition** | Hiển thị 4 KPIs real-time |

**Main Flow:**
1. Manager truy cập "Manager Dashboard"
2. Hệ thống hiển thị 4 KPIs (xem chi tiết tại Mục 10):
   - **Total Traffic Volume** — Tổng lưu lượng trong ngày
   - **Automation Rate** — % cổng mở tự động bằng AI
   - **Security Alerts `** — Tổng lượt xe bị từ chối (không khớp whitelist)
   - **Vehicles Inside** — Xe cư dân đang ở trong khu

#### UC-11: Tra cứu nhật ký hệ thống

| Attribute | Description |
|-----------|-------------|
| **Actor** | Manager |
| **Description** | Manager tìm kiếm, lọc access logs chi tiết |
| **Precondition** | Manager đã đăng nhập |
| **Postcondition** | Hiển thị danh sách logs với filter |

**Main Flow:**
1. Manager vào trang "Access Logs"
2. Manager có thể filter theo: Ngày, Cổng, Phương thức, Biển số, Granted/Denied
3. Hệ thống query và hiển thị kết quả pagination
4. Manager click vào log cụ thể → Xem ảnh snapshot

---

#### UC-13 & UC-14: Nhận diện & Check-in tự động (AI Service)

| Attribute | Description |
|-----------|-------------|
| **Actor** | AI Service (automated) |
| **Description** | Xử lý ảnh camera, nhận diện biển số xe bằng AI OCR, kiểm tra whitelist và anti-passback |
| **Trigger** | Camera phát hiện xe trong vùng nhận diện |

**Main Flow:**
1. Camera capture ảnh → Gửi đến AI Service.
2. YOLOv8 detect vùng biển số xe trong ảnh.
3. OCR model đọc text biển số → Lấy `plate_text`.
4. Gửi kết quả đến Backend: `plate_text`, `image_snapshot`.
5. Backend áp dụng luồng 3 Bước:
   - **Bước 1**: Tìm `plate_text` trong bảng `vehicles` (Whitelist Vĩnh viễn). Kèm check anti-passback (`is_inside`). NẾU PASS -> MỞ CỔNG. NẾU FAIL -> Qua Bước 2.
   - **Bước 2**: Tìm `plate_text` trong `guest_registrations` (Whitelist Tạm thời). NẾU PASS -> MỞ CỔNG. NẾU FAIL -> Qua Bước 3.
   - **Bước 3**: Fallback, ghi log từ chối (vehicle_id, guest_reg_id, token_id đều NULL), giữ cổng ĐÓNG`
6. Ghi log `is_inside` (nếu mở cổng) & `access_log`.
7. Gửi action đến IoT Gateway.

---

### 4.3 Functional Requirements Matrix

| FR ID | Requirement | Use Case | Priority |
|-------|-------------|----------|----------|
| **FR_SYS_01** | Xử lý nhận diện AI (YOLOv8 + OCR) | UC-13 | Must |
| **FR_SYS_02** | Kiểm tra quyền truy cập (whitelist check) | UC-14 | Must |
| **FR_SYS_03** | Giao tiếp IoT Gateway (HTTP/MQTT) | UC-14 | Must |
| **FR_SYS_04** | Lưu trữ hình ảnh nhị phân (BYTEA) | All | Must |
| **FR_CIT_01** | Quản lý phương tiện cá nhân | UC-01 | Must |
| **FR_CIT_02** | Đăng ký khách có hẹn | UC-02 | Must |
| **FR_CIT_03** | Tạo mã OTP khẩn cấp | UC-03 | Must |
| **FR_CIT_04** | Mã QR định danh | - | Should |
| **FR_CIT_05** | Tra cứu lịch sử cá nhân | UC-04 | Should |
| **FR_SEC_01** | Giám sát real-time | UC-05 | Must |
| **FR_SEC_02** | Xác thực OTP/QR tự động bằng Camera OCR | UC-06 | Must |
| **FR_SEC_03** | Thao tác thủ công (bắt buộc `action_reason`) | UC-07 | Must |
| **FR_SEC_04** | Human-in-the-loop (sửa biển số AI) | UC-08 | Should |
| **FR_MAN_01** | Phê duyệt phương tiện | UC-09 | Must |
| **FR_MAN_02** | Manager Dashboard KPIs (4 chỉ số) | UC-10 | Must |
| **FR_MAN_03** | Tra cứu nhật ký hệ thống | UC-11 | Must |

---

## 5. NON-FUNCTIONAL REQUIREMENTS

### 5.1 Performance Requirements

| NFR ID | Requirement | Metric | Target |
|--------|-------------|--------|--------|
| **NFR_PERF_01** | Thời gian nhận diện AI | End-to-end latency | < 1000ms |
| **NFR_PERF_02** | API response time | 95th percentile | < 200ms |
| **NFR_PERF_03** | Database query time | Complex queries | < 100ms |
| **NFR_PERF_04** | Concurrent users | Simultaneous requests | > 100 |
| **NFR_PERF_05** | Image upload size | Max payload | 10MB |

### 5.2 Security Requirements

| NFR ID | Requirement | Implementation |
|--------|-------------|----------------|
| **NFR_SEC_01** | Authentication | JWT với expiry 7 ngày |
| **NFR_SEC_02** | Password storage | bcrypt với salt rounds = 10 |
| **NFR_SEC_03** | Authorization | Role-based access control (RBAC) |
| **NFR_SEC_04** | API protection | Rate limiting, CORS policy |
| **NFR_SEC_05** | Data encryption | HTTPS/TLS in transit |
| **NFR_SEC_06** | OTP security | 6 digits, unique, 3-min expiry |

### 5.3 Reliability Requirements

| NFR ID | Requirement | Target |
|--------|-------------|--------|
| **NFR_REL_01** | System availability | 99.5% uptime |
| **NFR_REL_02** | Data durability | PostgreSQL with WAL |
| **NFR_REL_03** | Graceful shutdown | Session handling |
| **NFR_REL_04** | Database connection pool | Max 20 connections |
| **NFR_REL_05** | Transaction rollback | On any operation failure |

### 5.4 Scalability Requirements

| NFR ID | Requirement | Approach |
|--------|-------------|----------|
| **NFR_SCA_01** | Horizontal scaling | Stateless API design |
| **NFR_SCA_02** | Database scaling | Connection pooling |
| **NFR_SCA_03** | Image storage | BYTEA (có thể migrate S3) |
| **NFR_SCA_04** | Microservices | Độc lập AI Service |

### 5.5 Maintainability Requirements

| NFR ID | Requirement | Implementation |
|--------|-------------|----------------|
| **NFR_MNT_01** | Code structure | MVC pattern |
| **NFR_MNT_02** | API documentation | RESTful conventions |
| **NFR_MNT_03** | Logging | Structured logs với timestamps |
| **NFR_MNT_04** | Configuration | Environment variables (.env) |
| **NFR_MNT_05** | Testing | PowerShell test suite |

---

## 6. DATA REQUIREMENTS

### 6.1 Entity-Relationship Diagram

### 6.2 Database Schema (13 Bảng — Đã tinh gọn)

#### 6.2.1 Enum Types

```sql
CREATE TYPE user_role_enum AS ENUM ('citizen', 'guard', 'manager');

-- Nhúng trực tiếp vào vehicles và guest_registrations (Xóa bảng vehicle_types riêng biệt)
CREATE TYPE vehicle_type_enum AS ENUM ('car', 'motorbike', 'bicycle', 'truck', 'emergency');

-- Dùng cho bảng lanes
CREATE TYPE direction_enum AS ENUM ('inbound', 'outbound');

CREATE TYPE access_method_enum AS ENUM ('ai_plate_recognition', 'ai_camera_otp', 'ai_camera_qr', 'manual_guard');
```

#### 6.2.2 Core Tables

| Table | Description | Columns |
|-------|-------------|---------|
| **zones** | Khu vực quản lý | zone_id, zone_name, description, created_at |
| **gates** | Cổng vật lý | gate_id, zone_id, gate_name, is_active |
| **lanes** | Làn xe *(Mới)* | lane_id (PK, VARCHAR, VD: 'MAIN-IN'), gate_id (FK → gates), lane_name, direction_enum |
| **iot_devices** | Thiết bị IoT | device_id, **lane_id (FK)**, device_name, device_type, ip_address, status |
| **users** | Người dùng (base) | user_id, username, password_hash, full_name, email, role, avatar_url |

#### 6.2.3 Role-Specific Tables (ISA relationship)

| Table | Description | Columns |
|-------|-------------|---------|
| **citizens** | Cư dân | user_id (PK/FK), zone_id(FK -> zones), address, phone_number, identity_card_number, is_house_owner |
| **security_guards** | Bảo vệ | user_id (PK/FK), **assigned_gate_id (INT FK → gates)** *(guard quản lý tất cả lane trong gate)*, employee_code, shift_start, shift_end |
| **managers** | Quản lý | user_id (PK/FK), managed_zone_id, department_name |

#### 6.2.4 Business Tables

| Table | Description | Columns |
|-------|-------------|---------|
| **vehicles** | Xe cư dân (whitelist) | vehicle_id, owner_user_id (FK), **vehicle_type_enum** *(nhúng trực tiếp)*, license_plate, vehicle_color, **is_inside** (cờ anti-passback), is_active, status, pending_changes, last_log_time |
| **guest_registrations** | Khách có hẹn | registration_id, host_id, guest_name, **vehicle_type_enum**, guest_license_plate, visit_start/end_time, status |
| **access_tokens** | Mã OTP | token_id, issued_by, token_data, valid_from, valid_until, is_used, used_at |

#### 6.2.5 Logging & Audit Tables

| Table | Description | Columns |
|-------|-------------|---------|
| **access_logs** | Log ra vào | log_id, **lane_id (FK -> lanes)**, vehicle_id, guest_reg_id, |
|**token_id (FK → access_tokens)**, guard_id, check_in_time, **detected_text** *(chuỗi OCR thô: biển số / OTP 6 số / UUID QR)*, image_snapshot_data (BYTEA), access_method, **action_reason** *(lý do mở thủ công — UC-07)* |
| **system_audit_logs** | Audit trail | **device_id(FK -> iot_devices)**, audit_id, actor_id, action_type, target_table, target_id, action_details, performed_at |

> **Đã xóa hoàn toàn:** `vehicle_types`, `ai_predictions`, `ai_models`, `houses`
> **FK cập nhật:** `access_logs.lane_id`, `iot_devices.lane_id`, `security_guards.assigned_gate_id` — trỏ vào bảng gates (INT FK).
> **Thay đổi chính:** `citizens dùng zone_id trực tiếp; access_logs.lane_id FK → lanes;`
security_guards.assigned_gate_id là INT FK → gates; system_audit_logs thêm device_id.`
### 6.3 Data Storage Strategy

#### Binary Image Storage (BYTEA)

Hệ thống lưu trữ hình ảnh trực tiếp trong PostgreSQL dưới dạng binary:

```javascript
// Chuyển đổi Base64 → Buffer → BYTEA
const base64ToBuffer = (base64String) => {
    const base64Data = base64String.includes(',') 
        ? base64String.split(',')[1] 
        : base64String;
    return Buffer.from(base64Data, 'base64');
};

// Đọc từ DB: ENCODE(image_data, 'base64') AS image_base64
```

**Lý do chọn BYTEA thay vì File System/S3:**
- Đảm bảo tính toàn vẹn dữ liệu (ACID)  
- Backup cùng database
- Không cần quản lý file paths
- Phù hợp với quy mô dự án

### 6.4 Data Integrity Constraints

```sql
-- Foreign Key với CASCADE
vehicles.owner_user_id REFERENCES users(user_id) ON DELETE CASCADE
access_logs.lane_id REFERENCES lanes(lane_id)
access_logs.token_id REFERENCES access_tokens(token_id) ON DELETE SET NULL

-- Foreign Key với SET NULL
citizens.zone_id REFERENCES zones(zone_id) ON DELETE SET NULL
managers.managed_zone_id REFERENCES zones(zone_id) ON DELETE SET NULL

-- UNIQUE constraints
users.username UNIQUE
vehicles.license_plate UNIQUE
citizens.phone_number UNIQUE
security_guards.employee_code UNIQUE
```

---

## 7. IOT GATEWAY IMPLEMENTATION (PYTHON)

### 7.1 Tổng quan IoT Gateway

IoT Gateway đóng vai trò cầu nối giữa Backend (Node.js) và Hardware (Arduino). Giao tiếp được thực hiện qua **WebSocket (Socket.IO)** thay vì HTTP riêng biệt:

```
┌──────────────┐     WebSocket (Socket.IO)    ┌──────────────┐        Serial         ┌──────────────┐
│   Backend    │ ◄──────────────────────────► │ IoT Gateway  │ ──────────────────►  │   Arduino    │
│  (Node.js)   │  manual_command / scan_result│ (Python)     │  ENTRY_GO / EXIT_GO      │  (C/C++)     │
│  Port 5000   │                              │TestingMode.py│  MANUAL_OPEN_IN/OUT       │Hardware.ino  │
│              │                              │              │  FORCE_CLOSE / ALARM_FAKE │              │
└──────────────┘                              └──────────────┘ ◄──────────────────  └──────────────┘
        ▲                                                         CAR_ARRIVED / CAR_EXITING
        │ HTTP POST
        │ /api/v1/gates/check-in
        │ {lane_id, plate_text, image_base64}
        │
┌──────────────┐
│  IoT Gateway │ (gửi kết quả AI lên Backend)
│ (Python)     │
└──────────────┘
```

**Luồng giao tiếp chi tiết:**
1. **Camera → Python**: Python đọc frame từ camera (OpenCV), gọi AI nhận diện biển số
2. **Python → Backend (HTTP)**: `POST /api/v1/gates/check-in` với `{lane_id, plate_text, image_base64}`
3. **Backend → Python (WebSocket)**: Phản hồi qua `scan_result` event; lệnh thủ công qua `manual_command` event
4. **Python → Arduino (Serial)**: `ENTRY_GO` / `EXIT_GO` (mở cổng vào/ra), `MANUAL_OPEN_IN/OUT` (mở thủ công), `FORCE_CLOSE` (đóng khẩn cấp), `ALARM_FAKE` (anti-passback)
5. **Arduino → Python (Serial)**: `CAR_ARRIVED` (Lane A - cổng vào) / `CAR_EXITING` (Lane B - cổng ra) khi sensor phát hiện xe

### 7.2 Cấu trúc file IoT Gateway

```
DemoTest_SmartToll/
├── Hardware.ino              # Arduino sketch (C++)
├── TestingMode.py            # IoT Gateway chính (Python) — đang sử dụng
└── SystemCommunication.py    # Phiên bản cũ — lưu lại tham khảo
```

**`TestingMode.py`** đảm nhiệm toàn bộ:
- Kết nối WebSocket đến Backend (Socket.IO client)
- Lắng nghe lệnh `manual_command` từ Backend (guard/manager mở/đóng cổng) — phân biệt lane IN/OUT
- Đọc **2 camera** (Làn VÀO + Làn RA) đa luồng (`CameraStream`)
- Quét **QR Code** bằng OpenCV `QRCodeDetector` trước, sau đó mới gọi AI nhận diện biển số
- Tự động phân loại kết quả AI: biển số xe vs OTP 6 số (với OCR corrections)
- Gửi HTTP POST check-in hoặc verify-camera-otp lên Backend
- Giao tiếp Serial với Arduino (hỗ trợ cổng vào + cổng ra)
- Stream video lên Frontend (React) qua `video_stream` WebSocket event (kèm `lane_id`)
- Chống quét trùng lặp bằng cơ chế **Cooldown** (5 giây) và **Auto-scan** định kỳ (1 giây)

### 7.3 Implementation Code

#### `DemoTest_SmartToll/TestingMode.py` - IoT Gateway chính

File này thực hiện toàn bộ luồng: đọc 2 camera → quét QR → gọi AI → phân loại (biển số / OTP) → gửi HTTP lên Backend → nhận kết quả → gửi Serial tới Arduino.

```python
import cv2, time, base64, requests, os, sys, serial, unicodedata, socketio, threading, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_service.ai_engine import extract_license_plates

NODEJS_BACKEND_URL = "http://localhost:5000/api/v1/gates/check-in"
OTP_BACKEND_URL = "http://localhost:5000/api/v1/gates/verify-camera-otp"
WEBSOCKET_URL = "http://localhost:5000"
ARDUINO_PORT = 'COM3'
BAUD_RATE = 9600
SCAN_INTERVAL = 1.0       # Auto-scan mỗi 1 giây
PLATE_COOLDOWN = 5.0      # Không quét lại biển cũ trong 5 giây

# Camera đa luồng (1 instance / lane)
class CameraStream:
    def __init__(self, src=0, name="Camera"):
        self.cap = cv2.VideoCapture(src)
        self.frame = None; self.running = True; self.name = name
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret: self.frame = frame
            time.sleep(0.01)
    def read(self): return self.frame
    def stop(self): self.running = False; self.thread.join(); self.cap.release()

# Kết nối Arduino + WebSocket
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1); time.sleep(2)
except: ser = None

sio = socketio.Client()

@sio.on('manual_command')
def on_manual_command(data):
    action = data.get('action')
    lane_id = data.get('lane_id', 'MAIN-IN')
    operator = data.get('operator_name', 'Guard')
    clean_op = format_lcd_text(operator)
    if action == 'OPEN' and ser:
        if "OUT" in lane_id:
            ser.write(f"MANUAL_OPEN_OUT:{clean_op}\n".encode())
        else:
            ser.write(f"MANUAL_OPEN_IN:{clean_op}\n".encode())
    elif action == 'CLOSE' and ser:
        ser.write(b"FORCE_CLOSE\n")

def format_lcd_text(text):
    """Xóa dấu, loại ký tự lạ, ép max 16 ký tự cho LCD"""
    if not text: return "Guest"
    text = text.replace('Đ', 'D').replace('đ', 'd')
    text = "".join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-zA-Z0-9\s.,-]', '', text).strip()
    return text[:16]

def send_to_backend(plate_text, proc_ms, image_path, lane_id):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {"lane_id": lane_id, "plate_text": plate_text,
               "image_base64": f"data:image/jpeg;base64,{img_b64}"}
    return requests.post(NODEJS_BACKEND_URL, json=payload).json(), f"data:image/jpeg;base64,{img_b64}"

def send_otp_to_backend(otp_code, proc_ms, image_path, lane_id):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {"lane_id": lane_id, "token_data": otp_code,
               "code_type": "otp_6digit", "image_base64": f"data:image/jpeg;base64,{img_b64}"}
    return requests.post(OTP_BACKEND_URL, json=payload).json(), f"data:image/jpeg;base64,{img_b64}"

def process_and_authorize(frame, lane_id, current_plate=""):
    temp_path = f"temp_{lane_id}_{int(time.time())}.jpg"
    cv2.imwrite(temp_path, frame)

    # 1. QUÉT QR CODE TRƯỚC
    qr_data, _, _ = cv2.QRCodeDetector().detectAndDecode(frame)
    if qr_data and len(qr_data) > 10:
        # Gửi QR UUID lên verify-camera-otp
        # → OPEN: ser.write(f"ENTRY_GO:..." / "EXIT_GO:...")
        # → ALARM: ser.write("ALARM_FAKE:QR_USED")
        ...

    # 2. AI NHẬN DIỆN BIỂN SỐ / OTP
    ai_result = extract_license_plates(temp_path, debug=False)
    if ai_result.get("status") == "success" and ai_result.get("plates"):
        raw_text = ai_result["plates"][0]
        clean = re.sub(r'[^a-zA-Z0-9]', '', raw_text).upper()

        # Phân loại: OTP 6 số vs Biển số xe
        is_otp = False
        if len(clean) == 6:
            ocr_fix = {'T':'7','O':'0','Q':'0','D':'0',
                       'I':'1','L':'1','Z':'2','B':'8','S':'5','A':'4','G':'6'}
            corrected = "".join([ocr_fix.get(c, c) for c in clean])
            if corrected.isdigit():
                is_otp = True; clean = corrected

        if is_otp:
            # Gửi OTP lên verify-camera-otp → mở/đóng cổng tương ứng lane_id
            backend_resp, img = send_otp_to_backend(clean, 0, temp_path, lane_id)
        else:
            # Gửi biển số lên check-in → phân loại resident/guest/anti-passback
            backend_resp, img = send_to_backend(raw_text, 0, temp_path, lane_id)
        # ... xử lý response, emit scan_result, gửi Serial command
    os.remove(temp_path)
    return found_plate

def main():
    sio.connect(WEBSOCKET_URL)
    cam_in  = CameraStream(src=0, name="Làn Vào")
    cam_out = CameraStream(src=1, name="Làn Ra")
    threading.Thread(target=stream_video_to_web, args=(cam_in, cam_out), daemon=True).start()

    while True:
        # Đọc lệnh Arduino (CAR_ARRIVED / CAR_EXITING) → process_and_authorize()
        # Auto-scan định kỳ mỗi SCAN_INTERVAL giây với PLATE_COOLDOWN chống trùng
        ...

if __name__ == "__main__":
    main()
```

**Các Serial command gửi tới Arduino:**

| Command | Ý nghĩa |
|---------|---------|
| `ENTRY_GO:TenChuXe\n` | Mở cổng vào (Lane A), hiện màn hình chào đón |
| `EXIT_GO:TenChuXe\n` | Mở cổng ra (Lane B), hiện "SAFE TRAVELS" |
| `MANUAL_OPEN_IN:Operator\n` | Mở thủ công cổng vào (không tự động đóng) |
| `MANUAL_OPEN_OUT:Operator\n` | Mở thủ công cổng ra (không tự động đóng) |
| `FORCE_CLOSE\n` | Đóng khẩn cấp tất cả cổng |
| `DENY_PLATE:BienSo\n` | Hiện màn hình "UNAUTHORIZED" |
| `ALARM_FAKE:Info\n` | Cảnh báo anti-passback (buzzer 5 lần, đóng cổng) |

**Các Serial signal nhận từ Arduino:**

| Signal | Ý nghĩa |
|--------|---------|
| `CAR_ARRIVED` | Sensor Lane A (cổng vào) phát hiện xe, trigger AI scan |
| `CAR_EXITING` | Sensor Lane B (cổng ra) phát hiện xe, trigger AI scan |

### 7.4 Arduino Sketch (`DemoTest_SmartToll/Hardware.ino`)

```cpp
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo sVao, sRa;

const int GATE_DOWN = 160;  // Goc dong
const int GATE_UP   = 60;   // Goc mo
const int buzzer = 10;
const int threshold = 10;

// --- LANE A (ENTRY) ---
const int trigInA = 2;  const int echoInA = 3;
const int trigOutA = 4; const int echoOutA = 5;

// --- LANE B (EXIT) ---
const int trigInB = 12; const int echoInB = A0;
const int trigOutB = 8;  const int echoOutB = 11;

bool vaoOpen = false;
bool raOpen = false;
bool manualOverride = false;  // Khóa tự động đóng khi mở thủ công

void setup() {
  Serial.begin(9600);
  lcd.init(); lcd.backlight();
  sVao.attach(6);
  sRa.attach(7);
  pinMode(buzzer, OUTPUT);
  sVao.write(GATE_DOWN);
  sRa.write(GATE_DOWN);
  updateLCD("BKEzPass Group 8", "System Ready");
  delay(2000); resetLCD();
}

void loop() {
  // --- 1. NHẬN LỆNH TỪ PYTHON ---
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() == 0) return;

    int separator = input.indexOf(':');
    String cmd = (separator != -1) ? input.substring(0, separator) : input;
    String payload = (separator != -1) ? input.substring(separator + 1) : "";
    if (payload.length() > 16) payload = payload.substring(0, 16);

    if (cmd == "ENTRY_GO") {            // Mở cổng vào (AI)
      sVao.write(GATE_UP); vaoOpen = true; manualOverride = false;
      updateLCD("WELCOME,", payload); beep(1);
    }
    else if (cmd == "EXIT_GO") {        // Mở cổng ra (AI)
      sRa.write(GATE_UP); raOpen = true; manualOverride = false;
      updateLCD("SAFE TRAVELS,", payload); beep(1);
    }
    else if (cmd == "MANUAL_OPEN_IN") { // Mở thủ công cổng vào
      sVao.write(GATE_UP); vaoOpen = true; manualOverride = true;
      updateLCD("EMERGENCY OPENED", "INBOUND GATE");
    }
    else if (cmd == "MANUAL_OPEN_OUT") { // Mở thủ công cổng ra
      sRa.write(GATE_UP); raOpen = true; manualOverride = true;
      updateLCD("EMERGENCY OPENED", "OUTBOUND GATE");
    }
    else if (cmd == "FORCE_CLOSE") {    // Đóng khẩn cấp
      sVao.write(GATE_DOWN); sRa.write(GATE_DOWN);
      vaoOpen = false; raOpen = false; manualOverride = false;
      updateLCD("EMERGENCY CLOSED", "By Operator");
      delay(2000); resetLCD();
    }
    else if (cmd == "DENY_PLATE") {     // Từ chối
      updateLCD("UNAUTHORIZED", payload); beep(1);
    }
    else if (cmd == "ALARM_FAKE") {     // Anti-passback
      sVao.write(GATE_DOWN); sRa.write(GATE_DOWN);
      vaoOpen = false; raOpen = false;
      updateLCD("ANTI-PASSBACK!", "WARNING: " + payload);
      for(int i=0;i<5;i++){digitalWrite(buzzer,HIGH);delay(250);digitalWrite(buzzer,LOW);delay(250);}
      resetLCD();
    }
  }

  // --- 2. XỬ LÝ CẢM BIẾN SIÊU ÂM ---
  int dInA = getDist(trigInA, echoInA);
  int dOutA = getDist(trigOutA, echoOutA);
  int dInB = getDist(trigInB, echoInB);
  int dOutB = getDist(trigOutB, echoOutB);

  // Phát hiện xe cổng vào (Lane A)
  if (dInA < threshold && !vaoOpen) {
    updateLCD("ENTRY DETECTED", "Scanning Plate...");
    Serial.println("CAR_ARRIVED");
    delay(1500);
  }
  // Phát hiện xe cổng ra (Lane B)
  if (dInB < threshold && !raOpen) {
    updateLCD("EXIT DETECTED", "Checking Plate...");
    Serial.println("CAR_EXITING");
    delay(1500);
  }

  // Tự động đóng cổng (chỉ khi không bị manual override)
  if (!manualOverride) {
    if (dOutA < threshold && vaoOpen) {
      delay(1000); sVao.write(GATE_DOWN); vaoOpen = false;
      updateLCD("GATE CLOSING", "Thank You"); beep(2); delay(2000); resetLCD();
    }
    if (dOutB < threshold && raOpen) {
      delay(1000); sRa.write(GATE_DOWN); raOpen = false;
      updateLCD("GATE CLOSING", "See You Again"); beep(2); delay(2000); resetLCD();
    }
  }
}

int getDist(int t, int e) {
  pinMode(t, OUTPUT); pinMode(e, INPUT);
  digitalWrite(t, LOW); delayMicroseconds(2);
  digitalWrite(t, HIGH); delayMicroseconds(10);
  digitalWrite(t, LOW);
  long duration = pulseIn(e, HIGH, 25000);
  if (duration == 0) return 999;
  return duration * 0.034 / 2;
}
void beep(int t) {
  for(int i=0;i<t;i++){digitalWrite(buzzer,HIGH);delay(100);digitalWrite(buzzer,LOW);delay(100);}
}
void updateLCD(String line1, String line2) {
  lcd.clear(); lcd.print(line1); lcd.setCursor(0,1); lcd.print(line2);
}
void resetLCD() { lcd.clear(); lcd.print("Ready to Respond"); }
```

**Hardware pin mapping:**

| Component | Arduino Pin |
|-----------|-------------|
| Servo Vào (Entry gate) | D6 |
| Servo Ra (Exit gate) | D7 |
| HC-SR04 Entry In Trig/Echo (Lane A) | D2 / D3 |
| HC-SR04 Entry Out Trig/Echo (Lane A) | D4 / D5 |
| HC-SR04 Exit In Trig/Echo (Lane B) | D12 / A0 |
| HC-SR04 Exit Out Trig/Echo (Lane B) | D8 / D11 |
| Buzzer | D10 |
| LCD I2C SDA/SCL | A4 / A5 |



---

## 8. PHỤ LỤC

### 8.1 API Endpoints Summary

| Module | Method | Endpoint | Description |
|--------|--------|----------|-------------|
| **Auth** | POST | /api/v1/auth/login | Đăng nhập |
| | GET | /api/v1/auth/me | Thông tin user hiện tại |
| **Citizens** | POST | /api/v1/citizens/tokens | Tạo OTP |
| | GET | /api/v1/citizens/tokens | Danh sách OTP |
| | GET | /api/v1/citizens/vehicles | Danh sách xe |
| | POST | /api/v1/citizens/vehicles | Đăng ký xe mới |
| | GET | /api/v1/citizens/guests | Danh sách khách |
| | POST | /api/v1/citizens/guests | Đăng ký khách |
| **Guards** | POST | /api/v1/guards/verify-otp | Xác thực OTP |
| | POST | /api/v1/guards/manual-action | Thao tác thủ công |
| | GET | /api/v1/guards/logs | Access logs gần nhất |
| | GET | /api/v1/guards/stats | Thống kê cổng |
| **Gates** | POST | /api/v1/gates/check-in | AI check-in |
| | POST | /api/v1/gates/verify-camera-otp | AI Service gửi kết quả OCR OTP/QR đọc từ màn hình điện thoại |
| **Managers** | GET | /api/v1/managers/analytics/* | Thống kê |
| | GET | /api/v1/managers/logs | Tra cứu logs |
| | GET | /api/v1/managers/vehicles/pending | Xe chờ duyệt |
| | POST | /api/v1/managers/vehicles/:id/approve | Phê duyệt xe |
| | GET | /api/v1/managers/analytics/overview | Dashboard KPIs (Traffic, Automation Rate, Alerts, Visitors) |

### 8.2 Test Coverage

Hệ thống có **37 automated tests** covering:
- Authentication (5 tests)
- Citizens Module (5 tests)
- Guards Module (6 tests)
- Gates Module (2 tests)
- Managers Module (13 tests)
- Authorization (3 tests)

### 8.3 Environment Variables

```env
# Database
DB_HOST=localhost
DB_PORT=5433
DB_USER=admin
DB_PASS=12345678
DB_NAME=iot_main_db_252

# Server
PORT=5000
NODE_ENV=development
CORS_ORIGIN=*

# JWT
JWT_SECRET=your-secret-key
JWT_EXPIRES_IN=7d

# IoT Gateway
IOT_GATEWAY_URL=http://localhost:8080
```

---

## 9. MANAGER DASHBOARD KPIs

Dashboard dành cho Manager được thiết kế **tối giản, loại bỏ các biểu đồ rườm rà**, tập trung hiển thị trực tiếp **4 chỉ số cốt lõi (KPIs)** phản ánh tình trạng vận hành khu dân cư theo thời gian thực.

| # | KPI | Mô tả | Nguồn dữ liệu |
|---|-----|--------|---------------|
| **1** | **Total Traffic Volume** | Tổng số lượt ra vào trong ngày (tính từ 0:00 đến hiện tại) | `COUNT(*) FROM access_logs WHERE DATE = TODAY` |
| **2** | **Automation Rate** | Tỉ lệ % cổng tự động mở thành công (do AI nhận diện biển số hoặc Camera OCR đọc OTP/QR) trên tổng số lượt | `COUNT(*) WHERE access_method IN ('ai_plate_recognition','ai_camera_otp','ai_camera_qr')` |
| **3** | **Security Alerts** | Tổng số lượt truy cập bị từ chối (xe không rõ nguồn gốc, OTP sai, anti-passback) | `COUNT(*) WHERE vehicle_id IS NULL AND guest_reg_id IS NULL AND token_id IS NULL` |
| **4** | **Vehicles Inside** | Số lượng xe cư dân đang ở trong khu (real-time, không phụ thuộc period) | `COUNT(*) FROM vehicles WHERE is_inside = true AND is_active = true` AND zone matching |

> **Ghi chú thiết kế:** Các KPIs chiết xuất trực tiếp từ bảng `access_logs` mà không cần bảng AI phụ trợ. Cột `action_reason` giúp bảo vệ giải thích minh bạch từng lượt mở cổng thủ công trên bảng lịch sử ca trực.

---

## 10. TRẠNG THÁI IMPLEMENT (Cập nhật 16/05/2026)

> Kết quả rà soát toàn bộ source code backend so với spec. Ngày kiểm tra: **16/05/2026**.

### 10.1 Đăng ký khách

#### Khách hẹn trước (`guest_registrations`)

| Endpoint | Mô tả | Trạng thái |
|----------|-------|-----------|
| `POST /api/v1/citizens/guests` | Citizen đăng ký khách hẹn trước | ✅ Đã implement |
| `GET /api/v1/citizens/guests` | Citizen xem danh sách khách | ✅ Đã implement |
| `DELETE /api/v1/citizens/guests/:id` | Citizen hủy đăng ký khách | ✅ Đã implement |
| `POST /api/v1/guards/guests` | Guard đăng ký khách thay cư dân | ✅ Đã implement |
| `POST /api/v1/managers/guests` | Manager đăng ký khách thay cư dân | ✅ Đã implement |

#### Khách vãng lai đột xuất (OTP — không hẹn trước)

| Endpoint | Mô tả | Trạng thái |
|----------|-------|-----------|
| `POST /api/v1/citizens/tokens` | Citizen tạo OTP 6 số (hạn 3 phút) | ✅ Đã implement |
| `GET /api/v1/citizens/tokens` | Citizen xem danh sách OTP đã tạo | ✅ Đã implement |
| `POST /api/v1/guards/verify-otp` | Guard nhập tay OTP (fallback khi Camera OCR thất bại) | ✅ Đã implement |
| `POST /api/v1/gates/verify-camera-otp` | AI Camera tự động đọc OTP (flow tự động — UC-06) | ✅ Đã implement |

---

### 10.2 Đăng ký xe mới + Cập nhật thông tin xe

| Endpoint | Mô tả | Trạng thái |
|----------|-------|-----------|
| `POST /api/v1/citizens/vehicles` | Đăng ký xe mới (`is_active = false`, chờ Manager duyệt) | ✅ Đã implement |
| `GET /api/v1/citizens/vehicles` | Xem danh sách xe của cư dân | ✅ Đã implement |
| `PATCH /api/v1/citizens/vehicles/:vehicleId` | Bật/tắt xe (`is_active`) | ✅ Đã implement |
| `PUT /api/v1/citizens/vehicles/:vehicleId` | Cập nhật thông tin xe (loại, màu, biển số) — chuyển sang `pending_update` chờ Manager duyệt | ✅ Đã implement |
| `DELETE /api/v1/citizens/vehicles/:vehicleId` | Xóa xe ngay lập tức — không cần Manager phê duyệt | ✅ Đã implement |

---

### 10.3 Manager Dashboard + Quản lý người dùng

#### Dashboard & Analytics

| Endpoint | Mô tả | Trạng thái |
|----------|-------|-----------|
| `GET /api/v1/managers/analytics/overview` | Thống kê tổng quan (today, week, active vehicles, pending) | ✅ Đã implement |
| `GET /api/v1/managers/analytics/traffic-by-day` | Lưu lượng theo ngày (7–90 ngày) | ✅ Đã implement |
| `GET /api/v1/managers/analytics/traffic-by-hour` | Lưu lượng theo giờ (24h gần nhất) | ✅ Đã implement |
| `GET /api/v1/managers/analytics/vehicle-types` | Phân bố loại xe (30 ngày) | ✅ Đã implement |
| `GET /api/v1/managers/analytics/access-methods` | Phân bố phương thức qua cổng (30 ngày) | ✅ Đã implement |

#### Quản lý phương tiện

| Endpoint | Mô tả | Trạng thái |
|----------|-------|-----------|
| `GET /api/v1/managers/vehicles/pending` | Danh sách xe chờ duyệt trong zone | ✅ Đã implement |
| `POST /api/v1/managers/vehicles/:id/approve` | Phê duyệt xe (`is_active = true`) | ✅ Đã implement |
| `POST /api/v1/managers/vehicles/:id/reject` | Từ chối xe (xóa khỏi hệ thống) | ✅ Đã implement |

#### Tra cứu & Audit

| Endpoint | Mô tả | Trạng thái |
|----------|-------|-----------|
| `GET /api/v1/managers/logs` | Tìm kiếm/lọc access logs | ✅ Đã implement |
| `GET /api/v1/managers/logs/:id` | Chi tiết log kèm ảnh Base64 | ✅ Đã implement |
| `GET /api/v1/managers/audit-logs` | Lịch sử thao tác hệ thống | ✅ Đã implement |
| `GET /api/v1/managers/gates` | Danh sách cổng + lanes trong zone | ✅ Đã implement |
| `GET /api/v1/managers/ai/performance` | Hiệu năng AI (30 ngày) | ✅ Đã implement |

#### Quản lý người dùng

| Endpoint | Mô tả | Trạng thái |
|----------|-------|-----------|
| `GET /api/v1/managers/users` | Danh sách users trong zone (`?role=citizen\|guard\|manager`) | ✅ Đã implement |
| `POST /api/v1/managers/users` | Tạo tài khoản mới (citizen / guard / manager) trong zone | ✅ Đã implement |
| `PATCH /api/v1/managers/users/:id` | Cập nhật thông tin user (base + role-specific) | ✅ Đã implement |
| `DELETE /api/v1/managers/users/:id` | Xóa user khỏi hệ thống (hard delete, cascade) | ✅ Đã implement |

---

### 10.4 Check QR cho người đi bộ / đi xe đạp (UC-15)

**Spec hiện tại:** Cư dân đi bộ/xe đạp tạo mã QR động chứa UUID (hạn **3 phút**) ngay lúc đứng tại cổng → đưa màn hình điện thoại vào Camera → AI decode QR → Backend kiểm tra UUID còn hiệu lực → mở cổng.

**Về thời điểm tạo QR:** Cư dân tạo QR **bất cứ khi nào cần qua cổng** (vào hoặc ra), không phân biệt chiều. Do QR chỉ hạn 3 phút nên phải tạo ngay lúc đang đứng tại cổng.

| Endpoint / Tính năng | Trạng thái |
|----------------------|-----------|
| `POST /api/v1/citizens/qr-code` — Tạo UUID token hạn 3 phút | ✅ Đã implement |
| `POST /api/v1/gates/verify-camera-otp` — Xử lý cả `otp_6digit` lẫn `qr_uuid` trong cùng endpoint | ✅ Đã implement |
| AI gửi `{ token_data: "uuid...", code_type: "qr_uuid" }` → BE tìm trong `access_tokens` → kiểm tra hết hạn | ✅ Đã implement |
| Anti-passback cho người đi bộ | ✅ Không áp dụng (đúng theo thiết kế — người đi bộ không có `vehicle_id`) |

---

**Kết thúc báo cáo.**
