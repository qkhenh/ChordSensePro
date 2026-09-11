# ChordSense Pro: Project Presentation Document / Tài Liệu Trình Bày Dự Án

> **English Title:** ChordSense Pro: An AI-Powered Context-Aware System for Musician's Interactive Practice Workspace with Harmonic Analysis Support  
> **Vietnamese Title:** ChordSense Pro: Hệ thống phân tích nhạc lý ngữ cảnh và Không gian luyện tập tương tác cho nhạc công hỗ trợ bởi AI  
> **Tagline / Khẩu hiệu:** *"Deep Analysis — Precision Practice — Complete Mastery" / "Phân tích sâu — Luyện tập chuẩn — Thành thạo toàn diện."*

---

# PHẦN 1: BẢN TRÌNH BÀY DỰ ÁN (TIẾNG VIỆT)

## 1. Bối Cảnh (Context)
* **Xu hướng số hóa học nhạc:** Hàng triệu người chơi nhạc (guitarist, pianist, bassist, producer) đang chuyển dần từ giáo trình giấy sang tự học và luyện tập cùng các nguồn nhạc số (YouTube, MP3/WAV tracks).
* **Sự trỗi dậy của công nghệ ACR (Automatic Chord Recognition):** Sự phát triển của Trí tuệ nhân tạo (AI) và Xử lý tín hiệu số (DSP) đã cho ra đời các công cụ dò hợp âm tự động như ChordAI, Chordify hay Song Master Pro.
* **Khoảng trống thị trường:** Hầu hết các giải pháp hiện nay xem việc nhận diện tên hợp âm là "đích đến cuối cùng". Thực tế với một nhạc công, hợp âm chỉ là **lớp dữ liệu thô (raw data)**. Để ứng dụng vào chơi đàn thực chiến, nhạc công cần hiểu bản chất hòa âm (hợp âm này là bậc mấy, dùng âm giai nào) và cần một môi trường luyện tập chuyên biệt mà không phải chuyển qua lại giữa nhiều phần mềm.

---

## 2. Vấn Đề (Problem Statement)
Thực trạng người chơi nhạc gặp phải khi sử dụng các công cụ nhận diện hợp âm hiện nay:
1. **Thiếu tư duy nhạc lý ngữ cảnh (Lack of Harmonic/Contextual Insight):**
   * Các công cụ phổ thông chỉ hiển thị ký hiệu hợp âm rời rạc (như `C`, `G`, `Am`, `Em`).
   * Không phân tích **bậc hòa âm** (Roman Numeral Analysis: I - V - vi - IV), khiến người chơi không hiểu được mối quan hệ hòa âm và gặp khó khăn khi muốn chuyển giọng (transpose) hoặc phân tích cấu trúc bài hát.
   * Không cung cấp gợi ý **âm giai (Scale Mapping)**, gây bế tắc cho người chơi khi muốn solo, ngẫu hứng (improvise) hoặc lót bè.
2. **Đứt gãy luồng luyện tập thực hành (Fragmented Practice Workflow):**
   * Người chơi phải mở cùng lúc nhiều ứng dụng: một tab để nghe nhạc, một ứng dụng để xem hợp âm, một công cụ Metronome để giữ nhịp, và một ứng dụng tra thế bấm.
   * Thiếu các tính năng luyện tập ngón đàn chuyên sâu: Tua chính xác theo ô nhịp, lặp đoạn (A/B Looping) chuẩn xác đến mili-giây, hoặc giảm tốc độ (Speed Shifting) mà không làm méo cao độ.
3. **Thiếu khả năng cá nhân hóa & Độ trễ cao (Rigidity & Latency):**
   * AI không bao giờ chính xác 100%. Tuy nhiên, các nền tảng web hiện tại không cho phép hoặc rất khó để người dùng sửa lại những hợp âm bị nhận diện sai hoặc đổi sang thế bấm (voicing) phù hợp với phong cách của mình.
   * Việc chuyển đổi giữa chế độ cơ bản (Basic) và nâng cao (Precise) thường phụ thuộc vào gọi lại API máy chủ, gây gián đoạn trải nghiệm người dùng.

---

## 3. Giải Pháp (Solution: ChordSense Pro)
**ChordSense Pro** giải quyết triệt để bài toán trên bằng cách xây dựng một **Web App khép kín (End-to-End Workspace)**, kết hợp giữa mô hình AI nhận diện âm thanh với động cơ phân tích nhạc lý ngữ cảnh và trình phát nhạc đồng bộ thời gian thực:

* **Động cơ Phân tích Nhạc lý Phía Client (Client-Side Harmonic Engine):**
  * Tự động tính toán **Bậc hòa âm** (Roman Numerals) theo Key gốc của bài hát (ví dụ: trong giọng C Major, hợp âm `Dm` sẽ được hiển thị là bậc ii, `G7` là bậc V7).
  * Gợi ý trực quan **Âm giai tương thích** (Scale Mapping: *Ionian, Dorian, Mixolydian...*) giúp nhạc công định hướng câu solo/fill ngay lập tức.
  * Hỗ trợ gạt chuyển chế độ **Basic / Precise** tức thì tại trình duyệt bằng thuật toán Down-mapping không có độ trễ.
* **Không gian Luyện tập Tương tác Đồng bộ (Interactive Workspace & Sync Player):**
  * **Smart Grid View:** Giao diện lưới ô nhịp (Bar-based) mượt mà, đồng bộ với thời gian thực bằng `requestAnimationFrame` (sai số < 100ms).
  * **Click-to-Seek & A/B Looping:** Nhấp vào bất kỳ ô nhịp nào để tua ngay đến mili-giây đó; chọn vùng ô nhịp để tạo vòng lặp mượt mà hỗ trợ tập ngón.
  * **Dynamic Metronome & Speed Shifting:** Tích hợp bộ đếm nhịp thông minh phát theo BPM thực và hỗ trợ giảm tốc độ phát (0.5x, 0.75x) bằng Web Audio API giữ nguyên cao độ.
* **Bách khoa Hợp âm Tương tác & Bộ Tổng hợp Âm thanh (Ultimate Chord Dictionary & Synth):**
  * Hiển thị thế bấm trực quan trên **cần đàn Guitar (Fretboard)** và **bàn phím Piano**.
  * Tích hợp Web MIDI / Audio Synth trực tiếp trên trình duyệt, cho phép người dùng click để nghe thử âm thanh thực tế của từng voicing.
* **Cá nhân hóa & Lưu trữ Đám mây (User UGC & Event-Based Architecture):**
  * Cho phép người dùng chỉnh sửa (Edit Marker) hợp âm sai hoặc thay đổi voicing theo sở thích.
  * Hệ thống tự động tính toán lại Bậc và Âm giai tương ứng ngay khi người dùng sửa đổi.
  * Lưu trữ cấu trúc bản nhạc dưới dạng **dòng thời gian sự kiện tuyệt đối (Event-based Timeline JSONB)** trên PostgreSQL, cho phép lưu trữ và quản lý Playlist cá nhân linh hoạt.

---

## 4. Đối Tượng Sử Dụng (Target Audience)
1. **Nhạc công biểu diễn & Thành viên Ban nhạc (Gigging Musicians & Band Members):**
   * Cần "mò bài" (transcribe) thần tốc từ bản thu YouTube/MP3 để kịp lịch tập hoặc biểu diễn.
   * Cần nắm bắt cấu trúc bậc hòa âm (I - IV - V...) để dịch giọng (transpose) ngay tức thì trên sân khấu theo tone của ca sĩ.
2. **Người học nhạc & Tự học nhạc cụ (Music Students & Self-Learners):**
   * Người mới hoặc trung cấp muốn vượt qua giai đoạn "chỉ biết bấm vẹt hợp âm" để hiểu sâu nhạc lý bài hát.
   * Cần công cụ lặp đoạn chậm rãi (A/B Loop, Metronome) để tập chuyển ngón trên những đoạn chuyển hợp âm nhanh hoặc kỹ thuật khó.
3. **Nhà sản xuất âm nhạc, Giảng viên & Nhạc sĩ Sáng tác (Music Producers, Songwriters & Educators):**
   * Phân tích tiến trình hòa âm (Chord Progressions) độc đáo từ các bài hát thịnh hành để tìm cảm hứng sáng tác và hòa âm phối khí.
   * Giảng viên âm nhạc có thể sử dụng làm công cụ trực quan để minh họa bài giảng về hòa âm và âm giai cho học viên.

---
---

# PART 2: PROJECT PRESENTATION (ENGLISH)

## 1. Context & Background
* **Digital Music Learning Trends:** Modern instrumentalists (guitarists, pianists, bassists, songwriters) increasingly rely on digital media (YouTube videos, streaming tracks, local audio files) for self-study, transcribing, and practice routines.
* **Rise of Automatic Chord Recognition (ACR):** Advances in Machine Learning and Digital Signal Processing (DSP) have enabled automatic chord transcription platforms like ChordAI, Chordify, and Song Master.
* **The Industry Gap:** Traditional ACR solutions treat chord label detection as the final deliverable. However, for a practicing musician, raw chord symbols are simply **unprocessed data**. True musicianship requires understanding the harmonic function (Roman numerals) and scale relationships within a structured, interactive practice environment.

---

## 2. Problem Statement
Current solutions create significant friction for musicians:
1. **Lack of Harmonic & Contextual Insight:**
   * Most tools only display isolated chord names (`C`, `G`, `Am`, `Em`) without identifying their contextual role.
   * They lack **Roman Numeral Analysis** (e.g., I - V - vi - IV), making it difficult for players to grasp progression logic or transpose across keys.
   * No **Scale Mapping** suggestions are provided, leaving players at a loss when attempting to improvise solos or construct vocal harmonies.
2. **Fragmented Practice Workflow:**
   * Musicians must juggle multiple disconnected tools: a media player for playback, an app for chord charts, a standalone metronome for timing, and a chord chart app for fingerings.
   * Crucial practice capabilities—such as millisecond-precision click-to-seek, A/B looping for difficult passages, and tempo shifting without pitch distortion—are either missing or poorly integrated.
3. **Rigidity and Latency:**
   * AI chord detection is inherently imperfect. Most web-based tools offer no way to override erroneous detections or customize voicings.
   * Switching between simplified ("Basic") and complex ("Precise") chord extensions often incurs server round-trips and sluggish UI response times.

---

## 3. The Solution: ChordSense Pro
**ChordSense Pro** transforms raw AI chord detection into actionable musical intelligence within an **all-in-one, end-to-end interactive workspace**:

* **Client-Side Harmonic Engine:**
  * Computes **Roman Numeral Analysis** relative to the detected song key (e.g., in the key of C Major, `Dm` is dynamically labeled as ii, `G7` as V7).
  * Delivers context-aware **Scale Mapping** (suggesting modes like *Ionian, Dorian, Mixolydian*) to empower instant improvisation and lead playing.
  * Provides zero-latency **Basic vs. Precise** chord simplification handled entirely in-browser.
* **Interactive Practice Workspace & Synchronized Player:**
  * **Smart Bar-based Grid View:** Rendered smoothly with sub-100ms sync using `requestAnimationFrame`, eliminating Virtual DOM bottlenecks.
  * **Click-to-Seek & A/B Looping:** Jump directly to any bar’s exact timestamp by clicking on the grid, and effortlessly loop challenging bars for finger muscle-memory training.
  * **Dynamic Metronome & Pitch-Preserving Speed Shift:** Generates an accurate Web Audio click synchronized with track BPM and allows tempo scaling (0.5x, 0.75x) without altering audio pitch.
* **Ultimate Chord Dictionary with Built-In Synthesizer:**
  * Displays intuitive visual diagrams for both **Guitar Fretboards** and **Piano Keyboards**.
  * Features an in-browser Web MIDI/Soundfont Synth that audibly plays voicing structures upon user interaction.
* **User Customization & Cloud Storage (UGC):**
  * Enables user overrides (Edit Marker modal) to correct chords, alter bass notes, or customize voicings.
  * Automatically recalculates harmonic degrees and scale modes upon user edits in real time.
  * Employs an **Event-based Timeline schema (PostgreSQL JSONB)** to store original AI analyses alongside personalized user custom charts and playlists.

---

## 4. Target Audience
1. **Gigging & Band Musicians:**
   * Quickly transcribe, learn, and internalize setlists from YouTube or audio demos.
   * Understand functional progressions to transpose on the fly during rehearsals or live stage performances.
2. **Music Students & Self-Taught Learners:**
   * Progress from mechanical finger-placement to genuine music theory comprehension.
   * Master complex rhythms and fast chord transitions through gradual-speed A/B loops and dynamic metronome support.
3. **Producers, Songwriters & Music Educators:**
   * Deconstruct harmonic progressions from hit records to discover fresh songwriting ideas.
   * Utilize the interactive grid and scale mapping as an engaging visual teaching aid for music theory instruction.

---

## Comparative Matrix / Bảng So Sánh Giá Trị

| Feature / Criteria | Traditional Tools (ChordAI, Chordify) | **ChordSense Pro** |
| :--- | :--- | :--- |
| **Output Data** | Raw chord labels (`C`, `G`, `Am`) | **Contextual Harmonic Intelligence (Roman Numerals + Scale Mapping + Voicings)** |
| **Practice Environment** | Disconnected tools (External metronome/players) | **All-in-One Integrated Workspace (Sync Player + Smart Grid + A/B Loop + Audio Synth)** |
| **Customization & Sync** | Fixed labels or local-only edits | **Live User Edits (Override) + Real-time Theory Recalculation + Cloud Playlist Sync** |
