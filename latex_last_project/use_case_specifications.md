### USE CASE 1: Ingest Audio from Multiple Sources, Analyze Chords and Personalize Harmonic Chart (HITL)

| Field | Details |
| :--- | :--- |
| **USE CASE 1** | **Ingest Audio from Multiple Sources, Analyze Chords and Personalize Harmonic Chart** |
| **Description** | The user adds a song from one of three sources: (1) a file uploaded from the device (MP3, WAV, FLAC, M4A, OGG); (2) a live microphone recording made in the browser; or (3) an external media stream (YouTube, SoundCloud, or a direct public audio URL). The gateway validates the request and enqueues an asynchronous job (about 15-45 seconds for a 3-4 minute song); a completion notification is pushed over WebSocket and the chord grid appears (HOTL). Free users see basic chords within a quota of 5 analyses per day; Paid users have unlimited ingestion and Education students draw from the pooled class quota; both also see Roman numerals and scale suggestions. |
| **Used by** | Practice Workspace & Repertoire Management (Use Case 2) |
| **Preconditions** | User is logged in; At least one valid source: an audio file (MP3, WAV, FLAC, M4A, or OGG) of at most 100 MB and 10 minutes; or a recording within the same limits with microphone permission granted; or a reachable external URL; Free User has not used all 5 analyses of the current day; Education Student has not exhausted the daily cap of 5 or the pooled class quota; Paid Users are not limited. |
| **Success End Condition** | The chord grid appears. Paid users also see Roman numerals and scale names. The song is saved to the user's library. Overrides are persisted to the cloud for Paid users and Students, and to local browser storage only for Free users. |
| **Failed End Condition** | An error message appears. No data is saved and any consumed quota is refunded. |
| **Actors** | Primary: Free User, Paid User (Pro), Teacher, Student. Secondary: AI Inference Pipeline, yt-dlp Media Resolver |
| **Trigger** | User clicks "Analyze Song" after selecting a file, finishing a recording, or pasting a link. |
| **DESCRIPTION** | **Step & Action** |
| | 1. User selects a source tab: Upload, Record, or Link |
| | 2. Upload: the user picks a file. Record: the user clicks "Record", grants microphone permission, plays or sings, stops, previews, and confirms. Link: the user pastes a URL |
| | 3. The client sends the file or recording as a multipart upload, or the URL as JSON, to the gateway |
| | 4. System checks the entitlements and quota and looks up the content hash; if the same content was already analysed, the cached result is returned immediately without consuming quota |
| | 5. Otherwise the system enqueues the job and immediately returns a job ID (HTTP 202); the browser does not wait on the request |
| | 6. The Normalizer validates the media with ffprobe, resolves an external URL through the SSRF-safe fetcher and yt-dlp, and transcodes every source to the canonical waveform of the pipeline |
| | 7. The AI pipeline runs asynchronously: Demucs → madmom → MERT inference |
| | 8. When done, the system pushes a completion notification over WebSocket |
| | 9. The page shows the Smart Grid with chords and annotations; the Client Harmonic Engine adds Roman numerals for entitled users |
| | 10. The song is saved to the user's library |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. External link cannot be resolved (private video, region block, unsupported site, or extractor failure): <br>&emsp; System tells the user and offers to upload a file or record the audio instead |
| | AF2. WebSocket connection is lost: <br>&emsp; The client falls back to polling the job status every 2 seconds until the job is done |
| | AF3. Microphone permission is denied: <br>&emsp; System explains how to enable it and offers file upload instead |
| | AF4. User is not satisfied with the recording preview: <br>&emsp; User discards it and records again; nothing is uploaded until confirmation |
| | EF1. Analysis takes more than 90 seconds (twice the expected p95): <br>&emsp; System marks the job as failed, refunds the quota, and shows a timeout error |
| | EF2. Free User has already used 5 analyses today: <br>&emsp; System rejects the request with HTTP 429, shows the next reset time, and offers the Pro upgrade |
| | EF3. Source is unsafe or invalid (non-HTTP scheme, address resolves to a private, loopback, or link-local range, redirect to such a range, corrupt file, unsupported format, or limit exceeded): <br>&emsp; System rejects the source before any fetch or processing and shows the specific reason |
| | EF4. Education pooled quota is exhausted: <br>&emsp; System blocks the new analysis, tells the Student, and notifies the Teacher |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Audio sources: local file upload (MP3, WAV, FLAC, M4A, OGG); live microphone recording; YouTube / SoundCloud / direct public audio URL; selecting a demo track from the catalog |
| | V2. Chord editing: visual dropdown picker modal; virtual Fretboard or Piano Keyboard; keyboard hotkeys |

---

### USE CASE 2: Practice with Interactive Workspace (Smart Grid, A/B Loop & Metronome)

| Field | Details |
| :--- | :--- |
| **USE CASE 2** | **Practice with Interactive Workspace (Smart Grid, A/B Loop & Metronome)** |
| **Description** | Musician or learner uses the interactive workspace to practice songs on the Smart Grid, loop difficult technical passages (A/B Looping), enable dynamic metronome clicks, and shift playback speed without pitch distortion. |
| **Used by** | None (Top-level core feature) |
| **Preconditions** | Song has been analyzed and loaded into Workspace; audio stream and chord timeline events are fully ready. Free Users have the basic Smart Grid and a basic single-range loop at the original speed; Paid Users (Pro) unlock the advanced A/B Loop (speed control, metronome synchronization, saved loop regions). |
| **Success End Condition** | User practices seamlessly; audio playback, metronome click, and visual grid highlight remain synchronized in real time (< 100ms latency); A/B loop executes accurately. |
| **Failed End Condition** | Browser experiences Virtual DOM lag or fails to decode Web Audio API; system gracefully falls back to basic HTML5 Audio playback. |
| **Actors** | Musician / Learner, Web Audio API Engine |
| **Trigger** | User clicks the "Play" button or clicks directly on a bar cell on the Smart Grid. |
| **DESCRIPTION** | **Step & Action** |
| | 1. User opens the song's Practice Workspace view. |
| | 2. System initializes the Web Audio Context, loads `tempo` and `first_beat_offset_ms` to interpolate the Bar Grid. |
| | 3. User presses Play (or Spacebar); system initiates audio playback and triggers `requestAnimationFrame` loop to highlight the active bar cell. |
| | 4. User toggles the Dynamic Metronome; system generates synthesized click pulses aligned with the track's BPM and time signature (see Use Case 18 for the full Metronome and Rhythm Engine). |
| | 5. User selects Bar 9 through Bar 16 to set an A/B Loop Range. |
| | 6. System constrains playback boundaries; when the playback playhead reaches point B, the player automatically seeks back to point A within 100ms. |
| | 7. User scales playback speed down to 0.75x; Web Audio Engine adjusts `playbackRate` using pitch-preserving time-stretching. |
| **EXTENSIONS** | **Step & Branching Action** |
| | 2a. Browser blocks automatic audio playback (AudioContext Autoplay Policy): <br>&emsp; 2a1. System displays an interactive modal requesting user click to unlock AudioContext. |
| | 4b. Free User selects the song-synchronized metronome, subdivisions, count-in, changes speed, or saves a loop region (Pro-only; the standalone Manual Mode metronome stays available, see Use Case 18): <br>&emsp; 4b1. System keeps the basic loop running and displays the Pro upgrade prompt. |
| | 5a. User wants to clear the active A/B loop: <br>&emsp; 5a1. User clicks "Clear Loop" button; system resumes normal full-track playback. |
| | 7a. Low client device performance causes visual rendering latency: <br>&emsp; 7a1. UI automatically reduces canvas repaint frame rate to prioritize glitch-free audio processing. |
| **VARIATIONS** | **Step & Branching Action** |
| | 3. User can control playback through: <br>&emsp; - Play/Pause transport controls on the toolbar <br>&emsp; - Keyboard shortcuts (Spacebar, Left/Right arrow keys) <br>&emsp; - Direct click on any grid bar (Click-to-Seek) |
| | 4. User can configure metronome click timbre: <br>&emsp; - Woodblock click <br>&emsp; - Mechanical electronic beep <br>&emsp; - Independent metronome volume control |

---

### USE CASE 3: Explore & Audition Chord Voicings with Web Audio Synth

| Field | Details |
| :--- | :--- |
| **USE CASE 3** | **Explore & Audition Chord Voicings with Web Audio Synth** |
| **Description** | User opens the chord dictionary to view fingering diagrams (Guitar Fretboard or Piano Keyboard) and triggers the real-time polyphonic synthesizer (Web Audio Synth / Soundfont) to audition the acoustic sound of specific voicings. |
| **Used by** | Ingest Audio & Personalize Chart (Use Case 1), Practice Workspace (Use Case 2) |
| **Preconditions** | User device has an active audio output; static Soundfonts are cached in browser storage. |
| **Success End Condition** | Visual fingering diagram renders accurately; polyphonic synth articulates the exact harmonic frequencies corresponding to the voicing notes. |
| **Failed End Condition** | Soundfont asset fails to load; system displays static fingering diagram with an audio fallback notice. |
| **Actors** | Musician / Learner, Browser Web Audio / Soundfont Synth |
| **Trigger** | User clicks a chord name on the Smart Grid or searches from the chord lookup bar. |
| **DESCRIPTION** | **Step & Action** |
| | 1. User clicks a chord symbol on the grid (e.g., `Cmaj9`) or searches within the Chord Dictionary. |
| | 2. System opens the Chord Dictionary modal, displaying Root, Quality, constituent notes, and Roman numeral harmonic function. |
| | 3. User chooses the target instrument view: Guitar Fretboard or Piano Keyboard. |
| | 4. System renders the visual fingering diagram for the standard root-position voicing. |
| | 5. User explores alternative voicing variations (Drop-2, Inversions, Open Chords). |
| | 6. User clicks the "Audition" button (or clicks directly on chord notes); Web Audio Synth triggers simultaneous voice playback. |
| **EXTENSIONS** | **Step & Branching Action** |
| | 1a. Selected chord is a slash chord (e.g., `F/G`): <br>&emsp; 1a1. System separates bass note `G` and upper structure `F major`, rendering compound fingering with highlighted bass register. |
| | 6a. Paid User connects an external MIDI controller: <br>&emsp; 6a1. Web MIDI API captures physical key presses and verifies accuracy against the target voicing (see Use Case 9 for the full MIDI connection and chord detection flow). |
| **VARIATIONS** | **Step & Branching Action** |
| | 3. User can switch display instrument: <br>&emsp; - Guitar Fretboard (6 strings, 12–24 frets) <br>&emsp; - Piano Keyboard (88 keys with color-coded key markers) <br>&emsp; - Bass guitar or Ukulele (if extended configuration is enabled) |
| | 6. User can audition playback as: <br>&emsp; - Simultaneous strum / block chord <br>&emsp; - Arpeggiated sequence (note-by-note roll) |

---

### USE CASE 4: Manage Repertoire, Custom Charts & Playlists

| Field | Details |
| :--- | :--- |
| **USE CASE 4** | **Manage Repertoire, Custom Charts & Playlists** |
| **Description** | Musician organizes repertoire, manages customized chord charts (with personal chord edits and markers), arranges songs into ordered setlists/playlists, and syncs across devices via cloud storage. |
| **Used by** | None (User Repertoire Management) |
| **Preconditions** | User is authenticated with a valid ChordSense Pro account. |
| **Success End Condition** | Song charts or playlists are successfully updated in PostgreSQL; changes synchronize across devices. |
| **Failed End Condition** | Network connection loss; system caches chart revisions locally in `IndexedDB / LocalStorage` and retries synchronization when connection recovers. |
| **Actors** | Musician / Learner, PostgreSQL Database Server |
| **Trigger** | User clicks "Create New Playlist" or selects "Save to Library" from the Practice Workspace. |
| **DESCRIPTION** | **Step & Action** |
| | 1. User navigates to the Personal Library / Repertoire page. |
| | 2. User clicks "Create Playlist" and enters a title (e.g., "Acoustic Gig Night 2026"). |
| | 3. System creates a new entry in the `playlists` database table. |
| | 4. User searches through analyzed tracks or verified charts in the public library. |
| | 5. User drags and drops selected songs into the playlist and arranges performance sequence (Setlist Order). |
| | 6. System updates the associative `playlist_items` table with normalized `order_index`. |
| | 7. User selects a track from the playlist to launch an interactive rehearsal session. |
| **EXTENSIONS** | **Step & Branching Action** |
| | 4a. Paid User saves a track containing user-edited chords (Custom Chart): <br>&emsp; 4a1. System forks a user-specific `song_charts` record with `is_customized = true` tied to the user account, preserving the public baseline intact. |
| | 4b. Free User saves a track containing user-edited chords: <br>&emsp; 4b1. System stores the overrides only in local browser storage (`IndexedDB`); no custom chord override is written to the cloud, and the Pro upgrade prompt is displayed. |
| | 6a. Server connection fails during setlist reordering: <br>&emsp; 6a1. Interface alerts user of offline mode, stores payload in LocalStorage, and automatically executes sync retry on network reconnection. |
| **VARIATIONS** | **Step & Branching Action** |
| | 2. Playlists can be created from: <br>&emsp; - An empty personal playlist <br>&emsp; - Cloning a shared public community playlist <br>&emsp; - Auto-generation by genre or musical key |
| | 7. Paid User can export a chart with overrides as: <br>&emsp; - ChordPro text <br>&emsp; - JSON event timeline |

---

### USE CASE 5: Manage & Standardize Chord Dictionary & Voicings (Music Platform Manager)

| Field | Details |
| :--- | :--- |
| **USE CASE 5** | **Manage & Standardize Chord Dictionary & Voicings** |
| **Description** | The Music Platform Manager (content role), with Expert advisors, adds, edits, standardizes, and sound-checks chord voicings (Guitar fretboard layouts, Piano voicings) in the system-wide core reference library. |
| **Used by** | Curriculum & Repertoire Governance |
| **Preconditions** | User is authenticated with `Music Platform Manager` credentials. |
| **Success End Condition** | New chord entry, fingering JSONB schema, and audio soundfont asset are validated and committed to `chord_dictionary`, propagating system-wide. |
| **Failed End Condition** | Fingering geometry is physically unplayable on instruments; system rejects submission with a biometric constraint error. |
| **Actors** | Music Platform Manager (Business and Content Admin), PostgreSQL Database |
| **Trigger** | Admin accesses the Content Management Portal and selects "Chord Dictionary Management". |
| **DESCRIPTION** | **Step & Action** |
| | 1. Admin logs into the Music PM Content Portal. |
| | 2. Admin selects "Add New Chord / Voicing" action. |
| | 3. Admin inputs foundational attributes: Root note (e.g., `Eb`), Quality (e.g., `maj9#11`), and Target Instrument (Guitar/Piano). |
| | 4. Admin marks fret positions, finger assignments (1, 2, 3, 4), open strings, and muted strings (Mute) on the visual Fretboard Editor. |
| | 5. System validates harmonic consistency: derives semitone intervals to ensure conformance with formal chord formula. |
| | 6. Admin uploads or associates the corresponding audio sample asset (Soundfont / Web Audio MIDI patch). |
| | 7. Admin clicks "Test Audio" to audition voicing clarity and sound fidelity. |
| | 8. Admin clicks "Publish"; system commits the record to `chord_dictionary` and invalidates the edge CDN cache. |
| **EXTENSIONS** | **Step & Branching Action** |
| | 5a. Fingering exceeds realistic human anatomical reach (e.g., fret span > 5 frets on guitar): <br>&emsp; 5a1. System displays "Unrealistic Fingering Span" warning requiring explicit confirmation or adjustment. |
| | 8a. Duplicate voicing signature exists under the same identifier: <br>&emsp; 8a1. System prompts to save the entry as an alternative voicing variation index (Inversion / Variation #N). |
| **VARIATIONS** | **Step & Branching Action** |
| | 4. Admin can input voicing data via: <br>&emsp; - Interactive virtual instrument editor <br>&emsp; - Tablature notation string (e.g., `x-3-2-0-0-0`) <br>&emsp; - Bulk JSON/CSV batch import |

---

### USE CASE 6: Review, Verify & Publish Public Song Charts (Music Platform Manager)

| Field | Details |
| :--- | :--- |
| **USE CASE 6** | **Review, Verify & Publish Public Song Charts** |
| **Description** | The Music Platform Manager reviews AI chord recognition accuracy on prominent songs, refines harmonic notations, and approves publication into the Verified Public Catalog for community use and school curriculum. |
| **Used by** | Curriculum & Repertoire Governance |
| **Preconditions** | User holds `Music Platform Manager` privileges; target song has completed preliminary AI pipeline inference. |
| **Success End Condition** | Song chart is marked `is_verified = true`, published to the public search index, and linked to educational curriculum tracks. |
| **Failed End Condition** | Track audio quality is excessively degraded or rhythmically corrupted beyond repair; song is rejected from publication. |
| **Actors** | Music Platform Manager (Business and Content Admin), AI Inference Service |
| **Trigger** | Admin selects a track from the "Quality Verification Queue". |
| **DESCRIPTION** | **Step & Action** |
| | 1. Admin opens the Song Verification Queue. |
| | 2. Admin selects a pending song and opens the "Master Audit Workspace". |
| | 3. System renders AI detection predictions alongside confidence scores and audio spectrogram display. |
| | 4. Admin audits playback against original audio, evaluating accuracy of key center, tempo, and measure downbeats. |
| | 5. Admin rectifies misclassified chords and adjusts downbeat boundary lines on the timeline. |
| | 6. Admin enters metadata: Artist, Genre, and Pedagogical Difficulty Level (Beginner, Intermediate, Advanced). |
| | 7. Admin clicks "Verify & Publish" button. |
| | 8. System saves the verified master chart and triggers automated sheet export generation (ChordPro / Lead Sheet). |
| **EXTENSIONS** | **Step & Branching Action** |
| | 4a. AI model misdetected the global key signature: <br>&emsp; 4a1. Admin re-assigns the tonic key; system automatically transposes the Roman numeral harmonic matrix accordingly. |
| | 7a. Song is subject to copyright infringement or content policy violation: <br>&emsp; 7a1. Admin flags song status as "Archived / Unlisted" and suppresses it from public discovery. |
| **VARIATIONS** | **Step & Branching Action** |
| | 8. Approved songs can be published to: <br>&emsp; - Public Community Repertoire Catalog <br>&emsp; - Formal Music Curriculum Tracks (Middle School Grades 6–9) |

---

### USE CASE 7: Monitor Infrastructure, Datastores, GPU Workload and Pipeline Performance (IT Admin)

| Field | Details |
| :--- | :--- |
| **USE CASE 7** | **Monitor Infrastructure, Datastores, GPU Workload and Pipeline Performance** |
| **Description** | The IT Administrator monitors and maintains the technical systems: the PostgreSQL connection pool, the MongoDB staging store, the Redis cache, the GPU worker pool, the Celery queue length, RAM and CPU, and container health, and takes corrective action such as scaling workers. |
| **Used by** | Infrastructure & DevOps Governance |
| **Preconditions** | User is logged in with the IT Administrator role; Telemetry collectors and worker daemons are running. |
| **Success End Condition** | The console shows current telemetry; after scaling or maintenance the queue backlog is back to a nominal level (fewer than 5 pending jobs) and processing time is within the NFR-05 limit. |
| **Failed End Condition** | If the telemetry collector is unreachable, the console shows a connection failure and marks the last known values as stale. |
| **Actors** | Primary: IT Administrator (System Operations and Infrastructure Admin). Secondary: Celery/Redis Job Queue, GPU Worker Nodes |
| **Trigger** | Administrator opens the Operations Console or receives a queue congestion alert. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Administrator opens the Operations Console |
| | 2. System shows the PostgreSQL pool (connections in use and idle, wait time), the MongoDB staging store (document count, TTL expiry), and the Redis cache (hit rate, memory, evictions) |
| | 3. System shows the GPU worker pool (utilization, VRAM), the Celery queue length and waiting time, per-stage durations (yt-dlp, Demucs, madmom, MERT), RAM, CPU, and container health |
| | 4. Administrator identifies a bottleneck (for example a growing queue in the Demucs stage after a traffic spike) |
| | 5. Administrator scales the GPU workers (manual slider or threshold policy) or resizes the connection pool |
| | 6. System starts additional containers and rebalances dispatching; the backlog returns to a nominal level |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. A worker task is stuck after a CUDA out-of-memory error: <br>&emsp; Administrator selects "Force Terminate and Re-queue"; the task is resubmitted with a smaller chunking window |
| | AF2. An external media provider returns HTTP 429: <br>&emsp; System rotates the egress proxy pool and alerts the Administrator |
| | AF3. Redis memory pressure: <br>&emsp; Administrator flushes a cache namespace after confirmation |
| | EF1. Telemetry collector is unreachable: <br>&emsp; Console shows a connection failure and marks values as stale |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Scaling: manual slider; threshold policy (Kubernetes HPA or Docker Swarm) |

---

### USE CASE 8: Protect API Gateway (Rate Limits, Token Quotas), Manage Access and Review Audit Logs (IT Admin)

| Field | Details |
| :--- | :--- |
| **USE CASE 8** | **Protect API Gateway** |
| **Description** | The IT Administrator operates the API gateway protections: rate limiting (token bucket per tier, endpoint, and IP address), distribution of technical token quotas derived from the active subscription policy, defence against brute-force and scraping attacks, role assignment and suspension of abusive accounts, and review of the security and audit logs. Business values such as prices and plan quotas belong to the Music PM (Use Case 15). |
| **Used by** | Infrastructure & DevOps Governance |
| **Preconditions** | User is logged in with the IT Administrator role. |
| **Success End Condition** | Protection settings are applied and every privileged action is recorded in the append-only audit log. |
| **Failed End Condition** | A security constraint blocks the action (for example demoting the last root administrator). |
| **Actors** | Primary: IT Administrator. Secondary: PostgreSQL Database |
| **Trigger** | Administrator opens the Security Console. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Administrator opens the Security Console |
| | 2. System shows request rates by tier and IP address, the number of HTTP 429 responses, and failed logins |
| | 3. Administrator identifies an abusive pattern (for example credential stuffing or scraping) |
| | 4. Administrator adjusts the token-bucket capacity and refill rate for a tier or endpoint; the base values come from the policy defined by the Music PM |
| | 5. Administrator blocks an IP range, suspends an account, or revokes JWT sessions |
| | 6. Administrator assigns or removes a role when required |
| | 7. Administrator reviews the audit log filtered by period, actor, and category and runs the integrity check |
| | 8. System appends an immutable audit entry for each change |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. A legitimate user is blocked: <br>&emsp; Administrator removes the block and the entry stays in the audit log |
| | EF1. Administrator tries to demote the last root administrator: <br>&emsp; System prevents the action |
| | EF2. Someone attempts to change or delete an audit entry: <br>&emsp; System rejects the attempt and raises an alert |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Audit log filters: timestamp range; source IP and user ID; category (authentication, role change, ingestion abuse, policy change) |

---

### USE CASE 9: Connect MIDI Keyboard & Detect Chords in Real Time (Web MIDI, HITL)

| Field | Details |
| :--- | :--- |
| **USE CASE 9** | **Connect MIDI Keyboard & Detect Chords in Real Time (Web MIDI, HITL)** |
| **Description** | Paid User (Pro) connects a Digital Piano or MIDI keyboard directly to the browser through the Web MIDI API. The Client-Side Harmonic Engine captures `noteOn` / `noteOff` events and infers the chord from the pitch-class structure of the sounding notes in real time. Everything executes in the browser: zero network latency and no AI audio inference. This feature is not available to Free Users. |
| **Used by** | Practice Workspace (Use Case 2) |
| **Preconditions** | User is authenticated as a Paid User (Pro); page is served over HTTPS; browser supports the Web MIDI API (e.g., Chromium-based browsers, Firefox); a MIDI device is connected (USB or Bluetooth). |
| **Success End Condition** | The chord being played is rendered on the Smart Grid within 50 ms of the `noteOn` event (excluding device/OS transport delay), together with its Roman numeral and suggested scale; no network request is needed for detection; the user may save the session as a chord chart. |
| **Failed End Condition** | MIDI permission denied, no device found, or browser unsupported (e.g., Safari); system explains the cause and offers the on-screen piano keyboard as an alternative input. |
| **Actors** | Paid User (Pro), MIDI Device (Digital Piano / MIDI Keyboard), Web MIDI API, Client-Side Harmonic Engine |
| **Trigger** | User opens the MIDI Practice tab and clicks "Connect MIDI Device". |
| **DESCRIPTION** | **Step & Action** |
| | 1. User clicks "Connect MIDI Device"; page calls `navigator.requestMIDIAccess()` (without system-exclusive access) and the browser asks for permission. |
| | 2. System lists the available MIDI inputs; user selects the instrument. |
| | 3. User plays notes or chords. |
| | 4. For each `noteOn` (velocity > 0) the Engine adds the note number to the active set; for each `noteOff` (or `noteOn` with velocity 0) it removes the note, unless the sustain pedal (CC 64) is held. |
| | 5. After a short grouping window (about 30 ms), the Engine reduces the active notes to pitch classes, takes the lowest note as the bass candidate, and matches the set against the chord dictionary to obtain root, quality, extension, and slash bass. |
| | 6. System renders the chord on the Smart Grid; the Harmonic Engine adds the Roman numeral and scale suggestion. |
| | 7. User optionally saves the session as a chord chart (the only step that uses the network, `POST /api/v1/charts`); raw MIDI is never uploaded. |
| **EXTENSIONS** | **Step & Branching Action** |
| | 1a. Free User clicks "Connect MIDI Device": <br>&emsp; 1a1. System blocks the action on the client and displays the Pro upgrade prompt; the server also rejects any MIDI session save (HTTP 403). |
| | 1b. Permission denied or `requestMIDIAccess` unavailable: <br>&emsp; 1b1. System displays enabling instructions or supported browsers and offers the on-screen keyboard fallback. |
| | 3a. MIDI device disconnects during the session (`statechange` event): <br>&emsp; 3a1. Engine clears the active note set to avoid stuck notes, shows a reconnect message, and resumes without page reload when the device returns. |
| | 5a. Note set is ambiguous (e.g., Am7 vs. C6/A): <br>&emsp; 5a1. Engine ranks candidates by bass note and current key, shows the best match, and lists alternatives. |
| **VARIATIONS** | **Step & Branching Action** |
| | 2. MIDI inputs can be: <br>&emsp; - USB-MIDI Digital Piano or controller <br>&emsp; - Bluetooth LE MIDI keyboard |

---

### USE CASE 10: Teacher: Create Class and Enroll Students (Education Package)

| Field | Details |
| :--- | :--- |
| **USE CASE 10** | **Teacher: Create Class and Enroll Students (Education Package)** |
| **Description** | A Teacher with an active Education subscription creates a class and fills the Student Seats (15 to 50, within the plan limit) by e-mail invitation, Class Code, or CSV import. The Teacher is the Master Admin of the class. |
| **Used by** | Shared Class Repertoire (Use Case 11) |
| **Preconditions** | User is logged in as a Teacher with an active Education plan (Classroom: up to 20 students; Studio: up to 50). |
| **Success End Condition** | The class exists and enrolled students occupy seats; the roster shows the status of each student (invited, pending approval, active). |
| **Failed End Condition** | No seat beyond the plan limit is ever consumed; invalid rows or expired invitations are reported. |
| **Actors** | Primary: Teacher. Secondary: Student, E-mail Service |
| **Trigger** | Teacher clicks "Create Class" in the Classes page. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Teacher opens the Classes page and clicks "Create Class", then enters the class name and level |
| | 2. System creates the class, assigns the Teacher Seat as Master Admin, and shows seat usage (for example 0 of 20) |
| | 3. Teacher chooses an enrollment method: (a) e-mail invitation - enters addresses, the system sends single-use links valid for 7 days; (b) Class Code - the system generates a code and the Teacher sets its expiry and an optional approval requirement; (c) CSV import - the Teacher uploads a file of names and e-mail addresses (at most 50 rows) |
| | 4. For a CSV file, the system validates every row, removes duplicates, sends the invitations, and shows a row-level report of rejected lines |
| | 5. Each Student opens the link or enters the code, registers or logs in, and (where regulation requires it) records guardian consent; the Student then occupies a seat |
| | 6. If approval is required, the Teacher approves pending students in the roster |
| | 7. Teacher may deactivate a student or reassign a seat (rate-limited per month) |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. Seat limit is reached: <br>&emsp; System rejects further enrollment and offers the upgrade from Classroom to Studio |
| | AF2. CSV file is partly invalid: <br>&emsp; Valid rows are processed and invalid rows are listed with the reason |
| | AF3. Class Code is shared outside the class: <br>&emsp; Teacher rotates the code; the old code stops working and enrolled students are not affected |
| | EF1. Subscription has lapsed: <br>&emsp; The class becomes read-only for a 30-day grace period and new enrollment is blocked |
| | EF2. Invitation link has expired: <br>&emsp; Teacher resends the invitation |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Enrollment methods: e-mail invitation; Class Code; CSV import of the school roster |
| | V2. Devices: Teacher on laptop; Students on laptop, tablet, or digital piano setup |

---

### USE CASE 11: Teacher: Publish Shared Class Repertoire and Annotate Master Chart

| Field | Details |
| :--- | :--- |
| **USE CASE 11** | **Teacher: Publish Shared Class Repertoire and Annotate Master Chart** |
| **Description** | The Teacher builds a repertoire list of analysed songs with assigned loop regions and target tempo, adds a master layer with reference chords, voicings, scale analysis, and notes, and publishes it. The list appears at once in every enrolled Student's workspace, and the master layer is read-only for Students. |
| **Used by** | Student Class Practice (Use Case 12) |
| **Preconditions** | Teacher is logged in; the class has at least one enrolled Student; The songs to assign are analysed, or the Teacher ingests them first (Use Case 1). |
| **Success End Condition** | The repertoire is published, the master layer is stored as a separate read-only document, and Students see the update within 2 seconds. |
| **Failed End Condition** | A draft remains unpublished and Students see no change. |
| **Actors** | Primary: Teacher. Secondary: Student (recipient) |
| **Trigger** | Teacher clicks "Publish" on a repertoire list. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Teacher opens the class repertoire and adds songs from the library |
| | 2. For each song the Teacher opens the master chart editor (Smart Grid in author mode) |
| | 3. Teacher corrects chords, selects reference voicings on the Fretboard or Piano diagram, and adds scale analysis and text notes on bars |
| | 4. Teacher sets assigned loop regions, target tempo, and an optional due date |
| | 5. Teacher clicks "Publish" |
| | 6. System stores the master layer as a separate JSONB document marked read-only for Students and saves the repertoire list |
| | 7. System pushes the change over WebSocket to every enrolled Student within 2 seconds; offline Students receive it on reconnection |
| | 8. Teacher may edit and republish later; Students see the new version |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. Teacher wants Students to experiment on a chart: <br>&emsp; Teacher unlocks editing for the class or for one Student; edits still go to the personal layer |
| | AF2. Student is offline during publication: <br>&emsp; The update is delivered when the Student reconnects |
| | EF1. A Student tries to modify the master layer through the API: <br>&emsp; System rejects the request (HTTP 403); the Student can write only to the personal layer |
| | EF2. A selected song has not finished analysis: <br>&emsp; System shows the job status and keeps the item as a draft until it is done |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Repertoire items: whole song; assigned bar range (A/B Loop); chord drill list |
| | V2. Annotation types: corrected chord; reference voicing; scale analysis; free text |

---

### USE CASE 12: Student: Join Class and Practice Assigned Repertoire

| Field | Details |
| :--- | :--- |
| **USE CASE 12** | **Student: Join Class and Practice Assigned Repertoire** |
| **Description** | A Student joins the Teacher's class by invitation or Class Code, opens the assigned songs on a laptop or tablet, and practices with the advanced A/B Loop, metronome, and Web MIDI or microphone input. Practice events are recorded for the Teacher's dashboard. |
| **Used by** | None (Education practice flow) |
| **Preconditions** | Student has an account or can register, and holds an invitation or Class Code; A seat is available in the class; For MIDI input: a browser with Web MIDI support and a connected device. |
| **Success End Condition** | The Student is enrolled and a practice session is recorded; notes in the personal layer are saved to the cloud. |
| **Failed End Condition** | If no seat is available or the code is invalid, the Student is told to contact the Teacher. |
| **Actors** | Primary: Student. Secondary: Teacher, MIDI Device |
| **Trigger** | Student opens an invitation link or enters a Class Code. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Student opens the invitation link or enters the Class Code |
| | 2. Student logs in or registers; guardian consent is recorded where required |
| | 3. System assigns a seat and opens the class workspace with the repertoire |
| | 4. Student opens a song; the Smart Grid shows the Teacher's read-only master layer and the Student's personal layer |
| | 5. Student sets a loop on the assigned region, changes the speed, and turns on the metronome |
| | 6. Optionally the Student connects a MIDI keyboard (Use Case 9) or uses microphone practice (Real-Time Practice Mode); correct and incorrect chords are flagged |
| | 7. The client records practice time, loop repetitions, and chord accuracy and sends them in batches |
| | 8. Student adds personal notes or chord overrides to the personal layer |
| | 9. At the end of the session a summary is shown and remaining events are sent |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. Tablet browser has no Web MIDI (for example Safari on iPad): <br>&emsp; System offers microphone practice or the on-screen keyboard |
| | AF2. Student is offline: <br>&emsp; Telemetry is buffered in IndexedDB and uploaded when the connection returns |
| | AF3. Student uses two devices (laptop and tablet): <br>&emsp; Both sessions are allowed (limit of two per seat) |
| | EF1. A third concurrent session is started: <br>&emsp; System refuses it and asks the Student to close another session |
| | EF2. Class Code expired or class full: <br>&emsp; System shows a message and asks the Student to contact the Teacher |
| | EF3. Student tries to export a chart: <br>&emsp; System blocks the action; export is not part of Student Seats |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Devices: laptop; tablet; digital piano connected to a supported browser |
| | V2. Practice input: Web MIDI keyboard; microphone; on-screen keyboard |

---

### USE CASE 13: Teacher: Monitor Student Practice Telemetry Dashboard

| Field | Details |
| :--- | :--- |
| **USE CASE 13** | **Teacher: Monitor Student Practice Telemetry Dashboard** |
| **Description** | The Teacher opens the Tracking Dashboard to see practice time, A/B Loop repetitions, and chord accuracy (from Web MIDI or microphone practice) for each Student and for the class, and uses the findings to prepare the next lesson. |
| **Used by** | Shared Class Repertoire (Use Case 11) |
| **Preconditions** | Teacher is logged in; the class has enrolled Students with recorded practice. |
| **Success End Condition** | The dashboard shows data that is at most 60 seconds old for the Teacher's own class only. |
| **Failed End Condition** | If data is delayed, the dashboard shows the time of the last update. |
| **Actors** | Primary: Teacher |
| **Trigger** | Teacher opens Class → Progress. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Teacher opens the Progress page of the class |
| | 2. Teacher selects a period (for example the last week) and optionally filters by Student or song |
| | 3. System aggregates the telemetry and shows total and active practice time, loop repetitions per section, chord accuracy by chord and by input type (MIDI or microphone), and the trend |
| | 4. Teacher sorts Students by lowest practice time or lowest accuracy |
| | 5. Teacher opens a Student's detail to see the bars and chords with most errors |
| | 6. Teacher adjusts the repertoire or master chart (Use Case 11) in response |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. A Student has no recorded practice: <br>&emsp; The dashboard shows "no practice recorded" rather than a zero accuracy |
| | AF2. A Student leaves the class: <br>&emsp; Access to that Student's telemetry is removed according to the retention rule |
| | EF1. Teacher requests another Teacher's class: <br>&emsp; System denies the request (HTTP 403) |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Views: per Student; class summary; per song |
| | V2. Periods: week; month; custom range |

---

### USE CASE 14: Music Platform Manager: Monitor Revenue and Product Usage

| Field | Details |
| :--- | :--- |
| **USE CASE 14** | **Music Platform Manager: Monitor Revenue and Product Usage** |
| **Description** | The Music PM reviews monthly recurring revenue (MRR) by plan (Pro, Education Classroom, Education Studio), subscriber counts of Free, Pro, and Education, churn rate, average revenue per user (ARPU), and customer lifetime value (LTV = ARPU x gross margin / churn rate), together with daily and monthly active users and the most used features. |
| **Used by** | Subscription Policy Configuration (Use Case 15) |
| **Preconditions** | User is logged in with the Music Platform Manager role; Payment events have been synchronized from the payment provider. |
| **Success End Condition** | Dashboards show revenue and usage figures for the selected period and can be exported as CSV. |
| **Failed End Condition** | If the payment feed is delayed, a banner shows the time of the last synchronization. |
| **Actors** | Primary: Music Platform Manager (Business and Content Admin). Secondary: Payment Provider |
| **Trigger** | Music PM opens Business Dashboard. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Music PM opens Business Dashboard → Revenue |
| | 2. System shows MRR by plan, subscriber counts, new and cancelled subscribers, churn rate, ARPU, and LTV |
| | 3. Music PM selects a period and compares it with the previous one |
| | 4. Music PM opens the Usage tab: DAU, MAU, the DAU/MAU ratio, feature adoption, and the ingestion-to-practice funnel, by tier (aggregated data only) |
| | 5. Music PM exports the figures as CSV |
| | 6. Music PM may open the Policy Console (Use Case 15) to act on the findings |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. Payment feed is delayed: <br>&emsp; A banner shows the last synchronization time and figures are marked provisional |
| | EF1. Music PM requests infrastructure metrics, rate-limit settings, or audit logs: <br>&emsp; System denies the request (HTTP 403); these belong to the IT Administrator |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Breakdowns: by plan; by tier; by period |
| | V2. Export formats: CSV |

---

### USE CASE 15: Music Platform Manager: Configure Subscription Policy (Prices, Quotas, Coupons)

| Field | Details |
| :--- | :--- |
| **USE CASE 15** | **Music Platform Manager: Configure Subscription Policy (Prices, Quotas, Coupons)** |
| **Description** | The Music PM creates a new version of the subscription policy: plan prices, the Free daily quota, Education pooled quotas and per-student caps, seat limits, and discount codes. The change is previewed, given an effective date, activated without a software release, and can be rolled back. |
| **Used by** | Gateway Protection (Use Case 8) |
| **Preconditions** | User is logged in with the Music Platform Manager role. |
| **Success End Condition** | The new policy version becomes active on its effective date, the gateway reads it, and an audit entry is written. |
| **Failed End Condition** | A validation error is shown and the active policy is unchanged. |
| **Actors** | Primary: Music Platform Manager |
| **Trigger** | Music PM opens the Policy Console and clicks "New version". |
| **DESCRIPTION** | **Step & Action** |
| | 1. Music PM opens the Policy Console and selects the active policy |
| | 2. Music PM clicks "New version" and edits prices, quotas, seat limits, and discount codes (type, value, expiry, redemption limit) |
| | 3. System validates the values (for example quotas are non-negative, the Studio price is higher than the Classroom price, seat limits stay in the 15-50 range) |
| | 4. System shows an impact preview: affected subscribers and the estimated worst-case compute cost per class |
| | 5. Music PM sets the effective date and confirms |
| | 6. System stores the version, publishes it to the gateway policy cache, and writes an audit entry |
| | 7. Music PM can roll back to the previous version if needed |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. Price increase affects existing subscribers: <br>&emsp; The new price applies at the next renewal and subscribers are notified in advance |
| | AF2. Discount code expires or reaches its redemption limit: <br>&emsp; System disables the code automatically |
| | EF1. Music PM tries to change rate limits or token buckets: <br>&emsp; System denies the change; this belongs to the IT Administrator (Use Case 8) |
| | EF2. Values are invalid: <br>&emsp; System rejects the version and lists the errors |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Policy elements: plan price; Free daily quota; Education pooled quota and per-student cap; seat limits; discount codes |

---

### USE CASE 16: Music Platform Manager: Handle User Feedback and Flag AI Quality Issues

| Field | Details |
| :--- | :--- |
| **USE CASE 16** | **Music Platform Manager: Handle User Feedback and Flag AI Quality Issues** |
| **Description** | The Music PM triages user feedback (reports of wrongly recognized chords and music-theory questions), replies to users, reviews the statistics by chord class and the AI quality KPIs, and flags chord classes that need re-evaluation. A flag creates a retraining request for the IT Administrator and, optionally, a review task for an Expert. |
| **Used by** | Deploy or Roll Back AI Model (Use Case 17) |
| **Preconditions** | Music PM is logged in; At least one feedback ticket exists. |
| **Success End Condition** | Tickets are answered and their status is updated; flagged chord classes appear in the IT Administrator's queue. |
| **Failed End Condition** | If a ticket cannot be answered, it stays open with a note. |
| **Actors** | Primary: Music Platform Manager. Secondary: Expert, IT Administrator |
| **Trigger** | Music PM opens the Feedback queue. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Music PM opens the Feedback queue, which contains the wrong-chord reports and the general feedback submitted by users (Use Case 19), sorted by priority |
| | 2. Music PM opens a wrong-chord report and sees the song, the time position, the detected chord, the chord the user expected, the genre note, and the model version; opening the ticket sets its status to "Under Review by Music PM" for the user |
| | 3. Music PM checks the report (with an Expert if needed) and decides whether it is valid |
| | 4. Music PM replies to the user and sets the final status: "Resolved", "Scheduled for Model Retraining" for a valid report that is flagged, or "Declined" with a reason; the user sees every change under "My Feedback" |
| | 5. Music PM opens the AI Quality tab: accuracy by chord class, confidence distribution, user correction rate, and counts of reports by chord class |
| | 6. For chord classes with accuracy below 75% or many valid reports, Music PM clicks "Flag for Re-evaluation" |
| | 7. System creates a retraining request for the IT Administrator and an optional review task for an Expert |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. Ticket is a music-theory question: <br>&emsp; Music PM answers it or assigns it to an Expert |
| | AF2. Several reports describe the same problem: <br>&emsp; Music PM merges them into one ticket |
| | EF1. The reporting user deleted their account: <br>&emsp; The ticket is anonymized and kept for statistics |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Ticket types: wrong-chord report; theory question; general support |

---

### USE CASE 17: Deploy or Roll Back AI Model Version (IT Admin)

| Field | Details |
| :--- | :--- |
| **USE CASE 17** | **Deploy or Roll Back AI Model Version (IT Admin)** |
| **Description** | The IT Administrator selects a new AI model version (LoRA adapter) from the Model Registry, for example after a retraining request raised by the Music PM, starts the update safely, and can switch back to the old model right away if something goes wrong. |
| **Used by** | Handle User Feedback and Flag AI Quality (Use Case 16) |
| **Preconditions** | Administrator is logged in with the IT Administrator role; A new model version has been uploaded to the Model Registry. |
| **Success End Condition** | The new version serves traffic and is marked stable, or the previous version is restored. |
| **Failed End Condition** | If the new version fails, the previous version stays active. |
| **Actors** | Primary: IT Administrator |
| **Trigger** | Administrator opens the Model Registry. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Administrator opens the Model Registry in the Operations Console |
| | 2. Administrator selects the new model version and checks its accuracy scores |
| | 3. Administrator clicks "Deploy"; the system switches to the new model without downtime |
| | 4. Administrator watches latency, error rate, and GPU usage |
| | 5. If everything is good, the Administrator marks the update as stable |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. The new model causes errors: <br>&emsp; Error rate exceeds the limit; Administrator clicks "Rollback"; system restores the previous model and saves an error report |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Rollout: all traffic at once; gradual percentage rollout |

---

### USE CASE 18: Metronome and Rhythm Engine (Standalone and Song-Synchronized)

| Field | Details |
| :--- | :--- |
| **USE CASE 18** | **Metronome and Rhythm Engine (Standalone and Song-Synchronized)** |
| **Description** | A metronome that works both as a standalone tool (Manual Mode) and in synchronization with an analysed song (Song Sync Mode). In Song Sync Mode it takes the BPM and the time signature (4/4, 3/4, 6/8, and others) from the AI analysis. Both modes support Tap Tempo and a BPM range of 30 to 280, an accented first beat, and a volume slider independent of the song. Song Sync Mode, subdivisions (quarter, eighth, triplet), and Count-in are available to Paid users and Education seats; Free users have the basic Manual Mode. |
| **Used by** | Practice Workspace (Use Case 2), A/B Loop and Practice Tools |
| **Preconditions** | User is logged in; The audio context has been unlocked by a user gesture; Song Sync Mode requires an analysed song to be open. |
| **Success End Condition** | Clicks sound on the Web Audio clock with an offset of at most 10 ms from the beat grid, with the accent on beat 1 and the chosen subdivision. |
| **Failed End Condition** | If the audio context cannot start, the system asks the user to click to unlock audio. |
| **Actors** | Primary: Free User (Manual Mode), Paid User (Pro), Teacher, Student. Secondary: Web Audio API |
| **Trigger** | User opens the Metronome panel or toggles the metronome in the workspace. |
| **DESCRIPTION** | **Step & Action** |
| | 1. User opens the Metronome panel (standalone or docked in the workspace) and clicks "Start"; the audio context resumes |
| | 2. System selects Manual Mode when no song is open, or Song Sync Mode when an analysed song is open |
| | 3. Song Sync Mode: the system loads the BPM and time signature from the analysis, shows the detection confidence, aligns the clicks with the beat grid of the song, and follows playback speed changes and the A/B Loop |
| | 4. Manual Mode: the user sets the BPM from 30 to 280 with the slider or number field, or uses Tap Tempo (the BPM is 60 divided by the mean interval of the last up to 8 taps; the tap history resets after a pause longer than 2 seconds); the user selects the time signature |
| | 5. System schedules the clicks with a look-ahead scheduler on the Web Audio clock; beat 1 receives a distinct accent timbre and pitch, other beats a normal click; in compound meter such as 6/8 the main pulses are dotted quarters with the accent on beat 1 and a secondary accent on beat 4 |
| | 6. The user selects a subdivision (quarter, eighth, or triplet); subdivision clicks are quieter than beat clicks |
| | 7. The user enables Count-in of 1 or 2 bars; before the song starts or before an A/B Loop pass only the clicks sound, and the music starts exactly on beat 1 |
| | 8. The user adjusts the metronome volume with a slider that does not change the song volume |
| | 9. If the detected BPM or time signature is wrong, the user overrides it manually |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. Tapped or typed BPM is outside 30-280: <br>&emsp; System clamps the value to the range and shows a message |
| | AF2. AI confidence is low or the time signature is ambiguous: <br>&emsp; System flags the value and the user can override it in Manual Mode |
| | AF3. Playback speed is reduced (for example 0.5x): <br>&emsp; In Song Sync Mode the click tempo is scaled proportionally so that the clicks stay on the song beats |
| | AF4. The browser tab loses focus: <br>&emsp; Timing is kept because scheduling uses the audio clock, not timers |
| | EF1. A Free User selects Song Sync Mode, subdivisions, or Count-in: <br>&emsp; System keeps Manual Mode running and shows the Pro upgrade prompt |
| | EF2. The browser blocks audio autoplay: <br>&emsp; System shows a modal asking the user to click to unlock audio |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Time signatures: 2/4; 3/4; 4/4; 5/4; 6/8; 7/8; 12/8 |
| | V2. Click timbre: woodblock; electronic beep |

---

### USE CASE 19: Report Chord Error, Send Feedback and Track Status (End User)

| Field | Details |
| :--- | :--- |
| **USE CASE 19** | **Report Chord Error, Send Feedback and Track Status (End User)** |
| **Description** | A user reports a wrongly recognized chord by selecting the bar or timestamp, stating the expected chord and a genre note, or sends general feedback (bug report or feature request) with a description and an optional screenshot. The user can follow the status of every submission: Submitted, Under Review by Music PM, and Resolved or Scheduled for Model Retraining. All submissions appear in the Music PM queue (Use Case 16). |
| **Used by** | Handle User Feedback and Flag AI Quality (Use Case 16) |
| **Preconditions** | User is logged in; For a chord report, an analysed song is open; The daily submission limit (20 per user) has not been reached. |
| **Success End Condition** | The submission is stored with a reference number, appears under "My Feedback" with its status, and reaches the Music PM queue. |
| **Failed End Condition** | A validation or rate-limit message is shown and nothing is stored. |
| **Actors** | Primary: Free User, Paid User (Pro), Teacher, Student. Secondary: Music Platform Manager |
| **Trigger** | User chooses "Report wrong chord" on a bar or timestamp, or opens the Feedback widget. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Chord report: the user right-clicks or long-presses a bar or timestamp marker and chooses "Report wrong chord" (or does so from the Edit Marker) |
| | 2. System shows the detected chord and the position as read-only fields and asks for the expected chord (chosen with the chord picker or typed) and an optional genre or style note, such as "V-Pop ballad" or "jazz" |
| | 3. System attaches the context automatically: song and chart ID, content hash, time position, model version, and source type; no audio is attached |
| | 4. The user confirms; the system stores the report with the status "Submitted" and shows a reference number |
| | 5. General feedback: the user opens the Feedback widget, selects the type (bug, feature request, or general comment), enters a description, and optionally attaches a screenshot (PNG or JPEG, up to 5 MB) after a notice that screenshots may contain personal information |
| | 6. The submission enters the Music PM queue (Use Case 16) |
| | 7. When the Music PM opens the ticket, the status becomes "Under Review by Music PM" and the user is notified in the application |
| | 8. At the end the status becomes "Resolved" (with a reply), "Scheduled for Model Retraining" (a valid chord report that was flagged for retraining), or "Declined" with the reason |
| | 9. The user opens "My Feedback" to see all submissions, their status, replies, and history |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. The user reports from the Edit Marker after correcting a chord: <br>&emsp; The expected chord is pre-filled with the corrected chord |
| | AF2. The user already reported the same bar: <br>&emsp; System links to the existing report instead of creating a duplicate |
| | AF3. The user is offline: <br>&emsp; The submission is queued locally and sent when the connection returns |
| | EF1. The daily limit of 20 submissions is exceeded: <br>&emsp; System rejects the submission (HTTP 429) and shows when the user can submit again |
| | EF2. The attachment has an unsupported type or is larger than 5 MB: <br>&emsp; System rejects the attachment and keeps the rest of the form |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Submission types: wrong-chord report; bug report; feature request; general comment |

---

### USE CASE 20: Edit Marker: Advanced Chord Editing, Forking and Custom Chords

| Field | Details |
| :--- | :--- |
| **USE CASE 20** | **Edit Marker: Advanced Chord Editing, Forking and Custom Chords** |
| **Description** | The user corrects a chord in a four-step workflow: (1) choose a harmonically equivalent name suggested by the system (enharmonic spellings, rootless upper structures, triad over bass); (2) choose how to save the change, as an in-place override or as a fork (a new chart); (3) otherwise pick the chord from the Chord Dictionary by root, quality, tension, alteration, and slash bass; (4) otherwise build a custom chord by selecting pitch classes on a virtual fretboard or keyboard and naming it, after which the Harmonic Engine derives its Roman numeral. Free users keep all changes in local browser storage; Paid users and Students are synchronized to the cloud; a Teacher edits the class master layer. |
| **Used by** | Ingest Audio (Use Case 1), Practice Workspace (Use Case 2) |
| **Preconditions** | User is logged in; A song analysis is open in the workspace. |
| **Success End Condition** | The chord is changed in the chosen target (current layer or a new forked chart); the Roman numeral and scale are recomputed within 50 ms; the AI baseline remains preserved. Changes are stored in the cloud for Paid users and Students, and in local browser storage only for Free users. |
| **Failed End Condition** | An invalid chord is rejected with a message and the chart is unchanged. |
| **Actors** | Primary: Free User, Paid User (Pro), Student; Teacher (master layer). Secondary: Chord Dictionary, Client-Side Harmonic Engine |
| **Trigger** | User clicks a chord cell (or long-presses or right-clicks it). |
| **DESCRIPTION** | **Step & Action** |
| | 1. User clicks a chord cell; the Edit Marker opens and shows the detected chord (for example "Cmaj9"), its pitch classes, and its Roman numeral |
| | 2. Step 1, alternative names: the Engine computes on the client the equivalent views of the same harmony and lists them with their Roman numerals and match type. An Exact match has the same pitch classes (for example "Em7/C" for Cmaj9: E-G-B-D over the bass C). A Partial match omits a chord tone and says which (for example "G/C" omits the third E, so it is a Cmaj9 without 3rd). Enharmonic spellings (for example C#maj7 and Dbmaj7) are ordered by the key signature. The user may click a suggestion |
| | 3. Step 2, save strategy: the user chooses (A) In-place override, which writes to the current editable layer while the AI baseline stays preserved and "Reset to AI" remains available, or (B) Fork as New Chart, which creates a copy in the personal library with a link to the source chart and a version label, while the original chart stays unchanged |
| | 4. Step 3, Chord Dictionary Picker: if no suggestion fits, the user opens the picker and selects Root, Quality, Tension/Extension, Alterations, and Slash Bass; a preview shows the fingering diagram and can be auditioned |
| | 5. Step 4, Custom Chord Creator: if the dictionary has no suitable chord, the user selects notes on the Virtual Fretboard or Keyboard; the system lists the selected pitch classes and proposes a name if one matches, and the user may accept it or type their own symbol |
| | 6. The Harmonic Engine derives the Roman numeral: if the pitch classes match a dictionary chord, its function is used; otherwise the scale degree of the chosen root (or the lowest note) in the song key is combined with the chord class given by its third and fifth, and the chord is marked "custom" |
| | 7. System validates the chord (at least one and at most twelve pitch classes, symbol not empty) and applies it to the chosen target |
| | 8. The Smart Grid shows the new chord and annotations within 50 ms |
| | 9. Persistence: for Paid users and Students the change is sent to the server in the background; for Free users it is written only to local browser storage; a Teacher's edit goes to the master layer and takes effect for Students when published (Use Case 11) |
| **EXTENSIONS** | **Step & Branching Action** |
| | AF1. User selects "Reset to AI": <br>&emsp; System restores the original AI chord from the preserved baseline and recalculates the analysis |
| | AF2. User wants to reuse a custom chord: <br>&emsp; User saves it to the personal chord palette, from which it can be applied in other songs (cloud for Paid users, local for Free users) |
| | AF3. No exact equivalent exists: <br>&emsp; System shows only partial matches or states that there is none, and the user continues with Step 3 |
| | EF1. The chord is invalid (empty note set or unspellable symbol): <br>&emsp; System rejects it with a message and changes nothing |
| | EF2. A Student edits a class chart: <br>&emsp; The change goes to the personal layer only; a direct write to the master layer is rejected (HTTP 403) |
| | EF3. Network fails during a cloud save: <br>&emsp; The change stays in local storage, is queued, and is retried when the connection returns |
| **VARIATIONS** | **Step & Branching Action** |
| | V1. Chord entry: suggestion list; Chord Dictionary Picker; Virtual Fretboard or Keyboard; quick text input |
| | V2. Save target: in-place override; fork as new chart; teacher master layer |
