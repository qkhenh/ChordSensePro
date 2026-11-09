### USE CASE 1: Ingest Audio, Analyze Chords and Personalize Harmonic Chart (HITL)

| Field | Details |
| :--- | :--- |
| **USE CASE 1** | **Ingest Audio, Analyze Chords and Personalize Harmonic Chart** |
| **Description** | Musician or learner ingests an audio source (audio file or YouTube URL); the AI pipeline automatically analyzes chords, beats, downbeats, and key, renders an interactive Smart Grid synchronized with Roman numerals and scales, while allowing the user to audit and personalize chord annotations via Human-in-the-Loop (HITL). |
| **Used by** | Practice Workspace & Repertoire Management (Use Case 2) |
| **Preconditions** | User is authenticated; provides a valid audio file (MP3/WAV/M4A) or accessible YouTube URL; backend AI inference worker is operational. |
| **Success End Condition** | System successfully renders an interactive Smart Grid synchronized with audio playback (< 100ms latency), displays full harmonic Roman Numerals and scale modes, and persists the custom chart to the user's personal library. |
| **Failed End Condition** | System notifies audio download/stem separation failure or unsupported media format; aborts pipeline without wasting additional GPU inference resources. |
| **Actors** | Musician / Learner, AI Inference Pipeline (Demucs + Madmom + MERT), YouTube Audio Service |
| **Trigger** | User clicks "Analyze Song" button after uploading an audio file or pasting a YouTube link. |
| **DESCRIPTION** | **Step & Action** |
| | 1. User inputs a YouTube URL or uploads a local audio file and optionally specifies the key center (if known beforehand). |
| | 2. System receives the request, allocates a unique Job ID, and enqueues the task into the background Celery/Redis queue. |
| | 3. AI Pipeline downloads the audio stream, executes source separation (Demucs), beat and downbeat tracking (Madmom), key/tempo extraction, and chord sequence recognition (MERT-v1-330M). |
| | 4. System returns event-based timeline JSONB and song metadata to the client browser. |
| | 5. Client Harmonic Engine computes harmonic Roman Numerals and determines corresponding modal scales for each chord event. |
| | 6. Web browser renders the interactive Smart Grid View synchronized with the Web Audio API player. |
| | 7. User audits the detected progression, clicks on a chord cell requiring adjustment, and opens the Edit Marker modal dialog. |
| | 8. User modifies the Root, Bass, or Chord Quality; the system instantly recalculates Roman Numerals and scale mode in real time. |
| | 9. User clicks to save the customized chord chart to their personal playlist. |
| **EXTENSIONS** | **Step & Branching Action** |
| | 1a. YouTube URL is copyright-restricted, private, or invalid: <br>&emsp; 1a1. System reports source download error and prompts user to upload a local audio file instead. |
| | 3a. AI processing pipeline exceeds timeout threshold: <br>&emsp; 3a1. System marks job status as failed, notifies the user, and refunds user quota. |
| | 7a. User prefers simplified triads over extended jazz voicings: <br>&emsp; 7a1. User toggles "Basic View", and the client instantly down-maps chords (e.g., G9sus4 → G) locally without re-querying the backend. |
| **VARIATIONS** | **Step & Branching Action** |
| | 1. User can provide audio sources via: <br>&emsp; - Local file upload (MP3, WAV, M4A) <br>&emsp; - Pasting YouTube video/music URL <br>&emsp; - Selecting a demo track from the system catalog |
| | 7. User can edit chords via: <br>&emsp; - Visual dropdown picker modal <br>&emsp; - Virtual Fretboard (Guitar) or Piano Keyboard interface <br>&emsp; - Keyboard hotkeys / quick text input |

---

### USE CASE 2: Practice with Interactive Workspace (Smart Grid, A/B Loop & Metronome)

| Field | Details |
| :--- | :--- |
| **USE CASE 2** | **Practice with Interactive Workspace (Smart Grid, A/B Loop & Metronome)** |
| **Description** | Musician or learner uses the interactive workspace to practice songs on the Smart Grid, loop difficult technical passages (A/B Looping), enable dynamic metronome clicks, and shift playback speed without pitch distortion. |
| **Used by** | None (Top-level core feature) |
| **Preconditions** | Song has been analyzed and loaded into Workspace; audio stream and chord timeline events are fully ready. |
| **Success End Condition** | User practices seamlessly; audio playback, metronome click, and visual grid highlight remain synchronized in real time (< 100ms latency); A/B loop executes accurately. |
| **Failed End Condition** | Browser experiences Virtual DOM lag or fails to decode Web Audio API; system gracefully falls back to basic HTML5 Audio playback. |
| **Actors** | Musician / Learner, Web Audio API Engine |
| **Trigger** | User clicks the "Play" button or clicks directly on a bar cell on the Smart Grid. |
| **DESCRIPTION** | **Step & Action** |
| | 1. User opens the song's Practice Workspace view. |
| | 2. System initializes the Web Audio Context, loads `tempo` and `first_beat_offset_ms` to interpolate the Bar Grid. |
| | 3. User presses Play (or Spacebar); system initiates audio playback and triggers `requestAnimationFrame` loop to highlight the active bar cell. |
| | 4. User toggles the Dynamic Metronome; system generates synthesized click pulses aligned with the track's original BPM. |
| | 5. User selects Bar 9 through Bar 16 to set an A/B Loop Range. |
| | 6. System constrains playback boundaries; when the playback playhead reaches point B, the player automatically seeks back to point A within 100ms. |
| | 7. User scales playback speed down to 0.75x; Web Audio Engine adjusts `playbackRate` using pitch-preserving time-stretching. |
| **EXTENSIONS** | **Step & Branching Action** |
| | 2a. Browser blocks automatic audio playback (AudioContext Autoplay Policy): <br>&emsp; 2a1. System displays an interactive modal requesting user click to unlock AudioContext. |
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
| | 6a. User connects an external MIDI controller: <br>&emsp; 6a1. Web MIDI API captures physical key presses and verifies accuracy against the target voicing. |
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
| | 4a. User saves a track containing user-edited chords (Custom Chart): <br>&emsp; 4a1. System forks a user-specific `song_charts` record with `is_customized = true` tied to the user account, preserving the public baseline intact. |
| | 6a. Server connection fails during setlist reordering: <br>&emsp; 6a1. Interface alerts user of offline mode, stores payload in LocalStorage, and automatically executes sync retry on network reconnection. |
| **VARIATIONS** | **Step & Branching Action** |
| | 2. Playlists can be created from: <br>&emsp; - An empty personal playlist <br>&emsp; - Cloning a shared public community playlist <br>&emsp; - Auto-generation by genre or musical key |

---

### USE CASE 5: Manage & Standardize Chord Dictionary & Voicings (Business Admin)

| Field | Details |
| :--- | :--- |
| **USE CASE 5** | **Manage & Standardize Chord Dictionary & Voicings** |
| **Description** | Music Content Manager / Curriculum Lead adds, edits, standardizes, and sound-checks chord voicings (Guitar fretboard layouts, Piano voicings) in the system-wide core reference library. |
| **Used by** | Curriculum & Repertoire Governance |
| **Preconditions** | User is authenticated with `Business Admin` credentials (`Music Content Manager` or `Curriculum Lead`). |
| **Success End Condition** | New chord entry, fingering JSONB schema, and audio soundfont asset are validated and committed to `chord_dictionary`, propagating system-wide. |
| **Failed End Condition** | Fingering geometry is physically unplayable on instruments; system rejects submission with a biometric constraint error. |
| **Actors** | Business Admin (Music Content Manager / Curriculum Lead), PostgreSQL Database |
| **Trigger** | Admin accesses the Content Management Portal and selects "Chord Dictionary Management". |
| **DESCRIPTION** | **Step & Action** |
| | 1. Admin logs into the Content Admin Portal. |
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

### USE CASE 6: Review, Verify & Publish Public Song Charts (Business Admin)

| Field | Details |
| :--- | :--- |
| **USE CASE 6** | **Review, Verify & Publish Public Song Charts** |
| **Description** | Music Content Manager reviews AI chord recognition accuracy on prominent songs, refines harmonic notations, and approves publication into the Verified Public Catalog for community use and school curriculum. |
| **Used by** | Curriculum & Repertoire Governance |
| **Preconditions** | User holds `Business Admin` privileges; target song has completed preliminary AI pipeline inference. |
| **Success End Condition** | Song chart is marked `is_verified = true`, published to the public search index, and linked to educational curriculum tracks. |
| **Failed End Condition** | Track audio quality is excessively degraded or rhythmically corrupted beyond repair; song is rejected from publication. |
| **Actors** | Business Admin (Music Content Manager / Curriculum Lead), AI Inference Service |
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

### USE CASE 7: Monitor AI Worker Queues, GPU Workload & Pipeline Performance (IT Admin)

| Field | Details |
| :--- | :--- |
| **USE CASE 7** | **Monitor AI Worker Queues, GPU Workload & Pipeline Performance** |
| **Description** | System & AI Operations Administrator monitors operational health of audio processing workers, GPU VRAM and CPU utilization, Celery/Redis queue backlogs, and pipeline SLA throughput. |
| **Used by** | Infrastructure & DevOps Governance |
| **Preconditions** | User is authenticated with `IT Admin / CIO` privileges; telemetry and background worker daemons are running. |
| **Success End Condition** | Dashboard reflects real-time telemetry (throughput, latency, queue backlog, GPU VRAM consumption); admin can trigger worker autoscaling. |
| **Failed End Condition** | Telemetry collector service (Prometheus / Metrics Server) becomes unreachable; dashboard displays telemetry connection failure. |
| **Actors** | IT Admin / CIO (System & AI Operations Administrator), Celery/Redis Job Queue, GPU Worker Nodes |
| **Trigger** | Admin opens the DevOps Monitoring Dashboard or receives an automated queue congestion alert. |
| **DESCRIPTION** | **Step & Action** |
| | 1. IT Admin opens the System & AI Operations Dashboard. |
| | 2. System renders real-time throughput metrics: active jobs in queue, currently executing tasks, and average processing latency. |
| | 3. Admin monitors pipeline sub-stages: YouTube streaming (`yt-dlp`), Stem separation (`Demucs`), Beat tracking (`Madmom`), and Chord inference (`MERT-v1-330M`). |
| | 4. Admin identifies bottleneck in the Demucs stem separation queue caused by sudden traffic spike. |
| | 5. Admin triggers GPU Worker autoscaling command. |
| | 6. System initializes supplementary worker container instances and rebalances task dispatching. |
| | 7. Queue backlog stabilizes to nominal levels (< 5 pending jobs); processing latency returns within SLA threshold (< 30 seconds per song). |
| **EXTENSIONS** | **Step & Branching Action** |
| | 3a. A worker task enters deadlock (Zombie Job) due to GPU CUDA Out-of-Memory: <br>&emsp; 3a1. Admin clicks "Force Terminate & Re-queue", releasing VRAM and resubmitting task with a smaller audio chunking window. |
| | 4a. Server egress IP receives HTTP 429 Too Many Requests from streaming provider: <br>&emsp; 4a1. System automatically rotates egress proxy IP pool and alerts admin on dashboard. |
| **VARIATIONS** | **Step & Branching Action** |
| | 5. Admin can scale workers via: <br>&emsp; - Manual scale slider on Web Ops Dashboard <br>&emsp; - Automated threshold-based scaling policies (Kubernetes HPA / Docker Swarm) |

---

### USE CASE 8: Manage System Users, Access Roles, Rate Limits & Audit Logs (IT Admin)

| Field | Details |
| :--- | :--- |
| **USE CASE 8** | **Manage System Users, Access Roles, Rate Limits & Audit Logs** |
| **Description** | IT Administrator manages user accounts, provisions role-based access control (Musician, Student, Business Admin, IT Admin), enforces ingestion rate limits, and inspects security audit logs. |
| **Used by** | Infrastructure & DevOps Governance |
| **Preconditions** | User is authenticated with `IT Admin / CIO` role permissions. |
| **Success End Condition** | Account privileges and API quota configurations are applied; all administrative mutations are recorded in the PostgreSQL audit log. |
| **Failed End Condition** | Security constraint violation (e.g., attempt to delete or demote the last remaining Root Superadmin); system prevents action. |
| **Actors** | IT Admin / CIO (System & AI Operations Administrator), PostgreSQL Database |
| **Trigger** | Admin opens the "User & Security Management" panel within the IT Admin Console. |
| **DESCRIPTION** | **Step & Action** |
| | 1. Admin opens user account directory in the administrative console. |
| | 2. Admin locates user account requiring role elevation (e.g., promoting a faculty instructor to `Business Admin`). |
| | 3. Admin assigns target role and scopes permissions (Content Management, Dictionary Editing). |
| | 4. Admin configures quota policy (e.g., Free Tier: 5 songs/day, Teacher Tier: unlimited ingestion). |
| | 5. Admin reviews Security & Audit Logs for suspicious access patterns (failed login brute force, API rate-limit abuse). |
| | 6. Admin clicks "Save Changes"; system commits updates and appends an immutable audit log entry. |
| **EXTENSIONS** | **Step & Branching Action** |
| | 2a. Account exhibits abusive behavior or API scraping: <br>&emsp; 2a1. Admin executes "Suspend User" action, revoking active JWT sessions and API keys. |
| **VARIATIONS** | **Step & Branching Action** |
| | 5. Audit logs can be filtered by: <br>&emsp; - Event timestamp range <br>&emsp; - Source IP address and User ID <br>&emsp; - Event category (Authentication, Role Elevation, Ingestion Abuse) |
