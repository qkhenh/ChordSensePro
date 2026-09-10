-- =============================================================
-- ChordSensePro DE — PostgreSQL init script
-- Auto-executed by Docker on first start (empty volume).
-- =============================================================

-- ── crawl_queue: URLs to be processed by DAG 1 ──────────────
CREATE TABLE IF NOT EXISTS crawl_queue (
    id              VARCHAR(36) PRIMARY KEY,
    source_url      TEXT UNIQUE NOT NULL,
    annotation_url  TEXT DEFAULT '',
    source_type     VARCHAR(20) DEFAULT 'local' NOT NULL,
    status          VARCHAR(20) DEFAULT 'pending' NOT NULL,
    priority        INTEGER DEFAULT 0 NOT NULL,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_crawl_status ON crawl_queue(status);
CREATE INDEX IF NOT EXISTS idx_crawl_url ON crawl_queue(source_url);

-- ── song_analyses: final output from DAG 2 ──────────────────
CREATE TABLE IF NOT EXISTS song_analyses (
    id              VARCHAR(36) PRIMARY KEY,
    user_id         VARCHAR(36) DEFAULT '',
    source_url      TEXT NOT NULL,
    song_title      VARCHAR(200) DEFAULT '',
    status          VARCHAR(20) DEFAULT 'pending',
    detected_key    VARCHAR(10),
    tempo_bpm       REAL,
    chord_timeline  JSONB DEFAULT '[]',
    chord_sheet     JSONB DEFAULT '{}',
    learning_plan   JSONB DEFAULT '[]',
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sa_url ON song_analyses(source_url);
CREATE INDEX IF NOT EXISTS idx_sa_status ON song_analyses(status);

-- ── chord_attempts: user practice attempts ───────────────────
CREATE TABLE IF NOT EXISTS chord_attempts (
    id              VARCHAR(36) PRIMARY KEY,
    user_id         VARCHAR(36) NOT NULL,
    chord_label     VARCHAR(20) NOT NULL,
    is_correct      BOOLEAN NOT NULL,
    confidence      REAL DEFAULT 0,
    response_time   REAL DEFAULT 0,
    attempted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ca_user ON chord_attempts(user_id);

-- ── user_chord_mastery: aggregated mastery per chord ─────────
CREATE TABLE IF NOT EXISTS user_chord_mastery (
    user_id         VARCHAR(36) NOT NULL,
    chord_label     VARCHAR(20) NOT NULL,
    mastery_level   REAL DEFAULT 0,
    total_attempts  INTEGER DEFAULT 0,
    correct_count   INTEGER DEFAULT 0,
    last_practiced  DATE,
    is_mastered     BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (user_id, chord_label)
);
