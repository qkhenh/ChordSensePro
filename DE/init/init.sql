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

-- == training_samples: OUTPUT of the training ETL (DAG 2) ==============================
-- One row = one labeled 2 s segment. Audio lives in an int16 .npy on the training volume
-- (audio_path + audio_idx); labels are the class ids of the 3 model heads.
-- Same DDL is run by TrainingSampleRepository.ensure_table() because this file only runs
-- on an empty Postgres volume.
CREATE TABLE IF NOT EXISTS training_samples (
    id            BIGSERIAL PRIMARY KEY,
    song_id       VARCHAR(36)  NOT NULL,
    source_type   VARCHAR(20)  NOT NULL,
    source_url    TEXT         NOT NULL,
    segment_idx   INTEGER      NOT NULL,
    start_ms      INTEGER      NOT NULL,
    end_ms        INTEGER      NOT NULL,
    audio_path    TEXT         NOT NULL,
    audio_idx     INTEGER      NOT NULL,
    sample_rate   INTEGER      NOT NULL DEFAULT 24000,
    label_raw     VARCHAR(64)  NOT NULL,
    label_status  VARCHAR(12)  NOT NULL,   -- ok | partial | no_chord | unmapped
    root_id       SMALLINT,                -- 0-11
    quality_id    SMALLINT,                -- 0-4
    ext_id        SMALLINT,                -- 0-14
    split         VARCHAR(5)   NOT NULL,   -- train | val | test (hash of source_url)
    align_status  VARCHAR(20)  NOT NULL DEFAULT 'n/a',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (song_id, segment_idx)
);
CREATE INDEX IF NOT EXISTS idx_ts_song   ON training_samples(song_id);
CREATE INDEX IF NOT EXISTS idx_ts_split  ON training_samples(split);
CREATE INDEX IF NOT EXISTS idx_ts_status ON training_samples(label_status);
CREATE INDEX IF NOT EXISTS idx_ts_url    ON training_samples(source_url);
-- NOTE: song_analyses below is no longer written by the training ETL (parked for the system side).


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

