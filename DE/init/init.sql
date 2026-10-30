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
    label_status  VARCHAR(12)  NOT NULL,
    root_id       SMALLINT,
    quality_id    SMALLINT,
    ext_id        SMALLINT,
    split         VARCHAR(5)   NOT NULL,
    align_status  VARCHAR(20)  NOT NULL DEFAULT 'n/a',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (song_id, segment_idx)
);
CREATE INDEX IF NOT EXISTS idx_ts_song   ON training_samples(song_id);
CREATE INDEX IF NOT EXISTS idx_ts_split  ON training_samples(split);
CREATE INDEX IF NOT EXISTS idx_ts_status ON training_samples(label_status);
CREATE INDEX IF NOT EXISTS idx_ts_url    ON training_samples(source_url);
TRUNCATE TABLE training_samples;

CREATE TABLE IF NOT EXISTS training_sections (
    song_id          VARCHAR(36)  PRIMARY KEY,
    source_type      VARCHAR(20)  NOT NULL,
    source_url       TEXT         NOT NULL,
    audio_path       TEXT         NOT NULL,
    raw_audio_path   TEXT,
    duration_ms      INTEGER      NOT NULL,
    align_offset_ms  INTEGER      NOT NULL DEFAULT 0,
    align_score      REAL,
    align_status     VARCHAR(20)  NOT NULL,
    split            VARCHAR(5)   NOT NULL,
    pipeline_version SMALLINT     NOT NULL DEFAULT 2,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS training_chords (
    id           BIGSERIAL    PRIMARY KEY,
    song_id      VARCHAR(36)  NOT NULL REFERENCES training_sections(song_id),
    start_ms     INTEGER      NOT NULL,
    end_ms       INTEGER      NOT NULL,
    label_raw    VARCHAR(64)  NOT NULL,
    label_status VARCHAR(12)  NOT NULL,
    root_id      SMALLINT,
    quality_id   SMALLINT,
    ext_id       SMALLINT
);
CREATE INDEX IF NOT EXISTS idx_tc_song ON training_chords(song_id);

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
