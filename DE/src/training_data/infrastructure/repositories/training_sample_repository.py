"""TrainingSampleRepository — PostgreSQL access for the ``training_samples`` table."""
from __future__ import annotations
from dataclasses import astuple

from psycopg2.extras import execute_values

from src.shared.infrastructure.postgres.client import get_connection
from src.training_data.domain.models.training_sample import TrainingSample

DDL = """
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
"""

_COLUMNS = (
    "song_id, source_type, source_url, segment_idx, start_ms, end_ms, audio_path, audio_idx, "
    "sample_rate, label_raw, label_status, root_id, quality_id, ext_id, split, align_status"
)


class TrainingSampleRepository:
    """Idempotent persistence of training samples (one song = one atomic replace)."""

    def ensure_table(self) -> None:
        """Create table if missing — init.sql only runs on an empty Postgres volume."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(DDL)

    def source_exists(self, source_url: str) -> bool:
        """True if this source already produced training samples."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM training_samples WHERE source_url = %s LIMIT 1", (source_url,))
                return cur.fetchone() is not None

    def replace_song(self, song_id: str, samples: list[TrainingSample]) -> int:
        """Delete then insert all samples of a song in one transaction. Returns rows inserted."""
        rows = [astuple(s) for s in samples]
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM training_samples WHERE song_id = %s", (song_id,))
                if rows:
                    execute_values(
                        cur,
                        f"INSERT INTO training_samples ({_COLUMNS}) VALUES %s",
                        rows,
                    )
        return len(rows)

    def stats(self) -> list[tuple]:
        """(source_type, split, label_status, count) aggregated — for monitoring."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source_type, split, label_status, COUNT(*) FROM training_samples "
                    "GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
                )
                return cur.fetchall()
