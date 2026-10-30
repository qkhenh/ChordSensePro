from __future__ import annotations

from psycopg2.extras import execute_values

from src.shared.infrastructure.postgres.client import get_connection
from src.training_data.domain.models.training_section import TrainingSection, TrainingChord

_SECTION_COLUMNS = (
    "song_id, source_type, source_url, audio_path, raw_audio_path, "
    "duration_ms, align_offset_ms, align_score, align_status, split, pipeline_version"
)

_CHORD_COLUMNS = (
    "song_id, start_ms, end_ms, label_raw, label_status, root_id, quality_id, ext_id"
)

DDL = """
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
"""


class TrainingSectionRepository:

    def ensure_tables(self) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(DDL)

    def source_exists(self, source_url: str) -> bool:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM training_sections WHERE source_url = %s LIMIT 1", (source_url,))
                return cur.fetchone() is not None

    def replace_song(self, song_id: str, section: TrainingSection, chords: list[TrainingChord]) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM training_chords WHERE song_id = %s", (song_id,))
                cur.execute("DELETE FROM training_sections WHERE song_id = %s", (song_id,))

                cur.execute(
                    f"INSERT INTO training_sections ({_SECTION_COLUMNS}) VALUES "
                    f"(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        section.song_id, section.source_type, section.source_url,
                        section.audio_path, section.raw_audio_path, section.duration_ms,
                        section.align_offset_ms, section.align_score, section.align_status,
                        section.split, section.pipeline_version,
                    ),
                )

                if chords:
                    rows = [
                        (c.song_id, c.start_ms, c.end_ms, c.label_raw,
                         c.label_status, c.root_id, c.quality_id, c.ext_id)
                        for c in chords
                    ]
                    execute_values(
                        cur,
                        f"INSERT INTO training_chords ({_CHORD_COLUMNS}) VALUES %s",
                        rows,
                    )
        return 1 + len(chords)

    def stats(self) -> list[tuple]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source_type, split, align_status, COUNT(*) FROM training_sections "
                    "GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
                )
                return cur.fetchall()

