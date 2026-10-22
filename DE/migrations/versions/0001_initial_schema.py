"""Initial gateway schema (schema ``app``).

Revision ID: 0001
Revises:
Create Date: 2026-10-22

Hand-written and frozen: later changes go in new revisions, never edit this one.
Compatible with PostgreSQL 16 (compose/prod) and 18 (dev).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

S = "app"


def _ts(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


def _uuid_pk() -> sa.Column:
    return sa.Column(
        "id", pg.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")
    )


def _fk(table: str, name: str, ondelete: str) -> sa.ForeignKey:
    return sa.ForeignKey(f"{S}.{table}.id", ondelete=ondelete, name=name)


def upgrade() -> None:
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {S}")

    op.create_table(
        "users",
        _uuid_pk(),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("name", sa.String(200)),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="free"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint(
            "role IN ('free', 'paid', 'teacher', 'student', 'expert', 'music_pm', 'it_admin')",
            name="ck_users_role",
        ),
        schema=S,
    )
    op.execute(f"CREATE UNIQUE INDEX ux_users_email_lower ON {S}.users (lower(email))")

    op.create_table(
        "analyses",
        sa.Column("content_hash", pg.CHAR(64), primary_key=True),
        sa.Column("result", pg.JSONB, nullable=False),
        sa.Column("model_version", sa.String(100), nullable=False),
        sa.Column("pipeline_version", sa.String(100), nullable=False),
        _ts("created_at"),
        sa.CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="ck_analyses_content_hash"),
        schema=S,
    )

    op.create_table(
        "charts",
        _uuid_pk(),
        sa.Column(
            "owner_id",
            pg.UUID(as_uuid=True),
            _fk("users", "fk_charts_owner_id_users", "CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("artist", sa.String(300)),
        sa.Column("source", pg.JSONB, nullable=False),
        sa.Column("metadata", pg.JSONB, nullable=False),
        sa.Column("ai_events", pg.JSONB, nullable=False),
        sa.Column("overrides", pg.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("is_customized", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column(
            "forked_from",
            pg.UUID(as_uuid=True),
            _fk("charts", "fk_charts_forked_from_charts", "SET NULL"),
        ),
        sa.Column("content_hash", pg.CHAR(64), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default=sa.text("1")),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint("version >= 1", name="ck_charts_version"),
        sa.CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="ck_charts_content_hash"),
        schema=S,
    )
    op.create_index("ix_charts_owner_updated", "charts", ["owner_id", "updated_at"], schema=S)
    op.create_index("ix_charts_content_hash", "charts", ["content_hash"], schema=S)
    op.create_index("ix_charts_forked_from", "charts", ["forked_from"], schema=S)

    op.create_table(
        "jobs",
        _uuid_pk(),
        sa.Column(
            "owner_id",
            pg.UUID(as_uuid=True),
            _fk("users", "fk_jobs_owner_id_users", "CASCADE"),
            nullable=False,
        ),
        sa.Column("source_kind", sa.String(20), nullable=False),
        sa.Column("source_url", sa.Text),
        sa.Column("content_hash", pg.CHAR(64)),
        sa.Column("state", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("stage", sa.String(20)),
        sa.Column("progress", sa.REAL, nullable=False, server_default=sa.text("0")),
        sa.Column("error", pg.JSONB),
        sa.Column(
            "result_chart_id",
            pg.UUID(as_uuid=True),
            _fk("charts", "fk_jobs_result_chart_id_charts", "SET NULL"),
        ),
        sa.Column("quota_charged", sa.Boolean, nullable=False, server_default=sa.text("true")),
        _ts("created_at"),
        _ts("updated_at"),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "source_kind IN ('file', 'url', 'recording', 'demo')", name="ck_jobs_source_kind"
        ),
        sa.CheckConstraint(
            "state IN ('queued', 'processing', 'done', 'failed')", name="ck_jobs_state"
        ),
        sa.CheckConstraint(
            "stage IS NULL OR stage IN "
            "('fetching', 'normalizing', 'separating', 'beat_tracking', 'recognizing')",
            name="ck_jobs_stage",
        ),
        sa.CheckConstraint("progress >= 0 AND progress <= 1", name="ck_jobs_progress"),
        sa.CheckConstraint(
            "(state = 'failed') = (error IS NOT NULL)", name="ck_jobs_error_iff_failed"
        ),
        schema=S,
    )
    op.create_index("ix_jobs_owner_created", "jobs", ["owner_id", "created_at"], schema=S)
    op.create_index(
        "ix_jobs_active",
        "jobs",
        ["state", "created_at"],
        schema=S,
        postgresql_where=sa.text("state IN ('queued', 'processing')"),
    )

    op.create_table(
        "usage_counters",
        sa.Column(
            "user_id",
            pg.UUID(as_uuid=True),
            _fk("users", "fk_usage_counters_user_id_users", "CASCADE"),
            primary_key=True,
        ),
        sa.Column("period_date", sa.Date, primary_key=True),
        sa.Column("analyses_used", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.CheckConstraint("analyses_used >= 0", name="ck_usage_counters_analyses_used"),
        schema=S,
    )

    op.create_table(
        "playlists",
        _uuid_pk(),
        sa.Column(
            "owner_id",
            pg.UUID(as_uuid=True),
            _fk("users", "fk_playlists_owner_id_users", "CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        _ts("created_at"),
        _ts("updated_at"),
        schema=S,
    )
    op.create_index("ix_playlists_owner", "playlists", ["owner_id"], schema=S)

    op.create_table(
        "playlist_items",
        sa.Column(
            "playlist_id",
            pg.UUID(as_uuid=True),
            _fk("playlists", "fk_playlist_items_playlist_id_playlists", "CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "chart_id",
            pg.UUID(as_uuid=True),
            _fk("charts", "fk_playlist_items_chart_id_charts", "CASCADE"),
            primary_key=True,
        ),
        sa.Column("order_index", sa.Integer, nullable=False),
        sa.CheckConstraint("order_index >= 0", name="ck_playlist_items_order_index"),
        sa.UniqueConstraint(
            "playlist_id",
            "order_index",
            name="uq_playlist_items_order",
            deferrable=True,
            initially="DEFERRED",
        ),
        schema=S,
    )

    op.create_table(
        "chord_dictionary",
        sa.Column("id", sa.String(80), primary_key=True),
        sa.Column("root", sa.String(2), nullable=False),
        sa.Column("quality", sa.String(20), nullable=False),
        sa.Column("extension", sa.String(20), nullable=False, server_default="none"),
        sa.Column("instrument", sa.String(20), nullable=False),
        sa.Column("variation", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("fingering", pg.JSONB, nullable=False),
        sa.Column("audio_midi", pg.JSONB),
        _ts("created_at"),
        sa.CheckConstraint(
            "root IN ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')",
            name="ck_chord_dictionary_root",
        ),
        sa.CheckConstraint(
            "instrument IN ('guitar', 'piano', 'bass', 'ukulele')",
            name="ck_chord_dictionary_instrument",
        ),
        sa.CheckConstraint("variation >= 0", name="ck_chord_dictionary_variation"),
        sa.UniqueConstraint(
            "root",
            "quality",
            "extension",
            "instrument",
            "variation",
            name="uq_chord_dictionary_voicing",
        ),
        schema=S,
    )
    op.create_index(
        "ix_chord_dictionary_search",
        "chord_dictionary",
        ["root", "quality", "instrument"],
        schema=S,
    )


def downgrade() -> None:
    # The schema itself is left in place: it may pre-exist and hold other objects.
    for table in (
        "chord_dictionary",
        "playlist_items",
        "playlists",
        "usage_counters",
        "jobs",
        "charts",
        "analyses",
        "users",
    ):
        op.drop_table(table, schema=S)
