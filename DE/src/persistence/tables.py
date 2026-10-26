"""Gateway tables (PostgreSQL schema ``app``), declared with SQLAlchemy Core.

Migrations in ``migrations/versions`` are written by hand and frozen; an integration
test compares the migrated database with this metadata so the two cannot drift.
Charts are read and written as whole JSONB documents (event-based timeline, never a
bar grid), layered into an immutable AI baseline and a personal override layer.
"""

from sqlalchemy import (
    REAL,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import CHAR, JSONB, UUID

SCHEMA = "app"

metadata = MetaData(schema=SCHEMA)

ROLES = ("free", "paid", "teacher", "student", "expert", "music_pm", "it_admin")
SOURCE_KINDS = ("file", "url", "recording", "demo")
JOB_STATES = ("queued", "processing", "done", "failed")
JOB_STAGES = ("fetching", "normalizing", "separating", "beat_tracking", "recognizing")
INSTRUMENTS = ("guitar", "piano", "bass", "ukulele")
ROOTS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SHA256_HEX = r"^[0-9a-f]{64}$"


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _uuid_pk() -> Column:
    return Column(
        "id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


def _created() -> Column:
    return Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now())


def _updated() -> Column:
    return Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now())


users = Table(
    "users",
    metadata,
    _uuid_pk(),
    Column("email", String(320), nullable=False),
    Column("name", String(200)),
    Column("password_hash", String(255), nullable=False),
    Column("role", String(20), nullable=False, server_default="free"),
    Column("is_active", Boolean, nullable=False, server_default=text("true")),
    _created(),
    _updated(),
    CheckConstraint(_in("role", ROLES), name="ck_users_role"),
    Index("ux_users_email_lower", func.lower(text("email")), unique=True),
)

# Shared AI results keyed by audio content hash: a repeated upload is served from here
# without consuming quota (UC1 step 4).
analyses = Table(
    "analyses",
    metadata,
    Column("content_hash", CHAR(64), primary_key=True),
    Column("result", JSONB, nullable=False),
    Column("model_version", String(100), nullable=False),
    Column("pipeline_version", String(100), nullable=False),
    _created(),
    CheckConstraint(f"content_hash ~ '{SHA256_HEX}'", name="ck_analyses_content_hash"),
)

# Cheap pre-decode cache key: sha256 of the raw upload -> canonical content_hash.
source_fingerprints = Table(
    "source_fingerprints",
    metadata,
    Column("fingerprint", CHAR(64), primary_key=True),
    Column(
        "content_hash",
        CHAR(64),
        ForeignKey(
            f"{SCHEMA}.analyses.content_hash",
            ondelete="CASCADE",
            name="fk_source_fingerprints_content_hash_analyses",
        ),
        nullable=False,
    ),
    _created(),
    CheckConstraint(f"fingerprint ~ '{SHA256_HEX}'", name="ck_source_fingerprints_fingerprint"),
)

charts = Table(
    "charts",
    metadata,
    _uuid_pk(),
    Column(
        "owner_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.users.id", ondelete="CASCADE", name="fk_charts_owner_id_users"),
        nullable=False,
    ),
    Column("title", String(300), nullable=False),
    Column("artist", String(300)),
    Column("source", JSONB, nullable=False),
    Column("metadata", JSONB, nullable=False),
    Column("ai_events", JSONB, nullable=False),
    Column("overrides", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("is_customized", Boolean, nullable=False, server_default=text("false")),
    Column(
        "forked_from",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.charts.id", ondelete="SET NULL", name="fk_charts_forked_from_charts"),
    ),
    Column("content_hash", CHAR(64), nullable=False),
    Column("version", Integer, nullable=False, server_default=text("1")),
    _created(),
    _updated(),
    CheckConstraint("version >= 1", name="ck_charts_version"),
    CheckConstraint(f"content_hash ~ '{SHA256_HEX}'", name="ck_charts_content_hash"),
    Index("ix_charts_owner_updated", "owner_id", "updated_at"),
    Index("ix_charts_content_hash", "content_hash"),
    Index("ix_charts_forked_from", "forked_from"),
)

jobs = Table(
    "jobs",
    metadata,
    _uuid_pk(),
    Column(
        "owner_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.users.id", ondelete="CASCADE", name="fk_jobs_owner_id_users"),
        nullable=False,
    ),
    Column("source_kind", String(20), nullable=False),
    Column("source_url", Text),
    Column("title", String(300), nullable=False, server_default="Untitled"),
    Column("input_path", Text),  # raw upload awaiting normalisation; deleted afterwards
    Column("fingerprint", CHAR(64)),  # sha256 of the uploaded bytes
    Column("content_hash", CHAR(64)),
    Column("state", String(20), nullable=False, server_default="queued"),
    Column("stage", String(20)),
    Column("progress", REAL, nullable=False, server_default=text("0")),
    Column("error", JSONB),
    Column(
        "result_chart_id",
        UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.charts.id", ondelete="SET NULL", name="fk_jobs_result_chart_id_charts"
        ),
    ),
    # Quota is charged on acceptance and refunded once on failure/timeout (UC1 EF1).
    Column("quota_charged", Boolean, nullable=False, server_default=text("true")),
    Column("charged_on", Date),  # UTC day the charge was taken from (refund goes back there)
    _created(),
    _updated(),
    Column("finished_at", DateTime(timezone=True)),
    CheckConstraint(_in("source_kind", SOURCE_KINDS), name="ck_jobs_source_kind"),
    CheckConstraint(_in("state", JOB_STATES), name="ck_jobs_state"),
    CheckConstraint("stage IS NULL OR " + _in("stage", JOB_STAGES), name="ck_jobs_stage"),
    CheckConstraint("progress >= 0 AND progress <= 1", name="ck_jobs_progress"),
    CheckConstraint("(state = 'failed') = (error IS NOT NULL)", name="ck_jobs_error_iff_failed"),
    CheckConstraint(
        f"fingerprint IS NULL OR fingerprint ~ '{SHA256_HEX}'", name="ck_jobs_fingerprint"
    ),
    Index("ix_jobs_owner_created", "owner_id", "created_at"),
    Index(
        "ix_jobs_active",
        "state",
        "created_at",
        postgresql_where=text("state IN ('queued', 'processing')"),
    ),
)

usage_counters = Table(
    "usage_counters",
    metadata,
    Column(
        "user_id",
        UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.users.id", ondelete="CASCADE", name="fk_usage_counters_user_id_users"
        ),
        primary_key=True,
    ),
    Column("period_date", Date, primary_key=True),
    Column("analyses_used", Integer, nullable=False, server_default=text("0")),
    CheckConstraint("analyses_used >= 0", name="ck_usage_counters_analyses_used"),
)

playlists = Table(
    "playlists",
    metadata,
    _uuid_pk(),
    Column(
        "owner_id",
        UUID(as_uuid=True),
        ForeignKey(f"{SCHEMA}.users.id", ondelete="CASCADE", name="fk_playlists_owner_id_users"),
        nullable=False,
    ),
    Column("name", String(200), nullable=False),
    _created(),
    _updated(),
    Index("ix_playlists_owner", "owner_id"),
)

playlist_items = Table(
    "playlist_items",
    metadata,
    Column(
        "playlist_id",
        UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.playlists.id",
            ondelete="CASCADE",
            name="fk_playlist_items_playlist_id_playlists",
        ),
        primary_key=True,
    ),
    Column(
        "chart_id",
        UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.charts.id", ondelete="CASCADE", name="fk_playlist_items_chart_id_charts"
        ),
        primary_key=True,
    ),
    Column("order_index", Integer, nullable=False),
    CheckConstraint("order_index >= 0", name="ck_playlist_items_order_index"),
    # Deferred so a reorder (swap) inside one transaction does not trip uniqueness.
    UniqueConstraint(
        "playlist_id",
        "order_index",
        name="uq_playlist_items_order",
        deferrable=True,
        initially="DEFERRED",
    ),
)

chord_dictionary = Table(
    "chord_dictionary",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("root", String(2), nullable=False),
    Column("quality", String(20), nullable=False),
    Column("extension", String(20), nullable=False, server_default="none"),
    Column("instrument", String(20), nullable=False),
    Column("variation", Integer, nullable=False, server_default=text("0")),
    Column("fingering", JSONB, nullable=False),
    Column("audio_midi", JSONB),
    _created(),
    CheckConstraint(_in("root", ROOTS), name="ck_chord_dictionary_root"),
    CheckConstraint(_in("instrument", INSTRUMENTS), name="ck_chord_dictionary_instrument"),
    CheckConstraint("variation >= 0", name="ck_chord_dictionary_variation"),
    UniqueConstraint(
        "root",
        "quality",
        "extension",
        "instrument",
        "variation",
        name="uq_chord_dictionary_voicing",
    ),
    Index("ix_chord_dictionary_search", "root", "quality", "instrument"),
)

__all__ = [
    "SCHEMA",
    "analyses",
    "charts",
    "chord_dictionary",
    "jobs",
    "metadata",
    "playlist_items",
    "playlists",
    "usage_counters",
    "users",
]
