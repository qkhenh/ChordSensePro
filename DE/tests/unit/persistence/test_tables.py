"""Static shape of the gateway schema (no database needed)."""

from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import JSONB

from src.persistence.tables import SCHEMA, metadata

EXPECTED_TABLES = {
    "users",
    "analyses",
    "charts",
    "jobs",
    "usage_counters",
    "playlists",
    "playlist_items",
    "chord_dictionary",
}


def test_all_gateway_tables_live_in_the_app_schema():
    assert SCHEMA == "app"
    assert {t.name for t in metadata.tables.values()} == EXPECTED_TABLES
    assert {t.schema for t in metadata.tables.values()} == {"app"}


def test_every_table_has_a_primary_key():
    for table in metadata.tables.values():
        assert table.primary_key.columns, table.name


def test_chart_layers_are_stored_as_jsonb():
    charts = metadata.tables["app.charts"]
    for column in ("source", "metadata", "ai_events", "overrides"):
        assert isinstance(charts.c[column].type, JSONB), column


def test_every_constraint_is_named_so_migrations_can_drop_it():
    for table in metadata.tables.values():
        for constraint in table.constraints:
            if isinstance(constraint, PrimaryKeyConstraint):
                continue  # PostgreSQL names these <table>_pkey deterministically
            assert constraint.name, f"{table.name}: {constraint!r}"
        for index in table.indexes:
            assert index.name, f"{table.name}: {index!r}"


def test_foreign_keys_stay_inside_the_app_schema():
    for table in metadata.tables.values():
        for fk in table.foreign_keys:
            assert fk.column.table.schema == "app", fk
