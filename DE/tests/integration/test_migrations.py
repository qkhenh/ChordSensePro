"""Alembic migrations against a real PostgreSQL (TEST_DATABASE_URL, database *_test)."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from src.persistence.tables import metadata

pytestmark = pytest.mark.integration

DE_DIR = Path(__file__).parents[2]
APP_TABLES = {
    "users",
    "analyses",
    "charts",
    "jobs",
    "usage_counters",
    "playlists",
    "playlist_items",
    "chord_dictionary",
    "source_fingerprints",
}


@pytest.fixture(scope="module")
def url() -> str:
    value = os.environ["TEST_DATABASE_URL"]
    assert value.rsplit("/", 1)[-1].endswith("_test"), "refusing to migrate a non-test database"
    return value


@pytest.fixture
def alembic_cfg(url) -> Config:
    cfg = Config(str(DE_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(DE_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


@pytest.fixture
def db(url, alembic_cfg):
    """Fresh schema for every test: base -> head."""
    command.downgrade(alembic_cfg, "base")
    command.upgrade(alembic_cfg, "head")

    def run(*statements: tuple[str, dict] | str):
        """Execute statements in ONE transaction; return rows of the last one."""

        async def go():
            engine = create_async_engine(url, poolclass=NullPool)
            try:
                async with engine.begin() as conn:
                    result = None
                    for stmt in statements:
                        sql, params = (stmt, {}) if isinstance(stmt, str) else stmt
                        result = await conn.execute(text(sql), params)
                    return result.fetchall() if result is not None and result.returns_rows else []
            finally:
                await engine.dispose()

        return asyncio.run(go())

    return run


def table_names(db) -> set[str]:
    rows = db("select table_name from information_schema.tables where table_schema = 'app'")
    return {r[0] for r in rows}


def new_user(db, email="a@example.com") -> str:
    user_id = str(uuid4())
    db(
        (
            "insert into app.users (id, email, password_hash) values (:id, :email, 'x')",
            {"id": user_id, "email": email},
        )
    )
    return user_id


def new_chart(db, owner_id, title="t") -> str:
    chart_id = str(uuid4())
    db(
        (
            """insert into app.charts (id, owner_id, title, source, metadata, ai_events, content_hash)
               values (:id, :o, :t, '{}', '{}', '[]', :h)""",
            {"id": chart_id, "o": owner_id, "t": title, "h": "a" * 64},
        )
    )
    return chart_id


class TestMigrationLifecycle:
    def test_upgrade_creates_every_gateway_table(self, db):
        assert table_names(db) == APP_TABLES | {"alembic_version"}

    def test_downgrade_removes_them_all(self, db, alembic_cfg):
        command.downgrade(alembic_cfg, "base")
        assert table_names(db) <= {"alembic_version"}

    def test_upgrade_downgrade_upgrade_is_clean(self, db, alembic_cfg):
        command.downgrade(alembic_cfg, "base")
        command.upgrade(alembic_cfg, "head")
        assert APP_TABLES <= table_names(db)

    def test_migrated_database_matches_the_declared_metadata(self, url, db):
        async def diff():
            engine = create_async_engine(url, poolclass=NullPool)
            try:
                async with engine.connect() as conn:

                    def compare(sync_conn):
                        ctx = MigrationContext.configure(
                            sync_conn,
                            opts={
                                "compare_type": True,
                                "include_schemas": True,
                                "version_table_schema": "app",
                                "include_name": lambda name, kind, parent: (
                                    kind != "schema" or name == "app"
                                ),
                            },
                        )
                        return compare_metadata(ctx, metadata)

                    return await conn.run_sync(compare)
            finally:
                await engine.dispose()

        assert asyncio.run(diff()) == []


class TestUsers:
    def test_email_is_unique_ignoring_case(self, db):
        new_user(db, "Ann@Example.com")
        with pytest.raises(IntegrityError):
            new_user(db, "ann@example.COM")

    def test_role_defaults_to_free_and_rejects_unknown_roles(self, db):
        uid = new_user(db)
        assert db(f"select role from app.users where id = '{uid}'")[0][0] == "free"
        with pytest.raises(IntegrityError):
            db(("update app.users set role = 'root' where id = :id", {"id": uid}))

    def test_deleting_a_user_cascades_to_their_data(self, db):
        uid = new_user(db)
        new_chart(db, uid)
        db(("delete from app.users where id = :id", {"id": uid}))
        assert db("select count(*) from app.charts")[0][0] == 0


class TestCharts:
    def test_owner_must_exist(self, db):
        with pytest.raises(IntegrityError):
            new_chart(db, str(uuid4()))

    def test_version_cannot_be_below_one(self, db):
        cid = new_chart(db, new_user(db))
        with pytest.raises(IntegrityError):
            db(("update app.charts set version = 0 where id = :id", {"id": cid}))

    def test_overrides_default_to_an_empty_list_and_layers_are_jsonb(self, db):
        cid = new_chart(db, new_user(db))
        row = db(f"select overrides, jsonb_typeof(ai_events) from app.charts where id = '{cid}'")[0]
        assert row[0] == []
        assert row[1] == "array"

    def test_deleting_the_source_chart_keeps_its_forks(self, db):
        uid = new_user(db)
        source = new_chart(db, uid, "source")
        fork = new_chart(db, uid, "fork")
        db(("update app.charts set forked_from = :s where id = :f", {"s": source, "f": fork}))
        db(("delete from app.charts where id = :id", {"id": source}))
        assert db(f"select forked_from from app.charts where id = '{fork}'")[0][0] is None

    def test_content_hash_must_be_sha256_hex(self, db):
        uid = new_user(db)
        with pytest.raises(IntegrityError):
            db(
                (
                    """insert into app.charts (owner_id, title, source, metadata, ai_events, content_hash)
                       values (:o, 't', '{}', '{}', '[]', 'nothex')""",
                    {"o": uid},
                )
            )


class TestJobs:
    def insert_job(self, db, uid, state="queued", error=None):
        return db(
            (
                """insert into app.jobs (owner_id, source_kind, state, error)
                   values (:o, 'file', :s, cast(:e as jsonb)) returning id""",
                {"o": uid, "s": state, "e": error},
            )
        )

    def test_new_jobs_start_queued_with_zero_progress_and_charged_quota(self, db):
        uid = new_user(db)
        jid = self.insert_job(db, uid)[0][0]
        row = db(f"select state, progress, quota_charged from app.jobs where id = '{jid}'")[0]
        assert tuple(row) == ("queued", 0.0, True)

    def test_unknown_state_is_rejected(self, db):
        with pytest.raises(IntegrityError):
            self.insert_job(db, new_user(db), state="exploded")

    def test_failed_requires_an_error_and_other_states_forbid_it(self, db):
        uid = new_user(db)
        with pytest.raises(IntegrityError):
            self.insert_job(db, uid, state="failed")
        with pytest.raises(IntegrityError):
            self.insert_job(db, uid, state="processing", error='{"code": "ai_failure"}')
        assert self.insert_job(db, uid, state="failed", error='{"code": "ai_failure"}')

    def test_progress_is_a_fraction(self, db):
        uid = new_user(db)
        jid = self.insert_job(db, uid)[0][0]
        with pytest.raises(IntegrityError):
            db(("update app.jobs set progress = 1.5 where id = :id", {"id": jid}))


class TestUsageCounters:
    UPSERT = """insert into app.usage_counters (user_id, period_date, analyses_used)
                values (:u, current_date, 1)
                on conflict (user_id, period_date)
                do update set analyses_used = app.usage_counters.analyses_used + 1
                returning analyses_used"""

    def test_atomic_upsert_increments_per_user_and_day(self, db):
        uid = new_user(db)
        counts = [db((self.UPSERT, {"u": uid}))[0][0] for _ in range(3)]
        assert counts == [1, 2, 3]

    def test_counter_cannot_go_negative(self, db):
        uid = new_user(db)
        db((self.UPSERT, {"u": uid}))
        with pytest.raises(IntegrityError):
            db(("update app.usage_counters set analyses_used = -1 where user_id = :u", {"u": uid}))


class TestPlaylists:
    def test_items_can_be_reordered_inside_one_transaction(self, db):
        uid = new_user(db)
        a, b = new_chart(db, uid, "a"), new_chart(db, uid, "b")
        pid = db(
            (
                "insert into app.playlists (owner_id, name) values (:o, 'set') returning id",
                {"o": uid},
            )
        )[0][0]
        db(
            ("insert into app.playlist_items values (:p, :c, 0)", {"p": pid, "c": a}),
            ("insert into app.playlist_items values (:p, :c, 1)", {"p": pid, "c": b}),
        )
        # Swapping positions violates uniqueness mid-statement unless it is deferrable.
        db(
            (
                "update app.playlist_items set order_index = 1 where playlist_id = :p and chart_id = :c",
                {"p": pid, "c": a},
            ),
            (
                "update app.playlist_items set order_index = 0 where playlist_id = :p and chart_id = :c",
                {"p": pid, "c": b},
            ),
        )
        order = db(
            (
                "select chart_id from app.playlist_items where playlist_id = :p order by order_index",
                {"p": pid},
            )
        )
        assert [str(r[0]) for r in order] == [b, a]

    def test_a_chart_appears_once_per_playlist(self, db):
        uid = new_user(db)
        cid = new_chart(db, uid)
        pid = db(
            ("insert into app.playlists (owner_id, name) values (:o, 'p') returning id", {"o": uid})
        )[0][0]
        db(("insert into app.playlist_items values (:p, :c, 0)", {"p": pid, "c": cid}))
        with pytest.raises(IntegrityError):
            db(("insert into app.playlist_items values (:p, :c, 1)", {"p": pid, "c": cid}))


class TestChordDictionary:
    INSERT = """insert into app.chord_dictionary (id, root, quality, extension, instrument, variation, fingering)
                values (:id, 'C', 'major', 'maj7', :ins, :var, '{"frets": [-1,3,2,0,0,0]}')"""

    def test_one_voicing_per_variation_slot(self, db):
        db((self.INSERT, {"id": "cmaj7_guitar_0", "ins": "guitar", "var": 0}))
        with pytest.raises(IntegrityError):
            db((self.INSERT, {"id": "cmaj7_guitar_dup", "ins": "guitar", "var": 0}))
        db((self.INSERT, {"id": "cmaj7_guitar_1", "ins": "guitar", "var": 1}))

    def test_instrument_is_constrained(self, db):
        with pytest.raises(IntegrityError):
            db((self.INSERT, {"id": "x", "ins": "kazoo", "var": 0}))

    def test_root_must_be_one_of_the_twelve_canonical_sharp_names(self, db):
        bad_root = """insert into app.chord_dictionary (id, root, quality, instrument, fingering)
                      values ('b', 'Db', 'major', 'piano', '{}')"""
        with pytest.raises(IntegrityError):
            db(bad_root)


JOB_COLUMNS = (
    "select column_name from information_schema.columns "
    "where table_schema='app' and table_name='jobs'"
)
NEW_JOB_COLUMNS = {"title", "input_path", "fingerprint", "charged_on"}


class TestIngestionMigration:
    def test_jobs_gain_the_ingestion_columns(self, db):
        assert NEW_JOB_COLUMNS <= {r[0] for r in db(JOB_COLUMNS)}

    def test_fingerprint_must_be_sha256_hex(self, db):
        with pytest.raises(IntegrityError):
            db(
                "insert into app.analyses (content_hash, result, model_version, pipeline_version) "
                f"values ('{'a' * 64}', '{{}}', 'm', 'p')",
                "insert into app.source_fingerprints (fingerprint, content_hash) "
                f"values ('NOT-HEX', '{'a' * 64}')",
            )

    def test_fingerprint_must_point_at_a_cached_analysis(self, db):
        with pytest.raises(IntegrityError):
            db(
                "insert into app.source_fingerprints (fingerprint, content_hash) "
                f"values ('{'b' * 64}', '{'c' * 64}')"
            )

    def test_deleting_the_analysis_drops_its_fingerprints(self, db):
        h = "d" * 64
        db(
            "insert into app.analyses (content_hash, result, model_version, pipeline_version) "
            f"values ('{h}', '{{}}', 'm', 'p')",
            "insert into app.source_fingerprints (fingerprint, content_hash) "
            f"values ('{'e' * 64}', '{h}')",
            f"delete from app.analyses where content_hash = '{h}'",
        )
        assert db("select count(*) from app.source_fingerprints")[0][0] == 0

    def test_downgrade_to_0001_restores_the_old_shape(self, db, alembic_cfg):
        command.downgrade(alembic_cfg, "0001")
        assert "source_fingerprints" not in table_names(db)
        assert not (NEW_JOB_COLUMNS & {r[0] for r in db(JOB_COLUMNS)})
