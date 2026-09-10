"""PostgreSQL connection pool using psycopg2.

Simple sync connection pool — no async, no ORM.
Usage:
    from src.shared.infrastructure.postgres.client import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM crawl_queue WHERE status = %s", ("pending",))
            rows = cur.fetchall()
"""
from __future__ import annotations
import os
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool

log = logging.getLogger(__name__)

_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    """Lazy init — pool created on first call, not at import time."""
    global _pool
    if _pool is None:
        host     = os.getenv("POSTGRES_HOST", "localhost")
        port     = int(os.getenv("POSTGRES_PORT", "5432"))
        db       = os.getenv("POSTGRES_DB", "chordsense")
        user     = os.getenv("POSTGRES_USER", "chordsense")
        password = os.getenv("POSTGRES_PASSWORD", "chordsense_dev")

        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=host,
            port=port,
            dbname=db,
            user=user,
            password=password,
        )
        log.info(f"PostgreSQL pool created: {user}@{host}:{port}/{db}")
    return _pool


@contextmanager
def get_connection():
    """Get a connection from the pool. Auto-commits on success, rollbacks on error.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)
