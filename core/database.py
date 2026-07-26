"""
core/database.py — Synchronous psycopg2 database connection
Works with Python 3.14 (unlike asyncpg)

FIX  the previous version used ONE global psycopg2 connection
shared across the whole app. FastAPI runs sync `def` route functions in a
worker thread pool, so concurrent requests were calling get_cursor() from
different threads at the same time and sharing that single connection's
socket. psycopg2 connections are not safe for concurrent multi-threaded use
-- that corrupts the wire protocol and surfaces as:

    psycopg2.DatabaseError: server closed the connection unexpectedly

This version uses psycopg2.pool.ThreadedConnectionPool, which is
thread-safe and hands each thread its own connection. The get_cursor()
interface is unchanged, so routers/questions.py needs no changes.
"""

import os
import psycopg2
import psycopg2.extras
import psycopg2.pool
from contextlib import contextmanager
from urllib.parse import urlparse, unquote

_pool = None


def _build_pool():
    db_url = os.environ["DATABASE_URL"].split("#")[0].strip()  # strip inline comments
    p = urlparse(db_url)
    return psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        # Tune this to (a) your Postgres plan's max_connections and
        # (b) how many Uvicorn/Gunicorn worker processes you run --
        # each process gets its OWN pool of this size.
        maxconn=int(os.environ.get("DB_POOL_MAX", 10)),
        host=p.hostname,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=unquote(p.username or ""),
        password=unquote(p.password or ""),
        cursor_factory=psycopg2.extras.RealDictCursor,
        sslmode="require",
    )


def get_pool():
    global _pool
    if _pool is None:
        _pool = _build_pool()
    return _pool


@contextmanager
def get_cursor():
    pool = get_pool()
    conn = pool.getconn()
    conn.autocommit = True
    broken = False
    try:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    except psycopg2.OperationalError:
        # Connection is dead (e.g. "server closed the connection
        # unexpectedly"). Discard it instead of returning a broken
        # connection to the pool for the next request to trip over.
        broken = True
        raise
    finally:
        pool.putconn(conn, close=broken)
