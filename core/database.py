"""
core/database.py — Synchronous psycopg2 database connection
Works with Python 3.14 (unlike asyncpg)
"""

import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from fastapi import Request


_conn = None


def get_connection():
    global _conn
    if _conn is None or _conn.closed:
        db_url = os.environ["DATABASE_URL"].split("#")[0].strip()  # strip inline comments
        from urllib.parse import urlparse, unquote
        p = urlparse(db_url)
        _conn = psycopg2.connect(
            host     = p.hostname,
            port     = p.port or 5432,
            dbname   = p.path.lstrip("/"),
            user     = unquote(p.username or ""),
            password = unquote(p.password or ""),
            cursor_factory = psycopg2.extras.RealDictCursor,
            sslmode  = "require",
        )
        _conn.autocommit = True
    return _conn


@contextmanager
def get_cursor():
    conn = get_connection()
    cur  = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()


# FastAPI dependency — kept for compatibility with existing router code
def get_pool(request: Request = None):
    return None   # not used with psycopg2 sync approach