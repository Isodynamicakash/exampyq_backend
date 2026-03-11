"""
services/job_cleanup.py
=======================
Background cleanup task — removes stale job directories from /tmp.

Why this matters on Railway free tier:
  - /tmp is RAM-backed (tmpfs) — it counts toward your 512 MB RAM limit
  - Each ZIP job can be 5–50 MB of images in /tmp
  - Without cleanup, 10 uploads = potential OOM crash

Strategy:
  - Jobs older than JOB_TTL_SECONDS (default 2 hours) are deleted
  - Jobs with status "saved" or "failed" are deleted after SHORT_TTL (15 min)
  - Runs every CLEANUP_INTERVAL_SECONDS (default 10 min)
  - Called from FastAPI lifespan so it runs as a background task
"""

import asyncio
import shutil
import time
import logging
from pathlib import Path

from services.pipeline import JOBS_ROOT, get_job, _update_job

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
JOB_TTL_SECONDS       = 2 * 60 * 60   # 2 hours  — max age for any job
SHORT_TTL_SECONDS     = 15 * 60       # 15 min   — for saved/failed jobs
CLEANUP_INTERVAL      = 10 * 60       # run cleanup every 10 minutes


def _delete_job_dir(job_id: str, reason: str):
    """Delete a job's directory and mark it cleaned in job.json."""
    job_dir = JOBS_ROOT / job_id
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.info(f"[cleanup] Deleted job {job_id} ({reason})")
        except Exception as e:
            logger.warning(f"[cleanup] Failed to delete {job_id}: {e}")


def cleanup_stale_jobs():
    """
    Synchronous cleanup — scans JOBS_ROOT and removes stale job dirs.
    Returns count of deleted jobs.
    """
    if not JOBS_ROOT.exists():
        return 0

    deleted = 0
    now = time.time()

    for job_dir in JOBS_ROOT.iterdir():
        if not job_dir.is_dir():
            continue

        job_id = job_dir.name
        job    = get_job(job_id)

        if job is None:
            # No job.json — orphaned directory, delete if older than 1 hour
            try:
                age = now - job_dir.stat().st_mtime
                if age > 3600:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    deleted += 1
            except Exception:
                pass
            continue

        age    = now - job.get("created_at", now)
        status = job.get("status", "")

        # Saved or failed jobs → short TTL
        if status in ("saved", "failed") and age > SHORT_TTL_SECONDS:
            _delete_job_dir(job_id, f"status={status}, age={age:.0f}s")
            deleted += 1

        # Any job older than JOB_TTL → delete regardless
        elif age > JOB_TTL_SECONDS:
            _delete_job_dir(job_id, f"TTL exceeded, age={age:.0f}s")
            deleted += 1

    if deleted:
        logger.info(f"[cleanup] Removed {deleted} stale job(s)")
    return deleted


async def cleanup_loop():
    """
    Async background loop — runs cleanup every CLEANUP_INTERVAL seconds.
    Registered in FastAPI lifespan so it starts with the server.
    """
    logger.info(f"[cleanup] Background cleanup started (interval={CLEANUP_INTERVAL}s)")
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            # Run in thread so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, cleanup_stale_jobs)
        except asyncio.CancelledError:
            logger.info("[cleanup] Cleanup loop cancelled")
            break
        except Exception as e:
            logger.error(f"[cleanup] Unexpected error: {e}")