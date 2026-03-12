"""
services/pipeline.py
====================
Pipeline: ZIP (.tex + images) → parse → admin review → save to DB

FLOW:
  1. Admin uploads ZIP containing .tex + images/ folder
  2. ZIP is extracted to /tmp/examside_jobs/{job_id}/
  3. Parser reads .tex → list of question dicts
  4. Questions stored in memory (_jobs)
  5. Admin reviews in UI → POST /save-questions → saved to Supabase
  6. Images uploaded to R2 (if configured)

CHANGES FROM PREVIOUS VERSION (parser v2 compatibility):
  - has_diagram now checks BOTH q_images and sol_images
  - answers table: NUMERICAL questions write to numerical_answer column,
    MCQ/MSQ write to correct_option column (was always writing correct_option)
  - _mark_image_availability now checks q_images and sol_images separately
    so the two lists stay independent (were being merged into one flat list)
  - images table: position column now correctly distinguishes
    "question" vs "solution" images
"""

import os
import asyncio
import traceback
import uuid
import time
import re
import zipfile
from pathlib import Path
from typing import Optional

from services.parser import parse_tex
# LLM tagger — imported lazily so missing openai package is non-fatal
try:
    from services.llm_tagger import tag_questions_async as _tag_questions
    _LLM_TAGGING = True
except ImportError:
    _LLM_TAGGING = False

# Read OpenAI key once at startup from env — used as default for all pipelines.
# This means tagging works without the frontend sending any header.
_OPENAI_KEY_FROM_ENV: str = os.environ.get("OPENAI_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# File-based job store  (replaces in-memory dict)
#
# WHY: uvicorn --reload spawns a reloader process + a server process.
# POST /upload-zip writes a job in process A. GET /jobs/{id} may hit
# process B which has an empty in-memory dict → 404 right after upload.
#
# FIX: store each job as a JSON file under /tmp/examside_jobs/{job_id}/job.json
# All processes share the filesystem → no more 404s.
# ─────────────────────────────────────────────────────────────────────────────

import json as _json

import os as _os
JOBS_ROOT = Path(_os.environ.get("JOBS_ROOT", str(Path(_os.environ.get("TEMP", "/tmp")) / "examside_jobs")))
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def _job_path(job_id: str) -> Path:
    return JOBS_ROOT / job_id / "job.json"


def create_job(filename: str) -> str:
    job_id  = str(uuid.uuid4())
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "job_id":      job_id,
        "filename":    filename,
        "status":      "pending",
        "progress":    0,
        "questions":   [],
        "job_dir":     str(job_dir),
        "images_dir":  None,
        "image_count": 0,
        "error":       None,
        "created_at":  time.time(),
    }
    _job_path(job_id).write_text(_json.dumps(data))
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    p = _job_path(job_id)
    if not p.exists():
        return None
    try:
        return _json.loads(p.read_text())
    except Exception:
        return None


def _update_job(job_id: str, **kwargs):
    p = _job_path(job_id)
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text())
        data.update(kwargs)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(_json.dumps(data))
        tmp.replace(p)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Extract exam year + shift from .tex \title{} line
# ─────────────────────────────────────────────────────────────────────────────

def _extract_exam_meta(tex_content: str) -> dict:
    """
    Extract year, exam_date (YYYY-MM-DD), and shift from .tex title + body.

    shift is ONLY "Morning" | "Evening" | "" — never a month name.
    exam_date is the exact paper date for user-frontend date filtering.
    year is derived from exam_date when possible, else from title.

    Allen PDF patterns:
      title: JEE-MAIN EXAMINATION - JANUARY 2026
      body:  JEE-Main Exam Session-1 (January 2026)/21-01-2026/Morning Shift
      →  year="2026", exam_date="2026-01-21", shift="Morning"
    """
    _mmap = {
        'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
        'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
        'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
        'sep':9,'oct':10,'nov':11,'dec':12,
    }
    _mp = (r'(january|february|march|april|may|june|july|august|'
           r'september|october|november|december|jan|feb|mar|apr|'
           r'jun|jul|aug|sep|oct|nov|dec)')

    title = ""
    m = re.search(r'\\title\s*\{([^}]+)\}', tex_content)
    if m:
        title = m.group(1).strip()

    body = ""
    bm = re.search(r'\\begin\{document\}([\s\S]{0:800})', tex_content)
    if bm:
        body = bm.group(1)

    combined = (title + " " + body).strip()
    exam_date = ""
    shift     = ""
    year      = ""

    # exam_date — DD-MM-YYYY (Allen body header)
    dm = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', combined)
    if dm:
        exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
    else:
        dm = re.search(r'\b(\d{2})/(\d{2})/(20\d{2})\b', combined)
        if dm:
            exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
        else:
            dm = re.search(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_mp}\s+(20\d{{2}})\b',
                           combined, re.IGNORECASE)
            if dm:
                mo = _mmap.get(dm.group(2).lower(), 0)
                if mo:
                    exam_date = f"{dm.group(3)}-{mo:02d}-{int(dm.group(1)):02d}"
            else:
                dm = re.search(rf'\b{_mp}\s+(\d{{1,2}}),?\s+(20\d{{2}})\b',
                               combined, re.IGNORECASE)
                if dm:
                    mo = _mmap.get(dm.group(1).lower(), 0)
                    if mo:
                        exam_date = f"{dm.group(3)}-{mo:02d}-{int(dm.group(2)):02d}"

    # year from exam_date, else title
    if exam_date:
        year = exam_date[:4]
    else:
        m = re.search(r'\b(20\d{2})\b', combined)
        if m: year = m.group(1)

    # shift — ONLY Morning | Evening
    tl = combined.lower()
    if any(x in tl for x in ("morning", "shift 1", "shift-1", "shift1",
                               "session 1", "session-1", "session1")):
        shift = "Morning"
    elif any(x in tl for x in ("evening", "shift 2", "shift-2", "shift2",
                                 "session 2", "session-2", "session2")):
        shift = "Evening"

    return {"year": year, "exam_date": exam_date, "shift": shift}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline A — ZIP upload (.tex + images folder)  ← MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline_zip(job_id: str, zip_bytes: bytes, filename: str, pool=None, openai_api_key: str = ""):
    """
    1. Extract ZIP to /tmp/examside_jobs/{job_id}/
    2. Find .tex file inside
    3. Find images/ folder inside
    4. Run parser on .tex
    5. Mark which images exist
    """
    try:
        _update_job(job_id, status="processing", progress=10)

        # Extract ZIP
        job_dir = Path(f"/tmp/examside_jobs/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)

        zip_path = job_dir / "upload.zip"
        zip_path.write_bytes(zip_bytes)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(job_dir)
        zip_path.unlink()

        _update_job(job_id, progress=30, job_dir=str(job_dir))

        # Find .tex file
        tex_files = list(job_dir.rglob("*.tex"))
        if not tex_files:
            raise ValueError(
                "No .tex file found in ZIP. "
                "ZIP must contain the .tex file from MathPix."
            )
        # Pick the largest .tex (avoid tiny helper files)
        tex_path = max(tex_files, key=lambda p: p.stat().st_size)

        # Find images folder
        images_dir = _find_images_dir(job_dir, tex_path)
        img_count  = len(list(images_dir.glob("*"))) if images_dir else 0

        _update_job(job_id, progress=50,
                    images_dir=str(images_dir) if images_dir else None,
                    image_count=img_count)

        # Parse
        _update_job(job_id, status="parsing", progress=70)
        tex_content = tex_path.read_text(encoding="utf-8", errors="replace")
        questions   = parse_tex(str(tex_path))

        # Extract year/shift from \title{} and stamp onto every question
        # that doesn't already have a year (parser may have found one too)
        meta = _extract_exam_meta(tex_content)
        for q in questions:
            if not q.get("year")      and meta["year"]:      q["year"]      = meta["year"]
            if not q.get("shift")     and meta["shift"]:     q["shift"]     = meta["shift"]
            if not q.get("exam_date") and meta["exam_date"]: q["exam_date"] = meta["exam_date"]

        # Mark image availability per question (keeps q_images/sol_images separate)
        questions = _mark_image_availability(questions, images_dir)

        # ── LLM auto-tagging ──────────────────────────────────────────────────
        # FIX: removed "pool is not None" guard — tagger works without DB
        #      (uses hardcoded taxonomy + falls back gracefully).
        # FIX: removed single-subject call — tagger now uses each question's
        #      own q["subject"] field, so multi-subject papers tag correctly.
        print(f"[pipeline] _LLM_TAGGING={_LLM_TAGGING} key_len={len(openai_api_key)} env_key_len={len(_OPENAI_KEY_FROM_ENV)}", flush=True)
        if _LLM_TAGGING:
            try:
                _update_job(job_id, status="tagging", progress=85)
                effective_key = openai_api_key or _OPENAI_KEY_FROM_ENV
                print(f"[pipeline] Calling tagger with key prefix={effective_key[:15] if effective_key else 'EMPTY'}", flush=True)
                questions = await _tag_questions(
                    questions,
                    subject="",          # ignored — tagger reads per-question subject
                    pool=pool,
                    openai_api_key=effective_key,
                )
            except Exception:
                import traceback as _tb
                import logging as _log
                print(f"[pipeline] Tagging EXCEPTION: {_tb.format_exc()}", flush=True)
                _log.getLogger(__name__).error(f"[pipeline] Tagging failed: {_tb.format_exc()}")

        _update_job(job_id,
                    status    = "ready",
                    progress  = 100,
                    questions = questions)

    except Exception:
        err = traceback.format_exc()
        _update_job(job_id, status="failed", progress=0, error=err[:3000])


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline B — .tex only (no images)
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline_tex(job_id: str, tex_bytes: bytes, filename: str, pool=None, openai_api_key: str = ""):
    """
    Pipeline B — .tex file only (no images in upload).
    Identical feature set to ZIP and PDF pipelines:
      - exam meta extraction (year/shift/exam_date from \\title{})
      - images_dir set to job_dir/images/ so manual image uploads work
      - _mark_image_availability (all images will be missing until manually added)
      - LLM auto-tagging
    """
    try:
        _update_job(job_id, status="processing", progress=10)

        job_dir  = Path(f"/tmp/examside_jobs/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)

        tex_path = job_dir / "output.tex"
        tex_path.write_bytes(tex_bytes)

        # Pre-create images/ dir so manual upload-image endpoint can save here
        images_dir = job_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        _update_job(job_id, progress=50,
                    job_dir=str(job_dir),
                    images_dir=str(images_dir),
                    image_count=0)

        _update_job(job_id, status="parsing", progress=70)
        tex_content = tex_path.read_bytes().decode("utf-8", errors="replace")
        questions   = parse_tex(str(tex_path))

        # ── Stamp exam meta from \title{} (same as ZIP/PDF) ────────────────
        meta = _extract_exam_meta(tex_content)
        for q in questions:
            if not q.get("year")      and meta["year"]:      q["year"]      = meta["year"]
            if not q.get("shift")     and meta["shift"]:     q["shift"]     = meta["shift"]
            if not q.get("exam_date") and meta["exam_date"]: q["exam_date"] = meta["exam_date"]

        # ── Mark image availability (will all be missing — expected for tex-only)
        questions = _mark_image_availability(questions, images_dir)

        # ── LLM auto-tagging ────────────────────────────────────────────────
        print(f"[TEX pipeline] _LLM_TAGGING={_LLM_TAGGING} key_len={len(openai_api_key)} env_key_len={len(_OPENAI_KEY_FROM_ENV)}", flush=True)
        if _LLM_TAGGING:
            try:
                _update_job(job_id, status="tagging", progress=85)
                effective_key = openai_api_key or _OPENAI_KEY_FROM_ENV
                print(f"[TEX pipeline] Calling tagger, key prefix={effective_key[:15] if effective_key else 'EMPTY'}", flush=True)
                questions = await _tag_questions(
                    questions,
                    subject="",
                    pool=pool,
                    openai_api_key=effective_key,
                )
                print(f"[TEX pipeline] Tagging done", flush=True)
            except Exception:
                import traceback as _tb
                import logging as _log
                print(f"[TEX pipeline] Tagging EXCEPTION: {_tb.format_exc()}", flush=True)
                _log.getLogger(__name__).error(f"[pipeline] Tagging failed: {_tb.format_exc()}")

        _update_job(job_id,
                    status    = "ready",
                    progress  = 100,
                    questions = questions)

    except Exception:
        err = traceback.format_exc()
        _update_job(job_id, status="failed", progress=0, error=err[:3000])


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline C — PDF via MathPix
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline_pdf(job_id: str, pdf_bytes: bytes, filename: str, pool=None, openai_api_key: str = ""):
    """
    Full pipeline for raw PDF upload:
      1. Send PDF to MathPix API
      2. Poll until conversion done
      3. Download .tex + images
      4. Run normal parser on the .tex
    """
    try:
        if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
            _update_job(job_id, status="failed", progress=0,
                        error="MATHPIX_APP_ID / MATHPIX_APP_KEY not set in .env")
            return

        from services.mathpix import run_mathpix_pipeline

        _update_job(job_id, status="mathpix", progress=10)

        # MathPix: PDF → .tex + images/ downloaded to job_dir
        job_dir = await run_mathpix_pipeline(pdf_bytes, filename, job_id)

        _update_job(job_id, status="parsing", progress=60)

        # Find .tex
        tex_files = list(job_dir.glob("*.tex"))
        if not tex_files:
            _update_job(job_id, status="failed", progress=60,
                        error="MathPix succeeded but no .tex file found in output")
            return

        tex_path    = tex_files[0]
        tex_content = tex_path.read_text(encoding="utf-8", errors="replace")
        images_dir  = _find_images_dir(job_dir, tex_path)

        _update_job(job_id, status="parsing", progress=70,
                    images_dir=str(images_dir) if images_dir else None,
                    image_count=len(list(images_dir.glob("*"))) if images_dir else 0)

        questions = parse_tex(str(tex_path))

        # Stamp exam meta from title
        meta = _extract_exam_meta(tex_content)
        for q in questions:
            if not q.get("year")      and meta["year"]:      q["year"]      = meta["year"]
            if not q.get("shift")     and meta["shift"]:     q["shift"]     = meta["shift"]
            if not q.get("exam_date") and meta["exam_date"]: q["exam_date"] = meta["exam_date"]

        questions = _mark_image_availability(questions, images_dir)

        # ── LLM auto-tagging ────────────────────────────────────────────────
        print(f"[PDF pipeline] _LLM_TAGGING={_LLM_TAGGING} key_len={len(openai_api_key)} env_key_len={len(_OPENAI_KEY_FROM_ENV)}", flush=True)
        if _LLM_TAGGING:
            try:
                _update_job(job_id, status="tagging", progress=90)
                effective_key = openai_api_key or _OPENAI_KEY_FROM_ENV
                print(f"[PDF pipeline] Calling tagger, key prefix={effective_key[:15] if effective_key else 'EMPTY'}", flush=True)
                questions = await _tag_questions(
                    questions,
                    subject="",
                    pool=pool,
                    openai_api_key=effective_key,
                )
                print(f"[PDF pipeline] Tagging done", flush=True)
            except Exception:
                import traceback as _tb
                import logging as _log
                print(f"[PDF pipeline] Tagging EXCEPTION: {_tb.format_exc()}", flush=True)
                _log.getLogger(__name__).error(f"[pipeline] Tagging failed: {_tb.format_exc()}")

        _update_job(job_id, status="ready", progress=100, questions=questions)

    except Exception:
        err = traceback.format_exc()
        _update_job(job_id, status="failed", progress=0, error=err[:3000])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_images_dir(job_dir: Path, tex_path: Path) -> Optional[Path]:
    """Find images folder — tries common names next to .tex first."""
    for parent in (tex_path.parent, job_dir):
        for name in ("images", "img", "Images", "Img", "figures"):
            candidate = parent / name
            if candidate.is_dir():
                imgs = (list(candidate.glob("*.jpg")) +
                        list(candidate.glob("*.png")) +
                        list(candidate.glob("*.jpeg")))
                if imgs:
                    return candidate

    # Fallback: any subfolder with images
    for folder in sorted(job_dir.rglob("*")):
        if folder.is_dir():
            imgs = (list(folder.glob("*.jpg")) +
                    list(folder.glob("*.png")) +
                    list(folder.glob("*.jpeg")))
            if imgs:
                return folder
    return None


def _mark_image_availability(questions: list, images_dir: Optional[Path]) -> list:
    """
    Check which image IDs exist in images_dir.
    q_images and sol_images are NOT modified — only images_found/images_missing added.
    """
    if not images_dir or not images_dir.exists():
        for q in questions:
            all_imgs = q.get("q_images", []) + q.get("sol_images", [])
            q["images_found"]   = []
            q["images_missing"] = all_imgs
        return questions

    # Build lookup: stem (no ext) → full path
    available: dict[str, Path] = {}
    for f in images_dir.iterdir():
        if f.is_file():
            available[f.stem] = f
            available[f.name] = f

    def _is_available(img_id: str) -> bool:
        stem = Path(img_id).stem
        return stem in available or img_id in available

    for q in questions:
        found, missing = [], []
        for img_id in (q.get("q_images", []) + q.get("sol_images", [])):
            (found if _is_available(img_id) else missing).append(img_id)
        q["images_found"]   = found
        q["images_missing"] = missing

    return questions


def get_image_path(job_id: str, image_id: str) -> Optional[Path]:
    """
    Given a job_id and image_id, return the full path to the image file.
    Used by the temp-image endpoint to serve images to the frontend.
    Checks job["images_dir"] first, then falls back to JOBS_ROOT/job_id/images/
    so manually uploaded images are found even before images_dir is persisted.
    """
    job = get_job(job_id)

    # Build candidate dirs: job["images_dir"] + fallback JOBS_ROOT path
    candidate_dirs = []
    if job and job.get("images_dir"):
        candidate_dirs.append(Path(job["images_dir"]))
    # Always also check the default location (covers manual uploads before images_dir is set)
    default_dir = JOBS_ROOT / job_id / "images"
    if default_dir not in candidate_dirs:
        candidate_dirs.append(default_dir)

    for images_dir in candidate_dirs:
        if not images_dir.exists():
            continue
        # Try exact name, then with common extensions
        for ext in ("", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"):
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        # Try stem match (image_id might include extension already)
        stem = Path(image_id).stem
        for f in images_dir.iterdir():
            if f.stem == stem:
                return f

    return None


# ─────────────────────────────────────────────────────────────────────────────
# FK resolution helpers
# ─────────────────────────────────────────────────────────────────────────────

import psycopg2.extras

# ─────────────────────────────────────────────────────────────────────────────
# DB helpers — psycopg2 sync (pool=None, use get_cursor() directly)
# ─────────────────────────────────────────────────────────────────────────────

def _db_fetchone(cur, sql, *params):
    cur.execute(sql, params)
    return cur.fetchone()

def _db_execute(cur, sql, *params):
    cur.execute(sql, params)

def _resolve_paper_id(cur, exam_name: str, year: str, exam_date: str, shift: str) -> int:
    exam_name = (exam_name or "JEE Main").strip()
    year_int  = int(year) if year and str(year).strip().isdigit() else None
    date_val  = exam_date.strip() if exam_date else None
    shift_val = shift.strip() if shift else None

    # Get or create exam
    row = _db_fetchone(cur, "SELECT id FROM exams WHERE name = %s", exam_name)
    if row:
        exam_id = row["id"]
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", exam_name.lower()).strip("-")
        row = _db_fetchone(cur,
            """INSERT INTO exams (name, slug, exam_category)
               VALUES (%s, %s, 'engineering')
               ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
               RETURNING id""",
            exam_name, slug)
        exam_id = row["id"]

    # Get or create paper
    if date_val:
        row = _db_fetchone(cur,
            """SELECT id FROM papers
               WHERE exam_id=%s AND exam_date=%s::date
                 AND shift IS NOT DISTINCT FROM %s""",
            exam_id, date_val, shift_val)
    else:
        row = _db_fetchone(cur,
            """SELECT id FROM papers
               WHERE exam_id=%s AND year=%s AND shift IS NOT DISTINCT FROM %s
                 AND exam_date IS NULL""",
            exam_id, year_int, shift_val)

    if row:
        return row["id"]

    try:
        row = _db_fetchone(cur,
           """INSERT INTO papers (exam_id, year, shift, exam_date)
               VALUES (%s, %s, %s, %s::date)
                 ON CONFLICT (exam_id, year, shift, exam_date)  -- Added exam_date
                DO UPDATE SET year = COALESCE(EXCLUDED.year, papers.year)
                 RETURNING id""",
            exam_id, year_int, shift_val, date_val)
    except Exception:
        row = _db_fetchone(cur,
           """INSERT INTO papers (exam_id, year, shift, exam_date) -- Added exam_date here too
   VALUES (%s, %s, %s, %s::date)
   ON CONFLICT (exam_id, year, shift, exam_date)  -- Added exam_date
   DO UPDATE SET exam_id = EXCLUDED.exam_id
   RETURNING id""",
            exam_id, year_int, shift_val,date_val)
    return row["id"]


def _resolve_chapter_id(cur, chapter_name: str, subject_name: str, exam_id: int) -> int:
    subject_name = (subject_name or "Physics").strip().title()

    row = _db_fetchone(cur,
        "SELECT id FROM subjects WHERE exam_id=%s AND name=%s", exam_id, subject_name)
    if row:
        subject_id = row["id"]
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", subject_name.lower()).strip("-")
        row = _db_fetchone(cur,
            """INSERT INTO subjects (exam_id, name, slug)
               VALUES (%s, %s, %s)
               ON CONFLICT (exam_id, slug) DO UPDATE SET name = EXCLUDED.name
               RETURNING id""",
            exam_id, subject_name, slug)
        subject_id = row["id"]

    row = _db_fetchone(cur,
        "SELECT id FROM chapters WHERE subject_id=%s AND name=%s", subject_id, chapter_name)
    if row:
        return row["id"]

    chap_slug = re.sub(r"[^a-z0-9]+", "-", chapter_name.lower()).strip("-")
    row = _db_fetchone(cur,
        """INSERT INTO chapters (subject_id, name, slug)
           VALUES (%s, %s, %s)
           ON CONFLICT (subject_id, slug) DO UPDATE SET name = EXCLUDED.name
           RETURNING id""",
        subject_id, chapter_name, chap_slug)
    return row["id"]


def _resolve_topic_id(cur, topic_name: str, chapter_id: int) -> Optional[int]:
    if not topic_name or not topic_name.strip():
        return None
    topic_name = topic_name.strip()
    row = _db_fetchone(cur,
        "SELECT id FROM topics WHERE chapter_id=%s AND name=%s", chapter_id, topic_name)
    if row:
        return row["id"]
    slug = re.sub(r"[^a-z0-9]+", "-", topic_name.lower()).strip("-")
    row = _db_fetchone(cur,
        """INSERT INTO topics (chapter_id, name, slug)
           VALUES (%s, %s, %s)
           ON CONFLICT (chapter_id, slug) DO UPDATE SET name = EXCLUDED.name
           RETURNING id""",
        chapter_id, topic_name, slug)
    return row["id"]


# ─────────────────────────────────────────────────────────────────────────────
# Save to DB  (psycopg2 sync)
# ─────────────────────────────────────────────────────────────────────────────

async def save_questions_to_db(
    job_id: str,
    reviewed_questions: list[dict],
    pool,                          # ignored — we use psycopg2 get_cursor()
) -> list[int]:
    """
    Save verified questions to DB using psycopg2 (sync).
    Resolves chapter_name / year / shift text → real FK IDs automatically.
    Creates missing exam / paper / subject / chapter rows on the fly.
    Only saves questions where verified=True.
    Returns list of inserted question IDs.
    """
    from core.database import get_cursor

    job        = get_job(job_id)
    images_dir = Path(job["images_dir"]) if job and job.get("images_dir") else None

    r2_configured = all([
        os.environ.get("R2_ENDPOINT_URL"),
        os.environ.get("R2_ACCESS_KEY_ID"),
        os.environ.get("R2_SECRET_ACCESS_KEY"),
        os.environ.get("R2_BUCKET"),
    ])

    inserted_ids = []

    with get_cursor() as cur:
        for q in reviewed_questions:
            if not q.get("verified"):
                continue
            try:
                q_type     = q.get("q_type", "MCQ")
                has_diagram = bool(q.get("q_images") or q.get("sol_images"))

                # ── Resolve paper ─────────────────────────────────────────
                paper_id = _resolve_paper_id(
                    cur,
                    exam_name = q.get("exam_name") or "JEE Main",
                    year      = q.get("year", ""),
                    exam_date = q.get("exam_date", ""),
                    shift     = q.get("shift", ""),
                )

                # ── Resolve chapter ───────────────────────────────────────
                chapter_name = (q.get("chapter_name") or "").strip() or "Uncategorised"
                exam_row = _db_fetchone(cur, "SELECT exam_id FROM papers WHERE id=%s", paper_id)
                exam_id  = exam_row["exam_id"]

                chapter_id = _resolve_chapter_id(
                    cur,
                    chapter_name = chapter_name,
                    subject_name = q.get("subject", "Physics"),
                    exam_id      = exam_id,
                )

                # ── Resolve topic ─────────────────────────────────────────
                topic_id = _resolve_topic_id(
                    cur,
                    topic_name = q.get("topic_name") or q.get("topic", ""),
                    chapter_id = chapter_id,
                )

                # ── Slug ──────────────────────────────────────────────────
                base = re.sub(r"[^a-z0-9]+", "-", q.get("question", "")[:60].lower()).strip("-")
                slug = f"{base}-{int(time.time())}-{q.get('number', 0)}"

                # ── Insert question ───────────────────────────────────────
                opts = q.get("options", [])
                row = _db_fetchone(cur,
                    """INSERT INTO questions (
                           slug, paper_id, chapter_id, topic_id,
                           question_number, question_type,
                           marks_positive, marks_negative,
                           question_text, option_1, option_2, option_3, option_4,
                           difficulty, has_diagram, is_verified, is_active
                       ) VALUES (
                           %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                       ) RETURNING id""",
                    slug,
                    paper_id, chapter_id, topic_id,
                    q.get("number"),
                    q_type,
                    q.get("marks_correct", 4),
                    q.get("marks_wrong", -1),
                    q.get("question", ""),
                    opts[0] if len(opts) > 0 else None,
                    opts[1] if len(opts) > 1 else None,
                    opts[2] if len(opts) > 2 else None,
                    opts[3] if len(opts) > 3 else None,
                    q.get("difficulty") or None,
                    has_diagram, True, True,
                )
                question_id = row["id"]
                inserted_ids.append(question_id)

                # ── Insert answer ─────────────────────────────────────────
                _db_execute(cur,
                    """INSERT INTO answers (question_id, correct_option, solution_text)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (question_id) DO UPDATE
                           SET correct_option = EXCLUDED.correct_option,
                               solution_text  = EXCLUDED.solution_text""",
                    question_id,
                    q.get("answer") or None,
                    q.get("solution") or None,
                )

                # ── Upload images to R2 ───────────────────────────────────
                if r2_configured and images_dir and images_dir.exists():
                    try:
                        from services.r2_upload import upload_question_images
                        uploaded = upload_question_images(
                            job_dir       = images_dir.parent,
                            question_id   = question_id,
                            q_image_ids   = q.get("q_images", []),
                            sol_image_ids = q.get("sol_images", []),
                            images_dir    = images_dir,
                            opt_image_ids = q.get("opt_images", {}),  # {a,b,c,d} → option_1..4
                        )
                        # Build filename → R2 URL map (for [IMAGE:x] placeholder replacement)
                        img_url_map = {
                            img["image_id"]: img["url"]
                            for img in uploaded
                            if "image_id" in img
                        }

                        # Replace [IMAGE:filename] placeholders with full R2 URLs
                        def _replace_images(text):
                            if not text:
                                return text
                            import re as _re
                            def _sub(m):
                                fname = m.group(1)
                                if fname in img_url_map:
                                    return f"[IMAGE:{img_url_map[fname]}]"
                                stem = fname.rsplit(".", 1)[0]
                                for k, v in img_url_map.items():
                                    if k.rsplit(".", 1)[0] == stem or k == stem:
                                        return f"[IMAGE:{v}]"
                                return m.group(0)  # leave unchanged if not found
                            return _re.sub(r"\[IMAGE:([^\]]+)\]", _sub, text)

                        # Build option texts — opt_images override the whole option cell
                        # with [IMAGE:<r2_url>] so the frontend renders an image instead of text
                        OPT_KEY_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
                        opt_r2_urls = {}  # index 0..3 → r2 url
                        for img in uploaded:
                            if "opt_key" in img:
                                idx = OPT_KEY_MAP.get(img["opt_key"])
                                if idx is not None:
                                    opt_r2_urls[idx] = img["url"]

                        if img_url_map or opt_r2_urls:
                            new_question_text = _replace_images(q.get("question", ""))
                            # For options: if opt_r2_url set, replace entire cell with [IMAGE:url]
                            def _opt_text(i, raw):
                                if i in opt_r2_urls:
                                    return f"[IMAGE:{opt_r2_urls[i]}]"
                                return _replace_images(raw)

                            new_opt1 = _opt_text(0, opts[0] if len(opts) > 0 else None)
                            new_opt2 = _opt_text(1, opts[1] if len(opts) > 1 else None)
                            new_opt3 = _opt_text(2, opts[2] if len(opts) > 2 else None)
                            new_opt4 = _opt_text(3, opts[3] if len(opts) > 3 else None)
                            new_solution = _replace_images(q.get("solution") or None)

                            _db_execute(cur,
                                """UPDATE questions
                                   SET question_text = %s,
                                       option_1 = %s, option_2 = %s,
                                       option_3 = %s, option_4 = %s
                                   WHERE id = %s""",
                                new_question_text,
                                new_opt1, new_opt2, new_opt3, new_opt4,
                                question_id,
                            )
                            if new_solution:
                                _db_execute(cur,
                                    "UPDATE answers SET solution_text = %s WHERE question_id = %s",
                                    new_solution, question_id,
                                )

                        for img in uploaded:
                            _db_execute(cur,
                                """INSERT INTO images
                                       (question_id, image_url, position, width_px, height_px)
                                   VALUES (%s, %s, %s, %s, %s)
                                   ON CONFLICT DO NOTHING""",
                                question_id,
                                img["url"], img["position"],
                                img.get("width_px"), img.get("height_px"),
                            )
                    except Exception as e:
                        print(f"[WARN] Image upload failed for Q{question_id}: {e}", flush=True)

            except Exception as e:
                import traceback as _tb
                print(f"[save_questions] Q{q.get('number','?')} failed: {_tb.format_exc()}", flush=True)
                # Continue saving other questions even if one fails

    _update_job(job_id, questions_saved=len(inserted_ids))
    return inserted_ids