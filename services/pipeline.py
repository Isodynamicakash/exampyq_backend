"""
services/pipeline.py
====================
Pipeline: ZIP (.tex + images) → LLM parse + tag → admin review → save to DB

CHANGES:
  - Now uses Anthropic Claude Haiku (ANTHROPIC_API_KEY)
  - Generalized parser handles ANY MCQ exam format
  - Auto-tagging with GPT-4o-mini for chapter/topic/difficulty
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

from services.llm_parser import parse_latex_with_llm

# API key — read from env at startup. Frontend never sends this.
_ANTHROPIC_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

if not _ANTHROPIC_KEY:
    import logging as _log
    _log.getLogger(__name__).warning(
        "[pipeline] ANTHROPIC_API_KEY not set — LLM parsing will fail. "
        "Set it in Railway environment variables."
    )

import json as _json

JOBS_ROOT = Path(os.environ.get(
    "JOBS_ROOT",
    str(Path(os.environ.get("TEMP", "/tmp")) / "examside_jobs")
))
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


def _find_images_dir(job_dir: Path, tex_path: Path) -> Optional[Path]:
    for parent in (tex_path.parent, job_dir):
        for name in ("images", "img", "Images", "Img", "figures"):
            candidate = parent / name
            if candidate.is_dir():
                imgs = (list(candidate.glob("*.jpg")) +
                        list(candidate.glob("*.png")) +
                        list(candidate.glob("*.jpeg")))
                if imgs:
                    return candidate
    for folder in sorted(job_dir.rglob("*")):
        if folder.is_dir():
            imgs = (list(folder.glob("*.jpg")) +
                    list(folder.glob("*.png")) +
                    list(folder.glob("*.jpeg")))
            if imgs:
                return folder
    return None


def _mark_image_availability(questions: list, images_dir: Optional[Path]) -> list:
    if not images_dir or not images_dir.exists():
        for q in questions:
            all_imgs = q.get("q_images", []) + q.get("sol_images", [])
            q["images_found"]   = []
            q["images_missing"] = all_imgs
        return questions

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
    job = get_job(job_id)
    candidate_dirs = []
    if job and job.get("images_dir"):
        candidate_dirs.append(Path(job["images_dir"]))
    default_dir = JOBS_ROOT / job_id / "images"
    if default_dir not in candidate_dirs:
        candidate_dirs.append(default_dir)

    for images_dir in candidate_dirs:
        if not images_dir.exists():
            continue
        for ext in ("", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"):
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        stem = Path(image_id).stem
        for f in images_dir.iterdir():
            if f.stem == stem:
                return f
    return None


async def run_pipeline_zip(
    job_id: str,
    zip_bytes: bytes,
    filename: str,
    pool=None,
    openai_api_key: str = "",
):
    try:
        _update_job(job_id, status="processing", progress=10)

        job_dir  = Path(f"/tmp/examside_jobs/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)

        zip_path = job_dir / "upload.zip"
        zip_path.write_bytes(zip_bytes)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(job_dir)
        zip_path.unlink()

        _update_job(job_id, progress=20, job_dir=str(job_dir))

        tex_files = list(job_dir.rglob("*.tex"))
        if not tex_files:
            raise ValueError("No .tex file found in ZIP.")
        tex_path = max(tex_files, key=lambda p: p.stat().st_size)

        images_dir = _find_images_dir(job_dir, tex_path)
        img_count  = len(list(images_dir.glob("*"))) if images_dir else 0

        _update_job(job_id, progress=30,
                    images_dir=str(images_dir) if images_dir else None,
                    image_count=img_count)

        _update_job(job_id, status="parsing", progress=40)
        tex_content = tex_path.read_text(encoding="utf-8", errors="replace")

        print(f"[pipeline/zip] Calling Haiku LLM parser, key_set={bool(_ANTHROPIC_KEY)}", flush=True)
        questions = await parse_latex_with_llm(
            tex     = tex_content,
            api_key = _ANTHROPIC_KEY,
        )
        print(f"[pipeline/zip] Haiku parser returned {len(questions)} questions", flush=True)

        _update_job(job_id, progress=85)
        questions = _mark_image_availability(questions, images_dir)
        _update_job(job_id, status="ready", progress=100, questions=questions)

    except Exception:
        err = traceback.format_exc()
        print(f"[pipeline/zip] ERROR:\n{err}", flush=True)
        _update_job(job_id, status="failed", progress=0, error=err[:3000])


async def run_pipeline_tex(
    job_id: str,
    tex_bytes: bytes,
    filename: str,
    pool=None,
    openai_api_key: str = "",
):
    try:
        _update_job(job_id, status="processing", progress=10)

        job_dir  = Path(f"/tmp/examside_jobs/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)

        tex_path = job_dir / "output.tex"
        tex_path.write_bytes(tex_bytes)

        images_dir = job_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        _update_job(job_id, progress=20,
                    job_dir=str(job_dir),
                    images_dir=str(images_dir),
                    image_count=0)

        _update_job(job_id, status="parsing", progress=40)
        tex_content = tex_bytes.decode("utf-8", errors="replace")

        print(f"[pipeline/tex] Calling Haiku LLM parser, key_set={bool(_ANTHROPIC_KEY)}", flush=True)
        questions = await parse_latex_with_llm(
            tex     = tex_content,
            api_key = _ANTHROPIC_KEY,
        )
        print(f"[pipeline/tex] Haiku parser returned {len(questions)} questions", flush=True)

        _update_job(job_id, progress=85)
        questions = _mark_image_availability(questions, images_dir)
        _update_job(job_id, status="ready", progress=100, questions=questions)

    except Exception:
        err = traceback.format_exc()
        print(f"[pipeline/tex] ERROR:\n{err}", flush=True)
        _update_job(job_id, status="failed", progress=0, error=err[:3000])


async def run_pipeline_pdf(
    job_id: str,
    pdf_bytes: bytes,
    filename: str,
    pool=None,
    openai_api_key: str = "",
):
    try:
        if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
            _update_job(job_id, status="failed", progress=0,
                        error="MATHPIX_APP_ID / MATHPIX_APP_KEY not set in .env")
            return

        from services.mathpix import run_mathpix_pipeline

        _update_job(job_id, status="mathpix", progress=10)
        job_dir = await run_mathpix_pipeline(pdf_bytes, filename, job_id)

        _update_job(job_id, status="parsing", progress=50)

        tex_files = list(job_dir.glob("*.tex"))
        if not tex_files:
            _update_job(job_id, status="failed", progress=50,
                        error="MathPix succeeded but no .tex file found")
            return

        tex_path    = tex_files[0]
        tex_content = tex_path.read_text(encoding="utf-8", errors="replace")
        images_dir  = _find_images_dir(job_dir, tex_path)

        _update_job(job_id, progress=60,
                    images_dir=str(images_dir) if images_dir else None,
                    image_count=len(list(images_dir.glob("*"))) if images_dir else 0)

        print(f"[pipeline/pdf] Calling Haiku LLM parser, key_set={bool(_ANTHROPIC_KEY)}", flush=True)
        questions = await parse_latex_with_llm(
            tex     = tex_content,
            api_key = _ANTHROPIC_KEY,
        )
        print(f"[pipeline/pdf] Haiku parser returned {len(questions)} questions", flush=True)

        _update_job(job_id, progress=90)
        questions = _mark_image_availability(questions, images_dir)
        _update_job(job_id, status="ready", progress=100, questions=questions)

    except Exception:
        err = traceback.format_exc()
        print(f"[pipeline/pdf] ERROR:\n{err}", flush=True)
        _update_job(job_id, status="failed", progress=0, error=err[:3000])


import psycopg2.extras


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

    row = _db_fetchone(cur, "SELECT id FROM exams WHERE name = %s", exam_name)
    if row:
        exam_id = row["id"]
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", exam_name.lower()).strip("-")
        row  = _db_fetchone(cur,
            """INSERT INTO exams (name, slug, exam_category)
               VALUES (%s, %s, 'engineering')
               ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
               RETURNING id""",
            exam_name, slug)
        exam_id = row["id"]

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
               ON CONFLICT (exam_id, year, shift, exam_date)
               DO UPDATE SET year = COALESCE(EXCLUDED.year, papers.year)
               RETURNING id""",
            exam_id, year_int, shift_val, date_val)
    except Exception:
        row = _db_fetchone(cur,
            """INSERT INTO papers (exam_id, year, shift, exam_date)
               VALUES (%s, %s, %s, %s::date)
               ON CONFLICT (exam_id, year, shift, exam_date)
               DO UPDATE SET exam_id = EXCLUDED.exam_id
               RETURNING id""",
            exam_id, year_int, shift_val, date_val)
    return row["id"]


def _resolve_chapter_id(cur, chapter_name: str, subject_name: str, exam_id: int) -> int:
    subject_name = (subject_name or "Physics").strip().title()

    row = _db_fetchone(cur,
        "SELECT id FROM subjects WHERE exam_id=%s AND name=%s", exam_id, subject_name)
    if row:
        subject_id = row["id"]
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", subject_name.lower()).strip("-")
        row  = _db_fetchone(cur,
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
    row  = _db_fetchone(cur,
        """INSERT INTO topics (chapter_id, name, slug)
           VALUES (%s, %s, %s)
           ON CONFLICT (chapter_id, slug) DO UPDATE SET name = EXCLUDED.name
           RETURNING id""",
        chapter_id, topic_name, slug)
    return row["id"]


async def save_questions_to_db(
    job_id: str,
    reviewed_questions: list[dict],
    pool,
) -> list[int]:
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
                q_type      = q.get("q_type", "MCQ")
                has_diagram = bool(q.get("q_images") or q.get("sol_images"))

                paper_id = _resolve_paper_id(
                    cur,
                    exam_name = q.get("exam_name") or "JEE Main",
                    year      = q.get("year", ""),
                    exam_date = q.get("exam_date", ""),
                    shift     = q.get("shift", ""),
                )

                chapter_name = (q.get("chapter_name") or "").strip() or "Uncategorised"
                exam_row     = _db_fetchone(cur, "SELECT exam_id FROM papers WHERE id=%s", paper_id)
                exam_id      = exam_row["exam_id"]

                chapter_id = _resolve_chapter_id(
                    cur,
                    chapter_name = chapter_name,
                    subject_name = q.get("subject", "Physics"),
                    exam_id      = exam_id,
                )

                topic_id = _resolve_topic_id(
                    cur,
                    topic_name = q.get("topic_name") or q.get("topic", ""),
                    chapter_id = chapter_id,
                )

                base = re.sub(r"[^a-z0-9]+", "-", q.get("question", "")[:60].lower()).strip("-")
                slug = f"{base}-{int(time.time())}-{q.get('number', 0)}"

                opts = q.get("options", [])
                row  = _db_fetchone(cur,
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

                if r2_configured and images_dir and images_dir.exists():
                    try:
                        from services.r2_upload import upload_question_images
                        uploaded = upload_question_images(
                            job_dir       = images_dir.parent,
                            question_id   = question_id,
                            q_image_ids   = q.get("q_images", []),
                            sol_image_ids = q.get("sol_images", []),
                            images_dir    = images_dir,
                            opt_image_ids = q.get("opt_images", {}),
                        )
                        img_url_map = {
                            img["image_id"]: img["url"]
                            for img in uploaded
                            if "image_id" in img
                        }

                        def _replace_images(text):
                            if not text:
                                return text
                            def _sub(m):
                                fname = m.group(1)
                                if fname in img_url_map:
                                    return f"[IMAGE:{img_url_map[fname]}]"
                                stem = fname.rsplit(".", 1)[0]
                                for k, v in img_url_map.items():
                                    if k.rsplit(".", 1)[0] == stem or k == stem:
                                        return f"[IMAGE:{v}]"
                                return m.group(0)
                            return re.sub(r"\[IMAGE:([^\]]+)\]", _sub, text)

                        OPT_KEY_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}
                        opt_r2_urls = {}
                        for img in uploaded:
                            if "opt_key" in img:
                                idx = OPT_KEY_MAP.get(img["opt_key"])
                                if idx is not None:
                                    opt_r2_urls[idx] = img["url"]

                        if img_url_map or opt_r2_urls:
                            new_question_text = _replace_images(q.get("question", ""))

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

            except Exception:
                import traceback as _tb
                print(f"[save_questions] Q{q.get('number','?')} failed:\n{_tb.format_exc()}", flush=True)

    _update_job(job_id, questions_saved=len(inserted_ids))
    return inserted_ids