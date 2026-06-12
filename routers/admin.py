"""
routers/admin.py — Admin API endpoints

All routes require header:  x-admin-key: <ADMIN_KEY>

ENDPOINTS:
  POST  /api/admin/upload-zip                       Upload ZIP (.tex + images) → parse
  POST  /api/admin/upload-tex                       Upload .tex only → parse (no images)
  POST  /api/admin/upload-image                     Upload a single image into a job
  GET   /api/admin/jobs/{job_id}                    Poll job status
  GET   /api/admin/jobs/{job_id}/questions          Get parsed questions
  GET   /api/admin/temp-image/{job_id}/{image_id}   Serve image (NO auth — browser img tags)
  POST  /api/admin/save-questions                   Save verified questions to DB
  GET   /api/admin/questions                        List all questions (admin edit screen)
  GET   /api/admin/chapters                         List chapters
  GET   /api/admin/topics                           List topics
  GET   /api/admin/papers                           List papers
  GET   /api/admin/stats                            Dashboard stats
  GET   /api/admin/queue                            Unverified questions
  PATCH /api/admin/questions/{id}                   Edit saved question
  POST  /api/admin/questions/{id}/verify            Verify one question
  POST  /api/admin/bulk-verify                      Bulk verify

NOTES:
  - temp-image has NO auth: browsers load <img src> without custom headers.
  - image_id uses :path so FastAPI accepts IDs with dots/underscores/hyphens.
"""

import asyncio
import os
import uuid as _uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, File, Header
from fastapi.responses import FileResponse
from typing import Optional

from models.schemas import SaveQuestionsRequest
from services.pipeline import (
    create_job, get_job, _update_job,
    run_pipeline_zip, run_pipeline_tex, run_pipeline_pdf,
    save_questions_to_db, get_image_path,
)


router    = APIRouter(prefix="/api/admin", tags=["admin"])
ADMIN_KEY = os.environ.get("ADMIN_KEY", "changeme")


# ── Auth ──────────────────────────────────────────────────────────────────────

def require_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin key")


# ── Upload ZIP (.tex + images) ────────────────────────────────────────────────

@router.post("/upload-zip", dependencies=[Depends(require_admin)])
async def upload_zip(
    file: UploadFile = File(...),
    x_openai_key: str = Header(default=""),
    x_exam_type:  str = Header(default=""),
):
    """Main upload: ZIP containing .tex + images/ folder."""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file")

    data = await file.read()
    if len(data) > 500 * 1024 * 1024:
        raise HTTPException(400, "ZIP too large (max 500MB)")

    job_id = create_job(file.filename)
    pool = None  # psycopg2 backend — no async pool
    asyncio.create_task(run_pipeline_zip(job_id, data, file.filename, pool=pool,
                                         openai_api_key=x_openai_key, exam_type=x_exam_type))

    return {"job_id": job_id, "status": "processing", "mode": "zip"}


# ── Upload .tex only ──────────────────────────────────────────────────────────

@router.post("/upload-tex", dependencies=[Depends(require_admin)])
async def upload_tex(
    file: UploadFile = File(...),
    x_openai_key: str = Header(default=""),
    x_exam_type:  str = Header(default=""),
):
    """Upload just the .tex file — no images."""
    if not file.filename.lower().endswith(".tex"):
        raise HTTPException(400, "Only .tex files accepted")

    data   = await file.read()
    job_id = create_job(file.filename)
    pool   = None  # psycopg2 backend — no async pool
    asyncio.create_task(run_pipeline_tex(job_id, data, file.filename, pool=pool,
                                          openai_api_key=x_openai_key, exam_type=x_exam_type))

    return {"job_id": job_id, "status": "processing", "mode": "tex_only"}


# ── Upload raw PDF → MathPix → parse ─────────────────────────────────────────

@router.post("/upload-pdf", dependencies=[Depends(require_admin)])
async def upload_pdf(
    file: UploadFile = File(...),
    x_openai_key: str = Header(default=""),
    x_exam_type:  str = Header(default=""),
):
    """
    Upload a raw PDF — backend sends it to MathPix API, polls until done,
    downloads the .tex + images, then runs the normal parser pipeline.
    Requires MATHPIX_APP_ID and MATHPIX_APP_KEY in .env
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files accepted")

    if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
        raise HTTPException(400, "MATHPIX_APP_ID and MATHPIX_APP_KEY must be set in .env to use PDF upload")

    data   = await file.read()
    job_id = create_job(file.filename)
    pool = None  # psycopg2 backend — no async pool
    asyncio.create_task(run_pipeline_pdf(job_id, data, file.filename, pool=pool,
                                          openai_api_key=x_openai_key, exam_type=x_exam_type))

    return {"job_id": job_id, "status": "processing", "mode": "pdf"}


# ── Upload TEX file + multiple image files (new mode) ───────────────────────

@router.post("/upload-tex-images", dependencies=[Depends(require_admin)])
async def upload_tex_images(
    file:         UploadFile                    = File(...),
    images:       Optional[list[UploadFile]]    = File(default=None),
    x_openai_key: str                           = Header(default=""),
    x_exam_type:  str                           = Header(default=""),
):
    """
    Upload a .tex file + separate image files (no ZIP needed).
    Frontend TEX + 🖼 mode sends: file=tex, images=[img1, img2, ...]
    """
    if not file.filename.lower().endswith(".tex"):
        raise HTTPException(400, "file must be a .tex file")

    from pathlib import Path as _Path
    import zipfile as _zf, io as _io

    tex_data = await file.read()

    # Re-pack as a ZIP so the normal pipeline can handle it
    buf = _io.BytesIO()
    with _zf.ZipFile(buf, "w") as zf:
        zf.writestr(file.filename, tex_data)
        for img in (images or []):
            img_data = await img.read()
            zf.writestr(f"images/{img.filename}", img_data)
    buf.seek(0)
    zip_bytes = buf.read()

    job_id = create_job(file.filename)
    pool   = None  # psycopg2 backend — no async pool
    asyncio.create_task(
        run_pipeline_zip(job_id, zip_bytes, file.filename, pool=pool,
                         openai_api_key=x_openai_key, exam_type=x_exam_type)
    )
    return {"job_id": job_id, "status": "processing", "mode": "tex_images"}


# ── Upload single image into an existing job ──────────────────────────────────

@router.post("/upload-image", dependencies=[Depends(require_admin)])
async def upload_image(
    file:    UploadFile = File(...),
    job_id:  str        = None,
    section: str        = "question",
):
    """
    Upload a single image into an existing job's temp images dir.
    Pass job_id as query param: POST /api/admin/upload-image?job_id=xxx
    Returns image_id to use as [IMAGE:image_id] in question/solution text.
    """
    if not job_id:
        raise HTTPException(400, "job_id query param required")

    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    from services.pipeline import JOBS_ROOT
    job_dir    = JOBS_ROOT / job_id
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    suffix   = Path(file.filename or "img.jpg").suffix.lower() or ".jpg"
    image_id = f"manual_{_uuid.uuid4().hex[:12]}{suffix}"
    dest     = images_dir / image_id

    data = await file.read()
    dest.write_bytes(data)

    if not job.get("images_dir"):
        _update_job(job_id, images_dir=str(images_dir))

    return {"image_id": image_id, "filename": file.filename}


# ── Poll job status ───────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}", dependencies=[Depends(require_admin)])
async def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return {
        "job_id":         job["job_id"],
        "status":         job["status"],
        "filename":       job["filename"],
        "progress":       job["progress"],
        "question_count": len(job.get("questions") or []),
        "image_count":    job.get("image_count", 0),
        "error":          job.get("error"),
    }


# ── Get parsed questions ──────────────────────────────────────────────────────

@router.get("/jobs/{job_id}/questions", dependencies=[Depends(require_admin)])
async def job_questions(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "ready":
        raise HTTPException(400, f"Job not ready (status={job['status']})")
    return {"job_id": job_id, "questions": job["questions"]}


# ── Serve temp images (NO auth — browsers can't send custom headers in <img>) ─

@router.get("/temp-image/{job_id}/{image_id:path}")
async def temp_image(job_id: str, image_id: str):
    """
    Serve extracted/uploaded images back to the frontend for preview.
    NO auth — images are scoped to a UUID job_id and are temp /tmp files.
    :path on image_id accepts IDs with dots, underscores, hyphens.
    """
    path = get_image_path(job_id, image_id)
    if not path:
        raise HTTPException(404, f"Image '{image_id}' not found for job {job_id}")

    suffix = path.suffix.lower().lstrip(".")
    media  = {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "gif":  "image/gif",
        "webp": "image/webp",
        "svg":  "image/svg+xml",
    }.get(suffix, "image/png")

    return FileResponse(str(path), media_type=media)


# ── Permanent image upload for editing existing questions (no job_id needed) ──

@router.post("/upload-question-image", dependencies=[Depends(require_admin)])
async def upload_question_image(
    file:        UploadFile = File(...),
    question_id: int        = Query(...),
    section:     str        = Query(default="question"),  # "question" | "solution" | "opt_a".."opt_d"
):
    """
    Upload an image directly for an existing DB question.
    Stores permanently under JOBS_ROOT/persistent_images/.
    Saves a row in the images table and returns a permanent URL.
    No job_id required — works even after jobs have expired.
    """
    from services.pipeline import JOBS_ROOT
    from core.database import get_cursor

    # Store in a permanent folder that never gets cleaned up
    perm_dir = JOBS_ROOT / "persistent_images" / str(question_id)
    perm_dir.mkdir(parents=True, exist_ok=True)

    suffix   = Path(file.filename or "img.jpg").suffix.lower() or ".jpg"
    image_id = f"q{question_id}_{section}_{_uuid.uuid4().hex[:10]}{suffix}"
    dest     = perm_dir / image_id

    data = await file.read()
    dest.write_bytes(data)

    # Map section → position value stored in images table
    position_map = {
        "question": "question", "solution": "solution",
        "opt_a": "option_1", "opt_b": "option_2",
        "opt_c": "option_3", "opt_d": "option_4",
    }
    position = position_map.get(section, "question")

    # Save to images table so it's permanent and retrievable
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO images (question_id, image_url, position)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (question_id, image_id, position))

    return {
        "image_id":    image_id,
        "question_id": question_id,
        "section":     section,
        "url":         f"/api/admin/question-image/{question_id}/{image_id}",
    }


@router.get("/question-image/{question_id}/{image_id:path}")
async def serve_question_image(question_id: int, image_id: str):
    """
    Serve a permanently stored question image. NO auth — used in <img> tags.
    """
    from services.pipeline import JOBS_ROOT
    perm_dir = JOBS_ROOT / "persistent_images" / str(question_id)
    path = perm_dir / image_id
    if not path.exists():
        raise HTTPException(404, f"Image not found: {image_id}")

    suffix = path.suffix.lower().lstrip(".")
    media  = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "gif":  "image/gif",
        "webp":"image/webp", "svg":  "image/svg+xml",
    }.get(suffix, "image/png")

    return FileResponse(str(path), media_type=media)


# ── Admin: list all questions (for Edit Existing Questions screen) ────────────

@router.get("/questions", dependencies=[Depends(require_admin)])
def list_questions_admin(
    # Pagination — no artificial cap, offset-based
    limit:         int            = Query(default=100, ge=1),
    offset:        int            = Query(default=0, ge=0),
    # All paper-level filters
    exam_id:       Optional[int]  = Query(default=None),   # 1=JEE Mains, 3=NEET
    paper_id:      Optional[int]  = Query(default=None),   # exact paper
    exam_name:     Optional[str]  = Query(default=None),   # e.g. "JEE Main"
    year:          Optional[int]  = Query(default=None),
    shift:         Optional[str]  = Query(default=None),
    exam_date:     Optional[str]  = Query(default=None),   # "YYYY-MM-DD"
    # Question-level filters
    subject:       Optional[str]  = Query(default=None),
    chapter:       Optional[str]  = Query(default=None),
    topic:         Optional[str]  = Query(default=None),
    difficulty:    Optional[str]  = Query(default=None),
    question_type: Optional[str]  = Query(default=None),
    is_verified:   Optional[bool] = Query(default=None),
    search:        Optional[str]  = Query(default=None),
):
    """
    Returns questions for the admin Edit Existing screen.
    All filters from uploaded papers are supported.
    No artificial limit — returns all matching questions (use offset for paging).
    """
    from core.database import get_cursor
    with get_cursor() as cur:
        conditions = ["q.is_active = true"]
        params: list = []

        # ── Top-level exam scope ─────────────────────────────────────────────
        if exam_id is not None:
            conditions.append("p.exam_id = %s")
            params.append(exam_id)

        if paper_id is not None:
            conditions.append("q.paper_id = %s")
            params.append(paper_id)

        if exam_name:
            conditions.append("LOWER(e.name) = LOWER(%s)")
            params.append(exam_name)

        # ── Paper date/shift/year filters ────────────────────────────────────
        if year is not None:
            conditions.append("p.year = %s")
            params.append(year)

        if shift:
            conditions.append("LOWER(p.shift) = LOWER(%s)")
            params.append(shift)

        if exam_date:
            conditions.append("p.exam_date::text = %s")
            params.append(exam_date)

        # ── Question-level filters ────────────────────────────────────────────
        if subject:
            conditions.append("LOWER(s.name) = LOWER(%s)")
            params.append(subject)

        if chapter:
            conditions.append(
                "(LOWER(c.name) = LOWER(%s) OR LOWER(c.slug) = LOWER(%s))"
            )
            params.extend([chapter, chapter])

        if topic:
            conditions.append(
                "(LOWER(t.name) = LOWER(%s) OR LOWER(t.slug) = LOWER(%s))"
            )
            params.extend([topic, topic])

        if difficulty:
            conditions.append("LOWER(q.difficulty) = LOWER(%s)")
            params.append(difficulty)

        if question_type:
            conditions.append("q.question_type = UPPER(%s)")
            params.append(question_type)

        if is_verified is not None:
            conditions.append("q.is_verified = %s")
            params.append(is_verified)

        if search:
            conditions.append(
                "(q.question_text ILIKE %s OR CAST(q.question_number AS TEXT) ILIKE %s)"
            )
            params.extend([f"%{search}%", f"%{search}%"])

        where = " AND ".join(conditions)

        # Total count (no LIMIT)
        cur.execute(f"""
            SELECT COUNT(*) AS total
            FROM questions q
            LEFT JOIN papers   p ON p.id = q.paper_id
            LEFT JOIN exams    e ON e.id = p.exam_id
            LEFT JOIN chapters c ON c.id = q.chapter_id
            LEFT JOIN subjects s ON s.id = c.subject_id
            LEFT JOIN topics   t ON t.id = q.topic_id
            WHERE {where}
        """, params)
        total_row = cur.fetchone()
        total = int(total_row["total"]) if total_row else 0

        # Main query — all fields the frontend needs, no artificial limit
        cur.execute(f"""
            SELECT
                q.id,
                q.question_number,
                q.question_type          AS q_type,
                q.question_text          AS question,
                q.option_1,
                q.option_2,
                q.option_3,
                q.option_4,
                q.difficulty,
                q.marks_positive         AS marks_correct,
                q.marks_negative         AS marks_wrong,
                q.is_verified,
                COALESCE(p.id, 0)        AS paper_id,
                COALESCE(p.year, 0)      AS year,
                COALESCE(p.shift, '')    AS shift,
                p.exam_date::text        AS exam_date,
                COALESCE(e.id, 0)        AS exam_id,
                COALESCE(e.name, '')     AS exam_name,
                COALESCE(c.name, '')     AS chapter_name,
                COALESCE(s.name, '')     AS subject,
                COALESCE(t.name, '')     AS topic_name,
                COALESCE(a.correct_option, '') AS answer,
                COALESCE(a.solution_text, '')  AS solution
            FROM questions q
            LEFT JOIN papers   p ON p.id = q.paper_id
            LEFT JOIN exams    e ON e.id = p.exam_id
            LEFT JOIN chapters c ON c.id = q.chapter_id
            LEFT JOIN subjects s ON s.id = c.subject_id
            LEFT JOIN topics   t ON t.id = q.topic_id
            LEFT JOIN answers  a ON a.question_id = q.id
            WHERE {where}
            ORDER BY e.id ASC, p.exam_date DESC NULLS LAST, q.question_number ASC NULLS LAST
            LIMIT %s OFFSET %s
        """, params + [limit, offset])

        rows = cur.fetchall()

    questions = []
    for r in rows:
        d = dict(r)
        d["options"] = [
            d.pop("option_1") or "",
            d.pop("option_2") or "",
            d.pop("option_3") or "",
            d.pop("option_4") or "",
        ]
        d["q_images"]   = []
        d["sol_images"] = []
        d["opt_images"] = {}
        questions.append(d)

    return {"total": total, "count": len(questions), "questions": questions}


# ── Save verified questions to DB (new upload flow — requires job_id) ────────

@router.post("/save-questions",
             dependencies=[Depends(require_admin)])
async def save_questions(body: SaveQuestionsRequest):
    pool = None
    job = get_job(body.job_id)
    if not job:
        raise HTTPException(404, f"Job {body.job_id} not found")

    try:
        ids, failures = await save_questions_to_db(
            body.job_id,
            [q.model_dump() for q in body.questions],
            pool,
        )
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"[save-questions ERROR]\n{msg}", flush=True)
        raise HTTPException(500, msg)

    # failures is returned so the frontend can show WHY a question was rejected
    # (constraint name etc.) instead of silently treating HTTP 200 as success.
    return {
        "saved_count":  len(ids),
        "question_ids": ids,
        "failed_count": len(failures),
        "failed":       failures,
    }


# ── Create a brand-new question directly (no job_id — for Edit Existing screen) ──

@router.post("/create-question", dependencies=[Depends(require_admin)])
def create_question(body: dict):
    """
    Insert a new question directly into the DB without any job_id.
    Used when admin adds a blank question in the Edit Existing screen.
    Reuses the same paper/chapter/subject/topic resolution as the pipeline.
    """
    import re as _re, time as _time
    from core.database import get_cursor
    from services.pipeline import (
        _resolve_paper_id, _resolve_chapter_id, _resolve_topic_id,
        _db_fetchone, _db_execute,
    )

    with get_cursor() as cur:
        exam_name    = (body.get("exam_name") or "JEE Main").strip()
        year         = body.get("year") or ""
        exam_date    = body.get("exam_date") or ""
        shift        = body.get("shift") or ""
        subject      = (body.get("subject") or body.get("subject_name") or "Physics").strip()
        chapter_name = (body.get("chapter_name") or "Uncategorised").strip()
        topic_name   = body.get("topic_name") or ""
        q_type       = body.get("q_type") or "MCQ"
        difficulty   = body.get("difficulty") or None
        question     = body.get("question") or ""
        options      = body.get("options") or ["", "", "", ""]
        answer       = body.get("answer") or None
        solution     = body.get("solution") or None
        marks_pos    = body.get("marks_correct") or 4
        marks_neg    = body.get("marks_wrong") or -1
        q_number     = body.get("question_number") or body.get("number") or None

        paper_id = _resolve_paper_id(
            cur,
            exam_name = exam_name,
            year      = year,
            exam_date = exam_date,
            shift     = shift,
        )

        exam_row = _db_fetchone(cur, "SELECT exam_id FROM papers WHERE id=%s", paper_id)
        exam_id  = exam_row["exam_id"]

        chapter_id = _resolve_chapter_id(
            cur,
            chapter_name = chapter_name,
            subject_name = subject,
            exam_id      = exam_id,
        )

        topic_id = _resolve_topic_id(cur, topic_name, chapter_id)

        base = _re.sub(r"[^a-z0-9]+", "-", question[:60].lower()).strip("-") or "q"
        slug = f"{base}-{int(_time.time())}-manual"

        opts = options if isinstance(options, list) else ["", "", "", ""]

        row = _db_fetchone(cur,
            """INSERT INTO questions (
                   slug, paper_id, chapter_id, topic_id,
                   question_number, question_type,
                   marks_positive, marks_negative,
                   question_text, option_1, option_2, option_3, option_4,
                   difficulty, has_diagram, is_verified, is_active
               ) VALUES (
                   %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
               )
               ON CONFLICT (paper_id, question_number) DO UPDATE SET
                   slug           = EXCLUDED.slug,
                   chapter_id     = EXCLUDED.chapter_id,
                   topic_id       = EXCLUDED.topic_id,
                   question_type  = EXCLUDED.question_type,
                   marks_positive = EXCLUDED.marks_positive,
                   marks_negative = EXCLUDED.marks_negative,
                   question_text  = EXCLUDED.question_text,
                   option_1       = EXCLUDED.option_1,
                   option_2       = EXCLUDED.option_2,
                   option_3       = EXCLUDED.option_3,
                   option_4       = EXCLUDED.option_4,
                   difficulty     = EXCLUDED.difficulty,
                   is_verified    = EXCLUDED.is_verified,
                   is_active      = EXCLUDED.is_active
               RETURNING id""",
            slug,
            paper_id, chapter_id, topic_id,
            q_number, q_type,
            marks_pos, marks_neg,
            question,
            opts[0] if len(opts) > 0 else None,
            opts[1] if len(opts) > 1 else None,
            opts[2] if len(opts) > 2 else None,
            opts[3] if len(opts) > 3 else None,
            difficulty, False, True, True,
        )
        question_id = row["id"]

        if answer or solution:
            _db_execute(cur,
                """INSERT INTO answers (question_id, correct_option, solution_text)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (question_id) DO UPDATE
                       SET correct_option = EXCLUDED.correct_option,
                           solution_text  = EXCLUDED.solution_text""",
                question_id, answer, solution,
            )

    return {"created": True, "id": question_id, "slug": slug}


# ── Permanent image upload for editing existing questions (no job_id needed) ──

@router.post("/upload-question-image", dependencies=[Depends(require_admin)])
async def upload_question_image(
    file:        UploadFile = File(...),
    question_id: int        = Query(...),
    section:     str        = Query(default="question"),
):
    """
    Upload image for an existing DB question — permanent storage, no job_id.
    If R2 is configured: resizes + uploads to R2, saves URL to images table.
    If R2 is not configured: stores locally, serves via /question-image/ endpoint.
    section: "question" | "solution" | "opt_a" | "opt_b" | "opt_c" | "opt_d"
    """
    from services.pipeline import JOBS_ROOT
    from core.database import get_cursor

    perm_dir = JOBS_ROOT / "persistent_images" / str(question_id)
    perm_dir.mkdir(parents=True, exist_ok=True)

    suffix   = Path(file.filename or "img.jpg").suffix.lower() or ".jpg"
    image_id = f"q{question_id}_{section}_{_uuid.uuid4().hex[:10]}{suffix}"
    dest     = perm_dir / image_id

    data = await file.read()
    dest.write_bytes(data)

    # Position mapping for images table
    position_map = {
        "question": "question", "solution": "solution",
        "opt_a": "option_1", "opt_b": "option_2",
        "opt_c": "option_3", "opt_d": "option_4",
    }
    position = position_map.get(section, "question")

    # Try R2 upload if configured
    r2_configured = all([
        os.environ.get("R2_ENDPOINT_URL"),
        os.environ.get("R2_ACCESS_KEY_ID"),
        os.environ.get("R2_SECRET_ACCESS_KEY"),
        os.environ.get("R2_BUCKET"),
    ])

    final_url = image_id  # default: local ID (served via /question-image/)
    width_px  = None
    height_px = None

    if r2_configured:
        try:
            from services.r2_upload import upload_image as r2_upload_image
            result = r2_upload_image(dest, position, question_id)
            final_url = result["url"]
            width_px  = result.get("width_px")
            height_px = result.get("height_px")
        except Exception as e:
            print(f"[upload-question-image] R2 failed, falling back to local: {e}", flush=True)

    # Save to images table
    with get_cursor() as cur:
        cur.execute("""
            INSERT INTO images (question_id, image_url, position, width_px, height_px)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (question_id, final_url, position, width_px, height_px))

    return {
        "image_id":    final_url,   # always return final_url as image_id so frontend uses it directly
        "local_id":    image_id,
        "question_id": question_id,
        "section":     section,
        "r2_uploaded": r2_configured and final_url.startswith("http"),
    }


@router.get("/question-image/{question_id}/{image_id:path}")
async def serve_question_image(question_id: int, image_id: str):
    """Serve permanently stored question image. NO auth — used in <img> tags."""
    from services.pipeline import JOBS_ROOT
    perm_dir = JOBS_ROOT / "persistent_images" / str(question_id)
    path = perm_dir / image_id
    if not path.exists():
        raise HTTPException(404, f"Image not found: {image_id}")
    suffix = path.suffix.lower().lstrip(".")
    media  = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "gif":  "image/gif",
        "webp":"image/webp", "svg":  "image/svg+xml",
    }.get(suffix, "image/png")
    return FileResponse(str(path), media_type=media)


# ── Debug headers — see exactly what the server receives ─────────────────────

@router.get("/debug-headers", dependencies=[Depends(require_admin)])
async def debug_headers(
    request: Request,
    x_openai_key: str = Header(default="__NOT_SENT__"),
):
    """GET /api/admin/debug-headers — shows received headers & env key status."""
    from services.pipeline import _OPENAI_KEY_FROM_ENV, _LLM_TAGGING
    return {
        "x_openai_key_received": x_openai_key,
        "x_openai_key_length":   len(x_openai_key),
        "env_key_set":           bool(_OPENAI_KEY_FROM_ENV),
        "env_key_prefix":        _OPENAI_KEY_FROM_ENV[:12] + "..." if _OPENAI_KEY_FROM_ENV else "NONE",
        "_LLM_TAGGING":          _LLM_TAGGING,
        "all_headers":           dict(request.headers),
    }


# ── Test tagger endpoint — diagnose tagging issues ────────────────────────────

@router.post("/test-tagger", dependencies=[Depends(require_admin)])
async def test_tagger(x_openai_key: str = Header(default="")):
    """
    Quick diagnostic: tags 1 dummy Physics question and returns the result.
    POST /api/admin/test-tagger  with x-admin-key and x-openai-key headers.
    """
    import traceback as _tb
    from services.pipeline import _OPENAI_KEY_FROM_ENV, _LLM_TAGGING
    
    key = x_openai_key or _OPENAI_KEY_FROM_ENV
    
    diagnostics = {
        "_LLM_TAGGING":          _LLM_TAGGING,
        "key_provided":          bool(x_openai_key),
        "key_from_env":          bool(_OPENAI_KEY_FROM_ENV),
        "key_used_prefix":       key[:12] + "..." if key else "NONE",
        "openai_installed":      False,
        "api_call_result":       None,
        "error":                 None,
    }
    
    try:
        from openai import AsyncOpenAI
        diagnostics["openai_installed"] = True
    except ImportError as e:
        diagnostics["error"] = f"openai not installed: {e}"
        return diagnostics
    
    if not key:
        diagnostics["error"] = "No API key — set OPENAI_API_KEY in .env or send x-openai-key header"
        return diagnostics
    
    try:
        from services.llm_tagger import tag_questions_async
        dummy = [{
            "number": 1,
            "subject": "Physics",
            "question": "A ball is thrown upward with velocity 20 m/s. Find maximum height. g=10.",
            "options": ["20 m", "40 m", "10 m", "5 m"],
            "chapter_name": "",
            "topic_name": "",
            "difficulty": "",
        }]
        result = await tag_questions_async(dummy, subject="", pool=None, openai_api_key=key)
        q = result[0]
        diagnostics["api_call_result"] = {
            "chapter_name": q.get("chapter_name", ""),
            "topic_name":   q.get("topic_name", ""),
            "difficulty":   q.get("difficulty", ""),
        }
    except Exception:
        diagnostics["error"] = _tb.format_exc()
    
    return diagnostics


# ── Chapters list ─────────────────────────────────────────────────────────────

@router.get("/chapters", dependencies=[Depends(require_admin)])
def list_chapters():
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("""
            SELECT c.id, c.name, s.name AS subject_name, c.subject_id
            FROM chapters c
            JOIN subjects s ON s.id = c.subject_id
            ORDER BY s.name, c.name
        """)
        return [dict(r) for r in cur.fetchall()]


# ── Topics list ───────────────────────────────────────────────────────────────

@router.get("/topics", dependencies=[Depends(require_admin)])
def list_topics():
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("""
            SELECT t.id, t.name, c.name AS chapter_name, c.id AS chapter_id,
                   s.name AS subject_name
            FROM topics t
            JOIN chapters c ON c.id = t.chapter_id
            JOIN subjects s ON s.id = c.subject_id
            ORDER BY c.name, t.name
        """)
        return [dict(r) for r in cur.fetchall()]


# ── Papers list ───────────────────────────────────────────────────────────────

@router.get("/papers", dependencies=[Depends(require_admin)])
def list_papers():
    from core.database import get_cursor
    with get_cursor() as cur:
        try:
            cur.execute("""
                SELECT p.id, e.name AS exam_name, p.year, p.exam_date,
                       p.shift, p.exam_date::text AS exam_date_str
                FROM papers p
                JOIN exams e ON e.id = p.exam_id
                ORDER BY p.exam_date DESC NULLS LAST, p.year DESC, p.shift
            """)
        except Exception:
            cur.execute("""
                SELECT id, year, shift, exam_date::text AS exam_date_str
                FROM papers ORDER BY year DESC, shift
            """)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ── Dashboard stats ───────────────────────────────────────────────────────────

@router.get("/stats", dependencies=[Depends(require_admin)])
def stats():
    from core.database import get_cursor
    with get_cursor() as cur:
        try:
            cur.execute("""
            SELECT
                COUNT(*)                                    AS total,
                COUNT(*) FILTER (WHERE is_verified = true)  AS verified,
                COUNT(*) FILTER (WHERE is_verified = false) AS pending,
                COUNT(*) FILTER (WHERE chapter_id IS NULL)  AS untagged
            FROM questions
        """)
            row = cur.fetchone() or {}
        except Exception:
            row = {}
    return dict(row)


# ── Unverified queue ──────────────────────────────────────────────────────────

@router.get("/queue", dependencies=[Depends(require_admin)])
def queue():
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("""
            SELECT q.id, q.question_text, q.question_type,
                   q.chapter_id, q.is_verified, a.correct_option
            FROM questions q
            LEFT JOIN answers a ON a.question_id = q.id
            WHERE q.is_verified = false
            ORDER BY q.id DESC LIMIT 100
        """)
        return [dict(r) for r in cur.fetchall()]


# ── Edit question (full update) ───────────────────────────────────────────────

@router.put("/update-question/{question_id}", dependencies=[Depends(require_admin)])
def update_question_full(question_id: int, body: dict):
    """Full update of an existing question — uses psycopg2 get_cursor (sync)."""
    import json as _json
    from core.database import get_cursor

    allowed_cols = {
        "question", "solution", "answer", "options",
        "chapter_name", "topic_name", "subject_name",
        "exam_date", "year", "shift", "exam_name", "q_type",
        "difficulty", "q_images", "sol_images", "opt_images",
    }
    updates = {k: v for k, v in body.items() if k in allowed_cols}
    if not updates:
        raise HTTPException(400, "No valid fields to update")

    with get_cursor() as cur:
        opt_images  = updates.get("opt_images", {}) or {}
        OPT_KEY_COL = {"a": "option_1", "b": "option_2", "c": "option_3", "d": "option_4"}

        # ── opt_images: write image value into option columns ────────────────
        for opt_key, img_val in opt_images.items():
            col = OPT_KEY_COL.get(opt_key.lower())
            if col and img_val:
                cell_val = img_val if img_val.startswith("http") else f"[IMAGE:{img_val}]"
                cur.execute(f"UPDATE questions SET {col} = %s WHERE id = %s", (cell_val, question_id))

        # ── options array ────────────────────────────────────────────────────
        options = updates.get("options")
        if options and isinstance(options, list):
            for i, col in enumerate(["option_1", "option_2", "option_3", "option_4"]):
                opt_key = ["a", "b", "c", "d"][i]
                if opt_key not in opt_images:
                    val = options[i] if i < len(options) else None
                    cur.execute(f"UPDATE questions SET {col} = %s WHERE id = %s", (val, question_id))

        # ── solution ─────────────────────────────────────────────────────────
        if "solution" in updates:
            cur.execute(
                "UPDATE answers SET solution_text = %s WHERE question_id = %s",
                (updates["solution"], question_id)
            )

        # ── answer ───────────────────────────────────────────────────────────
        if "answer" in updates:
            cur.execute(
                """INSERT INTO answers (question_id, correct_option)
                   VALUES (%s, %s)
                   ON CONFLICT (question_id) DO UPDATE SET correct_option = EXCLUDED.correct_option""",
                (question_id, updates["answer"])
            )

        # ── direct question columns ──────────────────────────────────────────
        DIRECT = {
            "question":   "question_text",
            "q_type":     "question_type",
            "difficulty": "difficulty",
        }
        for field, col in DIRECT.items():
            if field in updates:
                val = updates[field]
                if isinstance(val, (list, dict)):
                    val = _json.dumps(val)
                cur.execute(f"UPDATE questions SET {col} = %s WHERE id = %s", (val, question_id))

    return {"updated": True, "id": question_id}


def edit_question(question_id: int, updates: dict):
    from core.database import get_cursor
    allowed = {
        "question_text", "option_1", "option_2", "option_3", "option_4",
        "difficulty", "chapter_id", "topic_id", "marks_positive", "marks_negative",
    }
    filtered = {k: v for k, v in updates.items() if k in allowed}
    if not filtered:
        raise HTTPException(400, "No valid fields to update")
    set_clause = ", ".join(f"{k} = %s" for k in filtered)
    with get_cursor() as cur:
        cur.execute(
            f"UPDATE questions SET {set_clause} WHERE id = %s",
            (*filtered.values(), question_id)
        )
    return {"updated": True}


# ── Verify one ────────────────────────────────────────────────────────────────

@router.post("/questions/{question_id}/verify", dependencies=[Depends(require_admin)])
def verify_question(question_id: int):
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("UPDATE questions SET is_verified = true WHERE id = %s", (question_id,))
    return {"verified": True}


# ── Bulk verify ───────────────────────────────────────────────────────────────

@router.post("/bulk-verify", dependencies=[Depends(require_admin)])
def bulk_verify(ids: list[int]):
    from core.database import get_cursor
    if len(ids) > 200:
        raise HTTPException(400, "Max 200 at once")
    with get_cursor() as cur:
        cur.execute("UPDATE questions SET is_verified = true WHERE id = ANY(%s)", (ids,))
    return {"verified_count": len(ids)}


@router.delete("/questions/{question_id}", dependencies=[Depends(require_admin)])
def delete_question(question_id: int):
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("SELECT id FROM questions WHERE id = %s", (question_id,))
        if not cur.fetchone():
            raise HTTPException(404, f"Question {question_id} not found")
        cur.execute("DELETE FROM questions WHERE id = %s", (question_id,))
    return {"deleted": question_id}
