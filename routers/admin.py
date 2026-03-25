"""
routers/admin.py — Admin API endpoints
(Updated: uses ANTHROPIC_API_KEY via pipeline._OPENAI_KEY)
"""

import asyncio
import os
import uuid as _uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Header
from fastapi.responses import FileResponse
from typing import Optional

from models.schemas import SaveQuestionsRequest, SaveQuestionsResponse
from services.pipeline import (
    create_job, get_job, _update_job,
    run_pipeline_zip, run_pipeline_tex, run_pipeline_pdf,
    save_questions_to_db, get_image_path,
    _OPENAI_KEY,
)


router    = APIRouter(prefix="/api/admin", tags=["admin"])
ADMIN_KEY = os.environ.get("ADMIN_KEY", "changeme")


def require_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin key")


@router.post("/upload-zip", dependencies=[Depends(require_admin)])
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file")
    data = await file.read()
    if len(data) > 500 * 1024 * 1024:
        raise HTTPException(400, "ZIP too large (max 500MB)")
    if not _OPENAI_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not set on server. Contact administrator.")
    job_id = create_job(file.filename)
    asyncio.create_task(run_pipeline_zip(job_id, data, file.filename))
    return {"job_id": job_id, "status": "processing", "mode": "zip"}


@router.post("/upload-tex", dependencies=[Depends(require_admin)])
async def upload_tex(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".tex"):
        raise HTTPException(400, "Only .tex files accepted")
    if not _OPENAI_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not set on server. Contact administrator.")
    data   = await file.read()
    job_id = create_job(file.filename)
    asyncio.create_task(run_pipeline_tex(job_id, data, file.filename))
    return {"job_id": job_id, "status": "processing", "mode": "tex_only"}


@router.post("/upload-pdf", dependencies=[Depends(require_admin)])
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files accepted")
    if not os.environ.get("MATHPIX_APP_ID") or not os.environ.get("MATHPIX_APP_KEY"):
        raise HTTPException(400, "MATHPIX_APP_ID and MATHPIX_APP_KEY must be set in .env")
    if not _OPENAI_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not set on server. Contact administrator.")
    data   = await file.read()
    job_id = create_job(file.filename)
    asyncio.create_task(run_pipeline_pdf(job_id, data, file.filename))
    return {"job_id": job_id, "status": "processing", "mode": "pdf"}


@router.post("/upload-tex-images", dependencies=[Depends(require_admin)])
async def upload_tex_images(
    file:   UploadFile               = File(...),
    images: Optional[list[UploadFile]] = File(default=None),
):
    if not file.filename.lower().endswith(".tex"):
        raise HTTPException(400, "file must be a .tex file")
    if not _OPENAI_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not set on server. Contact administrator.")
    import io
    import zipfile as _zf
    tex_data = await file.read()
    buf = io.BytesIO()
    with _zf.ZipFile(buf, "w") as zf:
        zf.writestr(file.filename, tex_data)
        for img in (images or []):
            img_data = await img.read()
            zf.writestr(f"images/{img.filename}", img_data)
    buf.seek(0)
    zip_bytes = buf.read()
    job_id = create_job(file.filename)
    asyncio.create_task(run_pipeline_zip(job_id, zip_bytes, file.filename))
    return {"job_id": job_id, "status": "processing", "mode": "tex_images"}


@router.post("/upload-image", dependencies=[Depends(require_admin)])
async def upload_image(
    file:    UploadFile = File(...),
    job_id:  str        = None,
    section: str        = "question",
):
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


@router.get("/jobs/{job_id}/questions", dependencies=[Depends(require_admin)])
async def job_questions(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] != "ready":
        raise HTTPException(400, f"Job not ready (status={job['status']})")
    return {"job_id": job_id, "questions": job["questions"]}


@router.get("/temp-image/{job_id}/{image_id:path}")
async def temp_image(job_id: str, image_id: str):
    path = get_image_path(job_id, image_id)
    if not path:
        raise HTTPException(404, f"Image '{image_id}' not found for job {job_id}")
    suffix = path.suffix.lower().lstrip(".")
    media  = {
        "jpg":  "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
        "gif":  "image/gif",  "webp": "image/webp", "svg": "image/svg+xml",
    }.get(suffix, "image/png")
    return FileResponse(str(path), media_type=media)


@router.get("/questions", dependencies=[Depends(require_admin)])
def list_questions_admin(
    limit:   int           = 200,
    offset:  int           = 0,
    subject: Optional[str] = None,
    search:  Optional[str] = None,
):
    from core.database import get_cursor
    with get_cursor() as cur:
        conditions = ["q.is_active = true"]
        params: list = []
        if subject:
            conditions.append("LOWER(s.name) = LOWER(%s)")
            params.append(subject)
        if search:
            conditions.append(
                "(q.question_text ILIKE %s OR CAST(q.question_number AS TEXT) ILIKE %s)"
            )
            params.extend([f"%{search}%", f"%{search}%"])
        where = " AND ".join(conditions)
        cur.execute(f"""
            SELECT COUNT(*) AS total FROM questions q
            LEFT JOIN papers   p ON p.id = q.paper_id
            LEFT JOIN chapters c ON c.id = q.chapter_id
            LEFT JOIN subjects s ON s.id = c.subject_id
            WHERE {where}
        """, params)
        total_row = cur.fetchone()
        total = int(total_row["total"]) if total_row else 0
        cur.execute(f"""
            SELECT
                q.id, q.question_number, q.question_type AS q_type,
                q.question_text AS question, q.option_1, q.option_2, q.option_3, q.option_4,
                q.difficulty, q.marks_positive AS marks_correct, q.marks_negative AS marks_wrong,
                q.is_verified, COALESCE(p.year, 0) AS year, COALESCE(p.shift, '') AS shift,
                p.exam_date::text AS exam_date, COALESCE(e.name, '') AS exam_name,
                COALESCE(c.name, '') AS chapter_name, COALESCE(s.name, '') AS subject,
                COALESCE(t.name, '') AS topic_name,
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
            ORDER BY p.exam_date DESC NULLS LAST, q.question_number ASC NULLS LAST
            LIMIT %s OFFSET %s
        """, params + [limit, offset])
        rows = cur.fetchall()
    questions = []
    for r in rows:
        d = dict(r)
        d["options"] = [
            d.pop("option_1") or "", d.pop("option_2") or "",
            d.pop("option_3") or "", d.pop("option_4") or "",
        ]
        d["q_images"] = []; d["sol_images"] = []; d["opt_images"] = {}
        questions.append(d)
    return {"total": total, "count": len(questions), "questions": questions}


@router.post("/save-questions", response_model=SaveQuestionsResponse,
             dependencies=[Depends(require_admin)])
async def save_questions(body: SaveQuestionsRequest):
    pool = None
    job  = get_job(body.job_id)
    if not job:
        raise HTTPException(404, f"Job {body.job_id} not found")
    try:
        ids = await save_questions_to_db(
            body.job_id,
            [q.model_dump() for q in body.questions],
            pool,
        )
    except Exception as e:
        import traceback
        msg = traceback.format_exc()
        print(f"[save-questions ERROR]\n{msg}", flush=True)
        raise HTTPException(500, msg)
    return SaveQuestionsResponse(saved_count=len(ids), question_ids=ids)


@router.get("/debug-config", dependencies=[Depends(require_admin)])
async def debug_config():
    return {
        "anthropic_key_set":    bool(_OPENAI_KEY),
        "anthropic_key_prefix": (_OPENAI_KEY[:12] + "...") if _OPENAI_KEY else "NOT SET",
        "mathpix_set":          bool(os.environ.get("MATHPIX_APP_ID")),
        "r2_configured":        all([
            os.environ.get("R2_ENDPOINT_URL"), os.environ.get("R2_ACCESS_KEY_ID"),
            os.environ.get("R2_SECRET_ACCESS_KEY"), os.environ.get("R2_BUCKET"),
        ]),
    }


@router.post("/test-parser", dependencies=[Depends(require_admin)])
async def test_parser():
    import traceback as _tb
    from services.llm_parser import parse_latex_with_llm
    dummy_tex = r"""
\title{JEE Main 2024 - January - 27-01-2024 - Morning Shift}
\begin{document}
\section*{PHYSICS}
\begin{enumerate}
\item A ball is thrown upward with velocity 20 m/s. Find maximum height. $g = 10 \text{ m/s}^2$
(1) 10 m \quad (2) 20 m \quad (3) 30 m \quad (4) 40 m
\end{enumerate}
\section*{Sol. (2)}
Using $v^2 = u^2 - 2gh$, at max height $v=0$: $h = \frac{u^2}{2g} = \frac{400}{20} = 20$ m
\end{document}
"""
    diagnostics = {"anthropic_key_set": bool(_OPENAI_KEY), "result": None, "error": None}
    if not _OPENAI_KEY:
        diagnostics["error"] = "ANTHROPIC_API_KEY not set in environment"
        return diagnostics
    try:
        questions = await parse_latex_with_llm(tex=dummy_tex, api_key=_OPENAI_KEY)
        diagnostics["result"] = {
            "questions_found": len(questions),
            "first_question":  questions[0] if questions else None,
        }
    except Exception:
        diagnostics["error"] = _tb.format_exc()
    return diagnostics


@router.get("/chapters", dependencies=[Depends(require_admin)])
def list_chapters():
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("""
            SELECT c.id, c.name, s.name AS subject_name, c.subject_id
            FROM chapters c JOIN subjects s ON s.id = c.subject_id
            ORDER BY s.name, c.name
        """)
        return [dict(r) for r in cur.fetchall()]


@router.get("/topics", dependencies=[Depends(require_admin)])
def list_topics():
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("""
            SELECT t.id, t.name, c.name AS chapter_name, c.id AS chapter_id, s.name AS subject_name
            FROM topics t JOIN chapters c ON c.id = t.chapter_id JOIN subjects s ON s.id = c.subject_id
            ORDER BY c.name, t.name
        """)
        return [dict(r) for r in cur.fetchall()]


@router.get("/papers", dependencies=[Depends(require_admin)])
def list_papers():
    from core.database import get_cursor
    with get_cursor() as cur:
        try:
            cur.execute("""
                SELECT p.id, e.name AS exam_name, p.year, p.exam_date,
                       p.shift, p.exam_date::text AS exam_date_str
                FROM papers p JOIN exams e ON e.id = p.exam_id
                ORDER BY p.exam_date DESC NULLS LAST, p.year DESC, p.shift
            """)
        except Exception:
            cur.execute("SELECT id, year, shift, exam_date::text AS exam_date_str FROM papers ORDER BY year DESC, shift")
        return [dict(r) for r in cur.fetchall()]


@router.get("/stats", dependencies=[Depends(require_admin)])
def stats():
    from core.database import get_cursor
    with get_cursor() as cur:
        try:
            cur.execute("""
                SELECT COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE is_verified = true)  AS verified,
                    COUNT(*) FILTER (WHERE is_verified = false) AS pending,
                    COUNT(*) FILTER (WHERE chapter_id IS NULL)  AS untagged
                FROM questions
            """)
            row = cur.fetchone() or {}
        except Exception:
            row = {}
    return dict(row)


@router.get("/queue", dependencies=[Depends(require_admin)])
def queue():
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("""
            SELECT q.id, q.question_text, q.question_type, q.chapter_id, q.is_verified, a.correct_option
            FROM questions q LEFT JOIN answers a ON a.question_id = q.id
            WHERE q.is_verified = false ORDER BY q.id DESC LIMIT 100
        """)
        return [dict(r) for r in cur.fetchall()]


@router.put("/update-question/{question_id}", dependencies=[Depends(require_admin)])
def update_question_full(question_id: int, body: dict):
    import json as _json
    from core.database import get_cursor
    allowed_cols = {
        "question","solution","answer","options","chapter_name","topic_name","subject_name",
        "exam_date","year","shift","exam_name","q_type","difficulty","q_images","sol_images","opt_images",
    }
    updates = {k: v for k, v in body.items() if k in allowed_cols}
    if not updates:
        raise HTTPException(400, "No valid fields to update")
    with get_cursor() as cur:
        opt_images  = updates.get("opt_images", {}) or {}
        OPT_KEY_COL = {"a":"option_1","b":"option_2","c":"option_3","d":"option_4"}
        for opt_key, img_val in opt_images.items():
            col = OPT_KEY_COL.get(opt_key.lower())
            if col and img_val:
                cell_val = img_val if img_val.startswith("http") else f"[IMAGE:{img_val}]"
                cur.execute(f"UPDATE questions SET {col} = %s WHERE id = %s", (cell_val, question_id))
        options = updates.get("options")
        if options and isinstance(options, list):
            for i, col in enumerate(["option_1","option_2","option_3","option_4"]):
                opt_key = ["a","b","c","d"][i]
                if opt_key not in opt_images:
                    val = options[i] if i < len(options) else None
                    cur.execute(f"UPDATE questions SET {col} = %s WHERE id = %s", (val, question_id))
        if "solution" in updates:
            cur.execute("UPDATE answers SET solution_text = %s WHERE question_id = %s",
                       (updates["solution"], question_id))
        if "answer" in updates:
            cur.execute("""INSERT INTO answers (question_id, correct_option)
                           VALUES (%s, %s) ON CONFLICT (question_id)
                           DO UPDATE SET correct_option = EXCLUDED.correct_option""",
                       (question_id, updates["answer"]))
        DIRECT = {"question":"question_text","q_type":"question_type","difficulty":"difficulty"}
        for field, col in DIRECT.items():
            if field in updates:
                val = updates[field]
                if isinstance(val, (list, dict)): val = _json.dumps(val)
                cur.execute(f"UPDATE questions SET {col} = %s WHERE id = %s", (val, question_id))
    return {"updated": True, "id": question_id}


@router.post("/questions/{question_id}/verify", dependencies=[Depends(require_admin)])
def verify_question(question_id: int):
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("UPDATE questions SET is_verified = true WHERE id = %s", (question_id,))
    return {"verified": True}


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