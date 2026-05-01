"""
routers/questions.py  v2

KEY CHANGE:
  /api/questions/filters  now returns ONLY:
    - years, shifts, dates  (from DB — these change as papers are uploaded)
  Subject / chapter / topic are STATIC in the frontend (EXAM_TAXONOMY constant).
  This means no DB round-trip for taxonomy — only for live paper data.

Filtering still accepts:
  - subject    (slug string)   — matched against DB chapters.subject_id via join
  - chapter    (slug string)   — matched against DB chapters.slug
  - topic      (slug string)   — matched against DB topics.slug
  All slugs are the same ones baked into the frontend EXAM_TAXONOMY constant.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import time

router = APIRouter(tags=["questions"])

_filters_cache:    dict = {}
_filters_cache_ts: dict = {}
_FILTERS_TTL = 300  # 5 minutes


# ── /api/questions/filters  (years / shifts / dates only) ─────────────────────
def _get_filters_cached(exam_id: Optional[int] = None):
    key = exam_id or "all"
    if key in _filters_cache and (time.time() - _filters_cache_ts.get(key, 0)) < _FILTERS_TTL:
        return _filters_cache[key]
    result = _build_filters(exam_id)
    _filters_cache[key]    = result
    _filters_cache_ts[key] = time.time()
    return result


def _build_filters(exam_id: Optional[int] = None) -> dict:
    from core.database import get_cursor
    exam_where = f"AND e.id = {int(exam_id)}" if exam_id else ""

    with get_cursor() as cur:
        cur.execute(f"""
            SELECT DISTINCT
                p.year,
                p.shift,
                p.exam_date::text AS exam_date
            FROM papers p
            JOIN exams e ON e.id = p.exam_id
            JOIN questions q ON q.paper_id = p.id
            WHERE q.is_active = true {exam_where}
            ORDER BY p.year DESC NULLS LAST, p.shift
        """)
        rows = cur.fetchall()

    years  = sorted({r["year"]  for r in rows if r["year"]  is not None}, reverse=True)
    shifts = sorted({r["shift"] for r in rows if r["shift"] is not None})

    seen = set()
    dates = []
    for r in rows:
        if r["exam_date"]:
            key = r["exam_date"]
        elif r["year"]:
            key = str(r["year"]) + ("|" + r["shift"] if r["shift"] else "")
        else:
            continue
        full_key = key + "_" + (r["shift"] or "")
        if full_key not in seen:
            seen.add(full_key)
            dates.append({
                "exam_date": r["exam_date"],
                "year":      r["year"],
                "shift":     r["shift"],
                "key":       key,
            })
    dates.sort(key=lambda x: (str(x["year"] or ""), x["shift"] or ""), reverse=True)

    return {
        "years":          years,
        "shifts":         shifts,
        "dates":          dates,
        "difficulties":   ["easy", "medium", "hard"],
        "question_types": ["MCQ", "MSQ", "NUMERICAL"],
        # Subject/chapter/topic intentionally NOT returned — they live in frontend static EXAM_TAXONOMY
    }


@router.get("/api/questions/filters")
def get_filters(exam_id: Optional[int] = Query(default=None)):
    """
    Returns years, shifts, exam dates for the given exam.
    Subject/chapter/topic filters are static in the frontend — not fetched.
    """
    return _get_filters_cached(exam_id)


# ── /api/questions  (paginated list) ─────────────────────────────────────────
@router.get("/api/questions")
def list_questions(
    exam_id:       Optional[int] = Query(default=None),
    subject:       List[str]     = Query(default=[]),    # subject slug
    chapter:       List[str]     = Query(default=[]),    # chapter slug
    topic:         List[str]     = Query(default=[]),    # topic slug
    year:          List[int]     = Query(default=[]),
    shift:         List[str]     = Query(default=[]),
    difficulty:    List[str]     = Query(default=[]),
    question_type: List[str]     = Query(default=[]),
    exam_date:     List[str]     = Query(default=[]),
    limit:         int           = Query(20, le=100),
    offset:        int           = Query(0,  ge=0),
):
    from core.database import get_cursor
    with get_cursor() as cur:

        conditions = ["q.is_active = true"]
        params: list = []
        def ph(n): return ",".join(["%s"] * n)

        if exam_id is not None:
            conditions.append("p.exam_id = %s")
            params.append(exam_id)

        if subject:
            # subject = slug of subject e.g. "physics", "chemistry"
            vals = [v.strip().lower() for v in subject]
            conditions.append(f"LOWER(s.slug) = ANY(ARRAY[{ph(len(vals))}]::text[])")
            params.extend(vals)

        if chapter:
            # chapter = slug of chapter e.g. "laws-of-motion"
            vals = [v.strip().lower() for v in chapter]
            conditions.append(f"LOWER(c.slug) = ANY(ARRAY[{ph(len(vals))}]::text[])")
            params.extend(vals)

        if topic:
            # topic = slug of topic e.g. "friction"
            vals = [v.strip().lower() for v in topic]
            conditions.append(f"LOWER(t.slug) = ANY(ARRAY[{ph(len(vals))}]::text[])")
            params.extend(vals)

        if year:
            conditions.append(f"p.year = ANY(ARRAY[{ph(len(year))}]::int[])")
            params.extend(int(y) for y in year)

        if shift:
            vals = [v.strip() for v in shift]
            conditions.append(f"LOWER(p.shift) = ANY(ARRAY[{ph(len(vals))}]::text[])")
            params.extend(v.lower() for v in vals)

        if difficulty:
            vals = [v.strip().lower() for v in difficulty]
            conditions.append(f"LOWER(q.difficulty) = ANY(ARRAY[{ph(len(vals))}]::text[])")
            params.extend(vals)

        if question_type:
            vals = [v.strip().upper() for v in question_type]
            conditions.append(f"q.question_type = ANY(ARRAY[{ph(len(vals))}]::text[])")
            params.extend(vals)

        if exam_date:
            date_conds = []
            for key in exam_date:
                key = key.strip()
                if "-" in key and len(key) == 10:
                    date_conds.append("p.exam_date::text = %s")
                    params.append(key)
                elif "|" in key:
                    parts = key.split("|", 1)
                    try:
                        date_conds.append("(p.year = %s AND LOWER(p.shift) = LOWER(%s))")
                        params.extend([int(parts[0]), parts[1]])
                    except (ValueError, IndexError):
                        pass
                else:
                    try:
                        date_conds.append("p.year = %s")
                        params.append(int(key))
                    except ValueError:
                        pass
            if date_conds:
                conditions.append("(" + " OR ".join(date_conds) + ")")

        where = " AND ".join(conditions)
        joins = """
            FROM questions q
            JOIN papers   p ON p.id  = q.paper_id
            JOIN exams    e ON e.id  = p.exam_id
            JOIN chapters c ON c.id  = q.chapter_id
            JOIN subjects s ON s.id  = c.subject_id
            LEFT JOIN topics t ON t.id = q.topic_id
        """

        cur.execute(f"SELECT COUNT(*) AS total {joins} WHERE {where}", params)
        total = int((cur.fetchone() or {}).get("total", 0))

        cur.execute(f"""
            SELECT
                q.id, q.slug, q.question_number, q.question_type,
                q.question_text,
                q.option_1, q.option_2, q.option_3, q.option_4,
                q.difficulty, q.marks_positive, q.marks_negative,
                q.has_diagram,
                p.year, p.shift,
                p.exam_date::text AS exam_date,
                e.name  AS exam_name,
                s.name  AS subject_name,
                s.slug  AS subject_slug,
                c.name  AS chapter_name,
                c.slug  AS chapter_slug,
                t.name  AS topic_name,
                t.slug  AS topic_slug
            {joins}
            WHERE {where}
            ORDER BY
                s.name ASC,
                p.exam_date DESC NULLS LAST,
                p.year DESC NULLS LAST,
                q.question_number ASC NULLS LAST
            LIMIT %s OFFSET %s
        """, params + [limit, offset])

        questions = [dict(r) for r in cur.fetchall()]

    return {"total": total, "count": len(questions), "offset": offset, "questions": questions}


# ── Single question endpoints (unchanged) ─────────────────────────────────────
@router.get("/api/questions/{slug}/answer")
def get_answer(slug: str):
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("""
            SELECT q.id, a.correct_option, a.solution_text,
                COALESCE(
                    json_agg(json_build_object('url',i.image_url,'position',i.position)
                        ORDER BY i.id) FILTER (WHERE i.id IS NOT NULL), '[]'::json
                ) AS images
            FROM questions q
            LEFT JOIN answers a ON a.question_id = q.id
            LEFT JOIN images  i ON i.question_id = q.id AND i.position = 'solution'
            WHERE q.slug = %s AND q.is_active = true
            GROUP BY q.id, a.id
        """, (slug,))
        row = cur.fetchone()
        if not row or row["id"] is None:
            raise HTTPException(status_code=404, detail="Question not found")
    return dict(row)


@router.get("/api/questions/{slug}")
def get_question(slug: str):
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                q.id, q.slug, q.question_number, q.question_type,
                q.question_text,
                q.option_1, q.option_2, q.option_3, q.option_4,
                q.difficulty, q.marks_positive, q.marks_negative,
                q.has_diagram,
                p.year, p.shift, p.exam_date::text AS exam_date,
                e.name AS exam_name,
                s.name AS subject_name, s.slug AS subject_slug,
                c.name AS chapter_name, c.slug AS chapter_slug,
                t.name AS topic_name,  t.slug AS topic_slug,
                COALESCE(
                    json_agg(json_build_object(
                        'url',i.image_url,'position',i.position,
                        'width',i.width_px,'height',i.height_px
                    ) ORDER BY i.id) FILTER (WHERE i.id IS NOT NULL), '[]'::json
                ) AS images
            FROM questions q
            JOIN papers   p ON p.id = q.paper_id
            JOIN exams    e ON e.id = p.exam_id
            JOIN chapters c ON c.id = q.chapter_id
            JOIN subjects s ON s.id = c.subject_id
            LEFT JOIN topics t ON t.id = q.topic_id
            LEFT JOIN images i ON i.question_id = q.id
            WHERE q.slug = %s AND q.is_active = true
            GROUP BY q.id, p.id, e.id, c.id, s.id, t.id
        """, (slug,))
        row = cur.fetchone()
        if not row or row["id"] is None:
            raise HTTPException(status_code=404, detail="Question not found")
    return dict(row)
