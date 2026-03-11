"""
routers/questions.py — Public student-facing question browser API

ENDPOINTS:
  GET /api/questions/filters  — all available filter options (cached)
  GET /api/questions          — filtered, paginated question list
  GET /api/questions/:slug/answer — answer + solution only
  GET /api/questions/:slug    — single question with images
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import time

router = APIRouter(prefix="/api/questions", tags=["questions"])

# ── Simple in-process cache for /filters ─────────────────────────────────────
# Filters rarely change (only when new papers uploaded).
# Cache for 5 minutes to avoid 6 DB queries on every page load.
_filters_cache: dict = {}
_filters_cache_ts: float = 0.0
_FILTERS_TTL = 300  # seconds


def _get_filters_cached():
    global _filters_cache, _filters_cache_ts
    if _filters_cache and (time.time() - _filters_cache_ts) < _FILTERS_TTL:
        return _filters_cache
    _filters_cache = _build_filters()
    _filters_cache_ts = time.time()
    return _filters_cache


def _build_filters() -> dict:
    """
    Single DB round-trip using JSON aggregation.
    Replaces 6 separate queries with 1.
    """
    from core.database import get_cursor
    with get_cursor() as cur:

        # All subjects that have active questions
        cur.execute("""
            SELECT DISTINCT s.id, s.name, s.slug
            FROM subjects s
            JOIN chapters c ON c.subject_id = s.id
            JOIN questions q ON q.chapter_id = c.id
            WHERE q.is_active = true
            ORDER BY s.name
        """)
        subjects = [dict(r) for r in cur.fetchall()]

        # All chapters that have active questions (with subject name for cascade)
        cur.execute("""
            SELECT DISTINCT c.id, c.name, c.slug,
                   s.name AS subject_name, s.slug AS subject_slug
            FROM chapters c
            JOIN subjects s ON s.id = c.subject_id
            JOIN questions q ON q.chapter_id = c.id
            WHERE q.is_active = true
            ORDER BY s.name, c.name
        """)
        chapters = [dict(r) for r in cur.fetchall()]

        # Topics (only chapters that have topics with active questions)
        cur.execute("""
            SELECT DISTINCT t.id, t.name, t.slug,
                   c.name AS chapter_name, c.slug AS chapter_slug
            FROM topics t
            JOIN chapters c ON c.id = t.chapter_id
            JOIN questions q ON q.topic_id = t.id
            WHERE q.is_active = true
            ORDER BY c.name, t.name
        """)
        topics = [dict(r) for r in cur.fetchall()]

        # Years + shifts + papers in ONE query
        cur.execute("""
            SELECT DISTINCT p.year, p.shift,
                   p.exam_date::text AS exam_date,
                   e.name AS exam_name
            FROM papers p
            JOIN exams e ON e.id = p.exam_id
            JOIN questions q ON q.paper_id = p.id
            WHERE q.is_active = true
            ORDER BY p.year DESC NULLS LAST, p.shift
        """)
        paper_rows = cur.fetchall()

    years  = sorted({r["year"]  for r in paper_rows if r["year"]  is not None}, reverse=True)
    shifts = sorted({r["shift"] for r in paper_rows if r["shift"] is not None})
    papers = [dict(r) for r in paper_rows if r["exam_date"]]

    # Build dates from exam_date if available, else from year+shift
    seen_dates = set()
    dates = []
    for r in paper_rows:
        if r["exam_date"]:
            key = r["exam_date"]
            label_date = r["exam_date"]
        elif r["year"]:
            # Use year|shift as synthetic key
            key = str(r["year"]) + ("|" + r["shift"] if r["shift"] else "")
            label_date = None
        else:
            continue
        full_key = key + "_" + (r["shift"] or "")
        if full_key not in seen_dates:
            seen_dates.add(full_key)
            dates.append({
                "exam_date": r["exam_date"],  # may be None
                "year": r["year"],
                "shift": r["shift"],
                # composite key used for filtering
                "key": key,
            })
    dates.sort(key=lambda x: (str(x["year"] or ""), x["shift"] or ""), reverse=True)

    return {
        "subjects":       subjects,
        "chapters":       chapters,
        "topics":         topics,
        "years":          years,
        "shifts":         shifts,
        "papers":         papers,
        "dates":          dates,
        "difficulties":   ["easy", "medium", "hard"],
        "question_types": ["MCQ", "MSQ", "NUMERICAL"],
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/filters")
def get_filters():
    """
    Returns all available filter values. Cached for 5 minutes.
    """
    return _get_filters_cached()


@router.get("")
def list_questions(
    subject:       list[str] = Query(default=[]),
    chapter:       list[str] = Query(default=[]),
    topic:         list[str] = Query(default=[]),
    year:          list[int] = Query(default=[]),
    shift:         list[str] = Query(default=[]),
    difficulty:    list[str] = Query(default=[]),
    question_type: list[str] = Query(default=[]),
    exam_date:     list[str] = Query(default=[]),
    limit:         int           = Query(20, le=100),
    offset:        int           = Query(0, ge=0),
):
    """
    Returns filtered, paginated questions (no answers).
    Subject filter STRICTLY scopes results — no cross-subject bleed.
    """
    from core.database import get_cursor
    with get_cursor() as cur:

        conditions = ["q.is_active = true"]
        params: list = []

        def _ph(n): return ",".join(["%s"]*n)  # placeholder helper

        if subject:
            vals = [v.strip() for v in subject]
            conditions.append(f"LOWER(s.name) = ANY(ARRAY[{_ph(len(vals))}]::text[])")
            params.extend([v.lower() for v in vals])

        if chapter:
            vals = [v.strip() for v in chapter]
            conditions.append(
                f"(LOWER(c.slug) = ANY(ARRAY[{_ph(len(vals))}]::text[]) "
                f"OR LOWER(c.name) = ANY(ARRAY[{_ph(len(vals))}]::text[]))"
            )
            params.extend([v.lower() for v in vals] * 2)

        if topic:
            vals = [v.strip() for v in topic]
            conditions.append(
                f"(LOWER(t.slug) = ANY(ARRAY[{_ph(len(vals))}]::text[]) "
                f"OR LOWER(t.name) = ANY(ARRAY[{_ph(len(vals))}]::text[]))"
            )
            params.extend([v.lower() for v in vals] * 2)

        if year:
            conditions.append(f"p.year = ANY(ARRAY[{_ph(len(year))}]::int[])")
            params.extend([int(y) for y in year])

        if shift:
            vals = [v.strip() for v in shift]
            conditions.append(f"LOWER(p.shift) = ANY(ARRAY[{_ph(len(vals))}]::text[])")
            params.extend([v.lower() for v in vals])

        if difficulty:
            vals = [v.strip().lower() for v in difficulty]
            conditions.append(f"LOWER(q.difficulty) = ANY(ARRAY[{_ph(len(vals))}]::text[])")
            params.extend(vals)

        if question_type:
            vals = [v.strip().upper() for v in question_type]
            conditions.append(f"q.question_type = ANY(ARRAY[{_ph(len(vals))}]::text[])")
            params.extend(vals)

        if exam_date:
            # values are: "YYYY-MM-DD" (real date), "YYYY|Shift N" (year+shift), or "YYYY" (year only)
            date_conds = []
            for key in exam_date:
                key = key.strip()
                if "-" in key and len(key) == 10:
                    # real exam_date e.g. "2024-01-27"
                    date_conds.append("p.exam_date::text = %s")
                    params.append(key)
                elif "|" in key:
                    # year|shift composite e.g. "2024|Shift 1"
                    parts = key.split("|", 1)
                    try:
                        y = int(parts[0])
                        s = parts[1]
                        date_conds.append("(p.year = %s AND LOWER(p.shift) = LOWER(%s))")
                        params.extend([y, s])
                    except (ValueError, IndexError):
                        pass
                else:
                    # just a year e.g. "2024"
                    try:
                        date_conds.append("p.year = %s")
                        params.append(int(key))
                    except ValueError:
                        pass
            if date_conds:
                conditions.append("(" + " OR ".join(date_conds) + ")")

        where = " AND ".join(conditions)

        # Count first (same WHERE, no LIMIT)
        cur.execute(f"""
            SELECT COUNT(*) AS total
            FROM questions q
            JOIN papers   p ON p.id = q.paper_id
            JOIN chapters c ON c.id = q.chapter_id
            JOIN subjects s ON s.id = c.subject_id
            LEFT JOIN topics t ON t.id = q.topic_id
            WHERE {where}
        """, params)
        row = cur.fetchone()
        total = int(row["total"]) if row else 0

        # Main query
        cur.execute(f"""
            SELECT
                q.id, q.slug, q.question_number, q.question_type,
                q.question_text,
                q.option_1, q.option_2, q.option_3, q.option_4,
                q.difficulty, q.marks_positive, q.marks_negative,
                q.has_diagram,
                p.year, p.shift, p.exam_date::text AS exam_date,
                e.name  AS exam_name,
                c.name  AS chapter_name,  c.slug AS chapter_slug,
                s.name  AS subject_name,  s.slug AS subject_slug,
                t.name  AS topic_name
            FROM questions q
            JOIN papers   p ON p.id = q.paper_id
            JOIN exams    e ON e.id = p.exam_id
            JOIN chapters c ON c.id = q.chapter_id
            JOIN subjects s ON s.id = c.subject_id
            LEFT JOIN topics t ON t.id = q.topic_id
            WHERE {where}
            ORDER BY
                s.name ASC,
                p.exam_date DESC NULLS LAST,
                p.year DESC NULLS LAST,
                q.question_number ASC NULLS LAST
            LIMIT %s OFFSET %s
        """, params + [limit, offset])

        questions = [dict(r) for r in cur.fetchall()]

    return {
        "total":     total,
        "count":     len(questions),
        "offset":    offset,
        "questions": questions,
    }


@router.get("/{slug}/answer")
def get_answer(slug: str):
    """
    Returns answer + solution (called when student clicks 'Show Answer').
    Grouped by q.id, a.id — never merges rows from different questions.
    """
    from core.database import get_cursor
    with get_cursor() as cur:
        cur.execute("""
            SELECT
                q.id,
                a.correct_option,
                a.solution_text,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'url',      i.image_url,
                            'position', i.position
                        ) ORDER BY i.id
                    ) FILTER (WHERE i.id IS NOT NULL),
                    '[]'::json
                ) AS images
            FROM questions q
            LEFT JOIN answers a ON a.question_id = q.id
            LEFT JOIN images  i ON i.question_id = q.id
                AND i.position = 'solution'
            WHERE q.slug = %s AND q.is_active = true
            GROUP BY q.id, a.id
        """, (slug,))
        row = cur.fetchone()

        # LEFT JOIN means we may get a row with all NULLs if question missing
        if not row or row["id"] is None:
            raise HTTPException(status_code=404, detail="Question not found")

    return dict(row)


@router.get("/{slug}")
def get_question(slug: str):
    """Single question with all images."""
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
                c.name AS chapter_name, s.name AS subject_name,
                t.name AS topic_name,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'url',      i.image_url,
                            'position', i.position,
                            'width',    i.width_px,
                            'height',   i.height_px
                        ) ORDER BY i.id
                    ) FILTER (WHERE i.id IS NOT NULL),
                    '[]'::json
                ) AS images
            FROM questions q
            JOIN papers   p ON p.id = q.paper_id
            JOIN exams    e ON e.id = p.exam_id
            JOIN chapters c ON c.id = q.chapter_id
            JOIN subjects s ON s.id = c.subject_id
            LEFT JOIN topics t ON t.id = q.topic_id
            LEFT JOIN images  i ON i.question_id = q.id
            WHERE q.slug = %s AND q.is_active = true
            GROUP BY q.id, p.id, e.id, c.id, s.id, t.id
        """, (slug,))
        row = cur.fetchone()
        if not row or row["id"] is None:
            raise HTTPException(status_code=404, detail="Question not found")
    return dict(row)