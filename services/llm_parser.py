"""
services/llm_parser.py
======================
Gemini-powered question paper parser with structured JSON output.

FLOW:
  1. Receive raw LaTeX text
  2. Extract metadata (year, shift, date, exam type)
  3. Regex-split by subject (reliable, free, instant)
  4. One Gemini call per subject — forced structured JSON via response_schema
  5. Validate + postprocess
  6. Return list of question dicts + parse_summary (how many expected vs got)
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logger.warning("[llm_parser] google-generativeai not installed — run: pip install google-generativeai")

from services.prompts import (
    PARSER_SYSTEM_PROMPT,
    PARSER_USER_PROMPT_TEMPLATE,
    format_taxonomy_for_prompt,
    get_expected_count,
)
from services.llm_tagger import _TAXONOMY, _normalise_subject

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_CHAR_LIMIT = 900_000   # Gemini 2.5 Flash has 1M context
PARSE_MODEL      = "gemini-2.5-flash"

# ─────────────────────────────────────────────────────────────────────────────
# Structured JSON schema — Gemini MUST fill every field
# ─────────────────────────────────────────────────────────────────────────────

# This is the single source of truth for what we expect from Gemini.
# Gemini's response_schema enforces this — no hallucinated fields, no missing fields.
QUESTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "parse_summary": {
            "type": "OBJECT",
            "description": "Summary of parsing — how many questions were found vs expected",
            "properties": {
                "subject":          {"type": "STRING", "description": "Subject name e.g. PHYSICS"},
                "expected_count":   {"type": "INTEGER", "description": "How many questions expected for this subject"},
                "found_count":      {"type": "INTEGER", "description": "How many questions actually found"},
                "section_a_count":  {"type": "INTEGER", "description": "MCQ questions found in SECTION-A"},
                "section_b_count":  {"type": "INTEGER", "description": "NUMERICAL questions found in SECTION-B"},
                "missing_numbers":  {
                    "type": "ARRAY",
                    "items": {"type": "INTEGER"},
                    "description": "Question numbers that seem to be missing"
                },
                "notes": {"type": "STRING", "description": "Any issues noticed during parsing"}
            },
            "required": ["subject", "expected_count", "found_count", "section_a_count", "section_b_count", "missing_numbers", "notes"]
        },
        "questions": {
            "type": "ARRAY",
            "description": "All extracted questions",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "number":        {"type": "INTEGER",  "description": "Question number as it appears in paper"},
                    "q_type":        {"type": "STRING",   "description": "MCQ | MSQ | NUMERICAL"},
                    "subject":       {"type": "STRING",   "description": "PHYSICS | CHEMISTRY | MATHEMATICS | BIOLOGY"},
                    "section":       {"type": "STRING",   "description": "SECTION-A for MCQ, SECTION-B for numerical"},
                    "year":          {"type": "STRING",   "description": "Exam year e.g. 2021"},
                    "shift":         {"type": "STRING",   "description": "Morning or Evening"},
                    "exam_date":     {"type": "STRING",   "description": "YYYY-MM-DD format"},
                    "question":      {"type": "STRING",   "description": "Full question LaTeX text"},
                    "options":       {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "4 options for MCQ/MSQ, empty array for NUMERICAL"
                    },
                    "answer":        {"type": "STRING",   "description": "1/2/3/4 for MCQ, numeric string for NUMERICAL, empty if not found"},
                    "solution":      {"type": "STRING",   "description": "Full solution LaTeX text"},
                    "chapter_name":  {"type": "STRING",   "description": "Chapter from taxonomy list"},
                    "topic_name":    {"type": "STRING",   "description": "Topic from taxonomy list"},
                    "difficulty":    {"type": "STRING",   "description": "easy | medium | hard"},
                    "q_images":      {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Image filenames referenced in question/options"
                    },
                    "sol_images":    {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Image filenames referenced in solution"
                    },
                    "marks_correct": {"type": "NUMBER",   "description": "Marks for correct answer, usually 4"},
                    "marks_wrong":   {"type": "NUMBER",   "description": "Marks for wrong answer, usually -1 for MCQ, 0 for NUMERICAL"},
                },
                "required": [
                    "number", "q_type", "subject", "section",
                    "year", "shift", "exam_date",
                    "question", "options", "answer", "solution",
                    "chapter_name", "topic_name", "difficulty",
                    "q_images", "sol_images", "marks_correct", "marks_wrong"
                ]
            }
        }
    },
    "required": ["parse_summary", "questions"]
}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
    'sep':9,'oct':10,'nov':11,'dec':12,
}

def _extract_meta_from_latex(tex: str) -> dict:
    _mp = (r'(january|february|march|april|may|june|july|august|'
           r'september|october|november|december|jan|feb|mar|apr|'
           r'jun|jul|aug|sep|oct|nov|dec)')
    title = ""
    m = re.search(r'\\title\s*\{([^}]+)\}', tex)
    if m:
        title = m.group(1)
    combined = title + " " + tex[:1500]

    exam_date = year = shift = ""

    dm = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', combined)
    if dm:
        exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
    else:
        dm = re.search(r'\b(\d{2})/(\d{2})/(20\d{2})\b', combined)
        if dm:
            exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
        else:
            dm = re.search(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_mp}\s+(20\d{{2}})\b', combined, re.IGNORECASE)
            if dm:
                mo = _MONTH_MAP.get(dm.group(2).lower(), 0)
                if mo:
                    exam_date = f"{dm.group(3)}-{mo:02d}-{int(dm.group(1)):02d}"
            else:
                dm = re.search(rf'\b{_mp}\s+(\d{{1,2}}),?\s+(20\d{{2}})\b', combined, re.IGNORECASE)
                if dm:
                    mo = _MONTH_MAP.get(dm.group(1).lower(), 0)
                    if mo:
                        exam_date = f"{dm.group(3)}-{mo:02d}-{int(dm.group(2)):02d}"

    year = exam_date[:4] if exam_date else ""
    if not year:
        m = re.search(r'\b(20\d{2})\b', combined)
        if m:
            year = m.group(1)

    tl = combined.lower()
    if any(x in tl for x in ("morning", "shift 1", "shift-1", "shift1", "session 1")):
        shift = "Morning"
    elif any(x in tl for x in ("evening", "shift 2", "shift-2", "shift2", "session 2")):
        shift = "Evening"

    exam_type = "JEE Main"
    if re.search(r'jee\s*advanced', combined, re.IGNORECASE):
        exam_type = "JEE Advanced"
    elif re.search(r'neet', combined, re.IGNORECASE):
        exam_type = "NEET"
    elif re.search(r'cuet', combined, re.IGNORECASE):
        exam_type = "CUET"

    subjects = []
    for subj in ("PHYSICS", "CHEMISTRY", "MATHEMATICS", "BIOLOGY"):
        if re.search(subj, combined, re.IGNORECASE):
            subjects.append(subj)
    if not subjects:
        subjects = ["PHYSICS", "CHEMISTRY", "MATHEMATICS"]

    return {
        "exam_date": exam_date,
        "year": year,
        "shift": shift,
        "exam_type": exam_type,
        "subjects": subjects,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image placeholder normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_image_refs(tex: str) -> str:
    def _rep(m):
        path = m.group(1).strip()
        basename = path.split("/")[-1].split("\\")[-1]
        return f"[IMAGE:{basename}]"
    return re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', _rep, tex)


# ─────────────────────────────────────────────────────────────────────────────
# Regex subject split — reliable, free, instant
# ─────────────────────────────────────────────────────────────────────────────

def _split_by_subject(tex: str, subjects: list) -> list[tuple[str, str]]:
    """
    Split LaTeX by subject boundaries using regex.
    Returns list of (subject_label, chunk_text) in paper order.
    """
    SUBJ_PAT = r'(?:PHYSICS|CHEMISTRY|MATHEMATICS|BIOLOGY)'
    patterns = [
        rf'(?=\\section\*?\{{[^}}]*{SUBJ_PAT}[^}}]*\}})',
        rf'(?=PART\s*[-–]?\s*[A-Z]\s*[-–]?\s*:?\s*{SUBJ_PAT})',
        rf'(?=\n\s*{SUBJ_PAT}\s*\n)',
        rf'(?=\\textbf\{{[^}}]*{SUBJ_PAT}[^}}]*\}})',
        rf'(?=%+\s*[-=]*\s*{SUBJ_PAT})',
        rf'(?=\n\n\s*{SUBJ_PAT}\b)',
    ]

    for pat in patterns:
        parts = re.split(pat, tex, flags=re.IGNORECASE)
        if len(parts) >= 2:
            result = []
            for part in parts:
                m = re.search(r'(PHYSICS|CHEMISTRY|MATHEMATICS|BIOLOGY)', part[:300], re.IGNORECASE)
                if m:
                    label = m.group(1).upper()
                    if label in ("MATHS", "MATH"):
                        label = "MATHEMATICS"
                    if label in subjects:
                        result.append((label, part))
            if len(result) >= 2:
                logger.info(f"[llm_parser] Regex split: {[s for s,_ in result]}")
                return result

    logger.warning("[llm_parser] No subject boundaries found — sending full tex per subject")
    return [(s, tex) for s in subjects]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini API call — forced structured JSON via response_schema
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini_sync(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = PARSE_MODEL,
    max_tokens: int = 16000,
) -> dict:
    """
    Synchronous Gemini API call with response_schema.
    Gemini MUST return JSON matching QUESTION_SCHEMA — guaranteed valid.
    Returns parsed dict with 'questions' and 'parse_summary'.
    """
    genai.configure(api_key=api_key)

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=QUESTION_SCHEMA,
            temperature=0.0,
            max_output_tokens=max_tokens,
        )
    )

    try:
        response = gemini_model.generate_content(user_prompt)
        raw = response.text

        # Parse the guaranteed-valid JSON
        data = json.loads(raw)

        questions   = data.get("questions", [])
        summary     = data.get("parse_summary", {})

        logger.info(
            f"[llm_parser] Gemini {model} | "
            f"subject={summary.get('subject','?')} | "
            f"expected={summary.get('expected_count','?')} | "
            f"found={summary.get('found_count','?')} | "
            f"section_a={summary.get('section_a_count','?')} | "
            f"section_b={summary.get('section_b_count','?')} | "
            f"missing={summary.get('missing_numbers',[])} | "
            f"notes={summary.get('notes','')}"
        )

        return {"questions": questions, "parse_summary": summary}

    except Exception as e:
        logger.error(f"[llm_parser] Gemini call failed: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Validation & fixing
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "section":       "SECTION-A",
    "year":          "",
    "shift":         "",
    "exam_date":     "",
    "options":       [],
    "answer":        "",
    "solution":      "",
    "chapter_name":  "",
    "topic_name":    "",
    "difficulty":    "medium",
    "q_images":      [],
    "sol_images":    [],
    "marks_correct": 4,
    "marks_wrong":   -1,
    "verified":      False,
    "chapter_id":    None,
    "topic":         "",
}

_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_QTYPES       = {"MCQ", "MSQ", "NUMERICAL"}
_VALID_SUBJECTS     = {"PHYSICS", "CHEMISTRY", "MATHEMATICS", "BIOLOGY"}
_NTA_LETTER = {"A":"1","B":"2","C":"3","D":"4","a":"1","b":"2","c":"3","d":"4"}


def _fix_newlines(text: str) -> str:
    if not isinstance(text, str): return text
    return text.replace('\\n', '\n')


def _coerce_answer(ans: str, q_type: str) -> str:
    if not ans: return ""
    ans = str(ans).strip()
    m = re.fullmatch(r'\(\s*(.+?)\s*\)', ans)
    if m: ans = m.group(1).strip()
    if re.fullmatch(r'[A-Da-d]', ans): return _NTA_LETTER.get(ans, ans)
    if re.fullmatch(r'[A-Da-d](?:\s*,\s*[A-Da-d])+', ans):
        return ",".join(_NTA_LETTER.get(c.strip(), c.strip()) for c in ans.split(","))
    return ans


def _extract_images_from_text(text: str) -> list:
    return re.findall(r'\[IMAGE:([^\]]+)\]', text or "")


def _validate_and_fix_question(q: dict, meta: dict) -> Optional[dict]:
    try:
        q["number"] = int(str(q.get("number", 0)).strip())
    except (ValueError, TypeError):
        return None
    if q["number"] < 1:
        return None

    if not str(q.get("question", "")).strip():
        return None

    # Fix escaped newlines from Gemini
    for field in ("question", "solution"):
        if q.get(field):
            q[field] = _fix_newlines(q[field])
    if isinstance(q.get("options"), list):
        q["options"] = [_fix_newlines(o) for o in q["options"]]

    # Apply defaults
    for field, default in _DEFAULTS.items():
        if field not in q or q[field] is None:
            import copy
            q[field] = copy.deepcopy(default)

    # Stamp metadata
    if not q.get("year")      and meta.get("year"):      q["year"]      = meta["year"]
    if not q.get("shift")     and meta.get("shift"):     q["shift"]     = meta["shift"]
    if not q.get("exam_date") and meta.get("exam_date"): q["exam_date"] = meta["exam_date"]

    # Normalise q_type
    qt = str(q.get("q_type", "MCQ")).strip().upper()
    q["q_type"] = qt if qt in _VALID_QTYPES else "MCQ"

    # Normalise subject
    subj = str(q.get("subject", "PHYSICS")).strip().upper()
    if subj in ("MATHS", "MATH"): subj = "MATHEMATICS"
    q["subject"] = subj if subj in _VALID_SUBJECTS else "PHYSICS"

    # Normalise section
    sec = str(q.get("section", "")).strip().upper()
    if "B" in sec or "NUMERICAL" in sec or "INTEGER" in sec:
        q["section"] = "SECTION-B"
        if q["q_type"] == "MCQ": q["q_type"] = "NUMERICAL"
    else:
        q["section"] = "SECTION-A"

    # Options
    opts = q.get("options", [])
    if not isinstance(opts, list): opts = []
    if q["q_type"] != "NUMERICAL":
        while len(opts) < 4: opts.append("")
        opts = [str(o) for o in opts[:4]]
    else:
        opts = []
    q["options"] = opts

    # Answer
    q["answer"] = _coerce_answer(str(q.get("answer", "")), q["q_type"])

    # Difficulty
    diff = str(q.get("difficulty", "medium")).strip().lower()
    q["difficulty"] = diff if diff in _VALID_DIFFICULTIES else "medium"

    # Marks
    if q.get("marks_correct") is None: q["marks_correct"] = 4
    if q.get("marks_wrong")   is None: q["marks_wrong"]   = -1

    # Images — merge from text + explicit lists
    def _merge_unique(*lists):
        seen = set(); result = []
        for lst in lists:
            for item in (lst or []):
                if item not in seen: seen.add(item); result.append(item)
        return result

    q_text_imgs = _extract_images_from_text(q["question"])
    opt_imgs    = []
    for o in q["options"]: opt_imgs.extend(_extract_images_from_text(o))
    sol_imgs    = _extract_images_from_text(q.get("solution", ""))

    q["q_images"]  = _merge_unique(q.get("q_images", []), q_text_imgs, opt_imgs)
    q["sol_images"] = _merge_unique(q.get("sol_images", []), sol_imgs)

    q["chapter_name"] = str(q.get("chapter_name", "") or "").strip()
    q["topic_name"]   = str(q.get("topic_name",   "") or "").strip()
    q["chapter_id"]   = None
    q["topic"]        = q.get("topic_name", "")
    q["verified"]     = False

    return q


# ─────────────────────────────────────────────────────────────────────────────
# Main parse function
# ─────────────────────────────────────────────────────────────────────────────

async def parse_latex_with_llm(
    tex: str,
    subject_hint: str = "",
    api_key: str = "",
    pool=None,
) -> list[dict]:
    """
    Parse a LaTeX exam paper using Gemini 2.5 Flash with structured JSON output.
    Returns list of question dicts ready for admin review / DB save.
    """
    if not _GEMINI_AVAILABLE:
        logger.error("[llm_parser] google-generativeai not installed")
        return []

    key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        logger.error("[llm_parser] No GEMINI_API_KEY set")
        return []

    t_start = time.time()
    logger.info(f"[llm_parser] Starting parse, tex={len(tex):,} chars")

    # Step 1: Normalise images
    tex = _normalise_image_refs(tex)

    # Step 2: Extract metadata
    meta = _extract_meta_from_latex(tex)
    if subject_hint:
        canonical = _normalise_subject(subject_hint)
        if canonical and canonical not in meta["subjects"]:
            meta["subjects"] = [canonical]
    logger.info(f"[llm_parser] Meta: {meta}")

    # Step 3: Taxonomy
    taxonomy = dict(_TAXONOMY)
    if pool is not None:
        try:
            from services.llm_tagger import _get_db_taxonomy
            for subj in meta["subjects"]:
                db_tax = await _get_db_taxonomy(pool, subj)
                if db_tax:
                    taxonomy[subj] = {**taxonomy.get(subj, {}), **db_tax}
        except Exception as e:
            logger.warning(f"[llm_parser] DB taxonomy failed: {e}")

    # Step 4: Regex subject split + one Gemini call per subject
    subjects       = meta["subjects"]
    subject_chunks = _split_by_subject(tex, subjects)

    loop          = asyncio.get_running_loop()
    all_questions = []
    all_summaries = []

    sem = asyncio.Semaphore(3)

    async def parse_subject(subj: str, subj_tex: str) -> tuple[list, dict]:
        async with sem:
            subj_taxonomy = format_taxonomy_for_prompt(taxonomy, [subj])
            subj_expected = get_expected_count(meta["exam_type"], meta["year"], [subj])

            user_prompt = PARSER_USER_PROMPT_TEMPLATE.format(
                exam_type      = meta["exam_type"],
                year           = meta["year"],
                exam_date      = meta["exam_date"],
                shift          = meta["shift"],
                subjects       = subj,
                expected_count = subj_expected,
                taxonomy_text  = subj_taxonomy,
                latex_content  = subj_tex,
            )
            try:
                result = await loop.run_in_executor(
                    None, _call_gemini_sync, key,
                    PARSER_SYSTEM_PROMPT, user_prompt,
                    PARSE_MODEL, 16000,
                )
                qs      = result.get("questions", [])
                summary = result.get("parse_summary", {})

                # Stamp correct subject always
                for q in qs:
                    q["subject"] = subj

                return qs, summary
            except Exception as e:
                logger.error(f"[llm_parser] {subj} failed: {e}")
                return [], {"subject": subj, "found_count": 0, "notes": str(e)}

    results = await asyncio.gather(
        *[parse_subject(subj, subj_tex) for subj, subj_tex in subject_chunks]
    )

    for qs, summary in results:
        all_questions.extend(qs)
        all_summaries.append(summary)

    # Step 5: Deduplicate by (subject, number)
    seen: dict[tuple, dict] = {}
    for q in all_questions:
        try:
            num  = int(str(q.get("number", 0)))
            subj = str(q.get("subject", "UNKNOWN"))
        except (ValueError, TypeError):
            continue
        k = (subj, num)
        if k not in seen or len(str(q.get("question", ""))) > len(str(seen[k].get("question", ""))):
            seen[k] = q

    # Sort by subject order then question number
    all_questions = [seen[k] for k in sorted(seen.keys(), key=lambda x: (
        subjects.index(x[0]) if x[0] in subjects else 99, x[1]
    ))]

    # Step 6: Validate
    validated = []
    for q in all_questions:
        fixed = _validate_and_fix_question(dict(q), meta)
        if fixed:
            validated.append(fixed)
        else:
            logger.warning(f"[llm_parser] Dropped: Q{q.get('number')} — {str(q.get('question',''))[:60]}")

    elapsed = time.time() - t_start

    # Step 7: Log full parse summary
    logger.info("=" * 60)
    logger.info(f"[llm_parser] PARSE COMPLETE in {elapsed:.1f}s")
    logger.info(f"[llm_parser] Total questions: {len(validated)}")
    for s in all_summaries:
        expected = s.get('expected_count', '?')
        found    = s.get('found_count', '?')
        missing  = s.get('missing_numbers', [])
        status   = "✓" if not missing else f"⚠ MISSING: {missing}"
        logger.info(
            f"[llm_parser]   {s.get('subject','?'):12s} | "
            f"expected={expected:>3} | found={found:>3} | {status}"
        )
        if s.get('notes'):
            logger.info(f"[llm_parser]   notes: {s.get('notes')}")
    logger.info("=" * 60)

    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Sync wrapper
# ─────────────────────────────────────────────────────────────────────────────

def parse_latex_sync(tex: str, subject_hint: str = "", api_key: str = "") -> list[dict]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, parse_latex_with_llm(tex, subject_hint, api_key))
                return future.result(timeout=300)
        else:
            return loop.run_until_complete(parse_latex_with_llm(tex, subject_hint, api_key))
    except Exception as e:
        logger.error(f"[llm_parser] parse_latex_sync failed: {e}")
        return []