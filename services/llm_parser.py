"""
services/llm_parser.py
======================
LLM-powered question paper parser.

Replaces the regex-based parser (services/parser.py) entirely.

FLOW:
  1. Receive raw LaTeX text
  2. Detect exam metadata (year, shift, date, exam type)
  3. Load taxonomy (hardcoded + DB if available)
  4. If paper fits in one GPT-4o context (~100k tokens) → single call
  5. If paper is large → split into subject-based or line-based chunks
  6. Merge results, validate, postprocess
  7. Return list of question dicts (same schema as old parser)

The taxonomy is imported from llm_tagger.py so there is ONE source of truth.
Chapter/topic/difficulty tagging is done IN THE SAME LLM CALL — no separate
tagging pass needed.
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
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    logger.warning("[llm_parser] openai package not installed")

from services.prompts import (
    PARSER_SYSTEM_PROMPT,
    PARSER_USER_PROMPT_TEMPLATE,
    PARSER_CHUNK_PROMPT_TEMPLATE,
    PARSER_MERGE_PROMPT_TEMPLATE,
    format_taxonomy_for_prompt,
    get_expected_count,
)
from services.llm_tagger import _TAXONOMY, _normalise_subject

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# GPT-4o context is 128k tokens. LaTeX chars ≈ 3 chars/token roughly.
# Keep chunks under ~280k chars to stay safe (≈ 90k tokens of content).
CHUNK_CHAR_LIMIT = 280_000

# Model to use — gpt-4o for best accuracy on complex LaTeX
PARSE_MODEL = "gpt-4o"

# Fallback model if gpt-4o quota exceeded
FALLBACK_MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction (reused from pipeline — no dependency on old parser)
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
    'sep':9,'oct':10,'nov':11,'dec':12,
}

def _extract_meta_from_latex(tex: str) -> dict:
    """Extract exam_date, year, shift, exam_type, subjects from raw LaTeX."""
    _mp = (r'(january|february|march|april|may|june|july|august|'
           r'september|october|november|december|jan|feb|mar|apr|'
           r'jun|jul|aug|sep|oct|nov|dec)')

    # Use title + first 1500 chars of body
    title = ""
    m = re.search(r'\\title\s*\{([^}]+)\}', tex)
    if m:
        title = m.group(1)
    combined = title + " " + tex[:1500]

    exam_date = year = shift = ""

    # DD-MM-YYYY
    dm = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', combined)
    if dm:
        exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
    else:
        dm = re.search(r'\b(\d{2})/(\d{2})/(20\d{2})\b', combined)
        if dm:
            exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
        else:
            dm = re.search(
                rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_mp}\s+(20\d{{2}})\b',
                combined, re.IGNORECASE
            )
            if dm:
                mo = _MONTH_MAP.get(dm.group(2).lower(), 0)
                if mo:
                    exam_date = f"{dm.group(3)}-{mo:02d}-{int(dm.group(1)):02d}"
            else:
                dm = re.search(
                    rf'\b{_mp}\s+(\d{{1,2}}),?\s+(20\d{{2}})\b',
                    combined, re.IGNORECASE
                )
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
    if any(x in tl for x in ("morning", "shift 1", "shift-1", "shift1", "session 1", "session-1")):
        shift = "Morning"
    elif any(x in tl for x in ("evening", "shift 2", "shift-2", "shift2", "session 2", "session-2")):
        shift = "Evening"

    # Exam type
    exam_type = "JEE Main"
    if re.search(r'jee\s*advanced', combined, re.IGNORECASE):
        exam_type = "JEE Advanced"
    elif re.search(r'neet', combined, re.IGNORECASE):
        exam_type = "NEET"
    elif re.search(r'cuet', combined, re.IGNORECASE):
        exam_type = "CUET"
    elif re.search(r'jee\s*main', combined, re.IGNORECASE):
        exam_type = "JEE Main"

    # Subjects present
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
    """
    Convert \\includegraphics[...]{path/to/img.png} → [IMAGE:img.png]
    so the LLM sees clean placeholders and can copy them into JSON fields.
    """
    def _rep(m):
        path = m.group(1).strip()
        basename = path.split("/")[-1].split("\\")[-1]
        return f"[IMAGE:{basename}]"

    return re.sub(
        r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',
        _rep,
        tex
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing / validation
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = {
    "number", "q_type", "subject", "question",
}

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


def _coerce_answer(ans: str, q_type: str) -> str:
    """Normalise answer to option number string or numeric string."""
    if not ans:
        return ""
    ans = str(ans).strip()
    # Strip wrapping parens: (2) → 2, (B) → B
    m = re.fullmatch(r'\(\s*(.+?)\s*\)', ans)
    if m:
        ans = m.group(1).strip()
    # Letter → number
    if re.fullmatch(r'[A-Da-d]', ans):
        return _NTA_LETTER.get(ans, ans)
    # Multi-letter MSQ: A,C → 1,3
    if re.fullmatch(r'[A-Da-d](?:\s*,\s*[A-Da-d])+', ans):
        return ",".join(_NTA_LETTER.get(c.strip(), c.strip())
                        for c in ans.split(","))
    return ans


def _extract_images_from_text(text: str) -> list:
    """Pull [IMAGE:xxx] ids from a text field."""
    return re.findall(r'\[IMAGE:([^\]]+)\]', text or "")


def _validate_and_fix_question(q: dict, meta: dict) -> Optional[dict]:
    """
    Validate a single question dict returned by LLM.
    Returns fixed dict or None if unfixable.
    """
    # Must have a number
    try:
        q["number"] = int(str(q.get("number", 0)).strip())
    except (ValueError, TypeError):
        return None
    if q["number"] < 1:
        return None

    # Must have question text
    if not str(q.get("question", "")).strip():
        return None

    # Apply defaults for missing fields
    for field, default in _DEFAULTS.items():
        if field not in q or q[field] is None:
            q[field] = default

    # Stamp metadata from paper if question didn't get it
    if not q.get("year")      and meta.get("year"):      q["year"]      = meta["year"]
    if not q.get("shift")     and meta.get("shift"):     q["shift"]     = meta["shift"]
    if not q.get("exam_date") and meta.get("exam_date"): q["exam_date"] = meta["exam_date"]

    # Normalise q_type
    qt = str(q.get("q_type", "MCQ")).strip().upper()
    q["q_type"] = qt if qt in _VALID_QTYPES else "MCQ"

    # Normalise subject
    subj = str(q.get("subject", "PHYSICS")).strip().upper()
    if subj in ("MATHS", "MATH"):
        subj = "MATHEMATICS"
    q["subject"] = subj if subj in _VALID_SUBJECTS else "PHYSICS"

    # Normalise section
    sec = str(q.get("section", "")).strip().upper()
    if "B" in sec or "NUMERICAL" in sec or "INTEGER" in sec:
        q["section"] = "SECTION-B"
        if q["q_type"] == "MCQ":
            q["q_type"] = "NUMERICAL"
    else:
        q["section"] = "SECTION-A"

    # Ensure options is a list of 4 strings for MCQ/MSQ
    opts = q.get("options", [])
    if not isinstance(opts, list):
        opts = []
    if q["q_type"] != "NUMERICAL":
        while len(opts) < 4:
            opts.append("")
        opts = [str(o) for o in opts[:4]]
    else:
        opts = []
    q["options"] = opts

    # Coerce answer
    q["answer"] = _coerce_answer(str(q.get("answer", "")), q["q_type"])

    # Difficulty
    diff = str(q.get("difficulty", "medium")).strip().lower()
    q["difficulty"] = diff if diff in _VALID_DIFFICULTIES else "medium"

    # Marks defaults by exam type
    if q.get("marks_correct") is None:
        q["marks_correct"] = 4
    if q.get("marks_wrong") is None:
        q["marks_wrong"] = -1

    # Build q_images / sol_images from placeholder scan
    q_text_imgs = _extract_images_from_text(q["question"])
    opt_imgs    = []
    for o in q["options"]:
        opt_imgs.extend(_extract_images_from_text(o))
    sol_imgs    = _extract_images_from_text(q.get("solution", ""))

    # Merge with any LLM-provided lists (deduplicated)
    def _merge_unique(*lists):
        seen = set()
        result = []
        for lst in lists:
            for item in (lst or []):
                if item not in seen:
                    seen.add(item)
                    result.append(item)
        return result

    q["q_images"]   = _merge_unique(q.get("q_images", []), q_text_imgs, opt_imgs)
    q["sol_images"]  = _merge_unique(q.get("sol_images", []), sol_imgs)

    # chapter_name / topic_name — keep as-is (LLM used taxonomy list)
    q["chapter_name"] = str(q.get("chapter_name", "") or "").strip()
    q["topic_name"]   = str(q.get("topic_name",   "") or "").strip()

    # Add fields expected by DB save layer
    q["chapter_id"] = None
    q["topic"]      = q.get("topic_name", "")
    q["verified"]   = False

    return q


def _parse_llm_json_response(raw: str) -> list:
    """
    Robustly parse LLM response that should be a JSON array.
    Handles markdown fences, leading/trailing text, truncated arrays.
    """
    if not raw:
        return []

    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()

    # Try direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        return []
    except json.JSONDecodeError:
        pass

    # Find JSON array in response
    start = raw.find("[")
    if start == -1:
        logger.error("[llm_parser] No JSON array found in LLM response")
        return []

    # Try increasingly truncated versions (handles cut-off at token limit)
    end = raw.rfind("]")
    if end > start:
        try:
            return json.loads(raw[start:end+1])
        except json.JSONDecodeError:
            pass

    # Try to fix truncated JSON by finding last complete object
    chunk = raw[start:]
    depth = 0
    last_complete = -1
    in_str = False
    escape = False
    for i, ch in enumerate(chunk):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete = i

    if last_complete > 0:
        truncated = chunk[:last_complete+1] + "]"
        try:
            return json.loads("[" + truncated.lstrip("["))
        except json.JSONDecodeError:
            pass

    logger.error("[llm_parser] Could not parse LLM response as JSON")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Core LLM call (sync, runs in executor)
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm_sync(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = PARSE_MODEL,
    max_tokens: int = 16000,
    temperature: float = 0.0,
) -> str:
    """Make a synchronous OpenAI API call. Returns raw response string."""
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"} if model.startswith("gpt-4o") else None,
        )
        content = resp.choices[0].message.content or ""
        logger.info(
            f"[llm_parser] model={model} "
            f"in_tokens={resp.usage.prompt_tokens} "
            f"out_tokens={resp.usage.completion_tokens}"
        )
        return content
    except Exception as e:
        logger.error(f"[llm_parser] LLM call failed with {model}: {e}")
        # Try fallback model
        if model != FALLBACK_MODEL:
            logger.info(f"[llm_parser] Retrying with {FALLBACK_MODEL}")
            try:
                resp = client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as e2:
                logger.error(f"[llm_parser] Fallback also failed: {e2}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Chunking strategy
# ─────────────────────────────────────────────────────────────────────────────

def _split_by_subject(tex: str) -> list[tuple[str, str]]:
    """
    Try to split LaTeX by subject sections.
    Returns list of (subject_label, chunk_text).
    """
    # Common subject boundary patterns
    patterns = [
        r'(?=\\section\*?\{[^}]*(?:PHYSICS|CHEMISTRY|MATHEMATICS|BIOLOGY)[^}]*\})',
        r'(?=PART\s*[-–]\s*[A-Z]\s*:?\s*(?:PHYSICS|CHEMISTRY|MATHEMATICS|BIOLOGY))',
        r'(?=\n\s*(?:PHYSICS|CHEMISTRY|MATHEMATICS|BIOLOGY)\s*\n)',
    ]
    for pat in patterns:
        parts = re.split(pat, tex, flags=re.IGNORECASE)
        if len(parts) >= 3:  # at least 3 parts (preamble + subjects)
            result = []
            for part in parts:
                m = re.search(r'(PHYSICS|CHEMISTRY|MATHEMATICS|BIOLOGY)', part[:200], re.IGNORECASE)
                label = m.group(1).upper() if m else "UNKNOWN"
                result.append((label, part))
            return result
    return []


def _split_by_lines(tex: str, chunk_size: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """
    Split large LaTeX into line-boundary chunks.
    Tries to split at \\item or question number boundaries.
    """
    if len(tex) <= chunk_size:
        return [tex]

    chunks = []
    start = 0
    while start < len(tex):
        end = min(start + chunk_size, len(tex))
        if end < len(tex):
            # Find last \\item or question boundary before end
            search_back = tex[max(start, end-5000):end]
            # Look for question number pattern like "\n42." or "\item"
            best_split = -1
            for pat in [r'\n\d{1,3}\.\s+', r'\\item\s+']:
                for m in re.finditer(pat, search_back):
                    best_split = max(best_split, m.start())
            if best_split > 0:
                end = max(start, end - 5000) + best_split
        chunks.append(tex[start:end])
        start = end
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main parse function (async, runs LLM calls in thread executor)
# ─────────────────────────────────────────────────────────────────────────────

async def parse_latex_with_llm(
    tex: str,
    subject_hint: str = "",
    api_key: str = "",
    pool=None,
) -> list[dict]:
    """
    Parse a LaTeX exam paper using GPT-4o.
    Returns list of question dicts ready for admin review / DB save.

    This replaces parse_tex() and parse_plain_pdf_text() entirely.
    Chapter/topic/difficulty tagging is done in the same LLM call.
    """
    if not _OPENAI_AVAILABLE:
        logger.error("[llm_parser] openai not installed — cannot parse")
        return []

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        logger.error("[llm_parser] No OPENAI_API_KEY set — cannot parse")
        return []

    t_start = time.time()
    logger.info(f"[llm_parser] Starting parse, tex length={len(tex):,} chars")

    # Step 1: Normalise image references
    tex = _normalise_image_refs(tex)

    # Step 2: Extract metadata
    meta = _extract_meta_from_latex(tex)
    if subject_hint:
        canonical = _normalise_subject(subject_hint)
        if canonical and canonical not in meta["subjects"]:
            meta["subjects"] = [canonical]

    logger.info(f"[llm_parser] Meta detected: {meta}")

    # Step 3: Build taxonomy text
    # Load DB taxonomy if pool available (same logic as llm_tagger)
    taxonomy = dict(_TAXONOMY)  # start with hardcoded
    if pool is not None:
        try:
            from services.llm_tagger import _get_db_taxonomy, _load_taxonomy
            for subj in meta["subjects"]:
                db_tax = await _get_db_taxonomy(pool, subj)
                if db_tax:
                    taxonomy[subj] = {**taxonomy.get(subj, {}), **db_tax}
        except Exception as e:
            logger.warning(f"[llm_parser] DB taxonomy load failed: {e}")

    taxonomy_text = format_taxonomy_for_prompt(taxonomy, meta["subjects"])
    expected_count = get_expected_count(meta["exam_type"], meta["year"], meta["subjects"])

    # Step 4: Decide single call vs chunked
    loop = asyncio.get_running_loop()
    all_questions = []

    if len(tex) <= CHUNK_CHAR_LIMIT:
        # ── Single call ───────────────────────────────────────────────────────
        logger.info("[llm_parser] Single-call mode")
        user_prompt = PARSER_USER_PROMPT_TEMPLATE.format(
            exam_type      = meta["exam_type"],
            year           = meta["year"],
            exam_date      = meta["exam_date"],
            shift          = meta["shift"],
            subjects       = ", ".join(meta["subjects"]),
            expected_count = expected_count,
            taxonomy_text  = taxonomy_text,
            latex_content  = tex,
        )
        raw = await loop.run_in_executor(
            None, _call_llm_sync, key,
            PARSER_SYSTEM_PROMPT, user_prompt,
            PARSE_MODEL, 16000, 0.0,
        )
        all_questions = _parse_llm_json_response(raw)

    else:
        # ── Chunked call ──────────────────────────────────────────────────────
        logger.info("[llm_parser] Chunked mode")
        subject_chunks = _split_by_subject(tex)

        if subject_chunks:
            chunks_with_labels = subject_chunks
        else:
            line_chunks = _split_by_lines(tex)
            chunks_with_labels = [(f"CHUNK_{i+1}", c) for i, c in enumerate(line_chunks)]

        logger.info(f"[llm_parser] {len(chunks_with_labels)} chunks")

        sem = asyncio.Semaphore(3)  # max 3 concurrent API calls

        async def parse_chunk(i, label, chunk_tex):
            async with sem:
                prompt = PARSER_CHUNK_PROMPT_TEMPLATE.format(
                    chunk_num      = i + 1,
                    total_chunks   = len(chunks_with_labels),
                    start_q        = "unknown",
                    exam_type      = meta["exam_type"],
                    year           = meta["year"],
                    exam_date      = meta["exam_date"],
                    shift          = meta["shift"],
                    subjects       = label,
                    taxonomy_text  = taxonomy_text,
                    latex_content  = chunk_tex,
                )
                try:
                    raw = await loop.run_in_executor(
                        None, _call_llm_sync, key,
                        PARSER_SYSTEM_PROMPT, prompt,
                        PARSE_MODEL, 16000, 0.0,
                    )
                    return _parse_llm_json_response(raw)
                except Exception as e:
                    logger.error(f"[llm_parser] Chunk {i+1} failed: {e}")
                    return []

        chunk_results = await asyncio.gather(
            *[parse_chunk(i, label, chunk)
              for i, (label, chunk) in enumerate(chunks_with_labels)]
        )

        # Flatten
        for chunk_qs in chunk_results:
            all_questions.extend(chunk_qs)

        # Deduplicate by question number (keep fuller version)
        seen: dict[int, dict] = {}
        for q in all_questions:
            try:
                num = int(str(q.get("number", 0)))
            except (ValueError, TypeError):
                continue
            if num not in seen:
                seen[num] = q
            else:
                # Keep whichever has more content
                existing = seen[num]
                if len(str(q.get("question", ""))) > len(str(existing.get("question", ""))):
                    seen[num] = q
        all_questions = [seen[k] for k in sorted(seen.keys())]

    # Step 5: Validate and fix each question
    validated = []
    for q in all_questions:
        fixed = _validate_and_fix_question(dict(q), meta)
        if fixed:
            validated.append(fixed)
        else:
            logger.warning(f"[llm_parser] Dropped invalid question: {q.get('number')} — {str(q.get('question',''))[:60]}")

    # Sort by number
    validated.sort(key=lambda q: float(str(q["number"]).replace(",", ".")))

    elapsed = time.time() - t_start
    logger.info(
        f"[llm_parser] Done: {len(validated)} questions validated "
        f"in {elapsed:.1f}s (raw={len(all_questions)})"
    )

    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Sync wrapper for compatibility (used by pipeline when called from sync context)
# ─────────────────────────────────────────────────────────────────────────────

def parse_latex_sync(tex: str, subject_hint: str = "", api_key: str = "") -> list[dict]:
    """Synchronous wrapper around parse_latex_with_llm for non-async callers."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't run nested event loops — use thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    parse_latex_with_llm(tex, subject_hint, api_key)
                )
                return future.result(timeout=300)
        else:
            return loop.run_until_complete(
                parse_latex_with_llm(tex, subject_hint, api_key)
            )
    except Exception as e:
        logger.error(f"[llm_parser] parse_latex_sync failed: {e}")
        return []