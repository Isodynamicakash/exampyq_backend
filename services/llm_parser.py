"""
services/llm_parser.py
======================
Gemini-powered question paper parser.

FLOW:
  1. Receive raw LaTeX
  2. Extract metadata
  3. ONE Gemini call — full paper, all subjects at once
  4. Gemini identifies subject boundaries itself
  5. Validate + return
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
    from google import genai
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai_old
        _GEMINI_AVAILABLE = True
        _USE_OLD_SDK = True
    except ImportError:
        _GEMINI_AVAILABLE = False
        _USE_OLD_SDK = False
        logger.warning("[llm_parser] google-genai not installed — run: pip install google-genai")
else:
    _USE_OLD_SDK = False

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

PARSE_MODEL = "gemini-2.5-flash-preview-04-17"

# ─────────────────────────────────────────────────────────────────────────────
# Few-shot example — shows Gemini exact output format with LaTeX
# This is the most important part — Gemini learns from this example
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLE = '''
EXAMPLE INPUT (partial LaTeX):
\\section*{PHYSICS}
\\section*{SECTION - A}
\\begin{enumerate}
  \\item A ball is thrown with velocity $v_0 = 20$ m/s at angle $\\theta = 30^{\\circ}$.
  Find the maximum height. $[g = 10 \\text{ m/s}^2]$\\\\
  (1) 5 m \\quad (2) 10 m \\quad (3) 15 m \\quad (4) 20 m

\\section*{Sol. (1)}
$H = \\frac{v_0^2 \\sin^2\\theta}{2g} = \\frac{400 \\times 0.25}{20} = 5$ m

  \\setcounter{enumi}{1}
  \\item A current of 2 A flows through a resistance of $5\\,\\Omega$.
  The power dissipated is:\\\\
  (1) 10 W \\quad (2) 20 W \\quad (3) 40 W \\quad (4) 5 W

\\section*{Sol. (2)}
$P = I^2 R = 4 \\times 5 = 20$ W

\\section*{SECTION - B}
  \\item The wavelength of light in nm is ______.\\\\
  Sol. 589

\\end{enumerate}

EXAMPLE OUTPUT (exact JSON you must return):
[
  {
    "number": 1,
    "q_type": "MCQ",
    "subject": "PHYSICS",
    "section": "SECTION-A",
    "year": "",
    "shift": "",
    "exam_date": "",
    "question": "A ball is thrown with velocity $v_0 = 20$ m/s at angle $\\\\theta = 30^{\\\\circ}$. Find the maximum height. $[g = 10 \\\\text{ m/s}^2]$",
    "options": ["5 m", "10 m", "15 m", "20 m"],
    "answer": "1",
    "solution": "$H = \\\\frac{v_0^2 \\\\sin^2\\\\theta}{2g} = \\\\frac{400 \\\\times 0.25}{20} = 5$ m",
    "chapter_name": "Kinematics",
    "topic_name": "Projectile Motion",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": -1
  },
  {
    "number": 2,
    "q_type": "MCQ",
    "subject": "PHYSICS",
    "section": "SECTION-A",
    "year": "",
    "shift": "",
    "exam_date": "",
    "question": "A current of 2 A flows through a resistance of $5\\\\,\\\\Omega$. The power dissipated is:",
    "options": ["10 W", "20 W", "40 W", "5 W"],
    "answer": "2",
    "solution": "$P = I^2 R = 4 \\\\times 5 = 20$ W",
    "chapter_name": "Current Electricity",
    "topic_name": "Power",
    "difficulty": "easy",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": -1
  },
  {
    "number": 3,
    "q_type": "NUMERICAL",
    "subject": "PHYSICS",
    "section": "SECTION-B",
    "year": "",
    "shift": "",
    "exam_date": "",
    "question": "The wavelength of light in nm is ______.",
    "options": [],
    "answer": "589",
    "solution": "",
    "chapter_name": "Optics",
    "topic_name": "Wave Optics",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": 0
  }
]

KEY RULES FROM EXAMPLE:
1. Options strip the "(1)" prefix — just the text: "5 m" not "(1) 5 m"
2. answer is "1","2","3","4" for MCQ (option number), numeric string for NUMERICAL
3. LaTeX backslashes are DOUBLED in JSON strings: \\frac becomes \\\\frac
4. Sol.(X) line gives answer X — solution text is everything after that line
5. SECTION-B questions are NUMERICAL with options:[]
6. question number comes from \\item order + \\setcounter{enumi}{N} (next item = N+1)
'''.strip()


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
    if any(x in tl for x in ("morning","shift 1","shift-1","shift1","session 1")):
        shift = "Morning"
    elif any(x in tl for x in ("evening","shift 2","shift-2","shift2","session 2")):
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


def _normalise_image_refs(tex: str) -> str:
    def _rep(m):
        path = m.group(1).strip()
        basename = path.split("/")[-1].split("\\")[-1]
        return f"[IMAGE:{basename}]"
    return re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', _rep, tex)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini API call — ONE call, full paper, new SDK
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini_sync(api_key: str, prompt: str, model: str = PARSE_MODEL) -> str:
    """
    Single Gemini call. Returns raw text response.
    Uses new google-genai SDK (not deprecated google-generativeai).
    """
    if not _USE_OLD_SDK:
        # New SDK: google-genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=65536,  # Gemini 2.5 Flash supports up to 65k output
            )
        )
        return response.text or ""
    else:
        # Fallback: old SDK (deprecated but still works)
        genai_old.configure(api_key=api_key)
        m = genai_old.GenerativeModel(model_name=model)
        response = m.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 65536}
        )
        return response.text or ""


# ─────────────────────────────────────────────────────────────────────────────
# Robust JSON extraction from response
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> list:
    """Extract JSON array from Gemini response — handles markdown fences, truncation."""
    if not raw:
        return []

    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()

    # Direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ("questions", "data", "results", "items"):
                if k in data and isinstance(data[k], list):
                    return data[k]
    except json.JSONDecodeError:
        pass

    # Find array start
    start = raw.find("[")
    if start == -1:
        logger.error("[llm_parser] No JSON array in response")
        return []

    # Try full array
    end = raw.rfind("]")
    if end > start:
        try:
            return json.loads(raw[start:end+1])
        except json.JSONDecodeError:
            pass

    # Truncated — find last complete object
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

    logger.error("[llm_parser] Could not parse JSON from response")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "section": "SECTION-A", "year": "", "shift": "", "exam_date": "",
    "options": [], "answer": "", "solution": "", "chapter_name": "",
    "topic_name": "", "difficulty": "medium", "q_images": [], "sol_images": [],
    "marks_correct": 4, "marks_wrong": -1, "verified": False, "chapter_id": None, "topic": "",
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


def _validate_and_fix(q: dict, meta: dict) -> Optional[dict]:
    try:
        q["number"] = int(str(q.get("number", 0)).strip())
    except (ValueError, TypeError):
        return None
    if q["number"] < 1: return None
    if not str(q.get("question", "")).strip(): return None

    for field in ("question", "solution"):
        if q.get(field): q[field] = _fix_newlines(q[field])
    if isinstance(q.get("options"), list):
        q["options"] = [_fix_newlines(o) for o in q["options"]]

    import copy
    for field, default in _DEFAULTS.items():
        if field not in q or q[field] is None:
            q[field] = copy.deepcopy(default)

    if not q.get("year")      and meta.get("year"):      q["year"]      = meta["year"]
    if not q.get("shift")     and meta.get("shift"):     q["shift"]     = meta["shift"]
    if not q.get("exam_date") and meta.get("exam_date"): q["exam_date"] = meta["exam_date"]

    qt = str(q.get("q_type", "MCQ")).strip().upper()
    q["q_type"] = qt if qt in _VALID_QTYPES else "MCQ"

    subj = str(q.get("subject", "PHYSICS")).strip().upper()
    if subj in ("MATHS", "MATH"): subj = "MATHEMATICS"
    q["subject"] = subj if subj in _VALID_SUBJECTS else "PHYSICS"

    sec = str(q.get("section", "")).strip().upper()
    if "B" in sec or "NUMERICAL" in sec or "INTEGER" in sec:
        q["section"] = "SECTION-B"
        if q["q_type"] == "MCQ": q["q_type"] = "NUMERICAL"
    else:
        q["section"] = "SECTION-A"

    opts = q.get("options", [])
    if not isinstance(opts, list): opts = []
    if q["q_type"] != "NUMERICAL":
        while len(opts) < 4: opts.append("")
        opts = [str(o) for o in opts[:4]]
    else:
        opts = []
    q["options"] = opts

    q["answer"] = _coerce_answer(str(q.get("answer", "")), q["q_type"])

    diff = str(q.get("difficulty", "medium")).strip().lower()
    q["difficulty"] = diff if diff in _VALID_DIFFICULTIES else "medium"

    if q.get("marks_correct") is None: q["marks_correct"] = 4
    if q.get("marks_wrong")   is None: q["marks_wrong"]   = -1

    def _imgs(text):
        return re.findall(r'\[IMAGE:([^\]]+)\]', text or "")

    def _merge(*lists):
        seen = set(); result = []
        for lst in lists:
            for item in (lst or []):
                if item not in seen: seen.add(item); result.append(item)
        return result

    q_imgs   = _imgs(q["question"])
    opt_imgs = [img for o in q["options"] for img in _imgs(o)]
    sol_imgs = _imgs(q.get("solution", ""))

    q["q_images"]  = _merge(q.get("q_images", []), q_imgs, opt_imgs)
    q["sol_images"] = _merge(q.get("sol_images", []), sol_imgs)
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
    Parse LaTeX exam paper using Gemini 2.5 Flash.
    ONE call — Gemini reads the full paper and extracts all questions.
    Few-shot example teaches exact format.
    """
    if not _GEMINI_AVAILABLE:
        logger.error("[llm_parser] google-genai not installed")
        return []

    key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        logger.error("[llm_parser] No GEMINI_API_KEY set")
        return []

    t_start = time.time()
    logger.info(f"[llm_parser] Starting parse, tex={len(tex):,} chars")

    tex  = _normalise_image_refs(tex)
    meta = _extract_meta_from_latex(tex)
    if subject_hint:
        canonical = _normalise_subject(subject_hint)
        if canonical and canonical not in meta["subjects"]:
            meta["subjects"] = [canonical]
    logger.info(f"[llm_parser] Meta: {meta}")

    # Taxonomy
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

    taxonomy_text  = format_taxonomy_for_prompt(taxonomy, meta["subjects"])
    expected_count = get_expected_count(meta["exam_type"], meta["year"], meta["subjects"])

    # Build single prompt with few-shot example
    prompt = f"""{PARSER_SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLE}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOW PARSE THIS ACTUAL EXAM PAPER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAM: {meta["exam_type"]} | Year: {meta["year"]} | Date: {meta["exam_date"]} | Shift: {meta["shift"]}
Subjects: {", ".join(meta["subjects"])}
Expected: {expected_count}

TAXONOMY (use ONLY these for chapter_name / topic_name):
{taxonomy_text}

CRITICAL RULES:
1. Extract ALL questions from ALL subjects
2. Follow the EXACT JSON format shown in the example above
3. Double all backslashes in LaTeX: \\frac → \\\\frac
4. Options: strip the "(1)" prefix, just keep the text
5. Answer: "1"/"2"/"3"/"4" for MCQ, numeric string for NUMERICAL
6. Return ONLY a raw JSON array — no markdown, no explanation

PAPER:
---BEGIN---
{tex}
---END---"""

    loop = asyncio.get_running_loop()
    logger.info("[llm_parser] Calling Gemini — single call, full paper")

    try:
        raw = await loop.run_in_executor(
            None, _call_gemini_sync, key, prompt, PARSE_MODEL
        )
        logger.info(f"[llm_parser] Response length: {len(raw):,} chars")
    except Exception as e:
        logger.error(f"[llm_parser] Gemini call failed: {e}")
        return []

    all_questions = _extract_json(raw)
    logger.info(f"[llm_parser] Raw questions extracted: {len(all_questions)}")

    # Validate
    validated = []
    for q in all_questions:
        fixed = _validate_and_fix(dict(q), meta)
        if fixed:
            validated.append(fixed)
        else:
            logger.warning(f"[llm_parser] Dropped Q{q.get('number')} — {str(q.get('question',''))[:60]}")

    # Sort by subject order then number
    subj_order = {s: i for i, s in enumerate(meta["subjects"])}
    validated.sort(key=lambda q: (
        subj_order.get(q.get("subject", ""), 99),
        int(str(q.get("number", 0)))
    ))

    elapsed = time.time() - t_start

    # Summary log
    logger.info("=" * 60)
    logger.info(f"[llm_parser] DONE in {elapsed:.1f}s — {len(validated)} questions")
    for subj in meta["subjects"]:
        subj_qs = [q for q in validated if q.get("subject") == subj]
        sec_a   = len([q for q in subj_qs if q.get("section") == "SECTION-A"])
        sec_b   = len([q for q in subj_qs if q.get("section") == "SECTION-B"])
        nums    = sorted([q["number"] for q in subj_qs])
        if nums:
            expected_nums = list(range(nums[0], nums[-1]+1))
            missing = [n for n in expected_nums if n not in nums]
        else:
            missing = []
        status = "✓" if not missing else f"⚠ MISSING: {missing[:10]}"
        logger.info(f"[llm_parser]   {subj:12s} | total={len(subj_qs):>3} | A={sec_a} B={sec_b} | {status}")
    logger.info("=" * 60)

    return validated


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