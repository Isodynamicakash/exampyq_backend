"""
services/llm_parser.py  —  Gemini-powered JEE/NEET paper parser
Single call, few-shot, new google-genai SDK.
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── SDK import ────────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    _USE_OLD_SDK = False
    _GEMINI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai_old
        _USE_OLD_SDK = True
        _GEMINI_AVAILABLE = True
    except ImportError:
        _USE_OLD_SDK = False
        _GEMINI_AVAILABLE = False
        logging.getLogger(__name__).warning(
            "[llm_parser] google-genai not installed — pip install google-genai"
        )

from services.prompts import (
    PARSER_SYSTEM_PROMPT,
    PARSER_USER_PROMPT_TEMPLATE,
    format_taxonomy_for_prompt,
    get_expected_count,
)
from services.llm_tagger import _TAXONOMY, _normalise_subject

# ── Model ─────────────────────────────────────────────────────────────────────
PARSE_MODEL = "gemini-2.5-flash"

# ── Few-shot example ──────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLE = r'''
EXAMPLE INPUT (partial LaTeX of a JEE Main paper):

\section*{PHYSICS}
\section*{SECTION - A}
\begin{enumerate}
  \item A ball is thrown with velocity $v_0 = 20$ m/s at angle $\theta = 30^{\circ}$.
  Find the maximum height. $[g = 10 \text{ m/s}^2]$\\
  (1) 5 m \quad (2) 10 m \quad (3) 15 m \quad (4) 20 m

\section*{Sol. (1)}
$H = \frac{v_0^2 \sin^2\theta}{2g} = \frac{400 \times 0.25}{20} = 5$ m

  \setcounter{enumi}{1}
  \item A current of 2 A flows through $5\,\Omega$. Power dissipated:\\
  (1) 10 W \quad (2) 20 W \quad (3) 40 W \quad (4) 5 W

\section*{Sol. (2)}
$P = I^2 R = 4 \times 5 = 20$ W

\section*{SECTION - B}
  \item The wavelength of sodium light in nm is \_\_\_\_.
Sol. 589

\end{enumerate}

EXAMPLE OUTPUT — return exactly this JSON format:
[
  {
    "number": 1,
    "q_type": "MCQ",
    "subject": "PHYSICS",
    "section": "SECTION-A",
    "year": "",
    "shift": "",
    "exam_date": "",
    "question": "A ball is thrown with velocity $v_0 = 20$ m/s at angle $\\theta = 30^{\\circ}$. Find the maximum height. $[g = 10 \\text{ m/s}^2]$",
    "options": ["5 m", "10 m", "15 m", "20 m"],
    "answer": "1",
    "solution": "$H = \\frac{v_0^2 \\sin^2\\theta}{2g} = 5$ m",
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
    "question": "A current of 2 A flows through $5\\,\\Omega$. Power dissipated:",
    "options": ["10 W", "20 W", "40 W", "5 W"],
    "answer": "2",
    "solution": "$P = I^2 R = 4 \\times 5 = 20$ W",
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
    "question": "The wavelength of sodium light in nm is ____.",
    "options": [],
    "answer": "589",
    "solution": "589 nm",
    "chapter_name": "Optics",
    "topic_name": "Wave Optics",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": 0
  }
]

RULES (memorise these):
1. options[] — strip "(1)" prefix, keep only the text
2. answer — "1"/"2"/"3"/"4" for MCQ, numeric string for NUMERICAL
3. In JSON strings, LaTeX \ becomes \\ : \frac → \\frac, \theta → \\theta
4. Sol.(X) = answer is X; solution text = lines after that
5. SECTION-B = NUMERICAL, options = []
6. question numbers: follow \item order, respecting \setcounter{enumi}{N}
7. Extract ALL subjects and ALL questions — do not stop early
'''.strip()

# ── Metadata extraction ───────────────────────────────────────────────────────
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
    m = re.search(r'\\title\s*\{([^}]+)\}', tex)
    combined = (m.group(1) if m else "") + " " + tex[:1500]

    exam_date = year = shift = ""
    dm = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', combined)
    if dm:
        exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
    else:
        dm = re.search(r'\b(\d{2})/(\d{2})/(20\d{2})\b', combined)
        if dm:
            exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
        else:
            dm = re.search(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_mp}\s+(20\d{{2}})\b', combined, re.I)
            if dm:
                mo = _MONTH_MAP.get(dm.group(2).lower(), 0)
                if mo: exam_date = f"{dm.group(3)}-{mo:02d}-{int(dm.group(1)):02d}"
            else:
                dm = re.search(rf'\b{_mp}\s+(\d{{1,2}}),?\s+(20\d{{2}})\b', combined, re.I)
                if dm:
                    mo = _MONTH_MAP.get(dm.group(1).lower(), 0)
                    if mo: exam_date = f"{dm.group(3)}-{mo:02d}-{int(dm.group(2)):02d}"

    year = exam_date[:4] if exam_date else ""
    if not year:
        m2 = re.search(r'\b(20\d{2})\b', combined)
        if m2: year = m2.group(1)

    tl = combined.lower()
    if any(x in tl for x in ("morning","shift 1","shift-1","shift1","session 1")):
        shift = "Morning"
    elif any(x in tl for x in ("evening","shift 2","shift-2","shift2","session 2")):
        shift = "Evening"

    exam_type = "JEE Main"
    if re.search(r'jee\s*advanced', combined, re.I): exam_type = "JEE Advanced"
    elif re.search(r'neet', combined, re.I): exam_type = "NEET"
    elif re.search(r'cuet', combined, re.I): exam_type = "CUET"

    subjects = [s for s in ("PHYSICS","CHEMISTRY","MATHEMATICS","BIOLOGY")
                if re.search(s, combined, re.I)]
    if not subjects:
        subjects = ["PHYSICS", "CHEMISTRY", "MATHEMATICS"]

    return {"exam_date": exam_date, "year": year, "shift": shift,
            "exam_type": exam_type, "subjects": subjects}


def _normalise_image_refs(tex: str) -> str:
    def _rep(m):
        basename = m.group(1).strip().split("/")[-1].split("\\")[-1]
        return f"[IMAGE:{basename}]"
    return re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', _rep, tex)


# ── Gemini call — new SDK, with retry + fallback ───────────────────────────────
def _call_gemini_sync(api_key: str, prompt: str, model: str = PARSE_MODEL) -> str:
    """
    Call Gemini. Retries on 503. Falls back to gemini-2.0-flash on 404.
    Collects all non-thought parts (handles thinking models like 2.5-flash).
    """
    import time as _time

    models_to_try = [model]
    if model != "gemini-2.0-flash":
        models_to_try.append("gemini-2.0-flash")

    for current_model in models_to_try:
        for attempt in range(3):
            try:
                logger.info(f"[llm_parser] model={current_model} attempt={attempt+1}")

                if not _USE_OLD_SDK:
                    client   = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=120000,
                        )
                    )
                    # Gemini 2.5 thinking model: skip thought parts, collect text parts
                    text = ""
                    try:
                        for part in response.candidates[0].content.parts:
                            if getattr(part, 'thought', False):
                                continue
                            if getattr(part, 'text', None):
                                text += part.text
                    except Exception:
                        pass
                    if not text:
                        text = response.text or ""
                    logger.info(f"[llm_parser] Response {len(text):,} chars")
                    return text

                else:  # old SDK fallback
                    genai_old.configure(api_key=api_key)
                    m = genai_old.GenerativeModel(model_name=current_model)
                    r = m.generate_content(
                        prompt,
                        generation_config={"temperature": 0.0, "max_output_tokens": 120000}
                    )
                    return r.text or ""

            except Exception as e:
                err = str(e)
                logger.warning(f"[llm_parser] {current_model} attempt={attempt+1}: {err[:120]}")
                if any(x in err for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                    if attempt < 2:
                        delay = [15, 45][attempt]
                        logger.info(f"[llm_parser] Retrying in {delay}s...")
                        _time.sleep(delay)
                        continue
                break  # 404 or other error → try next model

    raise RuntimeError("[llm_parser] All Gemini models failed")


# ── JSON extraction ───────────────────────────────────────────────────────────
def _extract_json(raw: str) -> list:
    """
    Extract JSON array from Gemini response.
    Handles: markdown fences, thinking preamble, truncation.
    """
    if not raw:
        return []

    # Remove markdown fences
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)

    # Find first '[' — skip any thinking/preamble text before it
    start = raw.find("[")
    if start == -1:
        logger.error("[llm_parser] No '[' found in response")
        logger.error(f"[llm_parser] Response sample: {raw[:300]}")
        return []

    raw = raw[start:]

    # Try full parse
    end = raw.rfind("]")
    if end > 0:
        try:
            result = json.loads(raw[:end+1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Truncated — recover last complete object
    depth = 0
    in_str = False
    escape = False
    last_complete = -1
    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
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
        try:
            result = json.loads("[" + raw[1:last_complete+1] + "]")
            if isinstance(result, list):
                logger.warning(f"[llm_parser] Recovered {len(result)} questions from truncated JSON")
                return result
        except json.JSONDecodeError:
            pass

    logger.error("[llm_parser] JSON parse failed completely")
    return []


# ── Validation ────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "section": "SECTION-A", "year": "", "shift": "", "exam_date": "",
    "options": [], "answer": "", "solution": "", "chapter_name": "",
    "topic_name": "", "difficulty": "medium", "q_images": [], "sol_images": [],
    "marks_correct": 4, "marks_wrong": -1, "verified": False,
    "chapter_id": None, "topic": "",
}
_VALID_Q    = {"MCQ", "MSQ", "NUMERICAL"}
_VALID_S    = {"PHYSICS", "CHEMISTRY", "MATHEMATICS", "BIOLOGY"}
_VALID_D    = {"easy", "medium", "hard"}
_NTA        = {"A":"1","B":"2","C":"3","D":"4","a":"1","b":"2","c":"3","d":"4"}


def _validate_and_fix(q: dict, meta: dict) -> Optional[dict]:
    import copy

    try:
        q["number"] = int(str(q.get("number", 0)).strip())
    except (ValueError, TypeError):
        return None
    if q["number"] < 1: return None
    if not str(q.get("question", "")).strip(): return None

    # Apply defaults
    for f, v in _DEFAULTS.items():
        if f not in q or q[f] is None:
            q[f] = copy.deepcopy(v)

    # Stamp metadata if missing
    for f in ("year", "shift", "exam_date"):
        if not q.get(f) and meta.get(f): q[f] = meta[f]

    # Fix escaped newlines
    for f in ("question", "solution"):
        if q.get(f): q[f] = q[f].replace('\\n', '\n')
    if isinstance(q.get("options"), list):
        q["options"] = [o.replace('\\n', '\n') for o in q["options"]]

    # q_type
    qt = str(q.get("q_type", "MCQ")).strip().upper()
    q["q_type"] = qt if qt in _VALID_Q else "MCQ"

    # subject
    subj = str(q.get("subject", "PHYSICS")).strip().upper()
    if subj in ("MATHS", "MATH"): subj = "MATHEMATICS"
    q["subject"] = subj if subj in _VALID_S else "PHYSICS"

    # section
    sec = str(q.get("section", "")).strip().upper()
    if "B" in sec or "NUMERICAL" in sec or "INTEGER" in sec:
        q["section"] = "SECTION-B"
        if q["q_type"] == "MCQ": q["q_type"] = "NUMERICAL"
    else:
        q["section"] = "SECTION-A"

    # options
    opts = q.get("options", [])
    if not isinstance(opts, list): opts = []
    if q["q_type"] != "NUMERICAL":
        while len(opts) < 4: opts.append("")
        opts = [str(o) for o in opts[:4]]
    else:
        opts = []
    q["options"] = opts

    # answer normalise
    ans = str(q.get("answer", "")).strip()
    m = re.fullmatch(r'\(\s*(.+?)\s*\)', ans)
    if m: ans = m.group(1).strip()
    if re.fullmatch(r'[A-Da-d]', ans): ans = _NTA.get(ans, ans)
    q["answer"] = ans

    # difficulty
    d = str(q.get("difficulty", "medium")).strip().lower()
    q["difficulty"] = d if d in _VALID_D else "medium"

    # marks
    if q.get("marks_correct") is None: q["marks_correct"] = 4
    if q.get("marks_wrong")   is None: q["marks_wrong"]   = -1

    # images
    def imgs(t): return re.findall(r'\[IMAGE:([^\]]+)\]', t or "")
    def merge(*ls):
        seen = set(); r = []
        for l in ls:
            for x in (l or []):
                if x not in seen: seen.add(x); r.append(x)
        return r

    q["q_images"]   = merge(q.get("q_images", []), imgs(q["question"]),
                            *[imgs(o) for o in q["options"]])
    q["sol_images"] = merge(q.get("sol_images", []), imgs(q.get("solution", "")))
    q["chapter_name"] = str(q.get("chapter_name") or "").strip()
    q["topic_name"]   = str(q.get("topic_name")   or "").strip()
    q["topic"]        = q["topic_name"]
    q["chapter_id"]   = None
    q["verified"]     = False
    return q


# ── Main ──────────────────────────────────────────────────────────────────────
async def parse_latex_with_llm(
    tex: str,
    subject_hint: str = "",
    api_key: str = "",
    pool=None,
) -> list[dict]:

    if not _GEMINI_AVAILABLE:
        logger.error("[llm_parser] google-genai not installed")
        return []

    key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        logger.error("[llm_parser] No GEMINI_API_KEY set")
        return []

    t0  = time.time()
    tex = _normalise_image_refs(tex)
    meta = _extract_meta_from_latex(tex)
    if subject_hint:
        c = _normalise_subject(subject_hint)
        if c and c not in meta["subjects"]: meta["subjects"] = [c]
    logger.info(f"[llm_parser] tex={len(tex):,}chars meta={meta}")

    # Taxonomy
    taxonomy = dict(_TAXONOMY)
    if pool:
        try:
            from services.llm_tagger import _get_db_taxonomy
            for s in meta["subjects"]:
                db = await _get_db_taxonomy(pool, s)
                if db: taxonomy[s] = {**taxonomy.get(s, {}), **db}
        except Exception as e:
            logger.warning(f"[llm_parser] DB taxonomy: {e}")

    taxonomy_text  = format_taxonomy_for_prompt(taxonomy, meta["subjects"])
    expected_count = get_expected_count(meta["exam_type"], meta["year"], meta["subjects"])

    prompt = f"""{PARSER_SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLE}

══════════════════════════════════════════════════════
PARSE THIS PAPER NOW
══════════════════════════════════════════════════════
Exam: {meta["exam_type"]} | Year: {meta["year"]} | Date: {meta["exam_date"]} | Shift: {meta["shift"]}
Subjects: {", ".join(meta["subjects"])} | Expected questions: {expected_count}

TAXONOMY — use ONLY these values for chapter_name / topic_name:
{taxonomy_text}

RULES:
1. Return a raw JSON array only — no markdown, no explanation, start with [
2. In JSON strings: \\ becomes \\\\ (double every backslash)
3. Options: strip "(1)" prefix — keep text only
4. Answer: "1"/"2"/"3"/"4" for MCQ, numeric string for NUMERICAL
5. SECTION-B = NUMERICAL, options = []
6. Extract ALL subjects, ALL questions — do NOT stop early

PAPER:
---BEGIN---
{tex}
---END---"""

    logger.info("[llm_parser] Calling Gemini")
    loop = asyncio.get_running_loop()
    try:
        raw = await loop.run_in_executor(None, _call_gemini_sync, key, prompt, PARSE_MODEL)
    except Exception as e:
        logger.error(f"[llm_parser] Gemini failed: {e}")
        return []

    logger.info(f"[llm_parser] Response {len(raw):,} chars | preview: {raw[:200]}")

    questions = _extract_json(raw)
    logger.info(f"[llm_parser] Extracted {len(questions)} raw questions")

    validated = []
    for q in questions:
        fixed = _validate_and_fix(dict(q), meta)
        if fixed: validated.append(fixed)

    # Sort
    so = {s: i for i, s in enumerate(meta["subjects"])}
    validated.sort(key=lambda q: (so.get(q.get("subject",""), 99), q.get("number", 0)))

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"[llm_parser] DONE {elapsed:.1f}s — {len(validated)} questions")
    for s in meta["subjects"]:
        qs = [q for q in validated if q.get("subject") == s]
        a  = len([q for q in qs if q.get("section") == "SECTION-A"])
        b  = len([q for q in qs if q.get("section") == "SECTION-B"])
        nums = sorted([q["number"] for q in qs])
        missing = [n for n in range(nums[0], nums[-1]+1) if n not in nums] if nums else []
        flag = "✓" if not missing else f"⚠ MISSING:{missing[:10]}"
        logger.info(f"[llm_parser]   {s:12s} | total={len(qs):>3} A={a} B={b} | {flag}")
    logger.info("=" * 60)

    return validated


def parse_latex_sync(tex: str, subject_hint: str = "", api_key: str = "") -> list[dict]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as p:
                return p.submit(asyncio.run,
                    parse_latex_with_llm(tex, subject_hint, api_key)).result(timeout=300)
        return loop.run_until_complete(parse_latex_with_llm(tex, subject_hint, api_key))
    except Exception as e:
        logger.error(f"[llm_parser] sync failed: {e}")
        return []