"""
services/llm_parser.py  —  Gemini-powered JEE/NEET paper parser
Single call, few-shot, new google-genai SDK.
"""

import asyncio
import copy
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

# ── Few-shot example — uses REAL paper format ─────────────────────────────────
FEW_SHOT_EXAMPLE = r'''
EXAMPLE INPUT (actual JEE Main LaTeX format):

\section*{26$^{th}$ Feb. 2021 | Shift - 1 PHYSICS}
\section*{SECTION - A}
\begin{enumerate}
  \item If $\lambda_{1}$ and $\lambda_{2}$ are the wavelengths of the third member of Lyman
  and first member of the Paschen series, then $\lambda_{1}: \lambda_{2}$ is :\\
  (1) $1: 3$\\
  (2) $1: 9$\\
  (3) $7: 135$\\
  (4) $7: 108$
\end{enumerate}

\section*{Sol. (3)}
For Lyman series $n_1=1, n_2=4$\\
$\frac{1}{\lambda_1} = R\left(\frac{1}{1} - \frac{1}{16}\right) = \frac{15R}{16}$

\begin{enumerate}
  \setcounter{enumi}{1}
  \item A wire has mass per unit length $0.135$ g/cm.\\
  \includegraphics[max width=\textwidth]{img-02_239}\\
  (1) $\frac{\theta_1 R_2 + \theta_2 R_1}{R_1 + R_2}$\\
  (2) $\frac{\theta_1 R_2 - \theta_2 R_1}{R_2 - R_1}$\\
  (3) $\frac{\theta_2 R_2 - \theta_1 R_1}{R_2 - R_1}$\\
  (4) $\frac{\theta_1 R_1 + \theta_2 R_2}{R_1 + R_2}$
\end{enumerate}

\section*{Sol. (1)}
At junction: $\theta = \frac{R_1\theta_2 + R_2\theta_1}{R_1 + R_2}$

\section*{SECTION - B}
\begin{enumerate}
  \setcounter{enumi}{20}
  \item A wave $y = -0.21\sin(x+30t)$ is produced in wire. Tension is $x \times 10^{-2}$ N. $x$ = \_\_\_\_.
\end{enumerate}

Sol. 1215\\
$v = \omega/k = 30$ m/s, $T = v^2\mu = 12.15$ N

EXAMPLE OUTPUT — return EXACTLY this JSON structure:
[
  {
    "number": 1,
    "q_type": "MCQ",
    "subject": "PHYSICS",
    "section": "SECTION-A",
    "year": "2021",
    "shift": "Morning",
    "exam_date": "2021-02-26",
    "question": "If $\\lambda_{1}$ and $\\lambda_{2}$ are the wavelengths of the third member of Lyman and first member of the Paschen series, then $\\lambda_{1}: \\lambda_{2}$ is :",
    "options": ["$1: 3$", "$1: 9$", "$7: 135$", "$7: 108$"],
    "answer": "3",
    "solution": "For Lyman series $n_1=1, n_2=4$\n$\\frac{1}{\\lambda_1} = R\\left(\\frac{1}{1} - \\frac{1}{16}\\right) = \\frac{15R}{16}$",
    "chapter_name": "Modern Physics",
    "topic_name": "Hydrogen Spectrum",
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
    "year": "2021",
    "shift": "Morning",
    "exam_date": "2021-02-26",
    "question": "A wire has mass per unit length $0.135$ g/cm. [IMAGE:img-02_239]",
    "options": ["$\\frac{\\theta_1 R_2 + \\theta_2 R_1}{R_1 + R_2}$", "$\\frac{\\theta_1 R_2 - \\theta_2 R_1}{R_2 - R_1}$", "$\\frac{\\theta_2 R_2 - \\theta_1 R_1}{R_2 - R_1}$", "$\\frac{\\theta_1 R_1 + \\theta_2 R_2}{R_1 + R_2}$"],
    "answer": "1",
    "solution": "At junction: $\\theta = \\frac{R_1\\theta_2 + R_2\\theta_1}{R_1 + R_2}$",
    "chapter_name": "Heat Transfer",
    "topic_name": "Thermal Resistance",
    "difficulty": "medium",
    "q_images": ["img-02_239"],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": -1
  },
  {
    "number": 21,
    "q_type": "NUMERICAL",
    "subject": "PHYSICS",
    "section": "SECTION-B",
    "year": "2021",
    "shift": "Morning",
    "exam_date": "2021-02-26",
    "question": "A wave $y = -0.21\\sin(x+30t)$ is produced in wire. Tension is $x \\times 10^{-2}$ N. $x$ = ____.",
    "options": [],
    "answer": "1215",
    "solution": "$v = \\omega/k = 30$ m/s, $T = v^2\\mu = 12.15$ N",
    "chapter_name": "Waves",
    "topic_name": "Wave Speed",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": 0
  }
]

STRICT RULES — follow exactly:
1. Return ONLY a valid JSON array. Start with [ end with ]. No markdown, no explanation.
2. options[]: strip "(1)" prefix — keep only the text/math after it
3. answer: "1"/"2"/"3"/"4" for MCQ, numeric string for NUMERICAL
4. \includegraphics{img-xyz} → add [IMAGE:img-xyz] in question text AND add "img-xyz" to q_images[]
5. \setcounter{enumi}{N} means next \item is question number N+1
6. "Sol. (X)" → answer="X". Sol text after that → solution field
7. SECTION-A = MCQ (marks_wrong: -1), SECTION-B = NUMERICAL (marks_wrong: 0, options: [])
8. Extract year/shift/date from section heading like "26th Feb. 2021 | Shift - 1"
9. DO NOT repeat questions. Each question appears exactly ONCE.
10. Extract ALL subjects and ALL questions — Physics 25, Chemistry 25, Maths 25 = 75 total
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
    combined = (m.group(1) if m else "") + " " + tex[:2000]

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


# ── Gemini call ───────────────────────────────────────────────────────────────
def _call_gemini_sync(api_key: str, prompt: str, model: str = PARSE_MODEL) -> str:
    """
    Call Gemini with retry on 503/429, fallback to gemini-2.0-flash on 404.
    Handles thinking models — skips thought parts, collects only text parts.
    """
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
                            temperature=0.1,   # 0.0 can cause issues on some models
                            max_output_tokens=32000,
                        )
                    )
                    # Thinking models: skip thought parts, collect only text parts
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
                    logger.info(f"[llm_parser] Got {len(text):,} chars")
                    return text

                else:
                    genai_old.configure(api_key=api_key)
                    m = genai_old.GenerativeModel(model_name=current_model)
                    r = m.generate_content(
                        prompt,
                        generation_config={"temperature": 0.1, "max_output_tokens": 32000}
                    )
                    return r.text or ""

            except Exception as e:
                err = str(e)
                logger.warning(f"[llm_parser] {current_model} attempt={attempt+1}: {err[:150]}")
                if any(x in err for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                    if attempt < 2:
                        delay = [15, 45][attempt]
                        logger.info(f"[llm_parser] Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                break  # 404 or non-retryable → try next model

    raise RuntimeError("[llm_parser] All Gemini models failed")


# ── JSON extraction ───────────────────────────────────────────────────────────
def _extract_json(raw: str) -> list:
    """
    Robust JSON extraction:
    - strips markdown fences
    - skips thinking preamble (finds first '[')
    - uses JSONDecodeError.pos to truncate broken JSON
    - falls back to manual depth tracking for recovery
    """
    if not raw:
        return []

    # Strip markdown fences
    raw = re.sub(r'```json', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'```', '', raw)
    raw = raw.strip()

    # Find first '[' — skip thinking/preamble
    start = raw.find("[")
    if start == -1:
        logger.error(f"[llm_parser] No '[' in response. Sample: {raw[:300]}")
        return []

    raw = raw[start:]  # now starts with '['

    # Attempt 1: full parse
    end = raw.rfind("]")
    if end > 0:
        try:
            result = json.loads(raw[:end+1])
            if isinstance(result, list):
                logger.info(f"[llm_parser] JSON parsed OK — {len(result)} items")
                return result
        except json.JSONDecodeError as je:
            logger.warning(f"[llm_parser] JSON error: {je.msg} at pos={je.pos}")
            # Attempt 2: use error position to truncate
            if je.pos and je.pos > 10:
                # Walk back to find last complete object '}'
                truncate_at = raw.rfind("}", 0, je.pos)
                if truncate_at > 0:
                    try:
                        result = json.loads(raw[:truncate_at+1] + "]")
                        if isinstance(result, list) and result:
                            logger.warning(f"[llm_parser] Recovered {len(result)} items via pos truncation")
                            return result
                    except json.JSONDecodeError:
                        pass

    # Attempt 3: manual depth tracking — find last complete object
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
            result = json.loads(raw[:last_complete+1] + "]")
            if isinstance(result, list) and result:
                logger.warning(f"[llm_parser] Recovered {len(result)} items via depth tracking")
                return result
        except json.JSONDecodeError:
            pass

    logger.error(f"[llm_parser] All JSON recovery failed. Raw sample: {raw[:500]}")
    return []


# ── Validation ────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "section": "SECTION-A", "year": "", "shift": "", "exam_date": "",
    "options": [], "answer": "", "solution": "", "chapter_name": "",
    "topic_name": "", "difficulty": "medium", "q_images": [], "sol_images": [],
    "marks_correct": 4, "marks_wrong": -1, "verified": False,
    "chapter_id": None, "topic": "",
}
_VALID_Q = {"MCQ", "MSQ", "NUMERICAL"}
_VALID_S = {"PHYSICS", "CHEMISTRY", "MATHEMATICS", "BIOLOGY"}
_VALID_D = {"easy", "medium", "hard"}
_NTA     = {"A":"1","B":"2","C":"3","D":"4","a":"1","b":"2","c":"3","d":"4"}

# Fix: exhaustive subject aliases including NEET subjects
_SUBJ_ALIASES = {
    "PHYSICS": "PHYSICS", "PHY": "PHYSICS",
    "CHEMISTRY": "CHEMISTRY", "CHEM": "CHEMISTRY",
    "MATHEMATICS": "MATHEMATICS", "MATHS": "MATHEMATICS", "MATH": "MATHEMATICS",
    "BIOLOGY": "BIOLOGY", "BIO": "BIOLOGY",
    "BOTANY": "BIOLOGY", "ZOOLOGY": "BIOLOGY",
}


def _validate_and_fix(q: dict, meta: dict) -> Optional[dict]:
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

    # subject — use alias map
    raw_subj = str(q.get("subject", "")).strip().upper()
    q["subject"] = _SUBJ_ALIASES.get(raw_subj, raw_subj if raw_subj in _VALID_S else "PHYSICS")

    # section — strict check: only "SECTION - B" or "SECTION-B" exact
    sec = str(q.get("section", "")).strip().upper().replace(" ", "")
    if sec in ("SECTIONB", "SECTION-B", "B"):
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
    if not q.get("marks_correct"): q["marks_correct"] = 4
    if q.get("marks_wrong") is None: q["marks_wrong"] = -1

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

    t0   = time.time()
    tex  = _normalise_image_refs(tex)
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

    loop = asyncio.get_running_loop()

    # ── Subject-wise calls — one per subject for reliability ──────────────────
    # One big call for 75 questions is unreliable (truncation, repetition).
    # Instead: 3 calls x 25 questions each = much more reliable.
    all_questions = []

    for subject in meta["subjects"]:
        subj_taxonomy = format_taxonomy_for_prompt(taxonomy, [subject])
        subj_expected = get_expected_count(meta["exam_type"], meta["year"], [subject])

        prompt = f"""{PARSER_SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLE}

══════════════════════════════════════════════════════
EXTRACT ONLY {subject} QUESTIONS FROM THIS PAPER
══════════════════════════════════════════════════════
Exam: {meta["exam_type"]} | Year: {meta["year"]} | Date: {meta["exam_date"]} | Shift: {meta["shift"]}
Subject: {subject} | Expected: {subj_expected} questions (SECTION-A: 20 MCQ + SECTION-B: 10 NUMERICAL = 30 total)

TAXONOMY for {subject} — use ONLY these for chapter_name / topic_name:
{subj_taxonomy}

CRITICAL:
- Return ONLY a valid JSON array starting with [ and ending with ]
- Extract ONLY {subject} questions — ignore PHYSICS/CHEMISTRY/MATHEMATICS/BIOLOGY questions from other subjects
- All {subj_expected} questions must be present — SECTION-A: 20 MCQ (Q1-20) + SECTION-B: 10 NUMERICAL (Q21-30)
- Each question appears EXACTLY ONCE — no duplicates
- subject field must be "{subject}" for all questions

PAPER:
---BEGIN---
{tex}
---END---"""

        logger.info(f"[llm_parser] Calling Gemini for {subject}")
        try:
            raw = await loop.run_in_executor(None, _call_gemini_sync, key, prompt, PARSE_MODEL)
        except Exception as e:
            logger.error(f"[llm_parser] {subject} Gemini call failed: {e}")
            continue

        logger.info(f"[llm_parser] {subject} response {len(raw):,} chars | preview: {raw[:200]}")
        subj_questions = _extract_json(raw)

        # Force subject field to be correct
        for q in subj_questions:
            q["subject"] = subject
        logger.info(f"[llm_parser] {subject}: extracted {len(subj_questions)} questions")
        all_questions.extend(subj_questions)

    questions = all_questions
    logger.info(f"[llm_parser] Total extracted: {len(questions)} questions")

    # Deduplicate by (subject, number) — keep first occurrence
    seen: dict[tuple, dict] = {}
    for q in questions:
        try:
            key_tuple = (str(q.get("subject", "")), int(str(q.get("number", 0))))
        except (ValueError, TypeError):
            continue
        if key_tuple not in seen:
            seen[key_tuple] = q
    questions = list(seen.values())
    logger.info(f"[llm_parser] After dedup: {len(questions)} questions")

    validated = []
    for q in questions:
        fixed = _validate_and_fix(dict(q), meta)
        if fixed: validated.append(fixed)

    # Sort by subject order then question number
    so = {s: i for i, s in enumerate(meta["subjects"])}
    validated.sort(key=lambda q: (so.get(q.get("subject", ""), 99), q.get("number", 0)))

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


# ── Sync wrapper ──────────────────────────────────────────────────────────────
def parse_latex_sync(tex: str, subject_hint: str = "", api_key: str = "") -> list[dict]:
    """Sync wrapper — safely handles both running and non-running event loops."""
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — use run_coroutine_threadsafe
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            parse_latex_with_llm(tex, subject_hint, api_key), loop
        )
        return future.result(timeout=300)
    except RuntimeError:
        # No running loop — safe to use asyncio.run
        return asyncio.run(parse_latex_with_llm(tex, subject_hint, api_key))
    except Exception as e:
        logger.error(f"[llm_parser] sync failed: {e}")
        return []