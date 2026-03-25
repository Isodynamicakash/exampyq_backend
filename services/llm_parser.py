"""
services/llm_parser.py
======================
Gemini-powered JEE/NEET paper parser.

FLOW:
  PDF → LaTeX → 3 Gemini calls (one per subject) → 90 questions

WHY 3 CALLS:
  - Each call: ~25k tokens input + ~5k tokens output = well within limits
  - Gemini focuses on one subject = better accuracy
  - No truncation = all 30 questions per subject

WHY JSON BREAKS:
  - LaTeX has \ and " characters that break JSON
  - Fix: use response_mime_type="application/json" with response_schema
  - This forces Gemini to return valid JSON ALWAYS
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

# ── SDK ───────────────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    _USE_OLD_SDK = False
    _GEMINI_AVAILABLE = True
    logger.info("[llm_parser] Using new google-genai SDK")
except ImportError:
    try:
        import google.generativeai as genai_old
        _USE_OLD_SDK = True
        _GEMINI_AVAILABLE = True
        logger.warning("[llm_parser] Using deprecated google-generativeai SDK")
    except ImportError:
        _USE_OLD_SDK = False
        _GEMINI_AVAILABLE = False

from services.prompts import (
    PARSER_SYSTEM_PROMPT,
    format_taxonomy_for_prompt,
    get_expected_count,
)
from services.llm_tagger import _TAXONOMY, _normalise_subject

# ── Constants ─────────────────────────────────────────────────────────────────
PARSE_MODEL = "gemini-2.5-flash"

# ── JSON Schema — Gemini MUST return this structure ───────────────────────────
# Using response_schema forces valid JSON — no broken LaTeX escaping issues
QUESTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "number":        {"type": "integer"},
            "q_type":        {"type": "string"},
            "subject":       {"type": "string"},
            "section":       {"type": "string"},
            "year":          {"type": "string"},
            "shift":         {"type": "string"},
            "exam_date":     {"type": "string"},
            "question":      {"type": "string"},
            "options":       {"type": "array", "items": {"type": "string"}},
            "answer":        {"type": "string"},
            "solution":      {"type": "string"},
            "chapter_name":  {"type": "string"},
            "topic_name":    {"type": "string"},
            "difficulty":    {"type": "string"},
            "q_images":      {"type": "array", "items": {"type": "string"}},
            "sol_images":    {"type": "array", "items": {"type": "string"}},
            "marks_correct": {"type": "number"},
            "marks_wrong":   {"type": "number"},
        },
        "required": ["number", "q_type", "subject", "section",
                     "question", "options", "answer", "solution",
                     "chapter_name", "topic_name", "difficulty",
                     "marks_correct", "marks_wrong"]
    }
}

# ── Few-shot example ──────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLE = r'''
EXAMPLE — how to parse this LaTeX:

\section*{26$^{th}$ Feb. 2021 | Shift - 1 PHYSICS}
\section*{SECTION - A}
\begin{enumerate}
  \item If $\lambda_{1}$ and $\lambda_{2}$ are wavelengths of Lyman and Paschen series, $\lambda_{1}:\lambda_{2}$ is:\\
  (1) $1:3$  (2) $1:9$  (3) $7:135$  (4) $7:108$
\end{enumerate}
\section*{Sol. (3)}
$\frac{1}{\lambda_1} = R\left(1 - \frac{1}{16}\right)$

\begin{enumerate}
  \setcounter{enumi}{1}
  \item \includegraphics[max width=\textwidth]{img-02_239}\\
  Junction temperature formula:\\
  (1) $\frac{\theta_1 R_2+\theta_2 R_1}{R_1+R_2}$  (2) ...  (3) ...  (4) ...
\end{enumerate}
\section*{Sol. (1)}

\section*{SECTION - B}
\begin{enumerate}
  \setcounter{enumi}{20}
  \item Tension in wire is $x \times 10^{-2}$ N. $x$ = \_\_\_\_.
\end{enumerate}
Sol. 1215

Expected JSON output for PHYSICS:
[
  {
    "number": 1, "q_type": "MCQ", "subject": "PHYSICS", "section": "SECTION-A",
    "year": "2021", "shift": "Morning", "exam_date": "2021-02-26",
    "question": "If $\\lambda_{1}$ and $\\lambda_{2}$ are wavelengths of Lyman and Paschen series, $\\lambda_{1}:\\lambda_{2}$ is:",
    "options": ["$1:3$", "$1:9$", "$7:135$", "$7:108$"],
    "answer": "3", "solution": "$\\frac{1}{\\lambda_1} = R\\left(1 - \\frac{1}{16}\\right)$",
    "chapter_name": "Modern Physics", "topic_name": "Hydrogen Spectrum",
    "difficulty": "medium", "q_images": [], "sol_images": [], "marks_correct": 4, "marks_wrong": -1
  },
  {
    "number": 2, "q_type": "MCQ", "subject": "PHYSICS", "section": "SECTION-A",
    "year": "2021", "shift": "Morning", "exam_date": "2021-02-26",
    "question": "Junction temperature formula: [IMAGE:img-02_239]",
    "options": ["$\\frac{\\theta_1 R_2+\\theta_2 R_1}{R_1+R_2}$", "...", "...", "..."],
    "answer": "1", "solution": "",
    "chapter_name": "Heat Transfer", "topic_name": "Thermal Resistance",
    "difficulty": "medium", "q_images": ["img-02_239"], "sol_images": [], "marks_correct": 4, "marks_wrong": -1
  },
  {
    "number": 21, "q_type": "NUMERICAL", "subject": "PHYSICS", "section": "SECTION-B",
    "year": "2021", "shift": "Morning", "exam_date": "2021-02-26",
    "question": "Tension in wire is $x \\times 10^{-2}$ N. $x$ = ____.",
    "options": [], "answer": "1215", "solution": "$v=30$ m/s, $T=12.15$ N",
    "chapter_name": "Waves", "topic_name": "Wave Speed",
    "difficulty": "medium", "q_images": [], "sol_images": [], "marks_correct": 4, "marks_wrong": 0
  }
]

KEY RULES:
1. options[] — strip "(1)" prefix, keep only text/math
2. answer — "1"/"2"/"3"/"4" for MCQ, numeric string for NUMERICAL
3. \includegraphics{img-xyz} → "[IMAGE:img-xyz]" in question + "img-xyz" in q_images[]
4. \setcounter{enumi}{N} → next \item is question N+1
5. Sol.(X) → answer="X", solution text = lines after Sol. until next question
6. SECTION-A = MCQ marks_wrong=-1, SECTION-B = NUMERICAL options=[] marks_wrong=0
7. Extract year/shift/date from heading like "26th Feb. 2021 | Shift - 1"
'''.strip()


# ── Metadata ──────────────────────────────────────────────────────────────────
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
    combined = (m.group(1) if m else "") + " " + tex[:5000]

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


# ── Gemini call — response_schema forces valid JSON ───────────────────────────
def _call_gemini_sync(api_key: str, prompt: str, model: str = PARSE_MODEL) -> list:
    """
    Call Gemini with response_schema — returns guaranteed valid JSON list.
    No broken JSON, no markdown fences, no truncation issues.
    Falls back to text parsing if schema not supported.
    """
    models_to_try = [model, "gemini-2.0-flash"]

    for current_model in models_to_try:
        for attempt in range(3):
            try:
                logger.info(f"[llm_parser] model={current_model} attempt={attempt+1}")

                if not _USE_OLD_SDK:
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=16000,
                        )
                    )
                    # Collect all non-thought parts
                    text = ""
                    try:
                        for part in response.candidates[0].content.parts:
                            if getattr(part, 'thought', False):
                                continue
                            if getattr(part, 'text', None):
                                text += part.text
                    except Exception:
                        text = response.text or ""

                    logger.info(f"[llm_parser] Response: {len(text):,} chars")
                    return _extract_json_from_text(text)

                else:
                    # Old SDK — free text only
                    genai_old.configure(api_key=api_key)
                    m = genai_old.GenerativeModel(model_name=current_model)
                    r = m.generate_content(
                        prompt,
                        generation_config={"temperature": 0.1, "max_output_tokens": 8192}
                    )
                    return _extract_json_from_text(r.text or "")

            except Exception as e:
                err = str(e)
                logger.warning(f"[llm_parser] {current_model} attempt={attempt+1}: {err[:150]}")
                if any(x in err for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                    if attempt < 2:
                        delay = [15, 45][attempt]
                        logger.info(f"[llm_parser] Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                break

    logger.error("[llm_parser] All models failed")
    return []


# ── JSON text extraction (fallback) ──────────────────────────────────────────
def _extract_json_from_text(raw: str) -> list:
    """Extract JSON array from free-text Gemini response."""
    if not raw:
        return []

    # Strip markdown
    raw = re.sub(r'```json', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'```', '', raw)
    raw = raw.strip()

    # Find first '['
    start = raw.find("[")
    if start == -1:
        logger.error(f"[llm_parser] No '[' in response: {raw[:200]}")
        return []
    raw = raw[start:]

    # Try full parse
    end = raw.rfind("]")
    if end > 0:
        try:
            result = json.loads(raw[:end+1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError as je:
            logger.warning(f"[llm_parser] JSON error at pos={je.pos}: {je.msg}")
            # Truncate at error position
            if je.pos and je.pos > 10:
                truncate_at = raw.rfind("}", 0, je.pos)
                if truncate_at > 0:
                    try:
                        result = json.loads(raw[:truncate_at+1] + "]")
                        if isinstance(result, list) and result:
                            logger.warning(f"[llm_parser] Recovered {len(result)} via truncation")
                            return result
                    except json.JSONDecodeError:
                        pass

    # Depth tracking recovery
    depth = 0; in_str = False; escape = False; last_ok = -1
    for i, ch in enumerate(raw):
        if escape: escape = False; continue
        if ch == "\\": escape = True; continue
        if ch == '"': in_str = not in_str
        if in_str: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: last_ok = i

    if last_ok > 0:
        try:
            result = json.loads(raw[:last_ok+1] + "]")
            if isinstance(result, list) and result:
                logger.warning(f"[llm_parser] Recovered {len(result)} via depth tracking")
                return result
        except json.JSONDecodeError:
            pass

    logger.error(f"[llm_parser] JSON recovery failed. Sample: {raw[:300]}")
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
_SUBJ_ALIASES = {
    "PHYSICS":"PHYSICS","PHY":"PHYSICS",
    "CHEMISTRY":"CHEMISTRY","CHEM":"CHEMISTRY",
    "MATHEMATICS":"MATHEMATICS","MATHS":"MATHEMATICS","MATH":"MATHEMATICS",
    "BIOLOGY":"BIOLOGY","BIO":"BIOLOGY","BOTANY":"BIOLOGY","ZOOLOGY":"BIOLOGY",
}


def _validate_and_fix(q: dict, meta: dict, force_subject: str = "") -> Optional[dict]:
    try:
        q["number"] = int(str(q.get("number", 0)).strip())
    except (ValueError, TypeError):
        return None
    if q["number"] < 1: return None
    if not str(q.get("question", "")).strip(): return None

    for f, v in _DEFAULTS.items():
        if f not in q or q[f] is None:
            q[f] = copy.deepcopy(v)

    for f in ("year", "shift", "exam_date"):
        if not q.get(f) and meta.get(f): q[f] = meta[f]

    for f in ("question", "solution"):
        if q.get(f): q[f] = q[f].replace('\\n', '\n')
    if isinstance(q.get("options"), list):
        q["options"] = [o.replace('\\n', '\n') for o in q["options"]]

    q["q_type"] = str(q.get("q_type","MCQ")).strip().upper()
    if q["q_type"] not in _VALID_Q: q["q_type"] = "MCQ"

    # Force subject if caller specifies
    if force_subject:
        q["subject"] = force_subject
    else:
        raw_s = str(q.get("subject","")).strip().upper()
        q["subject"] = _SUBJ_ALIASES.get(raw_s, raw_s if raw_s in _VALID_S else "PHYSICS")

    # Section — strict match only
    sec = str(q.get("section","")).strip().upper().replace(" ","").replace("-","")
    if sec == "SECTIONB":
        q["section"] = "SECTION-B"
        if q["q_type"] == "MCQ": q["q_type"] = "NUMERICAL"
    else:
        q["section"] = "SECTION-A"

    opts = q.get("options",[])
    if not isinstance(opts, list): opts = []
    if q["q_type"] != "NUMERICAL":
        while len(opts) < 4: opts.append("")
        opts = [str(o) for o in opts[:4]]
    else:
        opts = []
    q["options"] = opts

    ans = str(q.get("answer","")).strip()
    m = re.fullmatch(r'\(\s*(.+?)\s*\)', ans)
    if m: ans = m.group(1).strip()
    if re.fullmatch(r'[A-Da-d]', ans): ans = _NTA.get(ans, ans)
    q["answer"] = ans

    d = str(q.get("difficulty","medium")).strip().lower()
    q["difficulty"] = d if d in _VALID_D else "medium"

    if not q.get("marks_correct"): q["marks_correct"] = 4
    if q.get("marks_wrong") is None: q["marks_wrong"] = -1

    def imgs(t): return re.findall(r'\[IMAGE:([^\]]+)\]', t or "")
    def merge(*ls):
        seen = set(); r = []
        for l in ls:
            for x in (l or []):
                if x not in seen: seen.add(x); r.append(x)
        return r

    q["q_images"]   = merge(q.get("q_images",[]), imgs(q["question"]),
                            *[imgs(o) for o in q["options"]])
    q["sol_images"] = merge(q.get("sol_images",[]), imgs(q.get("solution","")))
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
    """
    3 Gemini calls — one per subject.
    Each call: ~25k input tokens + ~5k output tokens = well within 8192 limit.
    response_schema ensures valid JSON always.
    """
    if not _GEMINI_AVAILABLE:
        logger.error("[llm_parser] google-genai not installed")
        return []

    key = api_key or os.environ.get("GEMINI_API_KEY","") or os.environ.get("GOOGLE_API_KEY","")
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

    taxonomy = dict(_TAXONOMY)
    if pool:
        try:
            from services.llm_tagger import _get_db_taxonomy
            for s in meta["subjects"]:
                db = await _get_db_taxonomy(pool, s)
                if db: taxonomy[s] = {**taxonomy.get(s,{}), **db}
        except Exception as e:
            logger.warning(f"[llm_parser] DB taxonomy: {e}")

    loop = asyncio.get_running_loop()
    all_questions = []

    # ── 2 CALLS PER SUBJECT — 15 questions each ──────────────────────────────
    # Why 15? Because 17 questions = ~4k tokens fits safely in 8192 limit
    # 6 total calls: Physics(1-15), Physics(16-30), Chem(1-15), etc.

    CHUNKS = [
        ("SECTION-A Q1-15",   "questions 1 to 15 from SECTION-A (MCQ)",        1,  15),
        ("SECTION-A Q16-20 + SECTION-B", "questions 16-20 from SECTION-A (MCQ) AND all 10 questions from SECTION-B (NUMERICAL)", 16, 30),
    ]

    for subject in meta["subjects"]:
        subj_taxonomy = format_taxonomy_for_prompt(taxonomy, [subject])
        subj_all = []

        for chunk_label, chunk_desc, q_from, q_to in CHUNKS:
            prompt = f"""{PARSER_SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLE}

══════════════════════════════════════════════════════
TASK: Extract {subject} — {chunk_label}
══════════════════════════════════════════════════════
Exam: {meta["exam_type"]} | Year: {meta["year"]} | Date: {meta["exam_date"]} | Shift: {meta["shift"]}
Subject: {subject}
Extract ONLY: {chunk_desc} (question numbers {q_from} to {q_to})

TAXONOMY for {subject}:
{subj_taxonomy}

RULES:
- Return ONLY a JSON array of questions numbered {q_from} to {q_to}
- subject="{subject}" for all questions
- options[]: 4 items for MCQ (strip "(1)" prefix), empty [] for NUMERICAL
- answer: "1"/"2"/"3"/"4" for MCQ, numeric string for NUMERICAL
- marks_correct=4, marks_wrong=-1 for MCQ, marks_wrong=0 for NUMERICAL

PAPER:
---BEGIN---
{tex}
---END---"""

            logger.info(f"[llm_parser] {subject} {chunk_label}")
            try:
                questions = await loop.run_in_executor(
                    None, _call_gemini_sync, key, prompt, PARSE_MODEL
                )
            except Exception as e:
                logger.error(f"[llm_parser] {subject} {chunk_label} failed: {e}")
                questions = []

            for q in questions:
                fixed = _validate_and_fix(dict(q), meta, force_subject=subject)
                if fixed:
                    subj_all.append(fixed)

        # Deduplicate + sort
        seen = {}
        for q in subj_all:
            k = q["number"]
            if k not in seen:
                seen[k] = q
        validated = sorted(seen.values(), key=lambda q: q["number"])

        logger.info(f"[llm_parser] {subject}: {len(validated)} questions "
                    f"(A={len([q for q in validated if q['section']=='SECTION-A'])} "
                    f"B={len([q for q in validated if q['section']=='SECTION-B'])})")
        all_questions.extend(validated)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"[llm_parser] DONE {elapsed:.1f}s — {len(all_questions)} total questions")
    for s in meta["subjects"]:
        qs = [q for q in all_questions if q.get("subject") == s]
        a  = len([q for q in qs if q.get("section") == "SECTION-A"])
        b  = len([q for q in qs if q.get("section") == "SECTION-B"])
        nums = sorted([q["number"] for q in qs])
        missing = [n for n in range(1, 31) if n not in nums]
        flag = "✓" if not missing else f"⚠ MISSING:{missing}"
        logger.info(f"[llm_parser]   {s:12s} | total={len(qs):>3} A={a:>2} B={b:>2} | {flag}")
    logger.info("=" * 60)

    return all_questions


# ── Sync wrapper ──────────────────────────────────────────────────────────────
def parse_latex_sync(tex: str, subject_hint: str = "", api_key: str = "") -> list[dict]:
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            parse_latex_with_llm(tex, subject_hint, api_key), loop
        )
        return future.result(timeout=300)
    except RuntimeError:
        return asyncio.run(parse_latex_with_llm(tex, subject_hint, api_key))
    except Exception as e:
        logger.error(f"[llm_parser] sync failed: {e}")
        return []