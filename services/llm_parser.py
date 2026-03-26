"""
services/llm_parser.py
======================
Gemini-powered exam paper parser.

STRATEGY:
  1. Split LaTeX into ~4000 token chunks (split at paragraph boundaries)
  2. Call all chunks PARALLEL — each chunk fits in 8192 token output limit
  3. Gemini decides subject/chapter/topic/difficulty from each chunk
  4. Merge all results, deduplicate by (subject, number)
  5. Works for ANY exam format — JEE, NEET, CUET, etc.

WHY THIS WORKS:
  - 8192 token output limit = ~15 questions per call safely
  - Parallel calls = fast (all chunks at once)
  - No subject pre-filtering = works for any paper format
  - Overlap between chunks = no questions lost at boundaries
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
except ImportError:
    try:
        import google.generativeai as genai_old
        _USE_OLD_SDK = True
        _GEMINI_AVAILABLE = True
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
PARSE_MODEL      = "gemini-2.5-flash"
# ~4 chars per token, 8192 output tokens safe limit
# Input: 16000 tokens = ~64000 chars → split into ~20000 char chunks
CHUNK_CHARS      = 20000   # ~5000 tokens input per chunk
CHUNK_OVERLAP    = 2000    # overlap to avoid losing questions at boundaries
MAX_PARALLEL     = 4       # max simultaneous Gemini calls

# ── Few-shot example ──────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLE = r'''
EXAMPLE — parse this LaTeX chunk:

\section*{26$^{th}$ Feb. 2021 | Shift - 1 PHYSICS}
\section*{SECTION - A}
\begin{enumerate}
  \item If $\lambda_{1}$ and $\lambda_{2}$ are wavelengths of Lyman and Paschen series, $\lambda_{1}:\lambda_{2}$ is:\\
  (1) $1:3$ \quad (2) $1:9$ \quad (3) $7:135$ \quad (4) $7:108$
\end{enumerate}
\section*{Sol. (3)}
$\frac{1}{\lambda_1} = R\left(1 - \frac{1}{16}\right) = \frac{15R}{16}$

\begin{enumerate}
  \setcounter{enumi}{1}
  \item \includegraphics[max width=\textwidth]{img-02_239}\\
  (1) $\frac{\theta_1 R_2+\theta_2 R_1}{R_1+R_2}$ \quad (2) $\frac{\theta_1 R_2-\theta_2 R_1}{R_2-R_1}$ \quad (3) $\frac{\theta_2 R_2-\theta_1 R_1}{R_2-R_1}$ \quad (4) $\frac{\theta_1 R_1+\theta_2 R_2}{R_1+R_2}$
\end{enumerate}
\section*{Sol. (1)}
At junction: $\theta = \frac{R_1\theta_2 + R_2\theta_1}{R_1 + R_2}$

\section*{SECTION - B}
\begin{enumerate}
  \setcounter{enumi}{20}
  \item A wave $y = -0.21\sin(x+30t)$. Tension is $x \times 10^{-2}$ N. $x$ = \_\_\_\_.
\end{enumerate}
Sol. 1215

OUTPUT JSON:
[
  {
    "number": 1,
    "q_type": "MCQ",
    "subject": "PHYSICS",
    "section": "SECTION-A",
    "year": "2021",
    "shift": "Morning",
    "exam_date": "2021-02-26",
    "question": "If $\\lambda_{1}$ and $\\lambda_{2}$ are wavelengths of Lyman and Paschen series, $\\lambda_{1}:\\lambda_{2}$ is:",
    "options": ["$1:3$", "$1:9$", "$7:135$", "$7:108$"],
    "answer": "3",
    "solution": "$\\frac{1}{\\lambda_1} = R\\left(1 - \\frac{1}{16}\\right) = \\frac{15R}{16}$",
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
    "question": "[IMAGE:img-02_239]",
    "options": ["$\\frac{\\theta_1 R_2+\\theta_2 R_1}{R_1+R_2}$", "$\\frac{\\theta_1 R_2-\\theta_2 R_1}{R_2-R_1}$", "$\\frac{\\theta_2 R_2-\\theta_1 R_1}{R_2-R_1}$", "$\\frac{\\theta_1 R_1+\\theta_2 R_2}{R_1+R_2}$"],
    "answer": "1",
    "solution": "$\\theta = \\frac{R_1\\theta_2 + R_2\\theta_1}{R_1 + R_2}$",
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
    "question": "A wave $y = -0.21\\sin(x+30t)$. Tension is $x \\times 10^{-2}$ N. $x$ = ____.",
    "options": [],
    "answer": "1215",
    "solution": "$v=30$ m/s, $T=12.15$ N",
    "chapter_name": "Waves",
    "topic_name": "Wave Speed",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": 0
  }
]

RULES:
1. Return ONLY a JSON array — no markdown, no explanation
2. options[]: strip "(1)" prefix — keep only text/math after it
3. answer: "1"/"2"/"3"/"4" for MCQ, numeric string for NUMERICAL
4. \includegraphics{img-xyz} → question gets "[IMAGE:img-xyz]", q_images gets "img-xyz"
5. \setcounter{enumi}{N} → next \item is question N+1
6. Sol.(X) → answer="X"; text after Sol. until next question → solution
7. SECTION-A=MCQ (marks_wrong=-1), SECTION-B=NUMERICAL (options=[], marks_wrong=0)
8. year/shift/exam_date: extract from headings like "26th Feb. 2021 | Shift - 1"
9. subject: detect from headings like "PHYSICS", "CHEMISTRY", "MATHEMATICS", "BIOLOGY"
10. chapter_name/topic_name: use taxonomy provided, pick closest match
11. If chunk has NO questions (e.g. it's a cover page), return empty array []
12. DO NOT invent questions — only extract what is present in this chunk
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
    combined = (m.group(1) if m else "") + " " + tex[:3000]

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

    return {
        "exam_date": exam_date,
        "year": year,
        "shift": shift,
        "exam_type": exam_type,
    }


def _normalise_image_refs(tex: str) -> str:
    def _rep(m):
        basename = m.group(1).strip().split("/")[-1].split("\\")[-1]
        return f"[IMAGE:{basename}]"
    return re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', _rep, tex)


# ── Chunk splitter ────────────────────────────────────────────────────────────
def _split_into_chunks(tex: str, chunk_chars: int = CHUNK_CHARS,
                       overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split LaTeX into overlapping chunks at paragraph boundaries.
    Each chunk is ~chunk_chars characters.
    Overlap ensures questions at chunk boundaries are captured.
    """
    if len(tex) <= chunk_chars:
        return [tex]

    chunks = []
    start  = 0
    total  = len(tex)

    while start < total:
        end = min(start + chunk_chars, total)

        # Find a clean break point — prefer \n\n (paragraph) then \n
        if end < total:
            for sep in ["\n\n", "\n", " "]:
                idx = tex.rfind(sep, start + chunk_chars // 2, end)
                if idx > start:
                    end = idx + len(sep)
                    break

        chunk = tex[start:end]
        if chunk.strip():
            chunks.append(chunk)

        if end >= total:
            break

        # Next chunk starts with overlap
        next_start = end - overlap
        # Snap overlap start to clean boundary
        for sep in ["\n\n", "\n"]:
            idx = tex.find(sep, next_start)
            if idx != -1 and idx < end:
                next_start = idx + len(sep)
                break
        start = max(next_start, start + 1)  # always advance

    logger.info(f"[llm_parser] Split into {len(chunks)} chunks "
                f"(avg {sum(len(c) for c in chunks)//len(chunks):,} chars each)")
    return chunks


# ── Gemini call ───────────────────────────────────────────────────────────────
def _call_gemini_sync(api_key: str, prompt: str,
                      model: str = PARSE_MODEL) -> list:
    """Call Gemini, return parsed list of question dicts."""
    models_to_try = [model, "gemini-2.0-flash"]

    for current_model in models_to_try:
        for attempt in range(3):
            try:
                if not _USE_OLD_SDK:
                    client = genai.Client(api_key=api_key)

                    # Use generate_content_stream and collect ALL chunks
                    # This is required for thinking models (Gemini 2.5 Flash)
                    # where response.text only returns the first part
                    text_parts = []
                    try:
                        for chunk in client.models.generate_content_stream(
                            model=current_model,
                            contents=prompt,
                            config=genai_types.GenerateContentConfig(
                                temperature=0.1,
                                max_output_tokens=8192,
                                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                                    disable=True
                                ),
                            )
                        ):
                            try:
                                for part in chunk.candidates[0].content.parts:
                                    if getattr(part, 'thought', False):
                                        continue
                                    t = getattr(part, 'text', None)
                                    if t:
                                        text_parts.append(t)
                            except Exception:
                                t = getattr(chunk, 'text', None)
                                if t:
                                    text_parts.append(t)
                    except Exception as stream_err:
                        logger.warning(f"[llm_parser] Stream failed: {stream_err}, trying non-stream")
                        # Fallback to non-streaming
                        response = client.models.generate_content(
                            model=current_model,
                            contents=prompt,
                            config=genai_types.GenerateContentConfig(
                                temperature=0.1,
                                max_output_tokens=8192,
                                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                                    disable=True
                                ),
                            )
                        )
                        try:
                            for part in response.candidates[0].content.parts:
                                if getattr(part, 'thought', False): continue
                                t = getattr(part, 'text', None)
                                if t: text_parts.append(t)
                        except Exception:
                            t = response.text or ""
                            if t: text_parts.append(t)

                    text = "".join(text_parts)
                    logger.info(f"[llm_parser] parts={len(text_parts)} total_chars={len(text)}")
                else:
                    genai_old.configure(api_key=api_key)
                    r    = genai_old.GenerativeModel(model_name=current_model)
                    resp = r.generate_content(
                        prompt,
                        generation_config={"temperature": 0.1, "max_output_tokens": 8192}
                    )
                    text = resp.text or ""

                parsed = _parse_json(text)
                logger.info(f"[llm_parser] Raw response {len(text)} chars, parsed {len(parsed)} questions")
                if len(parsed) == 0 and text:
                    logger.info(f"[llm_parser] Response sample: {text[:400]}")
                return parsed

            except Exception as e:
                err = str(e)
                logger.warning(f"[llm_parser] {current_model} attempt={attempt+1}: {err[:120]}")
                if any(x in err for x in ["503","UNAVAILABLE","429","RESOURCE_EXHAUSTED"]):
                    if attempt < 2:
                        delay = [15, 45][attempt]
                        logger.info(f"[llm_parser] Retry in {delay}s...")
                        time.sleep(delay)
                        continue
                break  # non-retryable or 404 → try next model

    return []


# ── JSON parser ───────────────────────────────────────────────────────────────
def _parse_json(raw: str) -> list:
    """
    Robust JSON extraction from Gemini response.
    Handles markdown fences, thinking preamble, truncation.
    """
    if not raw: return []

    # Strip markdown fences
    raw = re.sub(r'```json\s*', '', raw, flags=re.I)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    # Find first '['
    start = raw.find("[")
    if start == -1:
        return []
    raw = raw[start:]

    # Attempt 1: full parse
    end = raw.rfind("]")
    if end > 0:
        try:
            r = json.loads(raw[:end+1])
            if isinstance(r, list): return r
        except json.JSONDecodeError as je:
            # Attempt 2: truncate at error pos
            if je.pos and je.pos > 10:
                trunc = raw.rfind("}", 0, je.pos)
                if trunc > 0:
                    try:
                        r = json.loads(raw[:trunc+1] + "]")
                        if isinstance(r, list) and r:
                            logger.warning(f"[llm_parser] JSON recovered {len(r)} via pos truncation")
                            return r
                    except json.JSONDecodeError:
                        pass

    # Attempt 3: depth tracking
    depth = 0; in_str = False; esc = False; last_ok = -1
    for i, ch in enumerate(raw):
        if esc: esc = False; continue
        if ch == "\\": esc = True; continue
        if ch == '"': in_str = not in_str
        if in_str: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: last_ok = i

    if last_ok > 0:
        try:
            r = json.loads(raw[:last_ok+1] + "]")
            if isinstance(r, list) and r:
                logger.warning(f"[llm_parser] JSON recovered {len(r)} via depth tracking")
                return r
        except json.JSONDecodeError:
            pass

    return []


# ── Validation ────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "section":"SECTION-A","year":"","shift":"","exam_date":"",
    "options":[],"answer":"","solution":"","chapter_name":"",
    "topic_name":"","difficulty":"medium","q_images":[],"sol_images":[],
    "marks_correct":4,"marks_wrong":-1,"verified":False,"chapter_id":None,"topic":"",
}
_VALID_Q = {"MCQ","MSQ","NUMERICAL"}
_VALID_S = {"PHYSICS","CHEMISTRY","MATHEMATICS","BIOLOGY"}
_VALID_D = {"easy","medium","hard"}
_NTA     = {"A":"1","B":"2","C":"3","D":"4","a":"1","b":"2","c":"3","d":"4"}
_SUBJ_ALIASES = {
    "PHYSICS":"PHYSICS","PHY":"PHYSICS",
    "CHEMISTRY":"CHEMISTRY","CHEM":"CHEMISTRY",
    "MATHEMATICS":"MATHEMATICS","MATHS":"MATHEMATICS","MATH":"MATHEMATICS",
    "BIOLOGY":"BIOLOGY","BIO":"BIOLOGY","BOTANY":"BIOLOGY","ZOOLOGY":"BIOLOGY",
}

def _validate(q: dict, meta: dict) -> Optional[dict]:
    try:    q["number"] = int(str(q.get("number",0)).strip())
    except: return None
    if q["number"] < 1: return None
    if not str(q.get("question","")).strip(): return None

    for f, v in _DEFAULTS.items():
        if f not in q or q[f] is None:
            q[f] = copy.deepcopy(v)

    # Stamp metadata from paper header if Gemini missed it
    for f in ("year","shift","exam_date"):
        if not q.get(f) and meta.get(f): q[f] = meta[f]

    # Fix escaped newlines
    for f in ("question","solution"):
        if q.get(f): q[f] = q[f].replace('\\n', '\n')
    if isinstance(q.get("options"), list):
        q["options"] = [o.replace('\\n', '\n') for o in q["options"]]

    q["q_type"] = str(q.get("q_type","MCQ")).strip().upper()
    if q["q_type"] not in _VALID_Q: q["q_type"] = "MCQ"

    raw_s = str(q.get("subject","")).strip().upper()
    q["subject"] = _SUBJ_ALIASES.get(raw_s, raw_s if raw_s in _VALID_S else "")
    if not q["subject"]: return None  # drop if no valid subject

    # Section — strict
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
    m   = re.fullmatch(r'\(\s*(.+?)\s*\)', ans)
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

    q["q_images"]  = merge(q.get("q_images",[]), imgs(q["question"]),
                           *[imgs(o) for o in q["options"]])
    q["sol_images"]= merge(q.get("sol_images",[]), imgs(q.get("solution","")))
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
    Parallel chunk-based parsing:
    1. Split tex into overlapping chunks (~20k chars each)
    2. Call all chunks in parallel (max 4 at a time)
    3. Merge + deduplicate by (subject, number)
    """
    if not _GEMINI_AVAILABLE:
        logger.error("[llm_parser] google-genai not installed")
        return []

    key = api_key or os.environ.get("GEMINI_API_KEY","") or os.environ.get("GOOGLE_API_KEY","")
    if not key:
        logger.error("[llm_parser] No GEMINI_API_KEY")
        return []

    t0  = time.time()
    tex = _normalise_image_refs(tex)
    meta = _extract_meta_from_latex(tex)
    logger.info(f"[llm_parser] tex={len(tex):,}chars meta={meta}")

    # Taxonomy
    taxonomy = dict(_TAXONOMY)
    if pool:
        try:
            from services.llm_tagger import _get_db_taxonomy
            for s in ("PHYSICS","CHEMISTRY","MATHEMATICS","BIOLOGY"):
                db = await _get_db_taxonomy(pool, s)
                if db: taxonomy[s] = {**taxonomy.get(s,{}), **db}
        except Exception as e:
            logger.warning(f"[llm_parser] DB taxonomy: {e}")
    taxonomy_text = format_taxonomy_for_prompt(taxonomy,
        list(taxonomy.keys()))

    # Split into chunks
    chunks = _split_into_chunks(tex, CHUNK_CHARS, CHUNK_OVERLAP)

    # Build prompts for each chunk
    def make_prompt(chunk: str, chunk_idx: int) -> str:
        return f"""{PARSER_SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLE}

══════════════════════════════════════════════════════
PARSE CHUNK {chunk_idx+1}/{len(chunks)}
══════════════════════════════════════════════════════
Paper metadata (use if not found in chunk):
  Exam: {meta["exam_type"]} | Year: {meta["year"]} | Date: {meta["exam_date"]} | Shift: {meta["shift"]}

TAXONOMY (use ONLY these for chapter_name / topic_name):
{taxonomy_text}

INSTRUCTIONS:
- Extract ALL questions present in this chunk
- Detect subject from headings (PHYSICS / CHEMISTRY / MATHEMATICS / BIOLOGY)
- If chunk has no questions, return []
- Return ONLY a valid JSON array

CHUNK:
---BEGIN---
{chunk}
---END---"""

    # Parallel execution using ThreadPoolExecutor
    # Each chunk gets its own thread — truly parallel Gemini calls
    loop = asyncio.get_running_loop()

    from concurrent.futures import ThreadPoolExecutor

    def call_chunk(idx: int, chunk: str) -> list:
        prompt = make_prompt(chunk, idx)
        logger.info(f"[llm_parser] Chunk {idx+1}/{len(chunks)} ({len(chunk):,} chars) starting")
        try:
            result = _call_gemini_sync(key, prompt, PARSE_MODEL)
            logger.info(f"[llm_parser] Chunk {idx+1}/{len(chunks)} → {len(result)} questions")
            return result
        except Exception as e:
            logger.error(f"[llm_parser] Chunk {idx+1} failed: {e}")
            return []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [
            loop.run_in_executor(executor, call_chunk, i, chunk)
            for i, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*futures)

    # Flatten all results
    all_raw = [q for chunk_qs in results for q in chunk_qs]
    logger.info(f"[llm_parser] Total raw: {len(all_raw)} questions from {len(chunks)} chunks")

    # Validate
    validated = []
    for q in all_raw:
        fixed = _validate(dict(q), meta)
        if fixed: validated.append(fixed)

    # Deduplicate by (subject, number) — keep the one with more content
    seen: dict[tuple, dict] = {}
    for q in validated:
        k = (q["subject"], q["number"])
        if k not in seen:
            seen[k] = q
        else:
            # Keep whichever has longer question+solution text
            existing = seen[k]
            if (len(q.get("solution","")) + len(q.get("question",""))) > \
               (len(existing.get("solution","")) + len(existing.get("question",""))):
                seen[k] = q

    final = sorted(seen.values(), key=lambda q: (
        ["PHYSICS","CHEMISTRY","MATHEMATICS","BIOLOGY"].index(q["subject"])
        if q["subject"] in ["PHYSICS","CHEMISTRY","MATHEMATICS","BIOLOGY"] else 99,
        q["number"]
    ))

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"[llm_parser] DONE {elapsed:.1f}s — {len(final)} questions")
    for s in ["PHYSICS","CHEMISTRY","MATHEMATICS","BIOLOGY"]:
        qs = [q for q in final if q.get("subject") == s]
        if not qs: continue
        a = len([q for q in qs if q["section"]=="SECTION-A"])
        b = len([q for q in qs if q["section"]=="SECTION-B"])
        nums    = sorted(q["number"] for q in qs)
        missing = [n for n in range(nums[0], nums[-1]+1) if n not in nums]
        flag    = "✓" if not missing else f"⚠ MISSING:{missing}"
        logger.info(f"[llm_parser]   {s:12s} | {len(qs):>3} total | A={a:>2} B={b:>2} | {flag}")
    logger.info("=" * 60)

    return final


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