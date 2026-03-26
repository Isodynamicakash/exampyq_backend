"""
services/llm_parser.py
======================
Gemini 3 Flash — single call, full paper, 65k output tokens.
Gemini decides how many questions exist (no hardcoded count).
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

# ── Model ─────────────────────────────────────────────────────────────────────
PARSE_MODEL = "gemini-3-flash-preview"

# ── Few-shot ──────────────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLE = r'''
EXAMPLE INPUT:
\section*{26$^{th}$ Feb. 2021 | Shift - 1 PHYSICS}
\section*{SECTION - A}
\begin{enumerate}
  \item If $\lambda_{1}$ and $\lambda_{2}$ are wavelengths of Lyman and Paschen series, $\lambda_{1}:\lambda_{2}$ is:\\
  (1) $1:3$ \quad (2) $1:9$ \quad (3) $7:135$ \quad (4) $7:108$
\end{enumerate}
\section*{Sol. (3)}
$\frac{1}{\lambda_1} = R\left(1-\frac{1}{16}\right)$

\begin{enumerate}
  \setcounter{enumi}{1}
  \item \includegraphics[max width=\textwidth]{img-02_239}\\
  (1) $\frac{\theta_1 R_2+\theta_2 R_1}{R_1+R_2}$ \quad (2) $\frac{\theta_1 R_2-\theta_2 R_1}{R_2-R_1}$ \quad (3) opt3 \quad (4) opt4
\end{enumerate}
\section*{Sol. (1)}

\section*{SECTION - B}
\begin{enumerate}
  \setcounter{enumi}{20}
  \item Tension is $x \times 10^{-2}$ N. $x$ = \_\_\_\_.
\end{enumerate}
Sol. 1215

EXAMPLE OUTPUT:
[
  {
    "number": 1, "q_type": "MCQ", "subject": "PHYSICS", "section": "SECTION-A",
    "year": "2021", "shift": "Morning", "exam_date": "2021-02-26",
    "question": "If $\\lambda_{1}$ and $\\lambda_{2}$ are wavelengths of Lyman and Paschen series, $\\lambda_{1}:\\lambda_{2}$ is:",
    "options": ["$1:3$", "$1:9$", "$7:135$", "$7:108$"],
    "answer": "3",
    "solution": "$\\frac{1}{\\lambda_1} = R\\left(1-\\frac{1}{16}\\right)$",
    "chapter_name": "Modern Physics", "topic_name": "Hydrogen Spectrum",
    "difficulty": "medium", "q_images": [], "sol_images": [],
    "marks_correct": 4, "marks_wrong": -1
  },
  {
    "number": 2, "q_type": "MCQ", "subject": "PHYSICS", "section": "SECTION-A",
    "year": "2021", "shift": "Morning", "exam_date": "2021-02-26",
    "question": "[IMAGE:img-02_239]",
    "options": ["$\\frac{\\theta_1 R_2+\\theta_2 R_1}{R_1+R_2}$", "$\\frac{\\theta_1 R_2-\\theta_2 R_1}{R_2-R_1}$", "opt3", "opt4"],
    "answer": "1", "solution": "",
    "chapter_name": "Heat Transfer", "topic_name": "Thermal Resistance",
    "difficulty": "medium", "q_images": ["img-02_239"], "sol_images": [],
    "marks_correct": 4, "marks_wrong": -1
  },
  {
    "number": 21, "q_type": "NUMERICAL", "subject": "PHYSICS", "section": "SECTION-B",
    "year": "2021", "shift": "Morning", "exam_date": "2021-02-26",
    "question": "Tension is $x \\times 10^{-2}$ N. $x$ = ____.",
    "options": [], "answer": "1215", "solution": "",
    "chapter_name": "Waves", "topic_name": "Wave Speed",
    "difficulty": "medium", "q_images": [], "sol_images": [],
    "marks_correct": 4, "marks_wrong": 0
  }
]

RULES:
1. Return ONLY a JSON array — no markdown, no explanation
2. options[]: strip "(1)"/"(A)" prefix, keep only text/math
3. answer: "1"/"2"/"3"/"4" for MCQ (option number), numeric string for NUMERICAL
4. \includegraphics{img-xyz} → "[IMAGE:img-xyz]" in question + "img-xyz" in q_images[]
5. \setcounter{enumi}{N} → next \item is question N+1
6. Answer can appear in ANY format — detect intelligently:
   - \section*{Sol. (3)} or Sol.(3) → answer="3"
   - \textbf{Ans.} (B) or Ans:(B) → answer="2"  (A=1,B=2,C=3,D=4)
   - Numeric: Sol. 42 → answer="42"
   - Answer key at end of paper: Q1.(C) → answer="3"
7. Solution text may be right after question OR at end of paper — find it either way
8. SECTION-A=MCQ marks_wrong=-1, SECTION-B=NUMERICAL options=[] marks_wrong=0
9. Extract year/shift/date from headings
10. Extract ALL questions from ALL subjects — do not stop early
11. Works for ANY exam format: JEE Main, JEE Advanced, NEET, CUET, coaching sheets
12. marks_correct and marks_wrong — use your knowledge of the exam year:
    JEE Main 2021+: MCQ=+4/-1, NUMERICAL=+4/0
    JEE Main 2017-20: MCQ=+4/-1
    JEE Advanced: varies by section — detect from paper or use +4/-1 as default
    NEET: +4/-1 for all
13. difficulty — judge each question independently on its own merit:
    easy:
      - Single concept, direct formula, one step
      - Definition/factual ("which statement is correct about...")
      - Standard textbook result, no manipulation needed
    medium:
      - 2-3 steps, requires understanding + some calculation
      - Known concept but needs careful application
    hard:
      - Multi-concept, non-obvious approach needed
      - Long calculation with multiple steps
      - Tricky or deceptive framing
      - Requires connecting 2+ topics
    Judge each question independently — do NOT default to medium when unsure.
    If a question is clearly one-step → easy. If it has a trick → hard.
14. chapter_name and topic_name — use your knowledge of the subject syllabus
    If taxonomy provided, pick closest match
    If not in taxonomy, use standard chapter names (e.g. "Kinematics", "Electrochemistry")
15. section field — detect intelligently:
    "SECTION - A" / "SECTION A" / "Part A" → SECTION-A (MCQ)
    "SECTION - B" / "SECTION B" / "Part B" / "Integer type" → SECTION-B (NUMERICAL)
    JEE Advanced may have Section 1, Section 2, Section 3 — map to MCQ or NUMERICAL based on instructions
16. q_type — detect from instructions in paper:
    "Only one correct" → MCQ
    "One or more correct" → MSQ
    "Integer answer" / "Numerical value" → NUMERICAL
17. shift field — detect from heading:
    "Shift 1" / "Morning" / "Session 1" → Morning
    "Shift 2" / "Evening" / "Session 2" → Evening
    If not found, leave empty string
'''.strip()

# ── Metadata ──────────────────────────────────────────────────────────────────
_MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
    'sep':9,'oct':10,'nov':11,'dec':12,
}

def _extract_meta(tex: str) -> dict:
    _mp = (r'(january|february|march|april|may|june|july|august|'
           r'september|october|november|december|jan|feb|mar|apr|'
           r'jun|jul|aug|sep|oct|nov|dec)')
    m = re.search(r'\\title\s*\{([^}]+)\}', tex)
    combined = (m.group(1) if m else "") + " " + tex[:3000]

    exam_date = year = shift = ""
    dm = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', combined)
    if dm: exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
    else:
        dm = re.search(r'\b(\d{2})/(\d{2})/(20\d{2})\b', combined)
        if dm: exam_date = f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}"
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

    return {"exam_date": exam_date, "year": year, "shift": shift, "exam_type": exam_type}


def _normalise_image_refs(tex: str) -> str:
    def _rep(m):
        return f"[IMAGE:{m.group(1).strip().split('/')[-1].split(chr(92))[-1]}]"
    return re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', _rep, tex)


# ── Gemini call ───────────────────────────────────────────────────────────────
def _call_gemini_sync(api_key: str, prompt: str, model: str = PARSE_MODEL) -> str:
    """Single Gemini call, returns raw text. Retries on 503."""
    import time as _time
    for attempt in range(3):
        try:
            logger.info(f"[llm_parser] model={model} attempt={attempt+1}")
            if not _USE_OLD_SDK:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=65536,
                    )
                )
                text = response.text or ""
                logger.info(f"[llm_parser] Response: {len(text):,} chars")
                return text
            else:
                genai_old.configure(api_key=api_key)
                r = genai_old.GenerativeModel(model_name=model)
                resp = r.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 65536}
                )
                return resp.text or ""
        except Exception as e:
            err = str(e)
            logger.warning(f"[llm_parser] attempt={attempt+1} error: {err[:150]}")
            if any(x in err for x in ["503","UNAVAILABLE","429","RESOURCE_EXHAUSTED"]):
                if attempt < 2:
                    delay = [15, 45][attempt]
                    logger.info(f"[llm_parser] Retrying in {delay}s...")
                    _time.sleep(delay)
                    continue
            raise
    return ""


# ── JSON extraction ───────────────────────────────────────────────────────────
def _fix_latex_escapes(raw: str) -> str:
    """Fix invalid LaTeX escape sequences in JSON strings.
    Gemini sometimes outputs \lambda instead of \\lambda."""
    result = []
    i = 0
    in_str = False
    prev_bs = False
    valid_esc = set('"' + "/" + "b" + "f" + "n" + "r" + "t" + "u" + "\\")
    while i < len(raw):
        ch = raw[i]
        if prev_bs:
            if ch in valid_esc:
                result.append(ch)
            else:
                # Insert extra backslash before this char
                result.insert(len(result) - 1, "\\")
                result.append(ch)
            prev_bs = False
        elif in_str and ch == "\\":
            result.append(ch)
            prev_bs = True
        elif ch == '"':
            in_str = not in_str
            result.append(ch)
            prev_bs = False
        else:
            result.append(ch)
            prev_bs = False
        i += 1
    return "".join(result)


def _parse_json(raw: str) -> list:
    if not raw: return []
    # Strip markdown fences
    raw = re.sub(r"```json", "", raw, flags=re.I)
    raw = re.sub(r"```", "", raw)
    raw = raw.strip()
    # Find first '['
    start = raw.find("[")
    if start == -1: return []
    raw = raw[start:]

    # Fix invalid LaTeX escape sequences before parsing
    raw = _fix_latex_escapes(raw)

    # Try progressively trimming trailing chars to find valid JSON
    end = raw.rfind("]")
    while end > 0:
        try:
            r = json.loads(raw[:end+1])
            if isinstance(r, list) and r: return r
        except json.JSONDecodeError as je:
            # Jump to error position and find last '}'
            if je.pos and je.pos > 100:
                trunc = raw.rfind("}", 0, je.pos)
                if trunc > 0:
                    try:
                        r = json.loads(raw[:trunc+1] + "]")
                        if isinstance(r, list) and r:
                            logger.warning(f"[llm_parser] Recovered {len(r)} questions (truncated at pos {je.pos})")
                            return r
                    except: pass
            break
        end = raw.rfind("]", 0, end)

    # Depth tracking last resort
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
                logger.warning(f"[llm_parser] Recovered {len(r)} via depth tracking")
                return r
        except: pass

    logger.error(f"[llm_parser] JSON failed. Sample: {raw[:400]}")
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
_SUBJ    = {
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

    for f in ("year","shift","exam_date"):
        if not q.get(f) and meta.get(f): q[f] = meta[f]

    for f in ("question","solution"):
        if q.get(f): q[f] = q[f].replace('\\n','\n')
    if isinstance(q.get("options"),list):
        q["options"] = [o.replace('\\n','\n') for o in q["options"]]

    q["q_type"] = str(q.get("q_type","MCQ")).strip().upper()
    if q["q_type"] not in _VALID_Q: q["q_type"] = "MCQ"

    raw_s = str(q.get("subject","")).strip().upper()
    q["subject"] = _SUBJ.get(raw_s, raw_s if raw_s in _VALID_S else "")
    if not q["subject"]: return None

    sec = str(q.get("section","")).strip().upper().replace(" ","").replace("-","")
    if sec == "SECTIONB":
        q["section"] = "SECTION-B"
        if q["q_type"] == "MCQ": q["q_type"] = "NUMERICAL"
    else:
        q["section"] = "SECTION-A"

    opts = q.get("options",[])
    if not isinstance(opts,list): opts = []
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
        seen=set(); r=[]
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

    if not _GEMINI_AVAILABLE:
        logger.error("[llm_parser] google-genai not installed")
        return []

    key = api_key or os.environ.get("GEMINI_API_KEY","") or os.environ.get("GOOGLE_API_KEY","")
    if not key:
        logger.error("[llm_parser] No GEMINI_API_KEY")
        return []

    t0  = time.time()
    tex = _normalise_image_refs(tex)
    meta = _extract_meta(tex)
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

    taxonomy_text = format_taxonomy_for_prompt(taxonomy, list(taxonomy.keys()))

    prompt = f"""{PARSER_SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLE}

══════════════════════════════════════════════════════
PARSE THIS EXAM PAPER
══════════════════════════════════════════════════════
Exam: {meta["exam_type"]} | Year: {meta["year"]} | Date: {meta["exam_date"]} | Shift: {meta["shift"]}

TAXONOMY (use ONLY these for chapter_name / topic_name):
{taxonomy_text}

INSTRUCTIONS:
- Extract ALL questions from ALL subjects
- Count questions yourself — do not assume a fixed number
- Return a single valid JSON array
- Start with [ and end with ]

PAPER:
---BEGIN---
{tex}
---END---"""

    loop = asyncio.get_running_loop()
    logger.info(f"[llm_parser] Calling {PARSE_MODEL} — single call, full paper")

    try:
        raw = await loop.run_in_executor(None, _call_gemini_sync, key, prompt, PARSE_MODEL)
    except Exception as e:
        logger.error(f"[llm_parser] Call failed: {e}")
        return []

    questions = _parse_json(raw)
    logger.info(f"[llm_parser] Extracted {len(questions)} raw questions")

    # Validate
    validated = []
    for q in questions:
        fixed = _validate(dict(q), meta)
        if fixed: validated.append(fixed)

    # Dedup by (subject, number)
    seen = {}
    for q in validated:
        k = (q["subject"], q["number"])
        if k not in seen: seen[k] = q

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
        nums = sorted(q["number"] for q in qs)
        missing = [n for n in range(nums[0], nums[-1]+1) if n not in nums]
        flag = "✓" if not missing else f"⚠ MISSING:{missing}"
        logger.info(f"[llm_parser]   {s:12s} | {len(qs):>3} total A={a} B={b} | {flag}")
    logger.info("=" * 60)
    return final


def parse_latex_sync(tex: str, subject_hint: str = "", api_key: str = "") -> list[dict]:
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            parse_latex_with_llm(tex, subject_hint, api_key), loop)
        return future.result(timeout=300)
    except RuntimeError:
        return asyncio.run(parse_latex_with_llm(tex, subject_hint, api_key))
    except Exception as e:
        logger.error(f"[llm_parser] sync failed: {e}")
        return []