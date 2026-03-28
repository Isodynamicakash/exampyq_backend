"""
services/llm_parser.py
======================
LaTeX parser - MARKDOWN OUTPUT ENGINE (v3.0)

WHY MARKDOWN INSTEAD OF JSON:
  - JSON requires every backslash to be doubled (\\frac, \\alpha ...).
    The LLM frequently forgets this, producing invalid JSON.
  - Markdown has NO escaping rules for LaTeX.  \frac{a}{b} is just text.
  - A simple regex post-processor splits the fixed-format blocks into dicts.
  - Image extraction logic is ported verbatim from parser.py.

PIPELINE:
  clean LaTeX
  -> detect exam type  (JEE_MAIN / JEE_ADVANCED / NEET)
  -> build prompt      (instructs LLM to output fixed Markdown blocks)
  -> single API call   -> raw Markdown text
  -> _parse_markdown_blocks()   -> list[dict]  (regex, no JSON)
  -> _postprocess_images()      -> replace \includegraphics with [IMAGE:id]
  -> _filter_no_answer()        -> drop questions with empty answer  (parser.py parity)
  -> _add_marks()               -> exam-type-aware marking scheme
  -> _sort_questions()
  -> return
"""

import os
import re
import anthropic

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

HAIKU_MODEL = "claude-haiku-4-5"
MAX_TOKENS  = 64000

# ══════════════════════════════════════════════════════════
# REGEX — Markdown post-processor
# ══════════════════════════════════════════════════════════

# Matches the header line of every question block:  ### Q42
RE_MD_QBLOCK = re.compile(r'^###\s+Q(\d+)\s*$', re.MULTILINE)

# Matches a field tag at the start of a line:  **QUESTION:** rest...
# group(1) = field name,  group(2) = rest of line (may be empty)
RE_MD_FIELD = re.compile(
    r'^\*\*('
    r'NUMBER|TYPE|SUBJECT|CHAPTER|TOPIC|DIFFICULTY'
    r'|ANSWER|QUESTION|OPTION1|OPTION2|OPTION3|OPTION4|SOLUTION'
    r'):\*\*[ \t]?(.*)',
    re.MULTILINE,
)

# ══════════════════════════════════════════════════════════
# REGEX — Image extraction  (identical to parser.py)
# ══════════════════════════════════════════════════════════

RE_INCLUDEGFX  = re.compile(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}')
RE_PLACEHOLDER = re.compile(r'\[IMAGE:([^\]]+)\]')


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def _detect_exam_type(tex: str) -> str:
    t = tex.lower()
    if "jee main"     in t or "jee-main"     in t: return "JEE_MAIN"
    if "jee advanced" in t or "jee-advanced" in t: return "JEE_ADVANCED"
    if "neet"         in t:                         return "NEET"
    if any(w in t for w in ("biology", "botany", "zoology")): return "NEET"
    if all(w in t for w in ("physics", "chemistry", "math")): return "JEE_MAIN"
    return "JEE_MAIN"


def _clean_latex(tex: str) -> str:
    s = tex.find(r'\begin{document}')
    if s != -1: tex = tex[s:]
    e = tex.find(r'\end{document}')
    if e != -1: tex = tex[:e]
    return tex.strip()

#hh
def _canon_subject(raw: str) -> str:
    s = raw.strip().upper()
    if s in ("MATHS", "MATH"): return "MATHEMATICS"
    return s


def _unique(lst: list) -> list:
    seen = set(); out = []
    for x in lst:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


# ══════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════

# Concrete examples embedded in the prompt so the LLM sees the exact
# format it must produce — including verbatim \includegraphics lines.
_EXAMPLE_MCQ = (
    "### Q1\n"
    "**NUMBER:** 1\n"
    "**TYPE:** MCQ\n"
    "**SUBJECT:** PHYSICS\n"
    "**CHAPTER:** Laws of Motion\n"
    "**TOPIC:** Newton's Second Law\n"
    "**DIFFICULTY:** medium\n"
    "**QUESTION:** A block of mass $m$ is on a frictionless surface. Force $F$ is applied.\n"
    r"\includegraphics[max width=\textwidth]{diagram_001.png}" + "\n"
    "**OPTION1:** $\\frac{F}{2m}$\n"
    "**OPTION2:** $\\frac{F}{m}$\n"
    "**OPTION3:** $Fm$\n"
    "**OPTION4:** $\\frac{2F}{m}$\n"
    "**ANSWER:** 2\n"
    "**SOLUTION:** Newton's second law: $F = ma \\Rightarrow a = \\frac{F}{m}$.\n"
    r"\includegraphics[max width=\textwidth]{sol_001.png}"
)

_EXAMPLE_MSQ = (
    "### Q11\n"
    "**NUMBER:** 11\n"
    "**TYPE:** MSQ\n"
    "**SUBJECT:** PHYSICS\n"
    "**CHAPTER:** Capacitance\n"
    "**TOPIC:** Charging and Discharging\n"
    "**DIFFICULTY:** hard\n"
    "**QUESTION:** In the circuit shown, after $S_3$ is pressed:\n"
    r"\includegraphics[max width=\textwidth]{circuit_011.png}" + "\n"
    "**OPTION1:** charge on upper plate of $C_1$ is $2CV_0$\n"
    "**OPTION2:** charge on upper plate of $C_1$ is $CV_0$\n"
    "**OPTION3:** charge on upper plate of $C_1$ is $0$\n"
    "**OPTION4:** charge on upper plate of $C_2$ is $-CV_0$\n"
    "**ANSWER:** 2,4\n"
    "**SOLUTION:** After $S_1$: $C_1$ charged to $2CV_0$. After $S_2$: both $CV_0$. After $S_3$: $C_2$ upper plate $-CV_0$."
)

_EXAMPLE_NUMERICAL = (
    "### Q16\n"
    "**NUMBER:** 16\n"
    "**TYPE:** NUMERICAL\n"
    "**SUBJECT:** PHYSICS\n"
    "**CHAPTER:** Work, Energy and Power\n"
    "**TOPIC:** Work-Energy Theorem\n"
    "**DIFFICULTY:** hard\n"
    "**QUESTION:** A particle of mass 0.2 kg starts from rest under constant power 0.5 W. Find speed after 5 s.\n"
    "**OPTION1:** \n"
    "**OPTION2:** \n"
    "**OPTION3:** \n"
    "**OPTION4:** \n"
    "**ANSWER:** 5\n"
    "**SOLUTION:** $W = Pt = 0.5 \\times 5 = 2.5$ J $= \\frac{1}{2}mv^2 \\Rightarrow v = 5$ m/s"
)


def _build_prompt(tex: str, exam_type: str) -> str:
    if exam_type in ("JEE_MAIN", "JEE_ADVANCED"):
        subject_rule = (
            "SUBJECT RULE - THIS IS JEE (NOT NEET)\n"
            "Valid subjects: PHYSICS, CHEMISTRY, MATHEMATICS\n"
            "BIOLOGY does not exist in JEE. Biology-looking topics go under CHEMISTRY."
        )
    else:
        subject_rule = (
            "SUBJECT RULE - THIS IS NEET\n"
            "Valid subjects: PHYSICS, CHEMISTRY, BIOLOGY"
        )

    return (
        "You are a LaTeX extractor for competitive exam papers.\n"
        "Your ONLY job: copy every question from the LaTeX source into the fixed Markdown format shown below.\n"
        "\n"
        "==============================================\n"
        "OUTPUT FORMAT - FOLLOW EXACTLY, NO EXCEPTIONS\n"
        "==============================================\n"
        "One block per question:\n"
        "\n"
        "### Q<number>\n"
        "**NUMBER:** <integer>\n"
        "**TYPE:** <MCQ|MSQ|NUMERICAL>\n"
        "**SUBJECT:** <PHYSICS|CHEMISTRY|MATHEMATICS|BIOLOGY>\n"
        "**CHAPTER:** <NCERT chapter name, or leave blank>\n"
        "**TOPIC:** <specific topic, or leave blank>\n"
        "**DIFFICULTY:** <easy|medium|hard>\n"
        "**QUESTION:** <question text - copy LaTeX verbatim>\n"
        "**OPTION1:** <option A text, blank for NUMERICAL>\n"
        "**OPTION2:** <option B text, blank for NUMERICAL>\n"
        "**OPTION3:** <option C text, blank for NUMERICAL>\n"
        "**OPTION4:** <option D text, blank for NUMERICAL>\n"
        "**ANSWER:** <see rules>\n"
        "**SOLUTION:** <solution text - copy LaTeX verbatim>\n"
        "\n"
        "EXAMPLES\n"
        "--------\n"
        + _EXAMPLE_MCQ + "\n\n"
        + _EXAMPLE_MSQ + "\n\n"
        + _EXAMPLE_NUMERICAL + "\n"
        "\n"
        "==============================================\n"
        "RULES\n"
        "==============================================\n"
        "\n"
        "LATEX VERBATIM RULE (MOST IMPORTANT):\n"
        "  Copy ALL LaTeX character-for-character. Do NOT modify, skip, or summarise anything.\n"
        r"  This includes \frac{}{}, $math$, \vec{}, \hat{}, \sqrt{}, \sum, \int, etc." + "\n"
        "  ESPECIALLY this:\n"
        r"    \includegraphics[max width=\textwidth]{filename}" + "\n"
        "  If an \\includegraphics line appears in the source, copy it EXACTLY into\n"
        "  the QUESTION field or SOLUTION field where it appears. NEVER skip it.\n"
        "\n"
        "FIELD RULES:\n"
        "  - ### Q<n> and every **FIELD:** tag must each be on their own line.\n"
        "  - Field value starts on the same line as the tag; may span multiple lines.\n"
        "  - A field ends when the next **FIELD:** tag or ### Q<n> line is seen.\n"
        "  - QUESTION: only the question statement. Do NOT include options in QUESTION.\n"
        "  - OPTION1-4: only the answer-choice text, no (A)/(B) labels.\n"
        "  - OPTION1-4: leave blank (empty) for NUMERICAL questions.\n"
        "\n"
        "ANSWER FORMAT:\n"
        "  MCQ       -> single digit:    1  or  2  or  3  or  4   (A=1 B=2 C=3 D=4)\n"
        "  MSQ       -> comma list:      1,3  or  2,4  or  1,2,4  etc.\n"
        "  NUMERICAL -> numeric value:   5   or  2.5  or  100\n"
        "\n"
        "TYPE DETECTION:\n"
        "  MCQ       = exactly one correct option (4 options present)\n"
        "  MSQ       = one or more correct options (4 options present)\n"
        "  NUMERICAL = no options, direct numeric answer\n"
        "\n"
        "OUTPUT DISCIPLINE:\n"
        "  - Output ONLY the blocks. No preamble. No commentary. No markdown fences.\n"
        "  - Extract EVERY question. Skip nothing.\n"
        "  - Number sequentially.\n"
        "\n"
        + subject_rule + "\n"
        "\n"
        "==============================================\n"
        "LaTeX SOURCE:\n"
        "==============================================\n"
        + tex
    )


# ══════════════════════════════════════════════════════════
# API CALL
# ══════════════════════════════════════════════════════════

def _call_api(prompt: str, api_key: str, exam_type: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    print(f"[LLM Parser] Calling API ({exam_type}, Markdown engine v3.0)...", flush=True)

    message = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": (
                f"You are a precise LaTeX extractor for {exam_type} papers (Markdown Engine v3.0). "
                "Output ONLY the fixed Markdown blocks. "
                "Copy every \\includegraphics command verbatim into the correct field. "
                "Do NOT skip images. Do NOT escape LaTeX."
            ),
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text.strip()
    print(f"[LLM Parser] Raw response: {len(text)} chars", flush=True)
    return text


# ══════════════════════════════════════════════════════════
# MARKDOWN -> DICT  (regex-based, no JSON)
# ══════════════════════════════════════════════════════════

def _extract_fields(block: str) -> dict:
    """
    Given the text of one ### Q<n> block, return a dict of field->value.
    Multi-line values are supported: a value runs from its **FIELD:** tag
    until the next **FIELD:** tag (or end of block).
    """
    fields      = {}
    current_key = None
    current_val = []

    def _commit():
        if current_key is not None:
            fields[current_key] = "\n".join(current_val).strip()

    for line in block.splitlines():
        m = RE_MD_FIELD.match(line)
        if m:
            _commit()
            current_key = m.group(1)
            current_val = [m.group(2)]  # rest of the tag line (often blank)
        else:
            if current_key is not None:
                current_val.append(line)

    _commit()
    return fields


def _parse_markdown_blocks(md: str) -> list[dict]:
    """
    Split LLM Markdown output on ### Q<n> headers.
    Parse each block with _extract_fields() and build raw question dicts.
    """
    boundaries = [
        (m.start(), int(m.group(1)))
        for m in RE_MD_QBLOCK.finditer(md)
    ]
    if not boundaries:
        print("[LLM Parser] No ### Q<n> headers found in response.", flush=True)
        return []

    questions = []
    for i, (start, qnum) in enumerate(boundaries):
        end   = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(md)
        block = md[start:end]

        fields = _extract_fields(block)
        if not fields:
            continue

        try:
            number = int(fields.get("NUMBER", qnum))
        except (ValueError, TypeError):
            number = qnum

        q_type = fields.get("TYPE", "MCQ").strip().upper()
        if q_type not in ("MCQ", "MSQ", "NUMERICAL"):
            q_type = "MCQ"

        options = [
            fields.get("OPTION1", "").strip(),
            fields.get("OPTION2", "").strip(),
            fields.get("OPTION3", "").strip(),
            fields.get("OPTION4", "").strip(),
        ]

        q = {
            "number":       number,
            "q_type":       q_type,
            "subject":      _canon_subject(fields.get("SUBJECT",    "")),
            "chapter_name": fields.get("CHAPTER",    "").strip(),
            "topic_name":   fields.get("TOPIC",      "").strip(),
            "difficulty":   fields.get("DIFFICULTY", "medium").strip().lower(),
            "question":     fields.get("QUESTION",   "").strip(),
            "options":      options,
            "answer":       fields.get("ANSWER",     "").strip(),
            "solution":     fields.get("SOLUTION",   "").strip(),
            # filled by _postprocess_images()
            "q_images":     [],
            "sol_images":   [],
            # metadata stubs for frontend admin
            "section":      "",
            "year":         "",
            "shift":        "",
            "exam_date":    "",
            "chapter_id":   None,
            "topic":        "",
            "verified":     False,
        }
        questions.append(q)

    print(f"[LLM Parser] Parsed {len(questions)} blocks from Markdown.", flush=True)
    return questions


# ══════════════════════════════════════════════════════════
# IMAGE POST-PROCESSING  (ported verbatim from parser.py)
# ══════════════════════════════════════════════════════════

def _extract_images(text: str):
    r"""
    Find every \includegraphics[...]{filename} in text.
    Replace each occurrence with [IMAGE:basename].
    Return (modified_text, list_of_basename_ids).

    Identical to parser.py's _extract_images().
    """
    ids = []

    def _rep(m):
        img_id = os.path.basename(m.group(1).strip())
        ids.append(img_id)
        return f"[IMAGE:{img_id}]"

    return RE_INCLUDEGFX.sub(_rep, text).strip(), ids


def _postprocess_images(questions: list) -> list:
    """
    For every question dict:
      1. Run _extract_images() on question text, each option, and solution.
      2. q_images   = unique image ids from question + options.
         sol_images = unique image ids from solution.

    Mirrors the image-handling section of parser.py's _postprocess().
    """
    for q in questions:
        # question
        q["question"], qi = _extract_images(q.get("question", ""))

        # options
        cleaned_opts = []
        oi = []
        for opt in q.get("options", []):
            c, o = _extract_images(opt)
            cleaned_opts.append(c)
            oi.extend(o)
        q["options"] = cleaned_opts

        # solution
        q["solution"], si = _extract_images(q.get("solution", ""))

        # collect placeholder ids that may already exist in the text
        # (handles the case where the LLM wrote [IMAGE:x] directly)
        q_ph = RE_PLACEHOLDER.findall(q["question"])
        o_ph = []
        for opt in q["options"]:
            o_ph.extend(RE_PLACEHOLDER.findall(opt))
        s_ph = RE_PLACEHOLDER.findall(q["solution"])

        q["q_images"]   = _unique(q_ph + qi + o_ph + oi)
        q["sol_images"] = _unique(s_ph + si)

    return questions


# ══════════════════════════════════════════════════════════
# ANSWER FILTER  (mirrors parser.py _postprocess())
# ══════════════════════════════════════════════════════════

def _filter_no_answer(questions: list) -> list:
    """
    Drop questions whose answer field is empty.
    Identical to the guard in parser.py:
        if d["answer"].strip(): result.append(d)
    """
    before  = len(questions)
    out     = [q for q in questions if q.get("answer", "").strip()]
    dropped = before - len(out)
    if dropped:
        print(f"[LLM Parser] Dropped {dropped} question(s) with empty answer.", flush=True)
    return out


# ══════════════════════════════════════════════════════════
# MARKS ASSIGNMENT
# ══════════════════════════════════════════════════════════

def _add_marks(questions: list, exam_type: str) -> list:
    for q in questions:
        qt = q.get("q_type", "MCQ")
        if exam_type == "JEE_MAIN":
            q["marks_correct"] = 4
            q["marks_wrong"]   = -1 if qt in ("MCQ", "MSQ") else 0
        elif exam_type == "JEE_ADVANCED":
            if   qt == "MCQ":  q["marks_correct"] = 3;  q["marks_wrong"] = -1
            elif qt == "MSQ":  q["marks_correct"] = 4;  q["marks_wrong"] = -2
            else:              q["marks_correct"] = 3;  q["marks_wrong"] =  0
        else:  # NEET / default
            q["marks_correct"] = 4
            q["marks_wrong"]   = -1
    return questions


def _sort_questions(questions: list) -> list:
    try:
        return sorted(questions, key=lambda q: int(q.get("number", 0) or 0))
    except Exception:
        return questions


# ══════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════

async def parse_latex_with_llm(tex: str, api_key: str = None) -> list[dict]:
    """
    Parse a LaTeX exam paper and return a list of question dicts.

    Args:
        tex:     Full LaTeX source (preamble stripped automatically).
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Returns:
        List of question dicts. Each dict has:
            number, q_type, subject, chapter_name, topic_name, difficulty,
            question, options[4], answer, solution,
            q_images, sol_images,
            marks_correct, marks_wrong,
            section, year, shift, exam_date, chapter_id, topic, verified
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[LLM Parser] No API key provided.", flush=True)
        return []

    tex       = _clean_latex(tex)
    exam_type = _detect_exam_type(tex)
    print(f"[LLM Parser] Exam: {exam_type} | Source: {len(tex)} chars", flush=True)

    prompt    = _build_prompt(tex, exam_type)
    md_text   = _call_api(prompt, api_key, exam_type)

    if not md_text:
        print("[LLM Parser] Empty API response.", flush=True)
        return []

    questions = _parse_markdown_blocks(md_text)
    if not questions:
        print("[LLM Parser] No questions parsed.", flush=True)
        print(f"[LLM Parser] Preview:\n{md_text[:800]}", flush=True)
        return []

    questions = _postprocess_images(questions)   # \includegraphics -> [IMAGE:id]
    questions = _filter_no_answer(questions)      # drop empty-answer questions
    questions = _add_marks(questions, exam_type)  # marks_correct / marks_wrong
    questions = _sort_questions(questions)         # sort by number

    print(f"[LLM Parser] Done: {len(questions)} questions.", flush=True)
    return questions