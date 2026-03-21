r"""
services/parser.py
==================
Parser for MathPix-converted JEE/NEET .tex files.

KEY DESIGN PRINCIPLES
─────────────────────
1. A new question can start in ANY state (IDLE, IN_Q, IN_ANS, IN_SOL).
   When a question-start signal is seen, the current question is always
   flushed first, then a new one begins.

2. Question-start signals:
     a) \\item                    → enumerate-style question
     b) N. text (plain number)  → plain-numbered question
        Three forms recognised:
          "N. text"  — number, dot, space, then text on same line
          "N.\\"    — number, dot, optional backslash (text/image next line)
          "N."       — bare number+dot alone on a line (image on next line)
        The third form (bare "N.") is the fix for MathPix output where a
        question image immediately follows the number on the next line.

3. last_committed_num resets to 0 at each new subject section header.
   This is essential because Chemistry/Math restart from Q1.

4. \\setcounter{enumi}{N} is tracked as pending_setcounter (O(1), not
   O(n^2) index scan). When \\item follows: q_num = N+1.
   When \\item has NO preceding \\setcounter (e.g. Q74 is second \\item in
   same block as Q73): q_num = effective_last + 1.

5. Plain-number detection fires BEFORE any state-based text routing,
   including when state == IN_SOL. This fixes "questions embedded inside
   solution text" — the most common issue with MathPix output where Q7-13
   appear on lines that the old parser treated as solution continuation of Q6.

6. effective_last = max(last_committed_num, current.number if current else 0).
   last_committed_num only updates on flush() (when a question is saved).
   While Q51 is active but not yet flushed, last_committed_num is still 0
   from the section reset. effective_last correctly returns 51, so
   is_next_q(52) = True and plain-numbered Q52 starts correctly.
   Without this: is_next_q(52, last=0) → 52 > 35 → False → Q52 silently lost.

7. is_next_q() gap guard: num <= effective_last + 35. Allows gaps (e.g. Q6 →
   Q14 directly if some questions are in enumerate blocks) while rejecting
   large numbers that appear in math expressions like "100." inside solutions.

8. Sol. answer extraction (FIX):
   In these PDFs MathPix renders the answer as the first token after "Sol."
   on the very same line, e.g.:
       Sol. 2          ← MCQ answer
       Sol. 4          ← MCQ answer
       Sol. 150        ← NUMERICAL integer answer
       Sol. 1.58       ← NUMERICAL decimal answer
       Sol. 9          ← NUMERICAL integer answer
   RE_SOL_ANS matches when the entire rest of the line after "Sol." is ONLY
   an answer token (single digit 1-4, or any integer/decimal).  When matched:
     • current.answer is set (if not already set)
     • state transitions to IN_S  (solution body may follow on later lines)
   When the rest is non-trivial text (e.g. "Sol. Using conservation..."),
   it is treated as before — state→IN_S and the text goes to solution body.
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# REGEX PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

RE_TITLE        = re.compile(r'\\title\{(.+?)\}')
RE_BEGIN_DOC    = re.compile(r'^\s*\\begin\{document\}')
RE_END_DOC      = re.compile(r'^\s*\\end\{document\}')
RE_BEGIN_ENUM   = re.compile(r'^\s*\\begin\{enumerate\}')
RE_END_ENUM     = re.compile(r'^\s*\\end\{enumerate\}')
RE_BEGIN_FIGURE = re.compile(r'^\s*\\begin\{figure\}')
RE_END_FIGURE   = re.compile(r'^\s*\\end\{figure\}')
RE_BEGIN_ITEMIZE = re.compile(r'^\s*\\begin\{itemize\}')
RE_END_ITEMIZE   = re.compile(r'^\s*\\end\{itemize\}')
RE_BEGIN_CENTER  = re.compile(r'^\s*\\begin\{center\}')
RE_END_CENTER    = re.compile(r'^\s*\\end\{center\}')
RE_BEGIN_TABULAR = re.compile(r'^\s*\\begin\{tabular\}')
RE_END_TABULAR   = re.compile(r'^\s*\\end\{tabular\}')
RE_CAPTION      = re.compile(r'\\caption\{(.+?)\}')
RE_SETCOUNTER   = re.compile(r'^\s*\\setcounter\{enumi\}\{(\d+)\}')
RE_ITEM         = re.compile(r'^\s*\\item\s*(.*)')
RE_SECTION      = re.compile(r'^\s*\\section\*\{(.+?)\}')
RE_INCLUDEGFX   = re.compile(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}')
RE_PLACEHOLDER  = re.compile(r'\[IMAGE:([^\]]+)\]')

RE_SUBJECT = re.compile(
    r'(?:PART[-\s]*[A-Za-z]\s*:\s*)?(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|MATH|BIOLOGY)',
    re.IGNORECASE,
)
# Numerical/Integer section detection
RE_NUMERICAL_SEC = re.compile(
    r'SECTION[-\s]?(?:B|2|II)|INTEGER\s*(?:TYPE|ANSWER)?|NUMERICAL\s*(?:VALUE|TYPE|ANSWER)?',
    re.IGNORECASE,
)

RE_PLAIN_Q = re.compile(
    r'^(\d{1,3})\.\s+(\S.*)'       # alt-1: "N. text"
    r'|^(\d{1,3})\.\s*\\\\?\s*$'  # alt-2: "N.\\" or bare with backslash
    r'|^(\d{1,3})\.\s*$'             # alt-3: bare "N." alone (image on next line)
)  # "N. text" OR bare "N." alone on a line (image/text follows)
RE_OPTION  = re.compile(r'^\$?\(([1-4])\)\$?\s*(.*)')
# Matches: Answer (4)  Ans. (4)  Ans (4)  Ans. 280
#           Official Ans. by NTA (C)  Allen Ans. (1)  NTA Ans. (3)
RE_ANSWER  = re.compile(
    r'^(?:(?:Official|Allen|NTA)\s+)?Ans(?:wer)?\.?\s+(?:by\s+NTA\s+)?\((.+?)\)'  # paren form
    r'|^(?:(?:Official|Allen|NTA)\s+)?Ans(?:wer)?\.?\s+(?:by\s+NTA\s+)?(\d+(?:\.\d+)?)\s*$',  # bare number form
    re.IGNORECASE
)
# NTA answer overrides Allen answer when both present
RE_NTA_ANS = re.compile(
    r'^NTA\s+Ans(?:wer)?\.?\s+(?:by\s+NTA\s+)?\((.+?)\)'
    r'|^NTA\s+Ans(?:wer)?\.?\s+(?:by\s+NTA\s+)?(\d+(?:\.\d+)?)\s*$',
    re.IGNORECASE
)
RE_SOL     = re.compile(r'^Sol\.\s*(.*)', re.IGNORECASE)

# ── NEW: detect when the token right after "Sol." IS the answer ──────────────
# Matches lines of the form:
#   "Sol. 2"          → MCQ option 1-4
#   "Sol. 4"
#   "Sol. 150"        → numerical integer
#   "Sol. 1.58"       → numerical decimal
#   "Sol. 9"
#   "Sol. 47 %"       → numerical with % unit (strip unit, keep number)
#   "Sol. 20cc" / "Sol. 20" → keep number part
# Group 1 = the numeric answer token (may be followed by non-numeric unit junk)
# The key constraint: the REST of the line must be ONLY the number (+ optional
# non-alpha unit suffix).  If there is real text after it (words, formulas),
# this regex will NOT match and we fall through to normal solution handling.
RE_SOL_ANS = re.compile(
    r'^Sol\.\s+'                       # "Sol." + mandatory whitespace
    r'(\d+(?:\.\d+)?)'                 # group 1: integer or decimal number
    r'(?:\s*(?:%|cc|cm|m|kg|s|V|J|N|eV|K|Hz|mol|kbar|mN|rpm|nm|mm|pm))?'  # optional unit
    r'\s*$',                           # nothing else on the line
    re.IGNORECASE
)

# ── Patterns that appear INSIDE solution bodies and must go to solution ────────
# These are summary/confirmation lines that some publishers append after working:
#   "Correct Option (3)"
#   "Correct Answer : 1080"
#   "Correct Answer: 280"
#   "Correct Answer (3)"
# They must NEVER be treated as the authoritative answer because:
#   (a) The real answer was already captured from "Ans. (N)" / "Sol. N" earlier.
#   (b) They appear AFTER Sol. (i.e. inside solution text), not before it.
#   (c) For MCQ, picking the number out of "Correct Option (3)" while already
#       in IN_S state would silently overwrite or set a wrong answer.
RE_CORRECT_SOL_NOISE = re.compile(
    r'^Correct\s+(?:Option|Answer)\s*[:(]',
    re.IGNORECASE
)

# ── Alternative format: options as a. b. c. d. (used by some publishers) ─────
# Strictly lowercase to avoid matching uppercase reaction sub-labels
# like "A. Reaction with HCl" that appear inside question bodies.
# Only fired when state == IN_Q (same guard as RE_OPTION).
# Group 1 = letter (a/b/c/d), Group 2 = option text
RE_OPTION_ABCD = re.compile(r'^([abcd])\.\s+(.*)')

# ── Alternative format: "Answer: X" (colon form, used by some publishers) ────
# X can be: letter (a/b/c/d), multi-letter MSQ (a,b / b,d),
#           integer (9, 18), decimal (5.22), negative+unit (-85 kJ/mol),
#           float+unit (0.3675 g).
# Only fired when state != IN_S (same guard as RE_ANSWER).
RE_ANSWER_COLON = re.compile(r'^Answer\s*:\s*(.+?)\s*$', re.IGNORECASE)

# ── Alternative format: "Solution:" (colon form) ──────────────────────────────
# Bold "Solution:" in PDF → \section*{Solution:} in MathPix LaTeX.
# Group 1 = any text on same line after the colon (usually empty).
RE_SOLUTION_COLON = re.compile(r'^Solution\s*:\s*(.*)$', re.IGNORECASE)

# Letter mapping for abcd options and Answer: letter
_ABCD_LETTER = {'a': '1', 'b': '2', 'c': '3', 'd': '4'}

_NOISE_PATS = [
    re.compile(r'^answers?\s*[&and]*\s*solutions?\s*$', re.IGNORECASE),  # "Answers & Solutions" header
    re.compile(r'important\s*instructions?',               re.IGNORECASE),
    re.compile(r'^\(physics',                              re.IGNORECASE),
    re.compile(r'fact\s*based',                            re.IGNORECASE),
    re.compile(r'^sol\.',                                  re.IGNORECASE),
    re.compile(r'correct\s*(option|answer|ans)',           re.IGNORECASE),
    re.compile(r'download.*app',                            re.IGNORECASE),  # Allen app ads
    re.compile(r'scan.*qr|qr.*code',                        re.IGNORECASE),  # QR code ads
    # NOTE: "TEST PAPER WITH SOLUTION" is NOT noise — real questions follow it
]

_MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
    'sep':9,'oct':10,'nov':11,'dec':12,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedQuestion:
    number:     int
    q_type:     str
    subject:    str
    section:    str
    year:       str
    shift:      str
    exam_date:  str
    question:   str
    options:    list = field(default_factory=list)
    answer:     str  = ""
    solution:   str  = ""
    q_images:   list = field(default_factory=list)
    sol_images: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number":        self.number,
            "q_type":        self.q_type,
            "subject":       self.subject,
            "section":       self.section,
            "year":          self.year,
            "shift":         self.shift,
            "exam_date":     self.exam_date,
            "question":      self.question,
            "options":       self.options,
            "answer":        self.answer,
            "solution":      self.solution,
            "q_images":      self.q_images,
            "sol_images":    self.sol_images,
            "chapter_id":    None,
            "topic":         "",
            "difficulty":    "medium",
            "marks_correct": 4,
            "marks_wrong":   -1,
            "verified":      False,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_meta(title: str, body: str = "") -> dict:
    combined = (title + " " + body[:800]).strip()
    exam_date = shift = year = ""
    _mp = (r'(january|february|march|april|may|june|july|august|'
           r'september|october|november|december|jan|feb|mar|apr|'
           r'jun|jul|aug|sep|oct|nov|dec)')
    m = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', combined)
    if m:
        exam_date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    else:
        m = re.search(r'\b(\d{2})/(\d{2})/(20\d{2})\b', combined)
        if m:
            exam_date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        else:
            m = re.search(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_mp}\s+(20\d{{2}})\b', combined, re.I)
            if m:
                mo = _MONTH_MAP.get(m.group(2).lower(), 0)
                if mo:
                    exam_date = f"{m.group(3)}-{mo:02d}-{int(m.group(1)):02d}"
            else:
                m = re.search(rf'\b{_mp}\s+(\d{{1,2}}),?\s+(20\d{{2}})\b', combined, re.I)
                if m:
                    mo = _MONTH_MAP.get(m.group(1).lower(), 0)
                    if mo:
                        exam_date = f"{m.group(3)}-{mo:02d}-{int(m.group(2)):02d}"
    year = exam_date[:4] if exam_date else ""
    if not year:
        m = re.search(r'\b(20\d{2})\b', combined)
        if m:
            year = m.group(1)
    tl = combined.lower()
    if any(x in tl for x in ("morning","shift 1","shift-1","session 1","session-1")):
        shift = "Morning"
    elif any(x in tl for x in ("evening","shift 2","shift-2","session 2","session-2")):
        shift = "Evening"
    return {"year": year, "exam_date": exam_date, "shift": shift}


def _is_noise(val: str) -> bool:
    return any(p.search(val) for p in _NOISE_PATS)


def _tail(s: str) -> str:
    return re.sub(r'(\\\\)+\s*$', '', s).strip()


def _extract_images(text: str):
    ids = []
    def _rep(m):
        img_id = os.path.basename(m.group(1).strip())
        ids.append(img_id)
        return f"[IMAGE:{img_id}]"
    return RE_INCLUDEGFX.sub(_rep, text).strip(), ids


# NTA uses letters A/B/C/D for options — map to 1/2/3/4
_NTA_LETTER = {'A':'1','B':'2','C':'3','D':'4','a':'1','b':'2','c':'3','d':'4'}

def _parse_answer(raw: str) -> str:
    raw = _tail(raw).strip()
    # unwrap outer parens: '(4)' -> '4', '(C)' -> 'C'
    m = re.fullmatch(r'\((.+)\)', raw)
    if m:
        inner = m.group(1).strip()
        if not any(c in inner for c in ('\\', '$', '{')):
            raw = inner
    # NTA single letter: 'C' -> '3'
    if re.fullmatch(r'[A-Da-d]', raw):
        return _NTA_LETTER[raw]
    # NTA multi-letter MSQ: 'A,C' or 'A, C' -> '1,3'
    if re.fullmatch(r'[A-Da-d](?:\s*,\s*[A-Da-d])+', raw):
        return ','.join(_NTA_LETTER[c] for c in re.findall(r'[A-Da-d]', raw))
    return raw


def _unique(lst: list) -> list:
    seen = set(); out = []
    for x in lst:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _parse_answer_colon(raw: str) -> str:
    """Parse answer from 'Answer: X' colon format (used by some publishers).
    Handles:
      "b"          → "2"   (MCQ single letter)
      "a,b"        → "1,2" (MSQ multi-letter)
      "B,D"        → "2,4" (MSQ uppercase)
      "9"          → "9"   (numerical integer)
      "5.22"       → "5.22"(numerical decimal)
      "0.3675 g"   → "0.3675" (numerical + unit)
      "-85 kJ/mol" → "-85"    (negative + unit)
    """
    raw = raw.strip()
    # Single letter a/b/c/d
    if re.fullmatch(r'[a-dA-D]', raw):
        return _ABCD_LETTER[raw.lower()]
    # Multi-letter MSQ: "a,b"  "b,d"  "B,D"  "a, b"
    if re.fullmatch(r'[a-dA-D](?:\s*,\s*[a-dA-D])+', raw):
        return ','.join(_ABCD_LETTER[c.lower()] for c in re.findall(r'[a-dA-D]', raw))
    # Negative or positive number (strip trailing unit text)
    m = re.match(r'^(-?\d+(?:\.\d+)?)', raw)
    if m:
        return m.group(1)
    return raw



# ═══════════════════════════════════════════════════════════════════════════════
# PARSER STATES
# ═══════════════════════════════════════════════════════════════════════════════

class S:
    IDLE = "IDLE"
    IN_Q = "IN_Q"
    IN_A = "IN_A"
    IN_S = "IN_S"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_tex(tex_path: str) -> list:
    with open(tex_path, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]

    title = subject = ""
    subject   = "PHYSICS"
    section   = "SECTION-A"
    year = shift = exam_date = ""
    _body_buf = ""; _body_done = False

    questions: list[ParsedQuestion] = []
    current:   Optional[ParsedQuestion] = None
    state = S.IDLE
    in_doc = False

    # noise-block tracking
    in_noise_block         = False
    last_section_was_noise = False

    # current section type — changes when SECTION-B/INTEGER headers are seen
    current_q_type = "MCQ"   # "MCQ" or "NUMERICAL"

    # enumerate/itemize depth tracking
    # \item is only a question trigger when inside enumerate, not itemize
    enum_depth    = 0   # depth of \begin{enumerate} nesting
    itemize_depth = 0   # depth of \begin{itemize} nesting
    tabular_depth = 0   # depth of \begin{tabular} nesting (nested headers exist)

    # figure state
    in_figure = False; fig_caption = ""; fig_img_id = ""

    # question numbering
    last_committed_num:    int           = 0
    pending_setcounter:    Optional[int] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def flush():
        nonlocal current, state, last_committed_num
        if current is not None:
            current.question = current.question.strip()
            current.solution = current.solution.strip()
            questions.append(current)
            last_committed_num = current.number
        current = None
        state   = S.IDLE

    def start_q(num: int, text: str):
        nonlocal current, state
        flush()
        current = ParsedQuestion(
            number    = num,
            q_type    = current_q_type,
            subject   = subject,
            section   = section,
            year      = year,
            shift     = shift,
            exam_date = exam_date,
            question  = _tail(text),
        )
        state = S.IN_Q

    def append_q(text: str):
        if current and text.strip():
            sep = "\n" if current.question else ""
            current.question += sep + _tail(text)

    def append_sol(text: str):
        if not (current and text.strip()): return
        sep = "\n" if current.solution else ""
        current.solution += sep + _tail(text)

    def set_opt(num: int, text: str):
        if current is None: return
        while len(current.options) < num:
            current.options.append("")
        idx = num - 1
        if not current.options[idx]:
            current.options[idx] = text
        else:
            current.options[idx] += " " + text.strip()

    def is_next_q(num: int, from_setcounter: bool = False) -> bool:
        """
        Accept num as the next question number if:
          - from_setcounter=True: ALWAYS accept (\\setcounter is authoritative)
          - num > effective_last  (strictly forward)
          - num <= effective_last + 35  (gap guard against math false positives)

        effective_last = max(last_committed_num, current.number if current else 0)
        This is critical: last_committed_num only updates on flush(), but Q51 may be
        active (current.number=51) while last_committed_num is still 0 from the section
        reset. Without this, is_next_q(52, last=0) wrongly rejects Q52.
        """
        if num < 1:
            return False
        effective_last = max(last_committed_num,
                             current.number if current is not None else 0)
        if from_setcounter:
            return num > effective_last
        if effective_last == 0 and num == 1:
            return True
        return effective_last < num <= effective_last + 35

    # ── main loop ─────────────────────────────────────────────────────────────

    for ln in lines:
        stripped = ln.strip()

        # document fences
        if RE_BEGIN_DOC.match(ln):
            in_doc = True; continue
        if RE_END_DOC.match(ln):
            flush(); break

        m = RE_TITLE.match(ln)
        if m:
            title = m.group(1)
            meta  = _extract_meta(title)
            year, exam_date, shift = meta["year"], meta["exam_date"], meta["shift"]
            continue

        if not in_doc:
            continue

        # body buffer
        if not _body_done:
            _body_buf += ln + "\n"
            if len(_body_buf) >= 800: _body_done = True
            m2 = _extract_meta(title, _body_buf)
            if m2["exam_date"]: exam_date = m2["exam_date"]
            if m2["shift"]:     shift     = m2["shift"]
            if m2["year"]:      year      = m2["year"]

        if not stripped:
            continue

        # ── figure ────────────────────────────────────────────────────────────
        if RE_BEGIN_FIGURE.match(ln):
            in_figure = True; fig_caption = ""; fig_img_id = ""; continue

        if RE_END_FIGURE.match(ln):
            if in_figure and fig_img_id:
                cm = re.fullmatch(r'\(([1-4])\)', fig_caption.strip())
                if cm and current is not None:
                    set_opt(int(cm.group(1)), f"[IMAGE:{fig_img_id}]")
                elif current is not None:
                    (append_sol if state == S.IN_S else append_q)(f"[IMAGE:{fig_img_id}]")
            in_figure = False; fig_caption = ""; fig_img_id = ""; continue

        if in_figure:
            cm = RE_CAPTION.search(ln)
            if cm: fig_caption = cm.group(1).strip()
            gm = RE_INCLUDEGFX.search(ln)
            if gm: fig_img_id = os.path.basename(gm.group(1).strip())
            continue

        # non-figure environments
        if RE_BEGIN_CENTER.match(ln) or RE_END_CENTER.match(ln):
            continue

        # ── tabular fences ────────────────────────────────────────────────────
        # Track depth for nested tabulars (e.g. multi-line column headers).
        # Inside tabular: content is appended as-is; only \includegraphics is
        # still extracted. All question-detection patterns are suppressed to
        # prevent table rows like "(1) & value \\" from false-matching RE_OPTION.
        if RE_BEGIN_TABULAR.match(ln):
            tabular_depth += 1
            if current is not None:
                (append_sol if state == S.IN_S else append_q)(ln.strip())
            continue
        if RE_END_TABULAR.match(ln):
            tabular_depth = max(0, tabular_depth - 1)
            if current is not None:
                (append_sol if state == S.IN_S else append_q)(r'\end{tabular}')
            continue
        if tabular_depth > 0:
            if current is not None:
                if RE_INCLUDEGFX.search(ln):
                    for raw_id in RE_INCLUDEGFX.findall(ln):
                        img_id = os.path.basename(raw_id.strip())
                        ph = f"[IMAGE:{img_id}]"
                        (append_sol if state == S.IN_S else append_q)(ph)
                    continue
                (append_sol if state == S.IN_S else append_q)(ln.strip())
            continue

        # ── section headers ───────────────────────────────────────────────────
        m = RE_SECTION.match(ln)
        if m:
            sec_text = m.group(1).strip()

            # ── "Correct Option (N)" / "Correct Answer : N" inside solution ──
            # These confirmation lines appear after the working in solution
            # bodies and must be routed to solution, never treated as the
            # authoritative answer.  Check BEFORE RE_ANSWER so "Correct Answer"
            # can never accidentally set current.answer.
            if RE_CORRECT_SOL_NOISE.match(sec_text):
                if current is not None and state == S.IN_S:
                    append_sol(sec_text)
                continue

            # ── Ans. / NTA Ans. — the authoritative answer line ───────────────
            # Only fire when NOT already inside solution (IN_S).  Once Sol. has
            # started, any "Answer (N)" line is part of the worked solution, not
            # a separate answer declaration.
            am = RE_ANSWER.match(sec_text)
            if am and current is not None and state != S.IN_S:
                raw_ans = (am.group(1) or am.group(2) or '').strip()
                nta_m = RE_NTA_ANS.match(sec_text)
                if nta_m:
                    nta_raw = (nta_m.group(1) or nta_m.group(2) or '').strip()
                    current.answer = _parse_answer(nta_raw)
                elif not current.answer:
                    current.answer = _parse_answer(raw_ans)
                state = S.IN_A
                continue

            sm = RE_SOL.match(sec_text)
            if sm:
                last_section_was_noise = False   # ← critical: Sol. ends any noise context
                if current is not None:
                    # ── FIX: check if the "Sol." line carries the answer ──────
                    sol_ans_m = RE_SOL_ANS.match(sec_text)
                    if sol_ans_m:
                        ans_tok = sol_ans_m.group(1).strip()
                        if not current.answer:
                            current.answer = _parse_answer(ans_tok)
                        state = S.IN_S
                    else:
                        state = S.IN_S
                        rest = sm.group(1).strip()
                        if rest:
                            append_sol(rest)
                continue

            # ── "Solution:" colon form (alternative publisher format) ─────────
            # e.g. \section*{Solution:} — bold "Solution:" in PDF
            sc_m = RE_SOLUTION_COLON.match(sec_text)
            if sc_m:
                last_section_was_noise = False
                if current is not None:
                    state = S.IN_S
                    rest = sc_m.group(1).strip()
                    if rest:
                        append_sol(rest)
                continue

            # ── "Answer: X" colon form (alternative publisher format) ─────────
            # e.g. \section*{Answer: b}  \section*{Answer: 9}
            # Guard: only fire when NOT already in solution (IN_S).
            ac_m = RE_ANSWER_COLON.match(sec_text)
            if ac_m and current is not None and state != S.IN_S:
                if not current.answer:
                    current.answer = _parse_answer_colon(ac_m.group(1))
                state = S.IN_A
                continue

            if _is_noise(sec_text):
                last_section_was_noise = True
                continue
            last_section_was_noise = False

            subj_m = RE_SUBJECT.search(sec_text)
            if subj_m:
                flush()
                subject = subj_m.group(1).upper()
                if subject in ("MATHS", "MATH"):
                    subject = "MATHEMATICS"
                last_committed_num = 0   # ← CRITICAL RESET per section
                pending_setcounter = None
                current_q_type = "MCQ"
                continue

            if RE_NUMERICAL_SEC.search(sec_text):
                flush()
                section = "SECTION-B"
                current_q_type = "NUMERICAL"
                continue

            # ── Unrecognised \section*{...} while inside solution ─────────────
            # MathPix sometimes wraps bold mid-solution lines (e.g. step headers,
            # "Balanced Wheatstone bridge", etc.) in \section*{}.  Rather than
            # silently dropping them, append to solution so the working is intact.
            if current is not None and state == S.IN_S:
                append_sol(sec_text)
            continue

        # ── itemize fences (solution step lists — never contain questions) ────
        if RE_BEGIN_ITEMIZE.match(ln):
            itemize_depth += 1
            continue
        if RE_END_ITEMIZE.match(ln):
            itemize_depth = max(0, itemize_depth - 1)
            continue

        # ── enumerate fences ──────────────────────────────────────────────────
        if RE_BEGIN_ENUM.match(ln):
            if last_section_was_noise:
                in_noise_block = True
            else:
                enum_depth += 1
            pending_setcounter = None   # fresh block resets any pending counter
            continue

        if RE_END_ENUM.match(ln):
            if in_noise_block:
                in_noise_block = False
            else:
                enum_depth = max(0, enum_depth - 1)
            continue

        # ── \setcounter ───────────────────────────────────────────────────────
        m = RE_SETCOUNTER.match(ln)
        if m:
            # Only track setcounter{enumi} (not enumii/enumiii used in solution math)
            # enumii/enumiii are nested enumerate counters — not question numbers
            if itemize_depth == 0:
                pending_setcounter = int(m.group(1))
            continue

        # ── noise block ───────────────────────────────────────────────────────
        if in_noise_block:
            continue

        # ─────────────────────────────────────────────────────────────────────
        # CHECK FOR NEW QUESTION — fires in ANY state including IN_S
        # Priority 1: \item
        # Priority 2: plain "N. text"
        # This is the fix for questions embedded in solution text.
        # ─────────────────────────────────────────────────────────────────────

        # Priority 1: \item — only treat as question when inside enumerate
        # (not itemize, which MathPix uses for solution bullet points)
        m = RE_ITEM.match(ln)
        if m:
            if itemize_depth > 0:
                # Inside \begin{itemize} — this is solution content, not a question
                if current is not None and state == S.IN_S:
                    rest = m.group(1).strip()
                    if rest:
                        append_sol(rest)
                continue
            if enum_depth > 0 or pending_setcounter is not None:
                # Inside \begin{enumerate} — this IS a question
                from_sc = pending_setcounter is not None
                if pending_setcounter is not None:
                    q_num = pending_setcounter + 1
                else:
                    # Use effective_last (includes current active question number)
                    # so Q74's \item correctly computes 73+1=74, not 0+1=1
                    effective = max(last_committed_num,
                                    current.number if current is not None else 0)
                    q_num = effective + 1
                pending_setcounter = None
                if is_next_q(q_num, from_setcounter=from_sc):
                    start_q(q_num, m.group(1).strip())
            # else: stray \item outside any environment — ignore
            continue

        # Priority 2: plain numbered question "N. text" or bare "N.\\"
        pq = RE_PLAIN_Q.match(stripped)
        if pq:
            num  = int(pq.group(1) or pq.group(3) or pq.group(4))
            rest = (pq.group(2) or '').strip()
            if is_next_q(num):
                start_q(num, rest)
                pending_setcounter = None
                continue

        # ── "Correct Option (N)" / "Correct Answer : N" inline in solution ──
        # Same patterns as the \section* case but appearing as plain text lines.
        # Must go to solution body, never set current.answer.
        if RE_CORRECT_SOL_NOISE.match(stripped) and current is not None:
            if state == S.IN_S:
                append_sol(stripped)
            continue

        # ── Answer ────────────────────────────────────────────────────────────
        # Guard: do NOT fire when already IN_S.  Once the solution body has
        # started, lines like "Answer (3)" are part of the worked solution text,
        # not a fresh authoritative answer declaration.
        am = RE_ANSWER.match(stripped)
        if am and current is not None and state != S.IN_S:
            raw_ans = (am.group(1) or am.group(2) or '').strip()
            parsed  = _parse_answer(raw_ans)
            nta_m = RE_NTA_ANS.match(stripped)
            if nta_m:
                nta_raw = (nta_m.group(1) or nta_m.group(2) or '').strip()
                current.answer = _parse_answer(nta_raw)
                state = S.IN_A
            elif not current.answer:
                current.answer = parsed
                state = S.IN_A
            continue

        # ── "Answer: X" colon form inline ────────────────────────────────────
        # Guard: only fire when NOT already in solution (IN_S).
        ac_m = RE_ANSWER_COLON.match(stripped)
        if ac_m and current is not None and state != S.IN_S:
            if not current.answer:
                current.answer = _parse_answer_colon(ac_m.group(1))
            state = S.IN_A
            continue

        # ── Sol ───────────────────────────────────────────────────────────────
        sm = RE_SOL.match(stripped)
        if sm and current is not None:
            # ── FIX: check whether the token after "Sol." is the answer ──────
            sol_ans_m = RE_SOL_ANS.match(stripped)
            if sol_ans_m:
                ans_tok = sol_ans_m.group(1).strip()
                if not current.answer:
                    current.answer = _parse_answer(ans_tok)
                state = S.IN_S
            else:
                state = S.IN_S
                rest = _tail(sm.group(1))
                if rest:
                    append_sol(rest)
            continue

        # ── "Solution:" colon form inline ────────────────────────────────────
        sc_m = RE_SOLUTION_COLON.match(stripped)
        if sc_m and current is not None:
            state = S.IN_S
            rest = sc_m.group(1).strip()
            if rest:
                append_sol(rest)
            continue

        # ── Options (1)/(2)/(3)/(4) form ─────────────────────────────────────
        om = RE_OPTION.match(stripped)
        if om and current is not None and state == S.IN_Q:
            set_opt(int(om.group(1)), _tail(om.group(2)))
            continue

        # ── Options a./b./c./d. form (alternative publisher format) ──────────
        # Lowercase only — uppercase A./B. are reaction sub-labels in question body.
        # Gate by state == IN_Q so solution text like "a. The charge..." is safe.
        om_abcd = RE_OPTION_ABCD.match(stripped)
        if om_abcd and current is not None and state == S.IN_Q:
            letter = om_abcd.group(1).lower()
            opt_num = int(_ABCD_LETTER[letter])
            set_opt(opt_num, _tail(om_abcd.group(2)))
            continue

        # ── Standalone \includegraphics ───────────────────────────────────────
        if RE_INCLUDEGFX.search(ln):
            if current is not None:
                for raw_id in RE_INCLUDEGFX.findall(ln):
                    img_id = os.path.basename(raw_id.strip())
                    ph     = f"[IMAGE:{img_id}]"
                    if state == S.IN_S:
                        append_sol(ph)
                    elif state == S.IN_Q:
                        if current.options and not current.options[-1].strip():
                            current.options[-1] = ph
                        else:
                            append_q(ph)
                    else:
                        append_q(ph)
            continue

        # ── Bare option label "(1)\\" ─────────────────────────────────────────
        solo = re.match(r'^\(([1-4])\)\\*\s*$', stripped)
        if solo and current is not None and state == S.IN_Q:
            set_opt(int(solo.group(1)), "")
            continue

        # ── Continuation text ─────────────────────────────────────────────────
        if current is None:
            continue

        if state == S.IN_S:
            append_sol(ln)
        elif state == S.IN_Q:
            if current.options:
                current.options[-1] += " " + stripped
            else:
                append_q(ln)
        # IN_A: ignore

    flush()

    # ── Post-processing ────────────────────────────────────────────────────────
    RE_NOISE_LINE = re.compile(
        r'^\s*\\setcounter\{enum[iIvV]+\}\{[^}]+\}\s*$'
    )

    def _clean_sol(text: str) -> str:
        lines = [ln for ln in text.split('\n') if not RE_NOISE_LINE.match(ln)]
        return '\n'.join(lines).strip()

    result = []
    for q in questions:
        q.solution = _clean_sol(q.solution)
        q.question, qi = _extract_images(q.question)
        q.solution, si = _extract_images(q.solution)
        clean = []; oi = []
        for opt in q.options:
            c, o = _extract_images(opt)
            clean.append(c); oi.extend(o)
        q.options = clean

        q_ids = RE_PLACEHOLDER.findall(q.question)
        o_ids = []
        for opt in q.options:
            o_ids.extend(RE_PLACEHOLDER.findall(opt))
        s_ids = RE_PLACEHOLDER.findall(q.solution)

        q.q_images   = _unique(q_ids + o_ids + qi + oi)
        q.sol_images = _unique(s_ids + si)

        if q.q_type == "MCQ":
            a = q.answer.lower()
            if re.search(r'\b[a-d]\b', a) and ',' in a:
                # Legacy format: answer still contains letters e.g. "a,c"
                q.q_type = "MSQ"
            elif re.fullmatch(r'\d+(?:,\d+)+', a.strip()) and q.options:
                # New format: answer already converted to "1,2" or "2,4" etc.
                # Multiple digit options with commas = MSQ
                q.q_type = "MSQ"
            elif not q.options and re.fullmatch(r'-?[\d.]+', q.answer.strip()):
                # No options + pure number (including negative) = NUMERICAL
                q.q_type = "NUMERICAL"

        result.append(q.to_dict())

    return result


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python parser.py <file.tex>")
        sys.exit(1)

    qs = parse_tex(sys.argv[1])
    print(json.dumps(qs, indent=2, ensure_ascii=False))
    print(f"\n✓ Parsed {len(qs)} questions", file=sys.stderr)