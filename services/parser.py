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

9. Inline multi-option line (NEW):
   Some publishers (e.g. Selfstudys/SelfStudys PDFs) put all four MCQ options
   on a single line separated by large whitespace, e.g.:
       (1) 2          (2) 4          (3) 3          (4) 5
       (1) 1/√2       (2) 1/2        (3) 1          (4) √2
   RE_INLINE_OPTS detects this pattern and splits it into individual options.
   It fires in state IN_Q, just before the single-option RE_OPTION check.
   Guard: line must contain at least two "(N)" markers → no false positives
   on lines like "(1) some long answer text that wraps".

10. Auto-detect NUMERICAL from Sol. when no options present (NEW):
    After parsing, if a question was classified MCQ but has 0 options and
    the answer is a pure number (int or decimal, including large integers like
    25, 150), q_type is promoted to NUMERICAL. This handles Section-B style
    integer-answer questions in PDFs that don't have explicit SECTION-B headers.

11. Subject detection for single-subject PDFs (FIX):
    Single-subject PDFs (Maths-only, Chemistry-only, Biology-only) have the
    subject in \title{} or as plain body text ("Subject: Mathematics"), NOT
    inside a \section*{} header.  Three detection layers are applied:
      a) subject_hint parameter — caller passes subject inferred from filename.
      b) \title{} scan — e.g. "JEE Main 2020 Chemistry" sets subject directly.
      c) Body buffer scan (first 800 chars) — catches "Subject: Mathematics"
         plain-text lines that MathPix renders outside any \section*{}.
    Without this, every question in a Maths-only PDF got subject="PHYSICS",
    causing the LLM tagger to load the Physics taxonomy and find no chapter.
    Multi-subject PDFs (JEE full paper, NEET) are unaffected because their
    \section*{PHYSICS} / \section*{CHEMISTRY} / \section*{BIOLOGY} headers
    override whatever the body-scan sets.
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
# ── FIX 11: Broader pattern for \title{} and plain body text ─────────────────
# Matches any standalone subject word so "Subject: Mathematics" in body text
# and "7 jan 2020 shift 1 Maths" in \title{} are correctly detected.
# Only used by the title/body scan layers — \section*{} still uses RE_SUBJECT.
RE_SUBJECT_BROAD = re.compile(
    r'\b(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|MATH|BIOLOGY)\b',
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

# ── NEW: inline multi-option line ─────────────────────────────────────────────
# Detects lines where ALL options appear on one line with whitespace separation:
#   "(1) 2          (2) 4          (3) 3          (4) 5"
#   "(1) 1/√2       (2) 1/2        (3) 1          (4) √2"
#   "(1) $\frac{1}{\sqrt{2}}$   (2) $\frac{1}{2}$   (3) 1   (4) $\sqrt{2}$"
#
# Strategy: findall of "(N) <text-up-to-next-(N)-or-EOL>" in order.
# Guard: only fires when the line contains ≥2 of the markers (1),(2),(3),(4)
# in ascending order — prevents false positives on single-option lines.
#
# Each captured group: (option_number_str, option_text_str)
RE_INLINE_OPT_MARKER = re.compile(r'\(([1-4])\)')

def _split_inline_options(line: str):
    """
    If `line` contains multiple (1)/(2)/(3)/(4) option markers (the Selfstudys
    single-line format), split and return list of (int, str) tuples like
    [(1, 'text1'), (2, 'text2'), ...].  Returns [] if this is NOT a multi-option
    inline line (so the caller falls through to the normal single-option path).

    Rules:
    - Must find ≥2 markers in strictly ascending order starting from (1).
    - Each option's text = everything between this marker and the next marker
      (or end of string), stripped.
    - If any option text itself looks like a new question number ("N. text"),
      abort and return [] — we're inside a solution, not an option line.
    """
    markers = list(RE_INLINE_OPT_MARKER.finditer(line))
    if len(markers) < 2:
        return []

    # Check ascending order starting at 1
    nums = [int(m.group(1)) for m in markers]
    if nums[0] != 1:
        return []
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1] + 1:
            return []

    # Extract text for each option
    result = []
    for i, m in enumerate(markers):
        start = m.end()
        end   = markers[i+1].start() if i + 1 < len(markers) else len(line)
        text  = line[start:end].strip()
        # Safety: if text looks like "N. something" it's not option text
        if re.match(r'^\d{1,3}\.\s', text):
            return []
        result.append((nums[i], text))
    return result


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

# ── "( N )" bare answer on its own line after Solution: ──────────────────────
# Some publishers put the answer on the very next line after the bold
# "Solution:" header as a standalone "(N)" or "(c)" or "(90°)".
# This fires ONLY when state==IN_S and current.answer is still empty.
# Must NOT match option lines like "(1) some text" or math like "(x+y)".
RE_BARE_PAREN_ANS = re.compile(
    r'^\(\s*([a-dA-D]|-?\d+(?:\.\d+)?°?)\s*\)\s*$'
)

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

# ── FIX 11: canonical subject names for validation ───────────────────────────
_VALID_SUBJECTS = {"PHYSICS", "CHEMISTRY", "MATHEMATICS", "BIOLOGY"}


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


# ── FIX 11: helper to canonicalise a raw subject token ───────────────────────
def _canon_subject(raw: str) -> str:
    """Return uppercase canonical subject name, or '' if unrecognised."""
    s = raw.strip().upper()
    if s in ("MATHS", "MATH"):
        return "MATHEMATICS"
    return s if s in _VALID_SUBJECTS else ""


def _extract_subject_from_line(stripped: str) -> str:
    """
    Returns canonical subject name if `stripped` contains ONLY a subject
    identifier, possibly wrapped in LaTeX commands like \\textbf{},
    {\\bf }, \\textcolor{white}{...}.
    Returns '' when the line has other meaningful content.

    Used to catch mid-document subject headers that MathPix renders as
    plain/bold text instead of \\section*{PHYSICS}.  Examples from
    Selfstudys NEET PDFs:
      PHYSICS
      \\textbf{CHEMISTRY}
      {\\bf BIOLOGY}
      \\textbf{\\textcolor{white}{PHYSICS}}
    """
    bare = stripped
    bare = re.sub(r'\\textcolor\{[^}]+\}', '', bare)
    bare = re.sub(r'\\textbf\s*', '', bare)
    bare = re.sub(r'\{\\bf\s*', '', bare)
    bare = re.sub(r'[{}]', '', bare).strip()
    m = re.fullmatch(
        r'\s*(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|MATH|BIOLOGY)\s*',
        bare, re.IGNORECASE
    )
    if m:
        s = m.group(1).upper()
        return "MATHEMATICS" if s in ("MATHS", "MATH") else s
    return ""


def _strip_textbf(s: str) -> str:
    """Strip \\textbf{...} wrappers from a line, keeping inner content.

    MathPix renders bold PDF text as \\textbf{...}.  Publisher PDFs like
    Selfstudys use bold for Answer/Solution labels, so these lines arrive as:
      \\textbf{Answer:} a
      \\textbf{Solution:} Here, ...
      \\textbf{Subject: Chemistry}
    Without stripping, RE_ANSWER_COLON / RE_SOLUTION_COLON / RE_SOL all miss
    them — the answer is silently lost and solution text gets appended to the
    last option instead.
    """
    return re.sub(r'\\textbf\{([^}]*)\}', r'\1', s).strip()


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
    """Parse answer from 'Answer: X' / 'Solution: X' colon formats.

    Handles all publisher variants seen in JEE PDFs:
      "b"           → "2"      (MCQ letter)
      "(b)"         → "2"      (MCQ letter in parens)
      "(b )"        → "2"      (space before closing paren)
      "b."          → "2"      (letter with trailing dot)
      "a,b"         → "1,2"   (MSQ multi-letter)
      "9"           → "9"      (numerical integer)
      "(6)"         → "6"      (numerical in parens)
      "(90°)"       → "90"     (numerical + degree symbol)
      "5.22"        → "5.22"   (decimal)
      "0.3675 g"    → "0.3675" (numerical + unit suffix)
      "-85 kJ/mol"  → "-85"    (negative + unit)
    """
    raw = raw.strip()

    # ── Step 1: unwrap outer parens ────────────────────────────────────────────
    # "(b)" → "b",  "(6)" → "6",  "(b )" → "b",  "(90°)" → "90°"
    m = re.fullmatch(r'\(\s*(.+?)\s*\)', raw)
    if m:
        inner = m.group(1).strip()
        # Only unwrap if it's not a LaTeX expression
        if not any(c in inner for c in ('\\', '$', '{')):
            raw = inner

    # ── Step 2: strip trailing dot from bare letter: "b." → "b" ───────────────
    if re.fullmatch(r'[a-dA-D]\.', raw):
        raw = raw[0]

    # ── Step 3: strip trailing degree symbol: "90°" → "90" ───────────────────
    raw = raw.rstrip('°').strip()

    # ── Step 4: single letter a/b/c/d ─────────────────────────────────────────
    if re.fullmatch(r'[a-dA-D]', raw):
        return _ABCD_LETTER[raw.lower()]

    # ── Step 5: multi-letter MSQ: "a,b"  "b,d"  "B,D"  "a, b" ───────────────
    if re.fullmatch(r'[a-dA-D](?:\s*,\s*[a-dA-D])+', raw):
        return ','.join(_ABCD_LETTER[c.lower()] for c in re.findall(r'[a-dA-D]', raw))

    # ── Step 6: number with optional unit suffix ───────────────────────────────
    m2 = re.match(r'^(-?\d+(?:\.\d+)?)', raw)
    if m2:
        return m2.group(1)

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

def parse_tex(tex_path: str, subject_hint: str = "") -> list:
    """Parse a MathPix .tex file and return a list of question dicts.

    Args:
        tex_path:     Path to the .tex file.
        subject_hint: Optional subject override inferred from the filename
                      before calling, e.g. "Mathematics", "Chemistry",
                      "Biology".  When provided it is used as the initial
                      subject so single-subject PDFs that lack a
                      \\section*{MATHEMATICS} header still produce correctly-
                      labelled questions.  Multi-subject PDFs are unaffected
                      because their \\section*{PHYSICS} etc. headers always
                      override this value.
    """
    with open(tex_path, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]

    title = subject = ""

    # ── FIX 11a: resolve subject_hint before defaulting to "PHYSICS" ──────────
    if subject_hint:
        _hc = _canon_subject(subject_hint)
        subject = _hc if _hc else "PHYSICS"
    else:
        subject = "PHYSICS"

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
            # ── FIX 11b: detect subject from \title{} ─────────────────────────
            # Only update if still at the initial default (no hint given and no
            # \section*{} has fired yet).  Catches titles like
            # "JEE Main 2020 Chemistry" or "7 jan 2020 shift 1 Maths".
            if subject == "PHYSICS":
                _tm = RE_SUBJECT_BROAD.search(title)
                if _tm:
                    _tc = _canon_subject(_tm.group(1))
                    if _tc:
                        subject = _tc
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
            # ── FIX 11c: detect subject from plain body text ──────────────────
            # Catches "Subject: Mathematics" / "Subject: \textbf{Chemistry}"
            # which MathPix renders as plain body text, not \section*{}.
            # Only update if still at the "PHYSICS" default — never overwrite
            # a subject already set by subject_hint, \title{}, or \section*{}.
            if subject == "PHYSICS" and stripped:
                _bm = RE_SUBJECT_BROAD.search(stripped)
                if _bm:
                    _bc = _canon_subject(_bm.group(1))
                    if _bc:
                        subject = _bc

        if not stripped:
            continue

        # ── FIX 11d: mid-document plain-text subject header ───────────────────
        # Catches MathPix rendering of colored/bold subject boxes as plain text
        # or \textbf{} instead of \section*{}.  Fires ANYWHERE in the document
        # so CHEMISTRY (Q46) and BIOLOGY (Q91) in NEET papers are caught even
        # though they appear thousands of chars past the 800-char body buffer.
        # Guard: only fire when the ENTIRE line is a subject word (no other text).
        _subj_plain = _extract_subject_from_line(stripped)
        if _subj_plain:
            flush()
            subject = _subj_plain
            last_committed_num = 0   # ← same reset as \section*{} handler
            pending_setcounter = None
            current_q_type = "MCQ"
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
            # e.g. \section*{Solution: (c)}  \section*{Solution:(b)}
            # The rest after "Solution:" may itself carry the answer:
            #   "Solution: (c)"  → answer='3', nothing to append to solution
            #   "Solution: (6)"  → answer='6'
            #   "Solution: (90°)"→ answer='90'
            #   "Solution: b."   → answer='2'
            #   "Solution:"      → just start solution state
            sc_m = RE_SOLUTION_COLON.match(sec_text)
            if sc_m:
                last_section_was_noise = False
                if current is not None:
                    state = S.IN_S
                    rest = sc_m.group(1).strip()
                    if rest:
                        # Try to parse rest as an answer token
                        parsed_ans = _parse_answer_colon(rest)
                        # Accept as answer if it reduced to something simpler
                        # (i.e. not equal to rest, OR rest is a bare paren/letter form)
                        if RE_BARE_PAREN_ANS.match(rest) or re.fullmatch(r'[a-dA-D]\.?', rest):
                            if not current.answer:
                                current.answer = parsed_ans
                            # Don't append to solution — it's the answer token
                        else:
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

        # ── Strip \textbf{} wrappers before pattern matching ──────────────────
        # Bold labels like \textbf{Answer:} a and \textbf{Solution:} are
        # MathPix output for publisher PDFs (Selfstudys, Allen, etc.) that use
        # bold for Answer/Solution.  Normalise here so all downstream patterns
        # (RE_ANSWER_COLON, RE_SOLUTION_COLON, RE_SOL, RE_ANSWER) work correctly.
        # FIX 11d above already ran on the raw stripped line (needs raw for
        # standalone subject detection), so this normalisation is safe here.
        stripped = _strip_textbf(stripped)

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

        # ── "Solution:" colon form inline ─────────────────────────────────────
        # "Solution: (c)" / "Solution:(b)" / "Solution: (6)" / "Solution: b."
        sc_m = RE_SOLUTION_COLON.match(stripped)
        if sc_m and current is not None:
            state = S.IN_S
            rest = sc_m.group(1).strip()
            if rest:
                if RE_BARE_PAREN_ANS.match(rest) or re.fullmatch(r'[a-dA-D]\.?', rest):
                    if not current.answer:
                        current.answer = _parse_answer_colon(rest)
                else:
                    append_sol(rest)
            continue

        # ── NEW: Inline multi-option line (Selfstudys / single-line format) ──
        # Must be checked BEFORE the single-option RE_OPTION path.
        # Only fires in state IN_Q (same guard as RE_OPTION).
        # Only fires when ≥2 option markers found in ascending order from (1).
        # Example: "(1) 2          (2) 4          (3) 3          (4) 5"
        if current is not None and state == S.IN_Q:
            inline_opts = _split_inline_options(stripped)
            if inline_opts:
                for opt_num, opt_text in inline_opts:
                    set_opt(opt_num, _tail(opt_text))
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

        # ── Bare "(N)" answer line after Solution: ──────────────────────────────
        # Some publishers put answer on its own line: "(40)", "(c)", "(90°)"
        # Only fire when: (a) state==IN_S, (b) answer not yet captured,
        # (c) line is ONLY a paren-wrapped token (no trailing text).
        # Guard: check state==IN_S first so "(1) option text" in IN_Q is safe.
        if state == S.IN_S and current is not None and not current.answer:
            bpa = RE_BARE_PAREN_ANS.match(stripped)
            if bpa:
                current.answer = _parse_answer_colon(stripped)
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

        # ── Smart MCQ / NUMERICAL / MSQ classification ────────────────────────
        # Priority order:
        #  1. Already NUMERICAL (set by SECTION-B header) — never downgrade.
        #  2. Answer contains comma + digits → MSQ.
        #  3. Answer contains letter (a-d) with comma → legacy MSQ.
        #  4. No options + pure numeric answer → NUMERICAL (Section-B without
        #     header, e.g. "Sol. 25", "Sol. 150", "Sol. 1.58").
        #     This handles Selfstudys-style PDFs where integer questions appear
        #     inline with MCQ questions and have no options whatsoever.
        #  5. Has 4 options + answer 1-4 → MCQ (unchanged).
        if q.q_type != "NUMERICAL":
            a = q.answer.strip()
            a_lower = a.lower()

            if re.fullmatch(r'\d+(?:,\d+)+', a) and q.options:
                # "1,3" or "2,4" with options → MSQ
                q.q_type = "MSQ"
            elif re.search(r'\b[a-d]\b', a_lower) and ',' in a_lower:
                # "a,c" style legacy MSQ
                q.q_type = "MSQ"
            elif not q.options and re.fullmatch(r'-?\d+(?:\.\d+)?', a):
                # No options + pure number (int, decimal, negative) → NUMERICAL
                # Covers: "25", "150", "1.58", "-85", "0"
                q.q_type = "NUMERICAL"
            # else: leave as MCQ

        result.append(q.to_dict())

    return result


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python parser.py <file.tex> [subject_hint]")
        sys.exit(1)

    hint = sys.argv[2] if len(sys.argv) > 2 else ""
    qs = parse_tex(sys.argv[1], subject_hint=hint)
    print(json.dumps(qs, indent=2, ensure_ascii=False))
    print(f"\n✓ Parsed {len(qs)} questions", file=sys.stderr)