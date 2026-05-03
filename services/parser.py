r"""
services/parser.py - Enhanced NEET/JEE Question Paper Parser
Supports both LaTeX/MathPix format and Vedantu plain-text PDF format.
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional

# ══════════════════════════════════════════════════════════
# REGEX PATTERNS
# ══════════════════════════════════════════════════════════

RE_TITLE         = re.compile(r'\\title\{(.+?)\}')
RE_BEGIN_DOC     = re.compile(r'^\s*\\begin\{document\}')
RE_END_DOC       = re.compile(r'^\s*\\end\{document\}')
RE_BEGIN_ENUM    = re.compile(r'^\s*\\begin\{enumerate\}')
RE_END_ENUM      = re.compile(r'^\s*\\end\{enumerate\}')
RE_BEGIN_FIGURE  = re.compile(r'^\s*\\begin\{figure\}')
RE_END_FIGURE    = re.compile(r'^\s*\\end\{figure\}')
RE_BEGIN_ITEMIZE = re.compile(r'^\s*\\begin\{itemize\}')
RE_END_ITEMIZE   = re.compile(r'^\s*\\end\{itemize\}')
RE_BEGIN_CENTER  = re.compile(r'^\s*\\begin\{center\}')
RE_END_CENTER    = re.compile(r'^\s*\\end\{center\}')
RE_BEGIN_TABULAR = re.compile(r'^\s*\\begin\{tabular\}')
RE_END_TABULAR   = re.compile(r'^\s*\\end\{tabular\}')
RE_CAPTION       = re.compile(r'\\caption\{(.+?)\}')
RE_SETCOUNTER    = re.compile(r'^\s*\\setcounter\{enumii?\}\{(\d+)\}')
RE_ITEM          = re.compile(r'^\s*\\item\s*(.*)')
RE_SECTION       = re.compile(r'^\s*\\section\*\{(.*)\}\s*$')   # greedy: capture to LAST }
RE_INCLUDEGFX    = re.compile(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}')
RE_PLACEHOLDER   = re.compile(r'\[IMAGE:([^\]]+)\]')
# Add these new patterns
RE_ADV_QUESTION = re.compile(r'^\s*(\d+)\.\s*', re.MULTILINE) # Simple digit + dot for Advanced
RE_ADV_SECTION_HEADER = re.compile(r'SECTION\s*[-:]\s*\d+', re.IGNORECASE)
# Updated Answer pattern to capture "A, B, D" or "1, 2, 4"
RE_MULTI_ANSWER = re.compile(r'Ans\.\s*\(?([A-E,\s1-4]+)\)?', re.IGNORECASE)
RE_SUBJECT = re.compile(
    r'(?:PART[-\s]*[A-Za-z]\s*:\s*)?(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|MATH|BIOLOGY|ZOOLOGY|BOTANY)',
    re.IGNORECASE,
)
RE_SUBJECT_BROAD = re.compile(
    r'\b(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|MATH|BIOLOGY|ZOOLOGY|BOTANY)\b',
    re.IGNORECASE,
)

# ── FIXED: handle "SECTION - B", "SECTION-B", "Section-B" etc. ──────────────
RE_NUMERICAL_SEC = re.compile(
    r'SECTION\s*[-–]\s*(?:B|2|II)\b'
    r'|INTEGER\s*(?:TYPE|ANSWER)?'
    r'|NUMERICAL\s*(?:VALUE|TYPE|ANSWER)?',
    re.IGNORECASE,
)
# ── NEW: detect "SECTION - A" to reset section back to MCQ ──────────────────
RE_SECTION_A_PAT = re.compile(r'^SECTION\s*[-–]\s*(?:A|1|I)\s*$', re.IGNORECASE)

# Question start patterns
RE_PLAIN_Q = re.compile(
    r'^\*?(\d{1,3})\.\s+(\S.*)'
    r'|^\*?(\d{1,3})\.\s*\\\\?\s*$'
    r'|^\*?(\d{1,3})\.\s*$'
)
RE_QUESTION_PREFIX = re.compile(r'^Question\s+(\d{1,3})\.\s*(.*)', re.IGNORECASE)
RE_QUESTION_COLON  = re.compile(r'^Question\s+(\d{1,3}):\s*(.*)', re.IGNORECASE)

# Option patterns
RE_OPTION            = re.compile(r'^\$?\(([1-4])\)\$?\s*(.*)')
RE_OPTION_PAREN_ABCD = re.compile(r'^\(([abcdABCD])\)\s+(.*)', re.IGNORECASE)
RE_OPTION_ABCD       = re.compile(r'^([abcd])\.\s+(.*)')
RE_OPTIONS_HEADER    = re.compile(r'^\*?\*?Options\:?\*?\*?\s*$', re.IGNORECASE)
RE_INLINE_OPT_MARKER = re.compile(r'\(([1-4])\)')

# Answer patterns
RE_ANSWER = re.compile(
    r'^(?:(?:Official|Allen|NTA)\s+)?Ans(?:wer)?\.?\s+(?:by\s+NTA\s+)?\((.+?)\)'
    r'|^(?:(?:Official|Allen|NTA)\s+)?Ans(?:wer)?\.?\s+(?:by\s+NTA\s+)?(\d+(?:\.\d+)?)\s*$',
    re.IGNORECASE
)
RE_NTA_ANS = re.compile(
    r'^NTA\s+Ans(?:wer)?\.?\s+(?:by\s+NTA\s+)?\((.+?)\)'
    r'|^NTA\s+Ans(?:wer)?\.?\s+(?:by\s+NTA\s+)?(\d+(?:\.\d+)?)\s*$',
    re.IGNORECASE
)
RE_ANSWER_COLON = re.compile(
    r'^\*?\*?Answer\s*:\s*\*?\*?\s*(.+?)\s*\*?\*?$',
    re.IGNORECASE
)
# Add this line
RE_HENCE_OPT = re.compile(r'Hence\s+.*(?:Option|Solution)\s+(?:is\s+)?\(?([1-4])\)?', re.IGNORECASE)

# Solution patterns
RE_SOL          = re.compile(r'^Sol\.\s*(.*)', re.IGNORECASE)
RE_SOL_ANS      = re.compile(
    r'^Sol\.\s+(\d+(?:\.\d+)?)(?:\s*(?:%|cc|cm|m|kg|s|V|J|N|eV|K|Hz|mol|kbar|mN|rpm|nm|mm|pm))?\s*$',
    re.IGNORECASE
)
RE_SOLUTION_COLON  = re.compile(r'^\*?\*?Solution\s*:\s*\*?\*?(.*)$', re.IGNORECASE)
RE_CORRECT_SOL_NOISE = re.compile(r'^Correct\s+(?:Option|Answer)\s*[:(]', re.IGNORECASE)
RE_BARE_PAREN_ANS  = re.compile(r'^\(\s*([a-dA-D]|-?\d+(?:\.\d+)?°?)\s*\)\s*$')
# Multi-answer paren: (A, C) or (B, D) or (A,C,D) etc.
RE_MULTI_PAREN_ANS = re.compile(r'^\(\s*[a-dA-D](?:\s*,\s*[a-dA-D])+\s*\)\s*$')

# ── NEW: extract answer from Sol. (X) where X is option number 1-4 ──────────
RE_SOL_PAREN_OPT = re.compile(r'^\(\s*([1-4])\s*\)\s*$')
# Sol. (6) lone pairs  — leading paren-integer possibly followed by trailing text
RE_SOL_PAREN_INT_LEAD = re.compile(r'^\(\s*(\d+)\s*\)')

_ABCD_LETTER = {'a':'1','b':'2','c':'3','d':'4','A':'1','B':'2','C':'3','D':'4'}
_NTA_LETTER  = {'A':'1','B':'2','C':'3','D':'4','a':'1','b':'2','c':'3','d':'4'}

_NOISE_PATS = [
    re.compile(r'^answers?\s*[&and]*\s*solutions?\s*$', re.IGNORECASE),
    re.compile(r'important\s*instructions?', re.IGNORECASE),
    re.compile(r'^\(physics', re.IGNORECASE),
    re.compile(r'fact\s*based', re.IGNORECASE),
    re.compile(r'^sol\.', re.IGNORECASE),
    re.compile(r'correct\s*(option|answer|ans)', re.IGNORECASE),
    re.compile(r'download.*app', re.IGNORECASE),
    re.compile(r'scan.*qr|qr.*code', re.IGNORECASE),
    # Preamble/instructions sections in JEE/NEET papers
    re.compile(r'^instructions?\s*$', re.IGNORECASE),
    re.compile(r'^[A-C]\.\s*(general|question\s*paper|marking\s*scheme)', re.IGNORECASE),
    re.compile(r'please\s*read\s*the\s*instructions', re.IGNORECASE),
    re.compile(r'^(general\s*)?instructions?\s*(to\s*candidates?)?$', re.IGNORECASE),
]

_MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
    'sep':9,'oct':10,'nov':11,'dec':12,
}
_VALID_SUBJECTS = {"PHYSICS","CHEMISTRY","MATHEMATICS","BIOLOGY","ZOOLOGY","BOTANY"}

RE_NOISE_LINE = re.compile(r'^\s*\\setcounter\{enum[iIvV]+\}\{[^}]+\}\s*$')



# ══════════════════════════════════════════════════════════
# SSC CGL OFFICIAL PDF (MathPix .tex) PATTERNS & PARSER
# ══════════════════════════════════════════════════════════
# Real MathPix output for official SSC CGL answer sheets:
#   Q. 1 <question text>\\
#   Ans
#   ✓ 1.<option text>    ← correct answer (✓ U+2713)
#   X 2.<option text>    ← wrong
#   × 3.<option text>    ← wrong (× U+00D7)
#   X4.<option text>     ← wrong (no space)
#   Section : General Intelligence and Reasoning\\

RE_SSC_OFF_Q = re.compile(
    r'^Q\.\s*\$?\s*(\d{1,3})\s*(.*)',  # "Q. 1 text", "Q.1 text", "Q. $22 text"
    re.IGNORECASE
)
# Option line: checkmark/cross + digit + dot + text
# ✓ 1.text  or  X 2.text  or  × 3.text  or  X4.text (no space)
RE_SSC_OFF_OPT = re.compile(
    r'^([✓✔✗×Xx])\s*(\d)\s*[.\s]\s*(.*)'
)
# Bare option: '2. text' or '2.text' or '2.' (empty) — no mark prefix (wrong option)
RE_SSC_OFF_OPT_BARE = re.compile(r'^(\d)\s*\.\s*(.*)')  # digit.text (space optional, text may be empty)
# Ans line: "Ans" possibly followed by option on same line
# e.g. "Ans ✓ 1.text"  or  "Ans $1 . \div$"
RE_SSC_OFF_ANS = re.compile(r'^Ans\b\s*(.*)', re.IGNORECASE)

# Section header: "Section : General Intelligence and Reasoning"
RE_SSC_OFF_SEC = re.compile(
    r'^Section\s*:\s*(.+?)(?:\\\\)?\s*$',
    re.IGNORECASE
)

# Metadata noise lines in new SSC 2024 PDF format (Question ID, Option IDs, Status, Chosen Option)
RE_SSC_METADATA = re.compile(
    r'^(?:Question\s+ID|Option\s+\d+\s+ID|Status|Chosen\s+Option)\s*[:\-]',
    re.IGNORECASE
)

_SSC_CHECKMARKS = {'✓', '✔'}
_SSC_CROSSMARKS  = {'X', 'x', '×', '✗', '✕', '✗'}

# SSC official section name → SSC CGL subject
_SSC_OFF_SECTION_MAP = {
    "general intelligence and reasoning": "General Intelligence and Reasoning",
    "general awareness":                  "General Awareness",
    "quantitative aptitude":              "Quantitative Aptitude",
    "english comprehension":              "English Comprehension",
    # fallbacks for slight variations
    "general intelligence":               "General Intelligence and Reasoning",
    "reasoning":                          "General Intelligence and Reasoning",
    "english":                            "English Comprehension",
    "quant":                              "Quantitative Aptitude",
}

def _map_ssc_section(raw: str) -> str:
    key = raw.strip().lower()
    for k, v in _SSC_OFF_SECTION_MAP.items():
        if key.startswith(k) or k in key:
            return v
    return raw.strip().title()


def _is_ssc_official_tex(content: str) -> bool:
    """
    Detect the real MathPix SSC CGL official answer-key tex format.
    Signature: has "Q. <n>" style questions AND checkmark ✓ answer lines.
    """
    has_q_dot = bool(re.search(r'^Q\.\s*\d{1,3}\s+\S', content, re.MULTILINE))
    has_check  = '✓' in content or '✔' in content
    return has_q_dot and has_check


def parse_ssc_official_tex(content: str) -> list:
    """
    Parse the real MathPix .tex output of an SSC CGL official answer-key PDF.

    Handles all quirks found in real MathPix output:
    - Option 4 sometimes appears inside a \\begin{enumerate} block that also
      contains the next question (Q starts nested inside an enum/itemize).
    - Options sometimes appear as plain \\item lines with no mark prefix.
    - Section header may be absent for some sections (GA in this PDF).
    - \\section*{Q. N text} — MathPix wraps some questions in \\section*.
    - Correct option marked with ✓; wrong with X/×/x; unmarked items are wrong.
    - Both "X 1. text" and "✓ 2. text" and bare "3. text" forms appear.
    - Images via \\includegraphics{} → [IMAGE:name] placeholder.
    - Metadata lines (Question ID, Option N ID, Status, Chosen Option) skipped.
    """
    meta = _extract_meta('', content[:600])
    exam_date = meta["exam_date"]
    year      = meta["year"]
    shift     = meta["shift"]

    if not exam_date:
        dm = re.search(r'(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(20\d{2})', content)
        if dm:
            exam_date = f"{dm.group(3)}-{dm.group(2).zfill(2)}-{dm.group(1).zfill(2)}"
            year = dm.group(3)
    if not exam_date:
        dm2 = re.search(r'(\d{2})/(\d{2})/(20\d{2})', content)
        if dm2:
            exam_date = f"{dm2.group(3)}-{dm2.group(2)}-{dm2.group(1)}"
            year = dm2.group(3)
    if not shift:
        if re.search(r'AM\s*-', content): shift = "Morning"
        elif re.search(r'PM\s*-', content): shift = "Evening"

    lines     = content.split('\n')
    questions: list = []
    current   = None
    subject   = "General Intelligence and Reasoning"
    last_num  = 0
    global_num = 0
    in_ans_block   = False
    enum_counter   = 0
    in_enum        = False
    in_itemize     = False
    section_just_changed = False

    # ── compiled helpers ──────────────────────────────────────────────────────
    # "Q. 12 text" — also handles \section*{Q. 12 text} and "Q. $2{text}$" math-wrapped
    # Lookahead after digits prevents "Q. 231" matching as Q.23 — digit must be followed
    # by space, $, \, {, [, or end-of-string, NOT another digit.
    RE_Q_LINE = re.compile(
        r'^(?:\\section\*\{)?\s*Q\.\s*'
        r'\$?\s*(\d{1,2})(?=[\s\$\\}{\[]|$)'
        r'(?:\$?\{[^}]*\}\$?)?'
        r'\s*(.*?)(?:\})?$',
        re.IGNORECASE | re.DOTALL
    )
    # Fallback for MathPix-merged Q numbers: "Q. 231 text" (should be Q.2 + "31 text")
    # Only fires when there are 3+ consecutive digits after "Q. "
    RE_Q_MERGED = re.compile(
        r'^(?:\\section\*\{)?\s*Q\.\s*\$?\s*(\d{3,})\s+(.*?)(?:\})?$',
        re.IGNORECASE | re.DOTALL
    )
    # Option with mark: "✓ 2. text"  "X 3. text"  "× 4.text"  "x1.text"
    RE_OPT_MARKED = re.compile(
        r'^([✓✔✗×Xx])\s*(\d)\s*[.\s]\s*(.*)',
        re.DOTALL
    )
    # Bare option (no mark): "3. text"  "3.text"  — only inside ans block
    RE_OPT_BARE = re.compile(r'^(\d)\s*\.\s*(.*)')
    # Correct-option prefix with no space before digit: "✓3.text"
    RE_OPT_TIGHT = re.compile(r'^([✓✔✗×Xx])(\d)\s*[.\s]\s*(.*)', re.DOTALL)
    # Section header line
    RE_SEC = re.compile(r'^Section\s*:\s*(.+?)(?:\\\\)?\s*$', re.IGNORECASE)
    # \setcounter{enumii}{N}
    RE_SC  = re.compile(r'^\s*\\setcounter\{enumii?\}\{(\d+)\}')
    # Metadata noise
    RE_META = re.compile(
        r'^(?:Question\s+ID|Option\s+\d+\s+ID|Status|Chosen\s+Option)\s*[:\-]',
        re.IGNORECASE
    )

    CHECKMARKS = {'✓', '✔'}

    def flush():
        nonlocal current, in_ans_block, in_enum, in_itemize, enum_counter
        if current is not None:
            current.question = current.question.strip()
            questions.append(current)
        current = None
        in_ans_block = False
        in_enum = in_itemize = False
        enum_counter = 0

    def start_q(num: int, text: str):
        nonlocal current, in_ans_block, in_enum, in_itemize
        nonlocal enum_counter, last_num, section_just_changed, global_num
        flush()
        global_num += 1
        current = ParsedQuestion(
            number=global_num, q_type="MCQ", subject=subject,
            section="SECTION-A", year=year, shift=shift,
            exam_date=exam_date, question=_tail(_strip_bold(text)),
        )
        in_ans_block = False
        in_enum = in_itemize = False
        enum_counter = 0
        last_num = num
        section_just_changed = False

    def set_opt(n: int, raw_text: str, is_correct: bool):
        if not current: return
        while len(current.options) < n:
            current.options.append("")
        val = _strip_bold(raw_text).strip()
        val = re.sub(r'\\\\+\s*$', '', val).strip()
        if not current.options[n - 1]:
            current.options[n - 1] = val
        else:
            current.options[n - 1] += " " + val
        if is_correct and not current.answer:
            current.answer = str(n)

    def _normalise_opt_line(s: str) -> str:
        """
        Normalise a raw option line so our mark+digit patterns always match.
        Handles all MathPix SSC CGL quirks:
          "$\\boldsymbol{\\times}$ 1. One"   →  "× 1. One"
          "$\\times 1.text$"                  →  "× 1.text"
          "× $1 \\frac{9}{16}$"              →  "× 1 \\frac{9}{16}"
          "X $4 . \\frac{4}{5}$"             →  "X 4 . \\frac{4}{5}"
          "2. $\\frac{25}{9}$"               →  "2. \\frac{25}{9}"  (bare)
          "-3. $x-3$"                         →  "× 3. x-3"  (- = misread ✗)
        """
        s = s.strip()
        # Strip leading $ delimiter
        s = re.sub(r'^\$\s*', '', s)
        # Replace LaTeX cross variants with unicode ×
        s = re.sub(r'\\boldsymbol\s*\{?\s*\\times\s*\}?', '×', s)
        s = re.sub(r'\\times\b', '×', s)
        # Strip $ that appears after a mark and before a digit: "×$ 1." -> "× 1."
        s = re.sub(r'^([✓✔✗×Xx])\s*\$\s*', r'\1 ', s)
        # Strip ALL remaining $ signs (they're just LaTeX math delimiters here)
        s = s.replace('$', '').strip()
        # A leading "-" before a digit is MathPix misreading a cross mark
        s = re.sub(r'^-\s*(?=\d)', '× ', s)
        return s

    def try_opt_line(clean: str) -> bool:
        """Try to parse clean as an option line. Return True if consumed."""
        if not current: return False
        norm = _normalise_opt_line(clean)
        # marked: ✓/X/× + digit + text
        m = RE_OPT_MARKED.match(norm)
        if not m:
            m = RE_OPT_TIGHT.match(norm)
        if m:
            mark, n, txt = m.group(1), int(m.group(2)), m.group(3).strip()
            # Re-strip any $ residue from the text portion
            txt = re.sub(r'\\\\+\s*$', '', txt).strip()
            set_opt(n, _tail(txt), mark in CHECKMARKS)
            return True
        # bare digit.text — only if already in ans block
        if in_ans_block:
            m2 = RE_OPT_BARE.match(norm)
            if m2 and 1 <= int(m2.group(1)) <= 4:
                n2 = int(m2.group(1))
                txt2 = m2.group(2).strip()
                txt2 = re.sub(r'\\\\+\s*$', '', txt2).strip()
                set_opt(n2, _tail(txt2), False)
                return True
        return False

    def _parse_merged_q(line: str) -> tuple:
        """
        Handle MathPix-merged Q numbers: "Q. 231 text" -> (2, "31 text").
        Uses last_num to determine how many digits belong to the Q number.
        Returns (num, rest_text) or None.
        """
        m = RE_Q_MERGED.match(line)
        if not m: return None
        digits    = m.group(1)   # e.g. "231" or "2419"
        rest_text = m.group(2)   # e.g. "is related to 152..."
        expected  = last_num + 1
        # Try matching expected number as a prefix of digits first
        for n in (1, 2):
            if n <= len(digits) and int(digits[:n]) == expected:
                leftover = digits[n:]
                return int(digits[:n]), (leftover + " " + rest_text).strip()
        # Fallback: try 1-digit then 2-digit
        for n in (1, 2):
            if n <= len(digits):
                candidate = int(digits[:n])
                if 1 <= candidate <= 25:
                    leftover = digits[n:]
                    return candidate, (leftover + " " + rest_text).strip()
        return None

    def try_q_line(clean: str) -> bool:
        """Detect and start a question. Return True if consumed."""
        nonlocal section_just_changed, subject, last_num

        # Primary pattern: "Q. 12 text" with proper spacing
        m = RE_Q_LINE.match(clean)
        if m:
            num  = int(m.group(1))
            rest = re.sub(r'\\\\+\s*$', '', (m.group(2) or '').strip()).strip()
        else:
            # Fallback: "Q. 231 text" (MathPix merged Q-num with question text)
            parsed = _parse_merged_q(clean)
            if parsed:
                num, rest = parsed
                rest = re.sub(r'\\\\+\s*$', '', rest).strip()
            else:
                return False

        # Implicit section change: Q number drops back to 1 after reaching ~20+
        # (MathPix omitted the "Section : X" header for this section)
        _SSC_SECTION_ORDER = [
            "General Intelligence and Reasoning",
            "General Awareness",
            "Quantitative Aptitude",
            "English Comprehension",
        ]
        if num == 1 and last_num >= 20 and not section_just_changed:
            try:
                idx = _SSC_SECTION_ORDER.index(subject)
                if idx + 1 < len(_SSC_SECTION_ORDER):
                    subject = _SSC_SECTION_ORDER[idx + 1]
            except ValueError:
                pass
            section_just_changed = True
            last_num = 0

        # Accept Q if number increases, OR after section change, OR num==1
        if (num > last_num or last_num == 0 or section_just_changed or num == 1):
            start_q(num, rest)
            return True
        elif current:
            # number went backwards unexpectedly — append as question text continuation
            current.question += ("\n" if current.question else "") + rest
            return True
        return False

    # ── noise patterns to skip entirely ──────────────────────────────────────
    _SKIP_PATTERNS = [
        re.compile(r'^\\(?:begin|end)\{(?:document|center|tabular|figure|multirow|cline|hline)\b'),
        re.compile(r'^\\(?:hline|cline|multicolumn|multirow)\b'),
        re.compile(r'^\s*\\(?:usepackage|documentclass|graphicspath|setmainlanguage|setother)\b'),
        re.compile(r'^\s*\\(?:newunicodechar|IfFontExistsTF|setCJK|setDefault|setTransitions|newfontfamily)\b'),
        re.compile(r'^(?:Roll\s+Number|Candidate\s+Name|Venue\s+Name|Exam\s+(?:Date|Time)|Subject)\s*$', re.IGNORECASE),
        re.compile(r'^Combined\s+Graduate\s+Level', re.IGNORECASE),
        re.compile(r'^\\section\*\{(?:Statements?|Conclusions?|Comprehension|SubQuestion)\b', re.IGNORECASE),
        re.compile(r'^SubQuestion\s+No\s*:', re.IGNORECASE),
        re.compile(r'^(?:Question|Option\s+\d+)\s+ID\s*:', re.IGNORECASE),
        re.compile(r'^(?:Status|Chosen\s+Option)\s*:', re.IGNORECASE),
        re.compile(r'^\\end\{enumerate\}'),
        re.compile(r'^\\end\{itemize\}'),
    ]

    def is_noise(raw: str, clean: str) -> bool:
        for p in _SKIP_PATTERNS:
            if p.match(raw.strip()) or p.match(clean):
                return True
        # table content
        if clean.startswith('|') or clean.startswith('\\hline') or clean.startswith('\\cline'):
            return True
        # pure LaTeX command line
        if re.match(r'^\\[a-zA-Z]+(?:\[[^\]]*\]|\{[^}]*\})*\s*$', clean):
            return True
        return False

    # ── main loop ─────────────────────────────────────────────────────────────
    for raw_ln in lines:
        stripped = raw_ln.strip()
        clean    = _strip_bold(stripped)

        # always skip metadata noise
        if RE_META.match(clean):
            continue

        # ── Section header ────────────────────────────────────────────────────
        sec_m = RE_SEC.match(clean)
        if sec_m:
            subject = _map_ssc_section(sec_m.group(1))
            section_just_changed = True
            last_num = 0
            continue

        # ── \setcounter ───────────────────────────────────────────────────────
        sc_m = RE_SC.match(raw_ln)
        if sc_m:
            enum_counter = int(sc_m.group(1))
            continue

        # ── \begin{enumerate} / \begin{itemize} ───────────────────────────────
        if re.match(r'^\s*\\begin\{enumerate\}', raw_ln):
            in_enum = True
            continue
        if re.match(r'^\s*\\begin\{itemize\}', raw_ln):
            in_itemize = True
            continue
        if re.match(r'^\s*\\end\{(?:enumerate|itemize)\}', raw_ln):
            in_enum = in_itemize = False
            continue

        # ── \includegraphics ──────────────────────────────────────────────────
        img_matches = RE_INCLUDEGFX.findall(raw_ln)
        if img_matches:
            if current:
                for raw_id in img_matches:
                    ph = f"[IMAGE:{os.path.basename(raw_id.strip())}]"
                    if in_ans_block:
                        # Each image on its own line = a separate answer option
                        # If the last option is already non-empty, start a new slot.
                        # If it's empty (just created), fill it.
                        if not current.options or current.options[-1].strip():
                            current.options.append(ph)
                        else:
                            current.options[-1] = ph
                    else:
                        current.question += ("\n" if current.question else "") + ph
            continue

        # ── New question — HIGHEST PRIORITY — fires even inside enum/itemize ──
        # Must check BEFORE \item so that "Q. N text" inside an \item is caught
        if try_q_line(clean):
            continue

        # ── "Ans" standalone or "Ans ..." inline ─────────────────────────────
        ans_m = RE_SSC_OFF_ANS.match(clean)
        if ans_m and current:
            in_ans_block = True
            rest = re.sub(r'\\quad\s*', '', ans_m.group(1)).strip()
            if rest:
                try_opt_line(rest)
            continue

        # ── \item inside enumerate/itemize ────────────────────────────────────
        item_m = RE_ITEM.match(raw_ln)
        if item_m and current:
            item_text = _strip_bold(item_m.group(1).strip())
            item_text = re.sub(r'\\\\+\s*$', '', item_text).strip()

            # Check if the item TEXT itself starts a new question
            if try_q_line(item_text):
                continue

            if in_ans_block:
                # Numbered option item
                enum_counter += 1
                opt_num = enum_counter
                # Try marked option first
                if not try_opt_line(item_text):
                    # Plain text item — use enum_counter as option number
                    set_opt(opt_num, _tail(item_text), False)
            else:
                # Question text continuation
                current.question += ("\n" if current.question else "") + item_text
            continue

        # ── Skip structural noise ──────────────────────────────────────────────
        if not stripped:
            continue
        if is_noise(raw_ln, clean):
            continue

        # ── Option line (marked or bare) ──────────────────────────────────────
        if current and try_opt_line(clean):
            in_ans_block = True
            continue

        # ── Question text continuation ────────────────────────────────────────
        if current and not in_ans_block:
            # Skip \section*{noise}
            sec_noise = re.match(r'^\\section\*\{(.*)\}', clean)
            if sec_noise:
                inner = sec_noise.group(1).strip()
                # Only append if it looks like real question content
                if inner and not re.match(
                    r'^(?:Statements?|Conclusions?|Comprehension|SubQuestion|y\s+Tfr)',
                    inner, re.IGNORECASE
                ):
                    cont = re.sub(r'\\\\+\s*$', '', inner).strip()
                    if cont:
                        current.question += ("\n" if current.question else "") + cont
                continue
            cont = re.sub(r'\\\\+\s*$', '', clean).strip()
            if cont:
                current.question += ("\n" if current.question else "") + cont

    flush()
    return _postprocess(questions)

# ══════════════════════════════════════════════════════════
# DATA CLASS
# ══════════════════════════════════════════════════════════

@dataclass
class ParsedQuestion:
    number: int; q_type: str; subject: str; section: str
    year: str;   shift: str;  exam_date: str; question: str
    options:    list = field(default_factory=list)
    answer:     str  = ""
    solution:   str  = ""
    q_images:   list = field(default_factory=list)
    sol_images: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number, "q_type": self.q_type,
            "subject": self.subject, "section": self.section,
            "year": self.year, "shift": self.shift, "exam_date": self.exam_date,
            "question": self.question, "options": self.options,
            "answer": self.answer, "solution": self.solution,
            "q_images": self.q_images, "sol_images": self.sol_images,
            "chapter_id": None, "topic": "", "difficulty": "medium",
            "marks_correct": 4, "marks_wrong": -1, "verified": False,
        }


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def _extract_meta(title: str, body: str = "") -> dict:
    combined = (title + " " + body[:800]).strip()
    exam_date = shift = year = ""
    _mp = (r'(january|february|march|april|may|june|july|august|'
           r'september|october|november|december|jan|feb|mar|apr|'
           r'jun|jul|aug|sep|oct|nov|dec)')
    m = re.search(r'\b(\d{2})-(\d{2})-(20\d{2})\b', combined)
    if m: exam_date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    else:
        m = re.search(r'\b(\d{2})/(\d{2})/(20\d{2})\b', combined)
        if m: exam_date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        else:
            m = re.search(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_mp}\s+(20\d{{2}})\b', combined, re.I)
            if m:
                mo = _MONTH_MAP.get(m.group(2).lower(), 0)
                if mo: exam_date = f"{m.group(3)}-{mo:02d}-{int(m.group(1)):02d}"
            else:
                m = re.search(rf'\b{_mp}\s+(\d{{1,2}}),?\s+(20\d{{2}})\b', combined, re.I)
                if m:
                    mo = _MONTH_MAP.get(m.group(1).lower(), 0)
                    if mo: exam_date = f"{m.group(3)}-{mo:02d}-{int(m.group(2)):02d}"
    year = exam_date[:4] if exam_date else ""
    if not year:
        m = re.search(r'\b(20\d{2})\b', combined)
        if m: year = m.group(1)
    tl = combined.lower()
    if any(x in tl for x in ("morning","shift 1","shift-1","session 1","session-1")): shift = "Morning"
    elif any(x in tl for x in ("evening","shift 2","shift-2","session 2","session-2")): shift = "Evening"
    return {"year": year, "exam_date": exam_date, "shift": shift}


def _canon_subject(raw: str) -> str:
    s = raw.strip().upper()
    if s in ("MATHS", "MATH"): return "MATHEMATICS"
    if s in ("ZOOLOGY", "BOTANY"): return "BIOLOGY"
    return s if s in _VALID_SUBJECTS else ""


def _extract_subject_from_line(s: str) -> str:
    s = re.sub(r'\\textcolor\{[^}]+\}', '', s)
    s = re.sub(r'\\textbf\s*|\\bf\s*', '', s)
    s = re.sub(r'[{}]', '', s)
    s = re.sub(r'\*\*', '', s).strip()
    m = re.fullmatch(r'\s*(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|MATH|BIOLOGY|ZOOLOGY|BOTANY)\s*', s, re.IGNORECASE)
    if m:
        v = m.group(1).upper()
        return "MATHEMATICS" if v in ("MATHS","MATH") else v
    return ""


def _strip_bold(s: str) -> str:
    s = re.sub(r'\\textbf\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\*\*([^*]*)\*\*', r'\1', s)
    s = re.sub(r'\*\*', '', s)
    return s.strip()


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


def _parse_answer(raw: str) -> str:
    raw = re.sub(r'\*\*', '', _tail(raw)).strip()
    m = re.fullmatch(r'\((.+)\)', raw)
    if m:
        inner = m.group(1).strip()
        if not any(c in inner for c in ('\\','$','{')): raw = inner
    if re.fullmatch(r'[A-Da-d]', raw): return _NTA_LETTER[raw]
    if re.fullmatch(r'[A-Da-d](?:\s*,\s*[A-Da-d])+', raw):
        return ','.join(_NTA_LETTER[c] for c in re.findall(r'[A-Da-d]', raw))
    return raw


def _parse_answer_colon(raw: str) -> str:
    raw = re.sub(r'\*\*', '', raw).strip()
    m = re.fullmatch(r'\(\s*(.+?)\s*\)', raw)
    if m:
        inner = m.group(1).strip()
        if not any(c in inner for c in ('\\','$','{')): raw = inner
    if re.fullmatch(r'[a-dA-D]\.', raw): raw = raw[0]
    raw = raw.rstrip('°').strip()
    if re.fullmatch(r'[a-dA-D]', raw): return _ABCD_LETTER[raw.lower()]
    if re.fullmatch(r'[a-dA-D](?:\s*,\s*[a-dA-D])+', raw):
        return ','.join(_ABCD_LETTER[c.lower()] for c in re.findall(r'[a-dA-D]', raw))
    m2 = re.match(r'^(-?\d+(?:\.\d+)?)', raw)
    if m2: return m2.group(1)
    return raw


def _unique(lst: list) -> list:
    seen = set(); out = []
    for x in lst:
        if x not in seen: seen.add(x); out.append(x)
    return out


def _split_inline_options(line: str):
    markers = list(RE_INLINE_OPT_MARKER.finditer(line))
    if len(markers) < 2: return []
    nums = [int(m.group(1)) for m in markers]
    if nums[0] != 1: return []
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]+1: return []
    result = []
    for i, m in enumerate(markers):
        start = m.end()
        end   = markers[i+1].start() if i+1 < len(markers) else len(line)
        text  = line[start:end].strip()
        if re.match(r'^\d{1,3}\.\s', text): return []
        result.append((nums[i], text))
    return result


def _postprocess(questions: list) -> list:
    result = []
    for q in questions:
        sol_lines = [ln for ln in q.solution.split('\n') if not RE_NOISE_LINE.match(ln)]
        q.solution  = '\n'.join(sol_lines).strip()
        q.question  = q.question.strip()
        q.question, qi = _extract_images(q.question)
        q.solution, si = _extract_images(q.solution)
        co = []; oi = []
        for opt in q.options:
            c, o = _extract_images(opt); co.append(c); oi.extend(o)
        q.options = co
        q_ids = RE_PLACEHOLDER.findall(q.question)
        o_ids = []
        for opt in q.options: o_ids.extend(RE_PLACEHOLDER.findall(opt))
        s_ids = RE_PLACEHOLDER.findall(q.solution)
        q.q_images   = _unique(q_ids + o_ids + qi + oi)
        q.sol_images = _unique(s_ids + si)
        if q.q_type != "NUMERICAL":
            a = q.answer.strip(); al = a.lower()
            if re.fullmatch(r'\d+(?:,\d+)+', a) and q.options: q.q_type = "MSQ"
            elif re.search(r'\b[a-d]\b', al) and ',' in al: q.q_type = "MSQ"
            elif not q.options and re.fullmatch(r'-?\d+(?:\.\d+)?', a): q.q_type = "NUMERICAL"
        if q.answer and ("," in str(q.answer) or len(re.findall(r'[A-D1-4]', str(q.answer))) > 1):
            q.q_type = "MSQ"

        # Pad options to exactly 4 for MCQ questions — MathPix OCR sometimes
        # fails to extract all options from visual/figure-based questions.
        # Empty string placeholders ensure the frontend always has 4 slots.
        if q.q_type == "MCQ":
            while len(q.options) < 4:
                q.options.append("")

        d = q.to_dict()
        result.append(d)
    return result


# ══════════════════════════════════════════════════════════
# STATES
# ══════════════════════════════════════════════════════════
class S:
    IDLE="IDLE"; IN_Q="IN_Q"; IN_A="IN_A"; IN_S="IN_S"


# ══════════════════════════════════════════════════════════
# PLAIN-TEXT PARSER  (Vedantu PDF format)
# ══════════════════════════════════════════════════════════

def _parse_plain_text(text: str, subject_hint: str = "") -> list:
    """
    Parse Vedantu plain-text format. Key invariant:
    Question-start detection fires in EVERY state, including IN_S.
    This means the solution of Q1 is always terminated when Q2 begins.
    """
    lines = text.split('\n')
    questions = []
    current   = None
    state     = S.IDLE

    subject          = _canon_subject(subject_hint) if subject_hint else "PHYSICS"
    section          = "SECTION-A"
    current_q_type   = "MCQ"
    last_committed_num = 0
    in_options_block = False

    meta = _extract_meta(text[:600])
    year = meta["year"]; exam_date = meta["exam_date"]; shift = meta["shift"]

    # ── helpers ──────────────────────────────────────────
    def flush():
        nonlocal current, state, last_committed_num, in_options_block
        if current is not None:
            current.question = current.question.strip()
            current.solution = current.solution.strip()
            questions.append(current)
            last_committed_num = current.number
        current = None; state = S.IDLE; in_options_block = False

    def start_q(num, q_text):
        nonlocal current, state, in_options_block
        flush()
        current = ParsedQuestion(
            number=num, q_type=current_q_type, subject=subject,
            section=section, year=year, shift=shift,
            exam_date=exam_date, question=_tail(q_text),
        )
        state = S.IN_Q; in_options_block = False

    def is_next_q(num):
        if num < 1: return False
        eff = max(last_committed_num, current.number if current else 0)
        if eff == 0: return True
        return eff < num <= eff + 35
    def append_q(t):
        if current and t.strip():
            current.question += ("\n" if current.question else "") + _tail(t)

    def append_sol(t):
        if current and t.strip():
            current.solution += ("\n" if current.solution else "") + _tail(t)

    def set_opt_letter(letter, t):
        if not current: return
        _set_num(int(_ABCD_LETTER[letter.lower()]), t)

    def set_opt_num(n, t):
        _set_num(n, t)

    def _set_num(n, t):
        if not current: return
        while len(current.options) < n: current.options.append("")
        if not current.options[n-1]: current.options[n-1] = t
        else: current.options[n-1] += " " + t.strip()

    # ── main loop ─────────────────────────────────────────
    for raw_ln in lines:
        stripped = raw_ln.strip()
        clean    = _strip_bold(stripped)
        if not clean: continue
        if RE_ADV_SECTION_HEADER.match(clean) or "Maximum Marks" in clean or "marking scheme" in clean.lower():continue
        

        # ── P0: Subject transition ────────────────────────
        subj = _extract_subject_from_line(clean)
        if subj and not RE_QUESTION_COLON.match(clean) \
                and not RE_QUESTION_PREFIX.match(clean) \
                and not RE_PLAIN_Q.match(clean):
            if subj in ("ZOOLOGY","BOTANY"): subj = "BIOLOGY"
            flush(); subject = subj; last_committed_num = 0
            current_q_type = "MCQ"; in_options_block = False
            continue

        # ── P1: New question — fires in ANY state ─────────
        qc = RE_QUESTION_COLON.match(clean)
        if qc:
            num = int(qc.group(1)); rest = (qc.group(2) or '').strip()
            if is_next_q(num): start_q(num, rest); continue

        qp = RE_QUESTION_PREFIX.match(clean)
        if qp:
            num = int(qp.group(1)); rest = (qp.group(2) or '').strip()
            if is_next_q(num): start_q(num, rest); continue

        pq = RE_PLAIN_Q.match(clean)
        if pq:
            num = int(pq.group(1) or pq.group(3) or pq.group(4))
            rest = (pq.group(2) or '').strip()
            if is_next_q(num): start_q(num, rest); continue

        if current is None: continue

        # ── P2: "Options:" header ─────────────────────────
        if RE_OPTIONS_HEADER.match(clean):
            in_options_block = True
            if state != S.IN_Q: state = S.IN_Q
            continue

        # ── P3: Answer ────────────────────────────────────
        if state != S.IN_S:
            # NEW: JEE Advanced MSQ / Multi-answer detection
            adv_am = RE_MULTI_ANSWER.match(clean)
            if adv_am:
                current.answer = adv_am.group(1).strip()
                # If it contains commas or multiple letters/numbers, mark as MSQ
                if "," in current.answer or len(re.findall(r'[A-E1-4]', current.answer)) > 1:
                    current.q_type = "MSQ"
                state = S.IN_A
                in_options_block = False
                continue
            ac_m = RE_ANSWER_COLON.match(clean)
            if ac_m:
                raw_ans = re.sub(r'\*\*', '', ac_m.group(1)).strip()
                if not current.answer: current.answer = _parse_answer_colon(raw_ans)
                state = S.IN_A; in_options_block = False; continue

            am = RE_ANSWER.match(clean)
            if am:
                raw = (am.group(1) or am.group(2) or '').strip()
                nta = RE_NTA_ANS.match(clean)
                if nta:
                    current.answer = _parse_answer((nta.group(1) or nta.group(2) or '').strip())
                elif not current.answer:
                    current.answer = _parse_answer(raw)
                state = S.IN_A; in_options_block = False; continue

        # ── P4: Solution ──────────────────────────────────
        sc_m = RE_SOLUTION_COLON.match(clean)
        if sc_m:
            state = S.IN_S; in_options_block = False
            rest = re.sub(r'\*\*', '', sc_m.group(1)).strip()
            if rest:
                if RE_BARE_PAREN_ANS.match(rest) or re.fullmatch(r'[a-dA-D]\.?', rest):
                    if not current.answer: current.answer = _parse_answer_colon(rest)
                else: append_sol(rest)
            continue

        sm = RE_SOL.match(clean)
        if sm:
            sol_ans_m = RE_SOL_ANS.match(clean)
            if sol_ans_m:
                if not current.answer: current.answer = _parse_answer(sol_ans_m.group(1).strip())
            state = S.IN_S; in_options_block = False
            rest = _tail(sm.group(1))
            if rest: append_sol(rest)
            continue

        # ── P5: Solution body ─────────────────────────────
        # ── P5: Solution body ─────────────────────────────
        if state == S.IN_S:
            if not current.answer:
                # ADD THIS CHECK:
                hence_m = RE_HENCE_OPT.search(clean)
                if hence_m:
                    current.answer = hence_m.group(1)
                else:
                    # Existing check for bare (1)
                    bpa = RE_BARE_PAREN_ANS.match(clean)
                    if bpa: 
                        current.answer = _parse_answer_colon(clean)
                        continue
            
            append_sol(raw_ln)
            continue

        # ── P6: Answer state — wait ───────────────────────
        if state == S.IN_A: continue

        # ── P7: Options and question text (IN_Q) ──────────
        if state == S.IN_Q:
            # Inline multi
            inline = _split_inline_options(clean)
            if inline:
                in_options_block = True
                for n, t in inline: set_opt_num(n, _tail(t))
                continue

            # (a) form — PRIMARY Vedantu format
            om_p = RE_OPTION_PAREN_ABCD.match(clean)
            if om_p:
                set_opt_letter(om_p.group(1), _tail(om_p.group(2)))
                in_options_block = True; continue

            # (1) form
            om = RE_OPTION.match(clean)
            if om:
                set_opt_num(int(om.group(1)), _tail(om.group(2)))
                in_options_block = True; continue

            # a. form
            om_d = RE_OPTION_ABCD.match(clean)
            if om_d:
                set_opt_letter(om_d.group(1), _tail(om_d.group(2)))
                in_options_block = True; continue

            # continuation of previous option
            if in_options_block and current.options:
                current.options[-1] += " " + clean; continue

            # question text
            append_q(raw_ln)

    flush()
    return questions


# ══════════════════════════════════════════════════════════
# LATEX PARSER (original MathPix .tex format)
# ══════════════════════════════════════════════════════════

def parse_tex(tex_path: str, subject_hint: str = "") -> list:
    """Parse a MathPix .tex file and return list of question dicts."""
    with open(tex_path, encoding="utf-8") as f:
        content = f.read()
    lines = [ln.rstrip() for ln in content.split('\n')]

    # Auto-detect format
    has_latex   = any(r'\begin{document}' in ln or r'\item' in ln for ln in lines[:50])
    has_vedantu = bool(re.search(r'^\*?\*?Question\s+\d+\s*:', content, re.MULTILINE|re.IGNORECASE))
    # SSC CGL official MathPix format: Q. <n> questions + ✓ checkmarks
    if _is_ssc_official_tex(content):
        return parse_ssc_official_tex(content)
    if has_vedantu and not has_latex:
        return parse_plain_pdf_text(content, subject_hint)

    subject = _canon_subject(subject_hint) if subject_hint else "PHYSICS"
    section = "SECTION-A"
    year = shift = exam_date = ""
    _body_buf = ""; _body_done = False

    questions = []; current = None; state = S.IDLE
    in_doc = False; in_noise_block = False; last_section_was_noise = False
    current_q_type = "MCQ"; in_options_block = False
    enum_depth = 0; itemize_depth = 0; tabular_depth = 0
    in_figure = False; fig_caption = ""; fig_img_id = ""
    last_committed_num = 0; pending_setcounter = None

    def flush():
        nonlocal current, state, last_committed_num, in_options_block
        if current:
            current.question = current.question.strip()
            current.solution = current.solution.strip()
            questions.append(current)
            last_committed_num = current.number
        current = None; state = S.IDLE; in_options_block = False

    def start_q(num, text):
        nonlocal current, state, in_options_block
        flush()
        current = ParsedQuestion(
            number=num, q_type=current_q_type, subject=subject,
            section=section, year=year, shift=shift,
            exam_date=exam_date, question=_tail(text),
        )
        state = S.IN_Q; in_options_block = False

    def append_q(t):
        if current and t.strip():
            current.question += ("\n" if current.question else "") + _tail(t)

    def append_sol(t):
        if current and t.strip():
            current.solution += ("\n" if current.solution else "") + _tail(t)

    def set_opt(n, t):
        if not current: return
        while len(current.options) < n: current.options.append("")
        if not current.options[n-1]: current.options[n-1] = t
        else: current.options[n-1] += " " + t.strip()

    def is_next_q(num, from_setcounter=False):
        if num < 1: return False
        eff = max(last_committed_num, current.number if current else 0)
        if from_setcounter: return num > eff
        if eff == 0 and num == 1: return True
        if eff == 0 and num > 35: return True
        return eff < num <= eff + 35

    for ln in lines:
        stripped = ln.strip()

        if RE_BEGIN_DOC.match(ln): in_doc = True; continue
        if RE_END_DOC.match(ln): flush(); break

        m = RE_TITLE.match(ln)
        if m:
            title = m.group(1)
            meta  = _extract_meta(title)
            year, exam_date, shift = meta["year"], meta["exam_date"], meta["shift"]
            if subject == "PHYSICS":
                _tm = RE_SUBJECT_BROAD.search(title)
                if _tm:
                    _tc = _canon_subject(_tm.group(1))
                    if _tc: subject = _tc
            continue

        if not in_doc: continue

        if not _body_done:
            _body_buf += ln + "\n"
            if len(_body_buf) >= 800: _body_done = True
            m2 = _extract_meta("", _body_buf)
            if m2["exam_date"]: exam_date = m2["exam_date"]
            if m2["shift"]: shift = m2["shift"]
            if m2["year"]: year = m2["year"]
            if subject == "PHYSICS" and stripped:
                _bm = RE_SUBJECT_BROAD.search(stripped)
                if _bm:
                    _bc = _canon_subject(_bm.group(1))
                    if _bc: subject = _bc

        if not stripped: continue

        clean = _strip_bold(stripped)

        _subj = _extract_subject_from_line(clean)
        if _subj and not re.search(r'Question\s+\d+', clean, re.I):
            flush(); subject = _subj; last_committed_num = 0
            pending_setcounter = None; current_q_type = "MCQ"; in_options_block = False
            continue

        if RE_BEGIN_FIGURE.match(ln): in_figure=True; fig_caption=""; fig_img_id=""; continue
        if RE_END_FIGURE.match(ln):
            if in_figure and fig_img_id:
                cm = re.fullmatch(r'\(([1-4])\)', fig_caption.strip())
                if cm and current: set_opt(int(cm.group(1)), f"[IMAGE:{fig_img_id}]")
                elif current: (append_sol if state==S.IN_S else append_q)(f"[IMAGE:{fig_img_id}]")
            in_figure=False; fig_caption=""; fig_img_id=""; continue
        if in_figure:
            cm = RE_CAPTION.search(ln)
            if cm: fig_caption = cm.group(1).strip()
            gm = RE_INCLUDEGFX.search(ln)
            if gm: fig_img_id = os.path.basename(gm.group(1).strip())
            continue

        if RE_BEGIN_CENTER.match(ln) or RE_END_CENTER.match(ln): continue

        if RE_BEGIN_TABULAR.match(ln):
            tabular_depth += 1
            if current: (append_sol if state==S.IN_S else append_q)(ln.strip())
            continue
        if RE_END_TABULAR.match(ln):
            tabular_depth = max(0, tabular_depth-1)
            if current: (append_sol if state==S.IN_S else append_q)(r'\end{tabular}')
            continue
        if tabular_depth > 0:
            if current:
                if RE_INCLUDEGFX.search(ln):
                    for raw_id in RE_INCLUDEGFX.findall(ln):
                        ph = f"[IMAGE:{os.path.basename(raw_id.strip())}]"
                        (append_sol if state==S.IN_S else append_q)(ph)
                    continue
                (append_sol if state==S.IN_S else append_q)(ln.strip())
            continue

        m = RE_SECTION.match(ln)
        if m:
            sec_text = m.group(1).strip()

            if RE_CORRECT_SOL_NOISE.match(sec_text):
                if current and state==S.IN_S: append_sol(sec_text)
                continue

            am = RE_ANSWER.match(sec_text)
            if am and current and state != S.IN_S:
                nta = RE_NTA_ANS.match(sec_text)
                if nta:
                    current.answer = _parse_answer((nta.group(1) or nta.group(2) or '').strip())
                elif not current.answer:
                    current.answer = _parse_answer((am.group(1) or am.group(2) or '').strip())
                state = S.IN_A; in_options_block = False; continue

            sm = RE_SOL.match(sec_text)
            if sm:
                last_section_was_noise = False
                _rest_raw = _tail(sm.group(1).strip())
                _clean_for_match = f"Sol. {_rest_raw}".strip()
                sol_ans_m = RE_SOL_ANS.match(_clean_for_match)
                _paren_opt = RE_SOL_PAREN_OPT.match(_rest_raw)
                _bare_paren = RE_BARE_PAREN_ANS.match(_rest_raw) if _rest_raw else None
                _multi_paren = RE_MULTI_PAREN_ANS.match(_rest_raw) if _rest_raw else None
                _lead_int = RE_SOL_PAREN_INT_LEAD.match(_rest_raw) if _rest_raw else None
                if current:
                    if sol_ans_m:
                        if not current.answer:
                            current.answer = _parse_answer(sol_ans_m.group(1).strip())
                    elif _paren_opt:
                        if not current.answer:
                            current.answer = _paren_opt.group(1)
                    elif _bare_paren or _multi_paren:
                        if not current.answer:
                            current.answer = _parse_answer(_rest_raw)
                    elif _lead_int:
                        if not current.answer:
                            current.answer = _lead_int.group(1)
                    state = S.IN_S
                    if _rest_raw and not sol_ans_m and not _paren_opt and not _bare_paren and not _multi_paren:
                        append_sol(_rest_raw)
                    in_options_block = False
                continue

            sc_m = RE_SOLUTION_COLON.match(sec_text)
            if sc_m:
                last_section_was_noise = False
                if current:
                    state = S.IN_S; in_options_block = False
                    rest = re.sub(r'\*\*','',sc_m.group(1)).strip()
                    if rest:
                        if RE_BARE_PAREN_ANS.match(rest) or re.fullmatch(r'[a-dA-D]\.?',rest):
                            if not current.answer: current.answer = _parse_answer_colon(rest)
                        else: append_sol(rest)
                continue

            ac_m = RE_ANSWER_COLON.match(sec_text)
            if ac_m and current and state != S.IN_S:
                if not current.answer: current.answer = _parse_answer_colon(re.sub(r'\*\*','',ac_m.group(1)).strip())
                state = S.IN_A; in_options_block = False; continue

            if _is_noise(sec_text): last_section_was_noise = True; continue

            last_section_was_noise = False

            subj_m = RE_SUBJECT.search(sec_text)
            if subj_m:
                flush(); subject = subj_m.group(1).upper()
                if subject in ("MATHS","MATH"): subject = "MATHEMATICS"
                if subject in ("ZOOLOGY","BOTANY"): subject = "BIOLOGY"
                last_committed_num=0; pending_setcounter=None
                current_q_type="MCQ"; in_options_block=False
                # ── FIXED: reset section back to SECTION-A on new subject ────
                section = "SECTION-A"
                continue

            # ── FIXED: "SECTION - A" resets type to MCQ ─────────────────────
            if RE_SECTION_A_PAT.match(sec_text):
                flush(); section = "SECTION-A"; current_q_type = "MCQ"
                last_committed_num = 0  # reset so Q1 is accepted after instructions block
                continue

            # SECTION-2 / SECTION - 2 = MSQ (multiple select), still MCQ q_type
            if re.match(r'^SECTION\s*[-–]?\s*2\b', sec_text, re.IGNORECASE):
                flush(); section="SECTION-A"; current_q_type="MCQ"; continue

            # ── FIXED: "SECTION - B" / INTEGER / NUMERICAL type ─────────────
            if RE_NUMERICAL_SEC.search(sec_text):
                flush(); section="SECTION-B"; current_q_type="NUMERICAL"; continue

            # Reset on SECTION - 1 / SECTION 1 style headers (paper sections)
            if re.match(r'^SECTION\s*[-–]?\s*1\b', sec_text, re.IGNORECASE) or \
               re.match(r'^SECTION\s*1\b', sec_text, re.IGNORECASE):
                flush(); section="SECTION-A"; current_q_type="MCQ"
                last_committed_num = 0
                continue

            # Guard: do NOT append plain section text (like stray headers) to
            # the current solution. Only append if it looks like actual content.
            if current and state==S.IN_S:
                # Skip bare section/header noise that wasn't caught above
                if not re.match(r'^SECTION', sec_text, re.IGNORECASE):
                    append_sol(sec_text)
            continue

        if RE_BEGIN_ITEMIZE.match(ln): itemize_depth+=1; continue
        if RE_END_ITEMIZE.match(ln): itemize_depth=max(0,itemize_depth-1); continue
        if RE_BEGIN_ENUM.match(ln):
            if last_section_was_noise: in_noise_block=True
            else: enum_depth+=1
            pending_setcounter=None; continue
        if RE_END_ENUM.match(ln):
            if in_noise_block: in_noise_block=False
            else: enum_depth=max(0,enum_depth-1)
            continue
        m = RE_SETCOUNTER.match(ln)
        if m:
            # Accept setcounter even when inside itemize (nested itemize+enumerate pattern)
            pending_setcounter=int(m.group(1))
            continue
        if in_noise_block: continue

        # Question detection (fires in ANY state)
        m = RE_ITEM.match(ln)
        if m:
            # itemize_depth > 0 but enum_depth > 0 means we're in the nested
            # \begin{itemize}\item\begin{enumerate}\item Q... pattern — treat as question
            if itemize_depth > 0 and enum_depth == 0:
                if current and state==S.IN_S:
                    rest = m.group(1).strip()
                    if rest: append_sol(rest)
                continue
            if enum_depth > 0 or pending_setcounter is not None:
                from_sc = pending_setcounter is not None
                if pending_setcounter is not None: q_num = pending_setcounter+1
                else:
                    eff = max(last_committed_num, current.number if current else 0)
                    q_num = eff+1
                pending_setcounter = None
                if is_next_q(q_num, from_setcounter=from_sc): start_q(q_num, m.group(1).strip())
            continue

        pq = RE_PLAIN_Q.match(clean)
        if pq:
            num = int(pq.group(1) or pq.group(3) or pq.group(4)); rest=(pq.group(2) or '').strip()
            if is_next_q(num): start_q(num,rest); pending_setcounter=None; continue

        qp = RE_QUESTION_PREFIX.match(clean)
        if qp:
            num=int(qp.group(1)); rest=(qp.group(2) or '').strip()
            if is_next_q(num): start_q(num,rest); pending_setcounter=None; continue

        qc = RE_QUESTION_COLON.match(clean)
        if qc:
            num=int(qc.group(1)); rest=(qc.group(2) or '').strip()
            if is_next_q(num): start_q(num,rest); pending_setcounter=None; continue

        if RE_OPTIONS_HEADER.match(clean):
            in_options_block=True
            if state!=S.IN_Q: state=S.IN_Q
            continue

        if RE_CORRECT_SOL_NOISE.match(clean) and current:
            if state==S.IN_S: append_sol(clean)
            continue

        am = RE_ANSWER.match(clean)
        if am and current and state!=S.IN_S:
            nta = RE_NTA_ANS.match(clean)
            if nta:
                current.answer = _parse_answer((nta.group(1) or nta.group(2) or '').strip())
            elif not current.answer:
                current.answer = _parse_answer((am.group(1) or am.group(2) or '').strip())
            state=S.IN_A; in_options_block=False; continue

        ac_m = RE_ANSWER_COLON.match(clean)
        if ac_m and current and state!=S.IN_S:
            if not current.answer: current.answer = _parse_answer_colon(re.sub(r'\*\*','',ac_m.group(1)).strip())
            state=S.IN_A; in_options_block=False; continue

        sm = RE_SOL.match(clean)
        if sm and current:
            _rest_raw = _tail(sm.group(1).strip())
            _clean_for_match = f"Sol. {_rest_raw}".strip()
            sol_ans_m = RE_SOL_ANS.match(_clean_for_match)
            _paren_opt = RE_SOL_PAREN_OPT.match(_rest_raw)
            _bare_paren = RE_BARE_PAREN_ANS.match(_rest_raw) if _rest_raw else None
            _multi_paren = RE_MULTI_PAREN_ANS.match(_rest_raw) if _rest_raw else None
            _lead_int = RE_SOL_PAREN_INT_LEAD.match(_rest_raw) if _rest_raw else None
            if sol_ans_m:
                if not current.answer: current.answer = _parse_answer(sol_ans_m.group(1).strip())
            elif _paren_opt:
                if not current.answer: current.answer = _paren_opt.group(1)
            elif _bare_paren or _multi_paren:
                if not current.answer: current.answer = _parse_answer(_rest_raw)
            elif _lead_int:
                if not current.answer: current.answer = _lead_int.group(1)
            state=S.IN_S; in_options_block=False
            if _rest_raw and not sol_ans_m and not _paren_opt and not _bare_paren and not _multi_paren:
                append_sol(_rest_raw)
            continue

        sc_m = RE_SOLUTION_COLON.match(clean)
        if sc_m and current:
            state=S.IN_S; in_options_block=False
            rest = re.sub(r'\*\*','',sc_m.group(1)).strip()
            if rest:
                if RE_BARE_PAREN_ANS.match(rest) or re.fullmatch(r'[a-dA-D]\.?',rest):
                    if not current.answer: current.answer = _parse_answer_colon(rest)
                else: append_sol(rest)
            continue

        if current and state==S.IN_Q:
            inline = _split_inline_options(clean)
            if inline:
                in_options_block=True
                for n,t in inline: set_opt(n, _tail(t))
                continue
            om_p = RE_OPTION_PAREN_ABCD.match(clean)
            if om_p:
                n = int(_ABCD_LETTER[om_p.group(1).lower()])
                set_opt(n, _tail(om_p.group(2))); in_options_block=True; continue
            om = RE_OPTION.match(clean)
            if om: set_opt(int(om.group(1)), _tail(om.group(2))); in_options_block=True; continue
            om_d = RE_OPTION_ABCD.match(clean)
            if om_d:
                n = int(_ABCD_LETTER[om_d.group(1).lower()])
                set_opt(n, _tail(om_d.group(2))); in_options_block=True; continue

        if RE_INCLUDEGFX.search(ln) and current:
            for raw_id in RE_INCLUDEGFX.findall(ln):
                ph = f"[IMAGE:{os.path.basename(raw_id.strip())}]"
                if state==S.IN_S: append_sol(ph)
                elif state==S.IN_Q:
                    if current.options and not current.options[-1].strip(): current.options[-1]=ph
                    else: append_q(ph)
                else: append_q(ph)
            continue

        solo = re.match(r'^\(([1-4])\)\\*\s*$', clean)
        if solo and current and state==S.IN_Q: set_opt(int(solo.group(1)),""); continue

        if state==S.IN_S and current and not current.answer:
            bpa = RE_BARE_PAREN_ANS.match(clean)
            if bpa: current.answer = _parse_answer_colon(clean); continue

        if not current: continue
        if state==S.IN_S: append_sol(ln)
        elif state==S.IN_Q:
            if in_options_block and current.options: current.options[-1]+=" "+clean
            else: append_q(ln)

    flush()
    return _postprocess(questions)


# ══════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════

def parse_plain_pdf_text(text: str, subject_hint: str = "") -> list:
    """
    Parse plain text extracted from a Vedantu-style NEET/JEE PDF.
    Returns list of question dicts ready for frontend admin.
    """
    questions = _parse_plain_text(text, subject_hint)
    return _postprocess(questions)


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python parser.py <file.tex|file.txt> [subject_hint]"); sys.exit(1)
    hint = sys.argv[2] if len(sys.argv) > 2 else ""
    fp   = sys.argv[1]
    if fp.endswith('.txt') or fp.endswith('.md'):
        with open(fp, encoding='utf-8') as f: content = f.read()
        qs = parse_plain_pdf_text(content, subject_hint=hint)
    else:
        qs = parse_tex(fp, subject_hint=hint)
    print(json.dumps(qs, indent=2, ensure_ascii=False))
    print(f"\n✓ Parsed {len(qs)} questions", file=sys.stderr)
