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
RE_SETCOUNTER    = re.compile(r'^\s*\\setcounter\{enumi\}\{(\d+)\}')
RE_ITEM          = re.compile(r'^\s*\\item\s*(.*)')
RE_SECTION       = re.compile(r'^\s*\\section\*\{(.+?)\}')
RE_INCLUDEGFX    = re.compile(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}')
RE_PLACEHOLDER   = re.compile(r'\[IMAGE:([^\]]+)\]')

RE_SUBJECT = re.compile(
    r'(?:PART[-\s]*[A-Za-z]\s*:\s*)?(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|MATH|BIOLOGY)',
    re.IGNORECASE,
)
RE_SUBJECT_BROAD = re.compile(
    r'\b(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|MATH|BIOLOGY)\b',
    re.IGNORECASE,
)
RE_NUMERICAL_SEC = re.compile(
    r'SECTION[-\s]?(?:B|2|II)|INTEGER\s*(?:TYPE|ANSWER)?|NUMERICAL\s*(?:VALUE|TYPE|ANSWER)?',
    re.IGNORECASE,
)

# Question start patterns
RE_PLAIN_Q = re.compile(
    r'^(\d{1,3})\.\s+(\S.*)'
    r'|^(\d{1,3})\.\s*\\\\?\s*$'
    r'|^(\d{1,3})\.\s*$'
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

# Solution patterns
RE_SOL          = re.compile(r'^Sol\.\s*(.*)', re.IGNORECASE)
RE_SOL_ANS      = re.compile(
    r'^Sol\.\s+(\d+(?:\.\d+)?)(?:\s*(?:%|cc|cm|m|kg|s|V|J|N|eV|K|Hz|mol|kbar|mN|rpm|nm|mm|pm))?\s*$',
    re.IGNORECASE
)
RE_SOLUTION_COLON  = re.compile(r'^\*?\*?Solution\s*:\s*\*?\*?(.*)$', re.IGNORECASE)
RE_CORRECT_SOL_NOISE = re.compile(r'^Correct\s+(?:Option|Answer)\s*[:(]', re.IGNORECASE)
RE_BARE_PAREN_ANS  = re.compile(r'^\(\s*([a-dA-D]|-?\d+(?:\.\d+)?°?)\s*\)\s*$')

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
]

_MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
    'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,
    'sep':9,'oct':10,'nov':11,'dec':12,
}
_VALID_SUBJECTS = {"PHYSICS","CHEMISTRY","MATHEMATICS","BIOLOGY"}

RE_NOISE_LINE = re.compile(r'^\s*\\setcounter\{enum[iIvV]+\}\{[^}]+\}\s*$')

# ── Detect "SECTION - A / B" header lines as they appear in these PDFs ──────
RE_SECTION_HEADER = re.compile(
    r'^\s*SECTION\s*[-–—]?\s*([AB12])\s*$',
    re.IGNORECASE,
)

# ── Detect page header/footer lines like "26th Feb. 2021 | Shift - 1" ───────
# These appear at subject boundaries and must be treated as hard boundaries,
# not appended to solution text.
RE_PAGE_HEADER = re.compile(
    r'\b(20\d{2})\b.*\bshift\b|\bshift\b.*\b(20\d{2})\b'
    r'|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+20\d{2}\b',
    re.IGNORECASE,
)


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
            # Allow optional dot after month abbrev: "26th Feb. 2021" or "26th Feb 2021"
            m = re.search(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_mp}\.?\s+(20\d{{2}})\b', combined, re.I)
            if m:
                mo = _MONTH_MAP.get(m.group(2).lower(), 0)
                if mo: exam_date = f"{m.group(3)}-{mo:02d}-{int(m.group(1)):02d}"
            else:
                m = re.search(rf'\b{_mp}\.?\s+(\d{{1,2}}),?\s+(20\d{{2}})\b', combined, re.I)
                if m:
                    mo = _MONTH_MAP.get(m.group(1).lower(), 0)
                    if mo: exam_date = f"{m.group(3)}-{mo:02d}-{int(m.group(2)):02d}"
    year = exam_date[:4] if exam_date else ""
    if not year:
        m = re.search(r'\b(20\d{2})\b', combined)
        if m: year = m.group(1)
    tl = combined.lower()
    # Match "Shift - 1", "Shift-1", "Shift 1", "Session 1" etc. with optional spaces around dash
    if any(x in tl for x in ("morning",)) or \
       re.search(r'shift\s*[-–]?\s*1\b|session\s*[-–]?\s*1\b', tl): shift = "Morning"
    elif re.search(r'shift\s*[-–]?\s*2\b|session\s*[-–]?\s*2\b', tl): shift = "Evening"
    return {"year": year, "exam_date": exam_date, "shift": shift}


def _canon_subject(raw: str) -> str:
    s = raw.strip().upper()
    if s in ("MATHS", "MATH"): return "MATHEMATICS"
    return s if s in _VALID_SUBJECTS else ""


def _extract_subject_from_line(s: str) -> str:
    s = re.sub(r'\\textcolor\{[^}]+\}', '', s)
    s = re.sub(r'\\textbf\s*|\\bf\s*', '', s)
    s = re.sub(r'[{}]', '', s)
    s = re.sub(r'\*\*', '', s).strip()
    m = re.fullmatch(r'\s*(PHYSICS|CHEMISTRY|MATHEMATICS|MATHS|MATH|BIOLOGY)\s*', s, re.IGNORECASE)
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

    # Post-process: when two consecutive slots are (empty, non-empty), the content
    # likely belongs to both — split it at the midpoint.  This happens when PDFs
    # render "(3) (4) content" where content straddles both options.
    final = []
    k = 0
    while k < len(result):
        num, text = result[k]
        if (text == '' and k + 1 < len(result) and
                result[k+1][0] == num + 1 and result[k+1][1]):
            # Next slot has all the content; split it by whitespace midpoint
            combined = result[k+1][1]
            tokens = combined.split()
            mid = max(1, len(tokens) // 2)
            final.append((num,       ' '.join(tokens[:mid])))
            final.append((num + 1,   ' '.join(tokens[mid:])))
            k += 2
            continue
        final.append((num, text))
        k += 1
    return final


# Regex to detect a line that is ONLY option-number markers, optionally with
# very short per-option separators (e.g. "(1)  (2)  (3)  (4)" or
# "(1) × (2) × (3) × (4) ×" where each chunk between markers is ≤ 4 chars).
_RE_MARKER_LINE = re.compile(
    r'^\s*\(1\)(?P<c1>[^(]*)\(2\)(?P<c2>[^(]*)\(3\)(?P<c3>[^(]*)\(4\)(?P<c4>[^(]*)\s*$'
)

def _is_marker_dominant(line: str) -> tuple:
    """
    Returns (True, c1,c2,c3,c4) if the line is a '(1)..(2)..(3)..(4)' marker line
    where the content between markers is short enough to be a separator/partial.
    Content slots are "short" if they have no alphabetic words longer than 3 chars
    AND the total non-marker content is ≤ 40% of the line length.
    Returns (False,...) otherwise.
    """
    m = _RE_MARKER_LINE.match(line.strip())
    if not m:
        return (False, '', '', '', '')
    c1, c2, c3, c4 = m.group('c1'), m.group('c2'), m.group('c3'), m.group('c4')
    chunks = [c1, c2, c3, c4]
    total_content = sum(len(c.strip()) for c in chunks)
    marker_overhead = len('(1)(2)(3)(4)')
    line_len = len(line.strip())
    # If content chars are more than 50% of line, it's a real inline option line
    if line_len > 0 and total_content / line_len > 0.5:
        return (False, c1, c2, c3, c4)
    # If any chunk has a long word (>4 chars), it carries real content
    for c in chunks:
        words = re.findall(r'[A-Za-z]+', c)
        if any(len(w) > 4 for w in words):
            return (False, c1, c2, c3, c4)
    return (True, c1.strip(), c2.strip(), c3.strip(), c4.strip())


def _split_prev_into_4(prev: str) -> list:
    """
    Split a line of space-separated tokens into 4 roughly equal parts.
    Returns list of 4 strings.
    """
    tokens = prev.split()
    n = len(tokens)
    if n == 0:
        return ['', '', '', '']
    if n <= 4:
        # Pad with empty strings
        result = tokens + [''] * (4 - n)
        return result
    # Distribute as evenly as possible
    chunk = n // 4
    extra = n % 4
    parts = []
    start = 0
    for i in range(4):
        size = chunk + (1 if i < extra else 0)
        parts.append(' '.join(tokens[start:start+size]))
        start += size
    return parts


def _preprocess_split_options(lines: list) -> list:
    """
    Pre-pass that normalises several broken option-line patterns into standard
    "(1) opt1  (2) opt2  (3) opt3  (4) opt4" lines before the main parser runs.

    Handled patterns
    ────────────────
    A. Fraction-style 2-3 line layout:
           "ρK  K  PK  ρP"          ← numerators
           "(1)  (2)  (3)  (4)"     ← marker-only (or near-empty) line
           "P  ρP  ρ  K"            ← denominators

    B. Two adjacent rows:
           "(1) text1  (2) text2"
           "(3) text3  (4) text4"

    C. Paired-marker with multi-line content between:
           "(1) (2)"                ← only markers 1 & 2 (with optional text)
           ...continuation lines...
           "(3) (4) [text]"         ← markers 3 & 4 (within a window of ≤12 lines)
       Content lines between each pair of markers are joined into opt1/opt2 and opt3/opt4.
    """
    # ── helpers ──────────────────────────────────────────────────────────────
    _RE_PAIR = re.compile(r'^\s*\(([1-4])\)\s+\(([1-4])\)\s*(.*)\s*$')
    _RE_ANS  = re.compile(r'^(?:Ans|Sol)\.', re.IGNORECASE)

    def _is_structural(s):
        return bool(RE_PLAIN_Q.match(s) or RE_SOL.match(s) or
                    RE_ANSWER.match(s) or RE_SECTION_HEADER.match(s) or
                    RE_SOLUTION_COLON.match(s) or _RE_ANS.match(s) or not s)

    def _markers_in(s):
        return sorted(set(int(x) for x in re.findall(r'\(([1-4])\)', s)))

    out = []
    i = 0
    while i < len(lines):
        ln      = lines[i]
        stripped = ln.strip()

        # ── Pattern A: marker-dominant (1)(2)(3)(4) line ─────────────────────
        is_marker, c1, c2, c3, c4 = _is_marker_dominant(stripped)
        if is_marker:
            prev = out[-1].strip() if out else ""
            nxt  = lines[i+1].strip() if i+1 < len(lines) else ""

            if not _is_structural(prev):
                out.pop()
                prev_parts = _split_prev_into_4(prev)

                nxt_is_marker  = _is_marker_dominant(nxt)[0]
                has_denom = not _is_structural(nxt) and not nxt_is_marker
                nxt_parts = _split_prev_into_4(nxt) if has_denom else ['','','','']

                def _comb(a, b, c):
                    return ' '.join(p for p in [a, b, c] if p.strip())

                merged = (f"(1) {_comb(prev_parts[0],c1,nxt_parts[0])}  "
                          f"(2) {_comb(prev_parts[1],c2,nxt_parts[1])}  "
                          f"(3) {_comb(prev_parts[2],c3,nxt_parts[2])}  "
                          f"(4) {_comb(prev_parts[3],c4,nxt_parts[3])}")
                out.append(merged)
                i += 2 if has_denom else 1
                continue
            out.append(ln); i += 1; continue

        # ── Pattern B: two adjacent rows (1)(2) / (3)(4), each possibly followed
        #              by subscript/continuation lines with no option markers ──
        mk = _markers_in(stripped)
        if mk == [1, 2] and i + 1 < len(lines):
            row1 = stripped
            j = i + 1
            # Collect subscript/continuation lines after row1 (up to 5 lines,
            # stop at structural or a line that contains any (1-4) markers)
            sub1_lines = []
            while j < len(lines) and j - i <= 6:
                s = lines[j].strip()
                if not s or _is_structural(s):
                    break
                mk_s = _markers_in(s)
                if mk_s:  # has any option markers — could be (3)(4) row
                    break
                sub1_lines.append(s)
                j += 1
            # Now check if this line is the (3)(4) row
            if j < len(lines):
                row2_raw = lines[j].strip()
                if _markers_in(row2_raw) == [3, 4]:
                    row2 = row2_raw
                    j += 1
                    # Collect subscripts after (3)(4) row
                    sub2_lines = []
                    while j < len(lines) and j - (j-1) <= 5:
                        s = lines[j].strip()
                        if not s or _is_structural(s) or _markers_in(s):
                            break
                        sub2_lines.append(s)
                        j += 1
                    # Build merged rows with their subscripts appended
                    r1 = ' '.join([row1] + sub1_lines).strip()
                    r2 = ' '.join([row2] + sub2_lines).strip()
                    out.append(r1 + '  ' + r2)
                    i = j
                    continue

        # ── Pattern C: paired-marker with multi-line option content ───────────
        # Detect "(1) (2)" or "(1) (2) extra" — exactly markers 1 and 2,
        # no markers 3 or 4 on the same line.
        pm = _RE_PAIR.match(stripped)
        if pm:
            na, nb = int(pm.group(1)), int(pm.group(2))
            if nb == na + 1 and na in (1, 3):
                # Collect content lines until we hit the complementary pair or a
                # structural line, within a window of 15 lines
                first_extra = pm.group(3).strip()  # text on same line after markers
                content_a = [first_extra] if first_extra else []
                content_b = []
                j = i + 1
                found_pair2 = -1
                pair2_extra = ''
                while j < len(lines) and j - i <= 15:
                    s = lines[j].strip()
                    if _is_structural(s):
                        break
                    pm2 = _RE_PAIR.match(s)
                    if pm2:
                        nc, nd = int(pm2.group(1)), int(pm2.group(2))
                        if nc == na + 2 and nd == na + 3:
                            found_pair2 = j
                            pair2_extra = pm2.group(3).strip()
                            break
                    j += 1

                if found_pair2 >= 0:
                    # Content lines between first pair and second pair belong to opt1+opt2
                    between = [lines[k].strip() for k in range(i+1, found_pair2)
                               if lines[k].strip() and not _is_structural(lines[k].strip())]
                    # Split between lines: first half → opt_a, second half → opt_b
                    mid = len(between) // 2
                    opt_a = ' '.join(filter(None, content_a + between[:mid])).strip()
                    opt_b = ' '.join(filter(None, between[mid:])).strip()

                    # Collect content after second pair marker
                    content_c = [pair2_extra] if pair2_extra else []
                    content_d = []
                    k = found_pair2 + 1
                    while k < len(lines) and k - found_pair2 <= 10:
                        s = lines[k].strip()
                        if _is_structural(s) or _RE_PAIR.match(s):
                            break
                        content_c.append(s)
                        k += 1
                    mid2 = len(content_c) // 2
                    opt_c = ' '.join(filter(None, content_c[:max(1,mid2)])).strip()
                    opt_d = ' '.join(filter(None, content_c[max(1,mid2):])).strip()

                    merged = (f"({na}) {opt_a}  ({nb}) {opt_b}  "
                              f"({na+2}) {opt_c}  ({nb+2}) {opt_d}")
                    out.append(merged)
                    i = k
                    continue
                # No matching pair2 found — fall through to normal emit

        out.append(ln)
        i += 1
    return out


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
        result.append(q.to_dict())
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

    FIX: When a SECTION - B / NUMERICAL section header is encountered,
    last_committed_num is reset to 0 so that Q1 of the new section is
    accepted even though Section A already emitted 20 questions.
    """
    lines = text.split('\n')
    lines = _preprocess_split_options(lines)   # normalise split-line options
    questions = []
    current   = None
    state     = S.IDLE

    subject          = _canon_subject(subject_hint) if subject_hint else "PHYSICS"
    section          = "SECTION-A"
    current_q_type   = "MCQ"
    last_committed_num = 0
    in_options_block = False
    _prev_was_page_header = False   # set True after a RE_PAGE_HEADER line

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
        if eff == 0 and num == 1: return True
        if eff == 0 and num > 35: return True   # Chemistry Q46, Biology Q91
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

        # ── P-2: Page header/footer (e.g. "26th Feb. 2021 | Shift - 1") ──────
        # These lines appear at every subject boundary in Vedantu PDFs.
        # They must be treated as hard boundaries: flush current question,
        # update meta from the line, then skip — never append to solution.
        if RE_PAGE_HEADER.search(clean):
            flush()
            _prev_was_page_header = True
            # Re-extract meta from this line to capture date/shift
            _m2 = _extract_meta(clean)
            if _m2["exam_date"]: exam_date = _m2["exam_date"]
            if _m2["shift"]:     shift     = _m2["shift"]
            if _m2["year"]:      year      = _m2["year"]
            continue

        # ── P-1: Bare "SECTION - A/B" header ─────────────────────────────────
        # Handles plain "SECTION - B" lines that appear between subjects in
        # these Vedantu PDFs.  Must be checked BEFORE subject-line detection
        # so we don't accidentally treat it as question text.
        sh_m = RE_SECTION_HEADER.match(clean)
        if sh_m and not RE_PLAIN_Q.match(clean) \
                and not RE_QUESTION_COLON.match(clean) \
                and not RE_QUESTION_PREFIX.match(clean):
            _prev_was_page_header = False
            sec_letter = sh_m.group(1).upper()
            if sec_letter in ('B', '2', 'II'):
                flush()
                section        = "SECTION-B"
                current_q_type = "NUMERICAL"
                last_committed_num = 0
                in_options_block   = False
            else:
                flush()
                section        = "SECTION-A"
                current_q_type = "MCQ"
                last_committed_num = 0
                in_options_block   = False
            continue

        # ── P0: Subject transition ────────────────────────────────────────────
        # A bare subject line is only treated as a boundary when:
        #   (a) no question is currently active (start of paper), OR
        #   (b) the immediately preceding non-empty line was a page-header, OR
        #   (c) the subject is genuinely different from the current one.
        # This prevents running-footer "PHYSICS" / "CHEMISTRY" labels that
        # appear mid-solution from incorrectly changing the subject.
        subj = _extract_subject_from_line(clean)
        if subj and not RE_QUESTION_COLON.match(clean) \
                and not RE_QUESTION_PREFIX.match(clean) \
                and not RE_PLAIN_Q.match(clean):
            # Only act on the subject line if it is a real boundary
            _is_genuine = (
                current is None            # no active question → definitely boundary
                or _prev_was_page_header   # immediately after date/shift header
                or subj != subject         # new subject (but only if no active Q mid-solution)
                    and state in (S.IDLE, S.IN_A)  # safe states to switch
            )
            if _is_genuine:
                flush()
                subject = subj
                last_committed_num = 0
                section        = "SECTION-A"
                current_q_type = "MCQ"
                in_options_block = False
                _prev_was_page_header = False
                continue
            # Not a genuine boundary — treat as regular text (fall through)
            _prev_was_page_header = False

        # ── P1: New question — fires in ANY state ─────────
        _prev_was_page_header = False   # clear flag — we're past the boundary zone
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
            if rest:
                # "Sol. (3)" — paren-wrapped option number is the answer, not sol text
                if RE_BARE_PAREN_ANS.match(rest) or re.fullmatch(r'[a-dA-D]\.?', rest):
                    if not current.answer: current.answer = _parse_answer_colon(rest)
                elif re.fullmatch(r'Bonus', rest, re.IGNORECASE):
                    if not current.answer: current.answer = 'Bonus'
                elif not sol_ans_m:
                    append_sol(rest)
            continue

        # ── P5: Solution body ─────────────────────────────
        if state == S.IN_S:
            if not current.answer:
                bpa = RE_BARE_PAREN_ANS.match(clean)
                if bpa: current.answer = _parse_answer_colon(clean); continue
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

        # ── Page header/footer line (e.g. "26th Feb. 2021 | Shift - 1") ──────
        if RE_PAGE_HEADER.search(clean):
            flush()
            _m2 = _extract_meta(clean)
            if _m2["exam_date"]: exam_date = _m2["exam_date"]
            if _m2["shift"]:     shift     = _m2["shift"]
            if _m2["year"]:      year      = _m2["year"]
            continue

        # ── Bare SECTION - A/B header in LaTeX plain text ────────────────────
        sh_m = RE_SECTION_HEADER.match(clean)
        if sh_m and not re.search(r'Question\s+\d+', clean, re.I):
            sec_letter = sh_m.group(1).upper()
            if sec_letter in ('B', '2', 'II'):
                flush()
                section            = "SECTION-B"
                current_q_type     = "NUMERICAL"
                last_committed_num = 0       # ← KEY FIX
                pending_setcounter = None
                in_options_block   = False
            else:
                flush()
                section            = "SECTION-A"
                current_q_type     = "MCQ"
                last_committed_num = 0
                pending_setcounter = None
                in_options_block   = False
            continue

        _subj = _extract_subject_from_line(clean)
        if _subj and not re.search(r'Question\s+\d+', clean, re.I):
            flush(); subject = _subj; last_committed_num = 0
            pending_setcounter = None; current_q_type = "MCQ"; in_options_block = False
            # reset section to A when subject changes
            section = "SECTION-A"
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
                if current:
                    sol_ans_m = RE_SOL_ANS.match(sec_text)
                    if sol_ans_m:
                        if not current.answer: current.answer = _parse_answer(sol_ans_m.group(1).strip())
                    state = S.IN_S
                    rest = sm.group(1).strip()
                    if rest:
                        if RE_BARE_PAREN_ANS.match(rest) or re.fullmatch(r'[a-dA-D]\.?', rest):
                            if not current.answer: current.answer = _parse_answer_colon(rest)
                        elif re.fullmatch(r'Bonus', rest, re.IGNORECASE):
                            if not current.answer: current.answer = 'Bonus'
                        elif not sol_ans_m:
                            append_sol(rest)
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
                last_committed_num=0; pending_setcounter=None
                current_q_type="MCQ"; in_options_block=False
                section = "SECTION-A"   # reset section on subject change
                continue
            if RE_NUMERICAL_SEC.search(sec_text):
                # ── KEY FIX for LaTeX parser ──────────────────
                flush()
                section            = "SECTION-B"
                current_q_type     = "NUMERICAL"
                last_committed_num = 0   # reset so Q1 of Section B is accepted
                pending_setcounter = None
                in_options_block   = False
                continue
            if current and state==S.IN_S: append_sol(sec_text)
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
            if itemize_depth==0: pending_setcounter=int(m.group(1))
            continue
        if in_noise_block: continue

        # Question detection (fires in ANY state)
        m = RE_ITEM.match(ln)
        if m:
            if itemize_depth > 0:
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
            sol_ans_m = RE_SOL_ANS.match(clean)
            if sol_ans_m:
                if not current.answer: current.answer = _parse_answer(sol_ans_m.group(1).strip())
            state=S.IN_S; in_options_block=False
            rest = _tail(sm.group(1))
            if rest:
                if RE_BARE_PAREN_ANS.match(rest) or re.fullmatch(r'[a-dA-D]\.?', rest):
                    if not current.answer: current.answer = _parse_answer_colon(rest)
                elif re.fullmatch(r'Bonus', rest, re.IGNORECASE):
                    if not current.answer: current.answer = 'Bonus'
                elif not sol_ans_m:
                    append_sol(rest)
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