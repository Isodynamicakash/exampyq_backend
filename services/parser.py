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

RE_PLAIN_Q = re.compile(r'^(\d{1,3})\.\s+(\S.*)|^(\d{1,3})\.\s*\\\\?\s*$')  # "N. text" OR bare "N.\\" (text on next line)
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
    enum_depth    = 0   # depth of egin{enumerate} nesting
    itemize_depth = 0   # depth of egin{itemize} nesting
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
          - from_setcounter=True: ALWAYS accept (\setcounter is authoritative)
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
            # Inside table: append raw content, still extract images
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

            # \section*{Answer (N)} or \section*{Ans. (N)}
            am = RE_ANSWER.match(sec_text)
            if am and current is not None:
                raw_ans = (am.group(1) or am.group(2) or '').strip()
                nta_m = RE_NTA_ANS.match(sec_text)
                if nta_m:
                    nta_raw = (nta_m.group(1) or nta_m.group(2) or '').strip()
                    current.answer = _parse_answer(nta_raw)
                elif not current.answer:
                    current.answer = _parse_answer(raw_ans)
                state = S.IN_A
                continue

            # \section*{Sol. ...}
            # ALWAYS resets last_section_was_noise — Sol. is never a noise block
            sm = RE_SOL.match(sec_text)
            if sm:
                last_section_was_noise = False   # ← critical: Sol. ends any noise context
                if current is not None:
                    state = S.IN_S
                    rest = sm.group(1).strip()
                    if rest:
                        append_sol(rest)
                continue

            # noise — but explicitly exclude Sol. (already handled above)
            if _is_noise(sec_text):
                last_section_was_noise = True
                continue
            last_section_was_noise = False

            # subject → RESET numbering
            subj_m = RE_SUBJECT.search(sec_text)
            if subj_m:
                flush()
                subject = subj_m.group(1).upper()
                if subject in ("MATHS", "MATH"):
                    subject = "MATHEMATICS"
                last_committed_num = 0   # ← CRITICAL RESET per section
                pending_setcounter = None
                current_q_type = "MCQ"  # reset to MCQ at start of each subject
                continue

            # Numerical / Integer section → switch q_type for all following questions
            if RE_NUMERICAL_SEC.search(sec_text):
                flush()
                section = "SECTION-B"
                current_q_type = "NUMERICAL"
                continue

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
            pending_setcounter = None
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
            # group(1)+group(2) = "N. text" form; group(3) = bare "N.\\" form
            num  = int(pq.group(1) or pq.group(3))
            rest = (pq.group(2) or '').strip()  # may be empty for bare "18.\\" case
            if is_next_q(num):
                start_q(num, rest)
                pending_setcounter = None
                continue
            # else: not a question start, fall through to text handling

        # ── Answer ────────────────────────────────────────────────────────────
        am = RE_ANSWER.match(stripped)
        if am and current is not None:
            raw_ans = (am.group(1) or am.group(2) or '').strip()
            parsed  = _parse_answer(raw_ans)
            # NTA answer always wins (overrides Allen/coaching answer if set earlier)
            nta_m = RE_NTA_ANS.match(stripped)
            if nta_m:
                nta_raw = (nta_m.group(1) or nta_m.group(2) or '').strip()
                current.answer = _parse_answer(nta_raw)
                state = S.IN_A
            elif not current.answer:
                # Only set if no answer recorded yet (first answer wins unless NTA)
                current.answer = parsed
                state = S.IN_A
            continue

        # ── Sol ───────────────────────────────────────────────────────────────
        sm = RE_SOL.match(stripped)
        if sm and current is not None:
            state = S.IN_S
            rest = _tail(sm.group(1))
            if rest:
                append_sol(rest)
            continue

        # ── Options ───────────────────────────────────────────────────────────
        om = RE_OPTION.match(stripped)
        if om and current is not None and state == S.IN_Q:
            set_opt(int(om.group(1)), _tail(om.group(2)))
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

        # Auto-upgrade type based on content
        if q.q_type == "MCQ":
            a = q.answer.lower()
            if re.search(r'\b[a-d]\b', a) and ',' in a:
                q.q_type = "MSQ"
            elif not q.options and re.fullmatch(r'[\d.]+', q.answer.strip()):
                q.q_type = "NUMERICAL"  # no options + pure number answer = numerical

        result.append(q.to_dict())

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python parser.py <file.tex>")
        sys.exit(1)

    qs = parse_tex(sys.argv[1])
    print(json.dumps(qs, indent=2, ensure_ascii=False))
    print(f"\n✓ Parsed {len(qs)} questions", file=sys.stderr)