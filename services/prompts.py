"""
services/prompts.py
===================
Master prompt definitions for LLM-based question paper parsing.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

PARSER_SYSTEM_PROMPT = """
You are an expert Indian competitive exam question paper parser.
You handle JEE Main, JEE Advanced, NEET, CUET, SSC CGL, and any other format.

YOUR MINDSET:
- Every paper is different. Do NOT assume a fixed format.
- Read the paper first, understand its structure, then extract.
- Use your intelligence to figure out where questions start/end, where answers are, what the sections mean.
- When in doubt, include the content — never skip a question.

════════════════════════════════════════════════════════
WHAT YOU WILL SEE (LaTeX source)
════════════════════════════════════════════════════════

The paper is raw LaTeX. Common patterns you may encounter — but not limited to:

SUBJECT HEADINGS (any of these mean a new subject starts):
  \\section*{PHYSICS}
  \\section*{26th Feb 2021 | Shift 1 CHEMISTRY}
  \\section*{PART A - MATHEMATICS}
  \\textbf{BIOLOGY}
  % or just a bold heading line

SECTION TYPES (may appear or may not):
  \\section*{SECTION - A}   → usually MCQ
  \\section*{SECTION - B}   → usually NUMERICAL/INTEGER
  \\section*{SECTION - C}   → could be MSQ or another type
  Some papers have no section labels — infer from content.

QUESTIONS:
  - Inside \\begin{enumerate}...\\end{enumerate}, each question is \\item
  - \\setcounter{enumi}{N} → next \\item is question N+1
  - Some papers use plain numbering: "1." "2." "Q.1" "Q1."
  - Question number resets per subject (Q1 for Physics, Q1 for Chemistry etc.)
    OR continues globally (Q1-Q30 Physics, Q31-Q60 Chemistry)
  - YOU must figure out the numbering scheme from context

OPTIONS:
  - (1) text (2) text (3) text (4) text   ← most common
  - (A) text (B) text (C) text (D) text   ← also common
  - \\item inside a nested enumerate       ← LaTeX style
  - options may span multiple lines
  - image-only options: [IMAGE:filename]

ANSWERS — APPEAR IN MANY WAYS:
  \\section*{Sol. (3)}          → answer = "3"
  \\section*{Sol. (A)}          → answer = "1" (A=1)
  \\section*{Ans. (2)}          → answer = "2"
  Sol. 42                       → answer = "42"   (numerical)
  Sol. 3.14                     → answer = "3.14" (numerical)
  Ans. (0.18)                   → answer = "0.18"
  Answer: C                     → answer = "3"
  Sol. Bonus                    → answer = ""  (bonus/dropped)
  Answer key table at end:
    1-(3)  2-(1)  3-(4) ...     → use this to fill missing answers
  Allen/Vedantu style:
    "Ans. (X)" after options    → answer = X
  NTA style:
    separate answer key section → match question number to answer

SOLUTIONS:
  Everything after the Sol./Ans. line until the next question = solution.
  May include equations, figures, tables. Capture ALL of it.

IMAGES:
  \\includegraphics has been pre-converted to [IMAGE:filename] placeholders.
  Keep these as-is in question/options/solution text.
  List filenames in q_images or sol_images accordingly.

════════════════════════════════════════════════════════
INTELLIGENCE RULES
════════════════════════════════════════════════════════

1. FIGURE OUT THE FORMAT FIRST
   Before extracting, mentally note:
   - How are questions numbered? (per-subject or global)
   - How are answers given? (inline Sol., end key, or both)
   - Are there sections A/B/C or not?
   - What exam is this? (determines expected count)

2. NEVER SKIP A QUESTION
   If you see \\item or a numbered line that looks like a question, extract it.
   Even if the format is unusual.

3. ANSWERS ARE ALWAYS SOMEWHERE
   If Sol./Ans. is not right after options, look:
   - At the end of the subject section
   - At the end of the entire paper
   - In a separate answer key block
   Match by question number.

4. LETTER → NUMBER MAPPING
   A or (A) → "1",  B or (B) → "2",  C or (C) → "3",  D or (D) → "4"

5. QUESTION TYPE INFERENCE
   - Has 4 options + letter/number answer → MCQ
   - Has 4 options + multiple answers (A,C) → MSQ
   - Has no options + numeric answer → NUMERICAL
   - Has no options + text answer → SHORT_ANSWER (use q_type: "NUMERICAL")
   - True/False questions → MCQ with 2 options

6. NUMBERING
   Use the question number AS IT APPEARS in the paper for that subject.
   If Physics Q1-Q25 and Chemistry also starts at Q1, that is correct — keep it.
   Do not renumber.

════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════
Return ONLY a raw JSON array. No markdown. No explanation. No preamble.
Every field must be present even if empty string or empty list.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

PARSER_USER_PROMPT_TEMPLATE = """
Extract ONLY the {subjects} questions from this exam paper.

STEP 1 — Find the {subjects} section in the paper.
STEP 2 — Understand how questions are numbered in this paper.
STEP 3 — Extract every question with its options, answer, and solution.
STEP 4 — If answer is not right after the question, look at the end of the section or paper.

EXAM INFO:
  Type  : {exam_type}
  Year  : {year}
  Date  : {exam_date}
  Shift : {shift}

EXPECTED for {subjects}: {expected_count}

TAXONOMY (use ONLY these exact strings for chapter_name and topic_name, or leave blank):
{taxonomy_text}

OUTPUT — return ONLY a raw JSON array, no markdown fences, no text before or after:
[
  {{
    "number": 1,
    "q_type": "MCQ",
    "subject": "{subjects}",
    "section": "SECTION-A",
    "year": "{year}",
    "shift": "{shift}",
    "exam_date": "{exam_date}",
    "question": "<full latex question text, preserve all LaTeX>",
    "options": ["<opt1>", "<opt2>", "<opt3>", "<opt4>"],
    "answer": "3",
    "solution": "<full solution text, all LaTeX preserved>",
    "chapter_name": "",
    "topic_name": "",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": -1
  }},
  {{
    "number": 21,
    "q_type": "NUMERICAL",
    "subject": "{subjects}",
    "section": "SECTION-B",
    "year": "{year}",
    "shift": "{shift}",
    "exam_date": "{exam_date}",
    "question": "<full latex question text>",
    "options": [],
    "answer": "1215",
    "solution": "<full solution text>",
    "chapter_name": "",
    "topic_name": "",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": 0
  }}
]

EXAM PAPER LaTeX:
---BEGIN---
{latex_content}
---END---
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK PROMPT — large papers only (>280k chars)
# ─────────────────────────────────────────────────────────────────────────────

PARSER_CHUNK_PROMPT_TEMPLATE = """
Extract ALL questions from this chunk of an exam paper. Chunk {chunk_num} of {total_chunks}.

EXAM: {exam_type} | Year: {year} | Date: {exam_date} | Shift: {shift}
Subjects in this chunk: {subjects}

INSTRUCTIONS:
- Figure out the question numbering scheme from context
- Extract every question you find with options, answer, solution
- Answer may be right after options (Sol./Ans.) or at end of chunk
- Preserve all LaTeX in question/options/solution fields
- If a question is cut off at chunk boundary, include what you have

TAXONOMY:
{taxonomy_text}

OUTPUT — raw JSON array only, no markdown:
[ {{
    "number": 1, "q_type": "MCQ", "subject": "PHYSICS", "section": "SECTION-A",
    "year": "{year}", "shift": "{shift}", "exam_date": "{exam_date}",
    "question": "", "options": [], "answer": "", "solution": "",
    "chapter_name": "", "topic_name": "", "difficulty": "medium",
    "q_images": [], "sol_images": [], "marks_correct": 4, "marks_wrong": -1
}} ]

CHUNK:
---BEGIN---
{latex_content}
---END---
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# MERGE PROMPT
# ─────────────────────────────────────────────────────────────────────────────

PARSER_MERGE_PROMPT_TEMPLATE = """
Merge these JSON arrays of questions from the same exam paper.
Remove duplicates (same number + subject), keep the version with more content.
Return ONLY the merged JSON array, no explanation.

CHUNKS:
{chunks_json}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMY FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def format_taxonomy_for_prompt(taxonomy: dict, subject_filter: list = None) -> str:
    lines = []
    for subject, chapters in taxonomy.items():
        if subject_filter and subject not in subject_filter:
            continue
        lines.append(f"\n{subject.upper()}:")
        for chapter, topics in sorted(chapters.items()):
            topic_preview = ", ".join(topics[:6])
            if len(topics) > 6:
                topic_preview += f" ... ({len(topics)} topics)"
            lines.append(f"  - {chapter}: {topic_preview}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED QUESTION COUNT
# ─────────────────────────────────────────────────────────────────────────────

def get_expected_count(exam_type: str, year: str, subjects: list) -> str:
    exam_upper = (exam_type or "").upper()
    try:
        yr = int(year) if year else 0
    except ValueError:
        yr = 0

    subj = subjects[0] if subjects else "this subject"

    if "JEE" in exam_upper and "ADVANCED" not in exam_upper:
        if yr >= 2021:
            return (
                f"25 questions for {subj}: "
                f"SECTION-A has 20 MCQ (Q1-Q20), SECTION-B has 5 NUMERICAL (Q21-Q25). "
                f"Total paper = 75 questions across 3 subjects."
            )
        elif yr >= 2017:
            return (
                f"30 MCQ questions for {subj} (all SECTION-A). "
                f"Total paper = 90 questions across 3 subjects."
            )
        else:
            return f"~30 questions for {subj}. Verify exact count from paper."

    elif "NEET" in exam_upper:
        if "BIOLOGY" in subj.upper():
            if yr >= 2021:
                return "100 Biology questions (50 Botany + 50 Zoology), attempt any 90."
            return "90 Biology questions (45 Botany + 45 Zoology)."
        else:
            if yr >= 2021:
                return f"50 questions for {subj} (attempt any 45): SECTION-A 35 MCQ + SECTION-B 15 MCQ attempt 10."
            return f"45 MCQ questions for {subj}."

    elif "ADVANCED" in exam_upper:
        return (
            "JEE Advanced has multiple papers and sections with varying question types. "
            "Extract ALL questions from all sections you find."
        )

    elif "SSC" in exam_upper or "CGL" in exam_upper:
        return f"Extract all {subj} questions found. SSC papers typically have 25 questions per subject."

    return f"Extract all {subj} questions you can find in the paper."