"""
services/prompts.py
===================
Master prompt definitions for LLM-based question paper parsing.

All prompts live here so they can be tuned without touching pipeline logic.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — sent once as the system message
# ─────────────────────────────────────────────────────────────────────────────

PARSER_SYSTEM_PROMPT = """
You are an expert JEE/NEET question paper parser with deep knowledge of Indian
competitive exam formats (JEE Main, JEE Advanced, NEET, CUET).

Your job:
  - Parse raw LaTeX text of an exam paper
  - Extract EVERY question with ALL its fields
  - Tag each question with chapter, topic, difficulty
  - Return a strict JSON array — nothing else

CRITICAL RULES:
1. NEVER skip a question. If the paper is JEE Main 2019 it has 90 questions
   (or 75 for papers after 2021 pattern change). Count carefully.
2. NEVER hallucinate answers — if answer is not present leave it empty string.
3. NEVER hallucinate chapter/topic — use only the provided taxonomy list.
4. LaTeX must be preserved exactly as-is in question/option/solution fields.
5. Image references like \\includegraphics{img1} or [IMAGE:img1.png] must be
   kept as [IMAGE:img1.png] placeholder in the relevant field.
6. For NUMERICAL questions options array should be [].
7. answer field for MCQ: "1","2","3","4" (option number). For MSQ: "1,3" etc.
   For NUMERICAL: the numeric value as string e.g. "42" or "3.14".
8. solution field: full solution LaTeX text if present, else empty string.
9. section field: "SECTION-A" for MCQ, "SECTION-B" for numerical/integer type.
10. q_type: "MCQ" | "MSQ" | "NUMERICAL"

ANSWER EXTRACTION RULES (extremely important):
- Look for patterns like: Ans. (3), Answer: (B), Sol. 42, Ans. by NTA (2)
- Look for answer keys at the END of the paper/section
- A line like "Sol. (2)" or "Sol. 3.14" means the answer is 2 or 3.14
- If answer appears AFTER the solution text, still extract it
- Allen/Vedantu papers often have "Ans. (X)" on a separate line after options
- NTA official papers have answer keys as separate sections
- Map letter answers: A→1, B→2, C→3, D→4

EXAM YEAR & PATTERN AWARENESS:
- JEE Main 2017-2020: 90 questions (30 Physics + 30 Chemistry + 30 Maths)
- JEE Main 2021+: 75 questions (20 MCQ + 5 Numerical per subject × 3 subjects)
- NEET: 180 questions (45 Physics + 45 Chemistry + 90 Biology)
- NEET 2021+: 200 questions attempt 180
- JEE Advanced: variable, check sections carefully
- If you detect the exam type and year, use this to VALIDATE your question count
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATE — formatted at runtime
# ─────────────────────────────────────────────────────────────────────────────

PARSER_USER_PROMPT_TEMPLATE = """
Parse the following LaTeX exam paper completely.

EXAM METADATA DETECTED:
  Exam type : {exam_type}
  Year      : {year}
  Date      : {exam_date}
  Shift     : {shift}
  Subject(s): {subjects}

EXPECTED QUESTION COUNT (use as validation):
  {expected_count}

TAXONOMY (use ONLY these for chapter/topic):
{taxonomy_text}

OUTPUT FORMAT — return ONLY a JSON array, no markdown, no explanation:
[
  {{
    "number": 1,
    "q_type": "MCQ",
    "subject": "PHYSICS",
    "section": "SECTION-A",
    "year": "2024",
    "shift": "Morning",
    "exam_date": "2024-01-27",
    "question": "<latex text>",
    "options": ["<opt1>", "<opt2>", "<opt3>", "<opt4>"],
    "answer": "2",
    "solution": "<latex text or empty>",
    "chapter_name": "Electrostatics",
    "topic_name": "Gauss Law",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": -1
  }},
  ...
]

LATEX PAPER TO PARSE:
---BEGIN---
{latex_content}
---END---
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK PROMPT — for papers too large for single context window
# ─────────────────────────────────────────────────────────────────────────────

PARSER_CHUNK_PROMPT_TEMPLATE = """
Parse this CHUNK of a LaTeX exam paper. This is chunk {chunk_num} of {total_chunks}.

Questions in this chunk start at number {start_q} (approximately).

EXAM METADATA:
  Exam type : {exam_type}
  Year      : {year}
  Date      : {exam_date}
  Shift     : {shift}
  Subject(s): {subjects}

TAXONOMY (use ONLY these for chapter/topic):
{taxonomy_text}

OUTPUT FORMAT — return ONLY a JSON array, no markdown, no explanation.
Same schema as full parse. If a question is cut off at chunk boundary, include
what you have and mark question field with [TRUNCATED] at end.

CHUNK CONTENT:
---BEGIN---
{latex_content}
---END---
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# MERGE / DEDUP PROMPT — when combining chunks
# ─────────────────────────────────────────────────────────────────────────────

PARSER_MERGE_PROMPT_TEMPLATE = """
You are given multiple JSON arrays of questions parsed from chunks of the same
exam paper. Merge them into one clean array:

1. Remove duplicates (same question number)
2. For [TRUNCATED] questions, keep the fuller version
3. Fill missing answers if visible in another chunk
4. Sort by question number ascending
5. Return ONLY the merged JSON array, no explanation

CHUNKS:
{chunks_json}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMY FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

def format_taxonomy_for_prompt(taxonomy: dict, subject_filter: list = None) -> str:
    """
    Format the taxonomy dict into a compact text list for the prompt.
    If subject_filter given, only include those subjects.
    """
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
# EXPECTED QUESTION COUNT helper
# ─────────────────────────────────────────────────────────────────────────────

def get_expected_count(exam_type: str, year: str, subjects: list) -> str:
    """Return human-readable expected question count string."""
    exam_upper = (exam_type or "").upper()
    try:
        yr = int(year) if year else 0
    except ValueError:
        yr = 0

    if "JEE" in exam_upper and "ADVANCED" not in exam_upper:
        if yr >= 2021:
            return (
                "JEE Main 2021+ pattern: 75 questions total "
                "(20 MCQ + 5 Numerical per subject × 3 subjects). "
                "Each subject: Q1-20 MCQ (4 marks), Q21-25 Numerical (4 marks)."
            )
        elif yr >= 2017:
            return (
                "JEE Main 2017-2020 pattern: 90 questions total "
                "(30 per subject × 3 subjects, all MCQ, 4 marks each)."
            )
        else:
            return "JEE Main: typically 90 questions. Verify from paper."

    elif "NEET" in exam_upper:
        if yr >= 2021:
            return (
                "NEET 2021+ pattern: 200 questions (attempt any 180). "
                "Physics: 50 (attempt 45), Chemistry: 50 (attempt 45), "
                "Biology: 100 (attempt 90)."
            )
        else:
            return "NEET: 180 questions (45 Physics + 45 Chemistry + 90 Biology)."

    elif "JEE" in exam_upper and "ADVANCED" in exam_upper:
        return (
            "JEE Advanced: variable structure with multiple sections. "
            "Extract ALL questions from all sections."
        )

    num_subjects = len(subjects) if subjects else 1
    return f"Unknown exam type. Extract all questions found. (~{30 * num_subjects} expected)"