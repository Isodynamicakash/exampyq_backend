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
  - Read the entire LaTeX exam paper
  - Extract EVERY SINGLE question — do not skip any
  - Each question gets its subject, options, answer, solution, chapter, topic
  - Return a JSON array — nothing else, no explanation, no markdown

CRITICAL RULES:
1. Extract ALL questions from ALL subjects in one pass.
2. NEVER skip a question. Count them — JEE Main 2021+ has 75, 2017-2020 has 90.
3. NEVER hallucinate answers — if answer is not in the paper, leave it "".
4. NEVER hallucinate chapter/topic — use only the provided taxonomy list.
5. LaTeX must be preserved exactly as-is in question/option/solution fields.
6. Image references [IMAGE:img1.png] must be kept as-is in the relevant field.
7. For NUMERICAL questions options array should be [].
8. answer field for MCQ: "1","2","3","4". For MSQ: "1,3". For NUMERICAL: "42".
9. section: "SECTION-A" for MCQ, "SECTION-B" for numerical/integer type.
10. q_type: "MCQ" | "MSQ" | "NUMERICAL"

ANSWER EXTRACTION RULES:
- Look for: Ans. (3), Answer: (B), Sol. 42, Ans. by NTA (2)
- Answer keys may appear at the END of each section or the full paper
- "Sol. (2)" or "Sol. 3.14" means answer is 2 or 3.14
- Allen/Vedantu papers often have "Ans. (X)" after options
- Map letter answers: A→1, B→2, C→3, D→4
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT TEMPLATE — single call, extract everything
# ─────────────────────────────────────────────────────────────────────────────

PARSER_USER_PROMPT_TEMPLATE = """
Extract ONLY the {subjects} questions from this exam paper.
Read the entire paper but return ONLY questions that belong to {subjects}.
Ignore questions from all other subjects completely.

EXAM INFO:
  Type  : {exam_type}
  Year  : {year}
  Date  : {exam_date}
  Shift : {shift}

EXPECTED for {subjects} only: {expected_count}

TAXONOMY — use ONLY these chapter/topic names for {subjects}:
{taxonomy_text}

OUTPUT — return ONLY a raw JSON array, no markdown, no explanation:
[
  {{
    "number": 1,
    "q_type": "MCQ",
    "subject": "{subjects}",
    "section": "SECTION-A",
    "year": "{year}",
    "shift": "{shift}",
    "exam_date": "{exam_date}",
    "question": "<latex text>",
    "options": ["<opt1>", "<opt2>", "<opt3>", "<opt4>"],
    "answer": "2",
    "solution": "<latex or empty>",
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

PAPER:
---BEGIN---
{latex_content}
---END---
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK PROMPT — only used if paper > 280k chars (very rare)
# ─────────────────────────────────────────────────────────────────────────────

PARSER_CHUNK_PROMPT_TEMPLATE = """
Extract ALL questions from this portion of an exam paper. Chunk {chunk_num} of {total_chunks}.

EXAM INFO:
  Type     : {exam_type}
  Year     : {year}
  Date     : {exam_date}
  Shift    : {shift}
  Subjects : {subjects}

TAXONOMY:
{taxonomy_text}

OUTPUT — return ONLY a raw JSON array, same schema as below, no markdown:
[ {{ "number": 1, "q_type": "MCQ", "subject": "PHYSICS", "section": "SECTION-A",
     "year": "", "shift": "", "exam_date": "", "question": "", "options": [],
     "answer": "", "solution": "", "chapter_name": "", "topic_name": "",
     "difficulty": "medium", "q_images": [], "sol_images": [],
     "marks_correct": 4, "marks_wrong": -1 }}, ... ]

CHUNK:
---BEGIN---
{latex_content}
---END---
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# MERGE PROMPT
# ─────────────────────────────────────────────────────────────────────────────

PARSER_MERGE_PROMPT_TEMPLATE = """
Merge these JSON arrays of questions from the same exam paper into one clean array.
Remove duplicates (same number+subject), keep the fuller version.
Return ONLY the merged JSON array.

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
# EXPECTED QUESTION COUNT helper
# ─────────────────────────────────────────────────────────────────────────────

def get_expected_count(exam_type: str, year: str, subjects: list) -> str:
    exam_upper = (exam_type or "").upper()
    try:
        yr = int(year) if year else 0
    except ValueError:
        yr = 0

    if "JEE" in exam_upper and "ADVANCED" not in exam_upper:
        if yr >= 2021:
            return "75 questions (20 MCQ + 5 Numerical per subject × 3 subjects)"
        elif yr >= 2017:
            return "90 questions (30 per subject × 3 subjects, all MCQ)"
        else:
            return "~90 questions, verify from paper"
    elif "NEET" in exam_upper:
        if yr >= 2021:
            return "200 questions (attempt 180)"
        else:
            return "180 questions (45 Physics + 45 Chemistry + 90 Biology)"
    elif "JEE" in exam_upper and "ADVANCED" in exam_upper:
        return "Variable — extract ALL questions from all sections"

    num_subjects = len(subjects) if subjects else 1
    return f"~{30 * num_subjects} questions expected"