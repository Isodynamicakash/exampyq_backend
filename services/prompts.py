"""
services/prompts.py
===================
Prompts for Gemini-based question paper parsing.
"""

PARSER_SYSTEM_PROMPT = """
You are a question paper parser. You read LaTeX exam papers and return a JSON array of questions.

Return ONLY a raw JSON array. No markdown. No explanation. Nothing before or after the array.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW THIS LaTeX FORMAT WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Subject starts at:
  \\section*{26th Feb. 2021 | Shift - 1 PHYSICS}
  \\section*{CHEMISTRY}
  \\section*{MATHEMATICS}

Section type:
  \\section*{SECTION - A}  → MCQ (q_type: "MCQ")
  \\section*{SECTION - B}  → Numerical (q_type: "NUMERICAL")

Questions are \\item inside \\begin{enumerate}.
\\setcounter{enumi}{N} means next \\item is question number N+1.

Answer is in \\section*{Sol. (X)} right after options:
  \\section*{Sol. (3)}   → answer = "3"
  \\section*{Sol. (A)}   → answer = "1"  (A=1, B=2, C=3, D=4)
  Sol. 42                → answer = "42" (numerical)
  \\section*{Sol. Bonus} → answer = ""

Solution text is everything after Sol./Ans. until next question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE INPUT → OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT:
\\section*{26$^{th}$ Feb. 2021 | Shift - 1 PHYSICS}
\\section*{SECTION - A}
\\begin{enumerate}
  \\item If $\\lambda_{1}$ and $\\lambda_{2}$ are the wavelengths of the third member of Lyman
  and first member of the Paschen series, then $\\lambda_{1}: \\lambda_{2}$ is :\\\\
  (1) $1: 3$\\\\
  (2) $1: 9$\\\\
  (3) $7: 135$\\\\
  (4) $7: 108$
\\end{enumerate}

\\section*{Sol. (3)}
For Lyman series $n_1=1, n_2=4$\\\\
$\\frac{1}{\\lambda_1} = R\\left(\\frac{1}{1} - \\frac{1}{16}\\right) = \\frac{15R}{16}$\\\\
$\\lambda_1 = \\frac{16}{15R}$

\\begin{enumerate}
  \\setcounter{enumi}{1}
  \\item The temperature $\\theta$ at the junction of two sheets with thermal resistances
  $R_1$ and $R_2$ and temperatures $\\theta_1$, $\\theta_2$ is:\\\\
  \\includegraphics[max width=\\textwidth]{img-02_239}\\\\
  (1) $\\frac{\\theta_1 R_2 + \\theta_2 R_1}{R_1 + R_2}$\\\\
  (2) $\\frac{\\theta_1 R_2 - \\theta_2 R_1}{R_2 - R_1}$\\\\
  (3) $\\frac{\\theta_2 R_2 - \\theta_1 R_1}{R_2 - R_1}$\\\\
  (4) $\\frac{\\theta_1 R_1 + \\theta_2 R_2}{R_1 + R_2}$
\\end{enumerate}

\\section*{Sol. (1)}
At junction temperature $\\theta$:\\\\
$\\frac{\\theta_2 - \\theta}{R_2} = \\frac{\\theta - \\theta_1}{R_1}$\\\\
$\\theta = \\frac{R_1\\theta_2 + R_2\\theta_1}{R_1 + R_2}$

\\section*{SECTION - B}
\\begin{enumerate}
  \\setcounter{enumi}{20}
  \\item The mass per unit length of a wire is $0.135$ g/cm. A wave
  $y = -0.21\\sin(x+30t)$ is produced. The tension in the wire is $x \\times 10^{-2}$ N.
  Value of $x$ is \\_\\_\\_\\_.
\\end{enumerate}

Sol. 1215\\\\
$v = \\omega/k = 30$ m/s\\\\
$T = v^2 \\mu = 900 \\times 0.135 \\times 10^{-1} = 12.15$ N $= 1215 \\times 10^{-2}$ N

OUTPUT:
[
  {
    "number": 1,
    "q_type": "MCQ",
    "subject": "PHYSICS",
    "section": "SECTION-A",
    "year": "2021",
    "shift": "Morning",
    "exam_date": "2021-02-26",
    "question": "If $\\\\lambda_{1}$ and $\\\\lambda_{2}$ are the wavelengths of the third member of Lyman and first member of the Paschen series, then $\\\\lambda_{1}: \\\\lambda_{2}$ is :",
    "options": ["$1: 3$", "$1: 9$", "$7: 135$", "$7: 108$"],
    "answer": "3",
    "solution": "For Lyman series $n_1=1, n_2=4$\\n$\\\\frac{1}{\\\\lambda_1} = R\\\\left(\\\\frac{1}{1} - \\\\frac{1}{16}\\\\right) = \\\\frac{15R}{16}$\\n$\\\\lambda_1 = \\\\frac{16}{15R}$",
    "chapter_name": "Modern Physics",
    "topic_name": "Hydrogen Spectrum",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": -1
  },
  {
    "number": 2,
    "q_type": "MCQ",
    "subject": "PHYSICS",
    "section": "SECTION-A",
    "year": "2021",
    "shift": "Morning",
    "exam_date": "2021-02-26",
    "question": "The temperature $\\\\theta$ at the junction of two sheets with thermal resistances $R_1$ and $R_2$ and temperatures $\\\\theta_1$, $\\\\theta_2$ is: [IMAGE:img-02_239]",
    "options": ["$\\\\frac{\\\\theta_1 R_2 + \\\\theta_2 R_1}{R_1 + R_2}$", "$\\\\frac{\\\\theta_1 R_2 - \\\\theta_2 R_1}{R_2 - R_1}$", "$\\\\frac{\\\\theta_2 R_2 - \\\\theta_1 R_1}{R_2 - R_1}$", "$\\\\frac{\\\\theta_1 R_1 + \\\\theta_2 R_2}{R_1 + R_2}$"],
    "answer": "1",
    "solution": "At junction temperature $\\\\theta$:\\n$\\\\frac{\\\\theta_2 - \\\\theta}{R_2} = \\\\frac{\\\\theta - \\\\theta_1}{R_1}$\\n$\\\\theta = \\\\frac{R_1\\\\theta_2 + R_2\\\\theta_1}{R_1 + R_2}$",
    "chapter_name": "Heat Transfer",
    "topic_name": "Thermal Resistance",
    "difficulty": "medium",
    "q_images": ["img-02_239"],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": -1
  },
  {
    "number": 21,
    "q_type": "NUMERICAL",
    "subject": "PHYSICS",
    "section": "SECTION-B",
    "year": "2021",
    "shift": "Morning",
    "exam_date": "2021-02-26",
    "question": "The mass per unit length of a wire is $0.135$ g/cm. A wave $y = -0.21\\\\sin(x+30t)$ is produced. The tension in the wire is $x \\\\times 10^{-2}$ N. Value of $x$ is ____.",
    "options": [],
    "answer": "1215",
    "solution": "$v = \\\\omega/k = 30$ m/s\\n$T = v^2 \\\\mu = 900 \\\\times 0.135 \\\\times 10^{-1} = 12.15$ N $= 1215 \\\\times 10^{-2}$ N",
    "chapter_name": "Waves",
    "topic_name": "Wave Speed",
    "difficulty": "medium",
    "q_images": [],
    "sol_images": [],
    "marks_correct": 4,
    "marks_wrong": 0
  }
]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES LEARNED FROM EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Options: remove "(1)" prefix — just the text/math
2. answer: "1"/"2"/"3"/"4" for MCQ, numeric string for NUMERICAL
3. LaTeX backslashes are DOUBLED in JSON: \\frac → \\\\frac
4. Newlines in solution: use \\n (literal backslash-n in JSON)
5. [IMAGE:filename] stays as-is in question/options/solution text
6. q_images: list of image filenames found in question/options
7. sol_images: list of image filenames found in solution
8. SECTION-B → q_type NUMERICAL, options [], marks_wrong 0
9. \\setcounter{enumi}{N} → next question number is N+1
10. Subject/date/shift from \\section*{...} heading
""".strip()


PARSER_USER_PROMPT_TEMPLATE = """
Extract ALL questions from this exam paper.

EXAM: {exam_type} | Year: {year} | Date: {exam_date} | Shift: {shift}
Subjects: {subjects}
Expected: {expected_count}

TAXONOMY — use ONLY these exact strings for chapter_name and topic_name:
{taxonomy_text}

Return ONLY the raw JSON array as shown in the example. Start with [ end with ].

PAPER:
---BEGIN---
{latex_content}
---END---
""".strip()


PARSER_CHUNK_PROMPT_TEMPLATE = """
Extract ALL questions from this chunk. Chunk {chunk_num} of {total_chunks}.

EXAM: {exam_type} | Year: {year} | Date: {exam_date} | Shift: {shift}
Subjects: {subjects}

TAXONOMY:
{taxonomy_text}

Return ONLY raw JSON array.

CHUNK:
---BEGIN---
{latex_content}
---END---
""".strip()


PARSER_MERGE_PROMPT_TEMPLATE = """
Merge these JSON arrays. Remove duplicates (same number+subject). Return ONLY merged JSON array.

{chunks_json}
""".strip()


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


def get_expected_count(exam_type: str, year: str, subjects: list) -> str:
    exam_upper = (exam_type or "").upper()
    try:
        yr = int(year) if year else 0
    except ValueError:
        yr = 0

    subj = subjects[0] if subjects else "this subject"

    if "JEE" in exam_upper and "ADVANCED" not in exam_upper:
        if yr >= 2021:
            return f"30 per subject (20 MCQ SECTION-A + 10 NUMERICAL SECTION-B) × {len(subjects)} subjects = {30*len(subjects)} total"
        elif yr >= 2017:
            return f"30 per subject × {len(subjects)} subjects = {30*len(subjects)} total"
        else:
            return "~90 questions total"
    elif "NEET" in exam_upper:
        if "BIOLOGY" in str(subj).upper():
            return "90 Biology questions (Botany + Zoology)"
        return f"45-50 questions for {subj}"
    elif "ADVANCED" in exam_upper:
        return "Variable — extract ALL questions from all sections"

    return f"Extract all {subj} questions you find"