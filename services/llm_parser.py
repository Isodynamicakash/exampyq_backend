"""
services/llm_parser.py
======================
Generalized LaTeX parser using Anthropic Claude Haiku.
Handles ANY MCQ exam format: JEE, NEET, CAT, SSC, Allen, etc.

CHANGES:
  - Uses Claude Haiku (claude-haiku-4-20250514) for cost-efficiency
  - Chunking strategy for large papers (10 questions per chunk)
  - Robust JSON extraction with fallback parsing
  - Auto-detects exam type, subject, date, shift from LaTeX
"""

import os
import re
import json
import asyncio
from typing import Optional
import anthropic

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

HAIKU_MODEL = "claude-haiku-4-20241022"
MAX_TOKENS = 4000
CHUNK_SIZE = 10  # questions per chunk


# ══════════════════════════════════════════════════════════
# HELPER: Extract JSON from LLM response
# ══════════════════════════════════════════════════════════

def extract_json(text: str) -> list:
    """Extract JSON array from LLM response, handling markdown fences."""
    # Remove markdown code fences
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE).strip()
    
    # Try direct parse
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except:
        pass
    
    # Find JSON array pattern
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    
    return []


# ══════════════════════════════════════════════════════════
# HELPER: Clean LaTeX for better parsing
# ══════════════════════════════════════════════════════════

def clean_latex(tex: str) -> str:
    """Remove LaTeX preamble and keep only document body."""
    # Extract content between \begin{document} and \end{document}
    doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', tex, re.DOTALL)
    if doc_match:
        return doc_match.group(1).strip()
    return tex


# ══════════════════════════════════════════════════════════
# HELPER: Split LaTeX into manageable chunks
# ══════════════════════════════════════════════════════════

def split_latex_chunks(tex: str, chunk_size: int = 10) -> list[str]:
    """
    Split LaTeX into chunks by detecting question boundaries.
    Tries to keep ~chunk_size questions per chunk.
    """
    # Find all question starts (handles multiple formats)
    question_pattern = r'(?:^|\n)(?:\\item|\\begin\{enumerate\}|\d+\.\s+)'
    splits = list(re.finditer(question_pattern, tex, re.MULTILINE))
    
    if len(splits) < 2:
        return [tex]  # Can't split meaningfully
    
    chunks = []
    chunk_start = 0
    
    for i in range(chunk_size, len(splits), chunk_size):
        chunk_end = splits[i].start()
        chunks.append(tex[chunk_start:chunk_end])
        chunk_start = chunk_end
    
    # Add remaining content
    if chunk_start < len(tex):
        chunks.append(tex[chunk_start:])
    
    return chunks if chunks else [tex]


# ══════════════════════════════════════════════════════════
# MAIN: Parse LaTeX with Claude Haiku
# ══════════════════════════════════════════════════════════

async def parse_latex_with_llm(
    tex: str,
    api_key: str,
    pool=None,
    subject_hint: str = "",
) -> list:
    """
    Parse ANY MCQ exam LaTeX using Claude Haiku.
    
    Returns list of question dicts with:
    - number, q_type, subject, question, options, answer, solution
    - Auto-detected: exam_name, year, exam_date, shift, chapter_name, topic_name, difficulty
    """
    
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY required for LLM parsing")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Clean and prepare LaTeX
    cleaned = clean_latex(tex)
    chunks = split_latex_chunks(cleaned, CHUNK_SIZE)
    
    print(f"[LLM Parser] Processing {len(chunks)} chunks (Haiku model)")
    
    all_questions = []
    
    for idx, chunk in enumerate(chunks):
        print(f"[LLM Parser] Chunk {idx+1}/{len(chunks)} - {len(chunk)} chars")
        
        prompt = f"""You are an expert MCQ exam parser. Extract questions from this LaTeX content.

**CRITICAL RULES:**
1. Return ONLY a valid JSON array, no markdown fences
2. Each question must have: number, q_type, subject, question, options (array), answer, solution
3. Auto-detect: exam_name (JEE/NEET/CAT/SSC/etc), year, exam_date (YYYY-MM-DD), shift
4. For chapter/topic: use best guess from content, or leave empty
5. Difficulty: "easy", "medium", or "hard" based on complexity
6. q_type: "MCQ" (single correct), "MSQ" (multiple correct), or "NUMERICAL" (integer answer)
7. Preserve ALL LaTeX math notation exactly: $...$, $$...$$, \\frac{{}}{{}}, \\sqrt{{}}, etc.
8. For images: use [IMAGE:filename] notation
9. Answer format:
   - MCQ: "1", "2", "3", or "4"
   - MSQ: "1,2" or "2,3,4" (comma-separated)
   - NUMERICAL: exact number like "25" or "3.14"

**LaTeX Content:**
```latex
{chunk}
```

**Subject Hint (if known):** {subject_hint or "Auto-detect from content"}

**Output Format (STRICT JSON):**
[
  {{
    "number": 1,
    "q_type": "MCQ",
    "subject": "PHYSICS",
    "exam_name": "JEE Main",
    "year": "2021",
    "exam_date": "2021-02-26",
    "shift": "Morning",
    "chapter_name": "Optics",
    "topic_name": "Young's Double Slit",
    "difficulty": "medium",
    "question": "A particle moves with velocity $v = 3t^2$. Find displacement.",
    "options": ["$10$ m", "$20$ m", "$30$ m", "$40$ m"],
    "answer": "2",
    "solution": "Using $s = \\int v dt = \\int 3t^2 dt = t^3$, at $t=2$, $s=8$ m."
  }}
]

**Remember:** ONLY output the JSON array, nothing else."""

        try:
            message = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            questions = extract_json(response_text)
            
            if questions:
                all_questions.extend(questions)
                print(f"[LLM Parser] ✓ Extracted {len(questions)} questions from chunk {idx+1}")
            else:
                print(f"[LLM Parser] ✗ No valid questions in chunk {idx+1}")
                print(f"[LLM Parser] Raw response preview: {response_text[:200]}")
        
        except Exception as e:
            print(f"[LLM Parser] ERROR in chunk {idx+1}: {e}")
            continue
    
    # Post-process: ensure required fields
    for q in all_questions:
        q.setdefault("q_type", "MCQ")
        q.setdefault("subject", subject_hint.upper() if subject_hint else "PHYSICS")
        q.setdefault("exam_name", "")
        q.setdefault("year", "")
        q.setdefault("exam_date", "")
        q.setdefault("shift", "")
        q.setdefault("chapter_name", "")
        q.setdefault("topic_name", "")
        q.setdefault("difficulty", "medium")
        q.setdefault("marks_correct", 4)
        q.setdefault("marks_wrong", -1)
        q.setdefault("verified", False)
        q.setdefault("q_images", [])
        q.setdefault("sol_images", [])
        q.setdefault("opt_images", {})
    
    print(f"[LLM Parser] ✓ Total questions extracted: {len(all_questions)}")
    return all_questions


# ══════════════════════════════════════════════════════════
# FALLBACK: If Haiku fails, try basic regex parser
# ══════════════════════════════════════════════════════════

def fallback_parser(tex: str, subject_hint: str = "") -> list:
    """
    Emergency fallback: basic regex extraction if LLM fails.
    NOT as good as LLM, but better than nothing.
    """
    print("[LLM Parser] WARNING: Using fallback regex parser")
    
    from services.parser import parse_tex
    import tempfile
    
    # Write to temp file and use existing parser
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
        f.write(tex)
        temp_path = f.name
    
    try:
        questions = parse_tex(temp_path, subject_hint)
        os.unlink(temp_path)
        return questions
    except:
        os.unlink(temp_path)
        return []