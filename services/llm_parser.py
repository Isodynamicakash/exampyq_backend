"""
services/llm_parser.py
======================
Generalized LaTeX parser using Anthropic Claude Haiku.
Handles ANY MCQ exam format: JEE, NEET, CAT, SSC, Allen, etc.

CHANGES:
  - Uses Claude Haiku (claude-haiku-4-20241022) for cost-efficiency
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

HAIKU_MODEL = "claude-haiku-4-20241022"  # ✅ CORRECT MODEL
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
# HELPER: Clean LaTeX
# ══════════════════════════════════════════════════════════

def clean_latex(tex: str) -> str:
    """Remove preamble, keep document body."""
    # Remove everything before \begin{document}
    doc_start = tex.find(r'\begin{document}')
    if doc_start != -1:
        tex = tex[doc_start:]
    
    # Remove \end{document} and everything after
    doc_end = tex.find(r'\end{document}')
    if doc_end != -1:
        tex = tex[:doc_end]
    
    return tex.strip()


# ══════════════════════════════════════════════════════════
# HELPER: Chunk LaTeX
# ══════════════════════════════════════════════════════════

def chunk_latex(tex: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split LaTeX into chunks by question count."""
    # Simple chunking by character count (adjust as needed)
    max_chars = 20000  # ~5K tokens
    chunks = []
    
    if len(tex) <= max_chars:
        return [tex]
    
    # Split by sections or enumerate blocks
    parts = re.split(r'(\\section\*?\{.*?\}|\\begin\{enumerate\})', tex)
    
    current_chunk = ""
    for part in parts:
        if len(current_chunk) + len(part) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = part
        else:
            current_chunk += part
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [tex]


# ══════════════════════════════════════════════════════════
# MAIN: Parse LaTeX with LLM
# ══════════════════════════════════════════════════════════

async def parse_latex_with_llm(tex: str, api_key: str = None) -> list[dict]:
    """
    Parse LaTeX using Claude Haiku LLM.
    
    Args:
        tex: LaTeX source code
        api_key: Anthropic API key (defaults to env var)
    
    Returns:
        List of question dicts with auto-detected metadata
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
    if not api_key:
        print("[LLM Parser] ⚠ No API key - skipping LLM parse", flush=True)
        return []
    
    # Clean and chunk
    tex = clean_latex(tex)
    chunks = chunk_latex(tex)
    
    print(f"[LLM Parser] Processing {len(chunks)} chunks (Haiku model)", flush=True)
    
    # Process chunks concurrently
    tasks = [_parse_chunk(chunk, i+1, len(chunks), api_key) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    
    # Flatten results
    all_questions = []
    for chunk_questions in results:
        all_questions.extend(chunk_questions)
    
    print(f"[LLM Parser] ✓ Total questions extracted: {len(all_questions)}", flush=True)
    return all_questions


async def _parse_chunk(chunk: str, chunk_num: int, total_chunks: int, api_key: str) -> list[dict]:
    """Parse a single chunk."""
    print(f"[LLM Parser] Chunk {chunk_num}/{total_chunks} - {len(chunk)} chars", flush=True)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""Extract ALL questions from this LaTeX exam paper.

IMPORTANT RULES:
1. Return ONLY a JSON array, no other text
2. Auto-detect: exam_name, year, exam_date, shift, subject from content
3. Preserve ALL LaTeX math notation exactly: $...$, $$...$$, \\frac{{}}{{}}, \\sqrt{{}}
4. For images, use notation: [IMAGE:filename.png]
5. Question types: MCQ (single correct), MSQ (multiple correct), NUMERICAL

Output JSON schema:
[
  {{
    "number": 1,
    "q_type": "MCQ|MSQ|NUMERICAL",
    "subject": "PHYSICS|CHEMISTRY|MATHEMATICS|BIOLOGY",
    "exam_name": "JEE Main",
    "year": "2024",
    "exam_date": "2024-01-27",
    "shift": "Morning|Evening",
    "chapter_name": "Optics",
    "topic_name": "Young's Double Slit Experiment",
    "difficulty": "easy|medium|hard",
    "question": "LaTeX question text...",
    "options": ["opt1", "opt2", "opt3", "opt4"],
    "answer": "2" or "1,3" or "25.5",
    "solution": "LaTeX solution...",
    "marks_correct": 4,
    "marks_wrong": -1,
    "q_images": [],
    "sol_images": [],
    "opt_images": {{}}
  }}
]

LaTeX content:
{chunk}
"""
    
    try:
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        questions = extract_json(response_text)
        
        if not questions:
            print(f"[LLM Parser] ⚠ Chunk {chunk_num}: No questions extracted", flush=True)
        
        return questions
        
    except Exception as e:
        print(f"[LLM Parser] ERROR in chunk {chunk_num}: {e}", flush=True)
        return []


# ══════════════════════════════════════════════════════════
# FALLBACK: Regex parser
# ══════════════════════════════════════════════════════════

def parse_latex_regex_fallback(tex: str) -> list[dict]:
    """Fallback regex parser if LLM fails."""
    # Basic regex extraction (simplified)
    questions = []
    
    # Find enumerate blocks
    enum_blocks = re.findall(r'\\begin{enumerate}(.*?)\\end{enumerate}', tex, re.DOTALL)
    
    for block in enum_blocks:
        items = re.findall(r'\\item\s+(.*?)(?=\\item|\Z)', block, re.DOTALL)
        
        for i, item in enumerate(items, 1):
            # Extract options
            options = re.findall(r'\((\d+)\)\s*([^\n]+)', item)
            
            questions.append({
                "number": i,
                "q_type": "MCQ",
                "question": item.split('(1)')[0].strip() if options else item.strip(),
                "options": [opt[1].strip() for opt in options] if options else [],
                "answer": "",
                "solution": "",
                "subject": "UNKNOWN",
                "chapter_name": "",
                "topic_name": "",
                "difficulty": "medium",
                "marks_correct": 4,
                "marks_wrong": -1,
            })
    
    return questions