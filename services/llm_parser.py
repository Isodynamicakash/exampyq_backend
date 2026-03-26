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

HAIKU_MODEL = "claude-haiku-4-5"  # ✅ CORRECT - Haiku 4.5
MAX_TOKENS = 8000  # Increased for full question extraction
CHUNK_SIZE = 10  # questions per chunk


# ══════════════════════════════════════════════════════════
# HELPER: Extract JSON from LLM response
# ══════════════════════════════════════════════════════════

def extract_json(text: str) -> list:
    """Extract JSON array from LLM response, handling markdown fences."""
    # Remove ALL markdown code fences (anywhere in text)
    text = re.sub(r'```(?:json)?', '', text)  # Remove ```json and ```
    text = text.strip()
    
    # Try direct parse first
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        # If direct parse fails, try to find JSON array
        pass
    
    # Find JSON array pattern (greedy match)
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
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
    
    prompt = f"""You are a LaTeX exam parser. Extract EVERY question from this content.

CRITICAL: Return ONLY a raw JSON array. NO markdown fences (no ```json), NO explanations, JUST the JSON array starting with [ and ending with ].

For each question, output:
{{
  "number": <question number>,
  "q_type": "MCQ",
  "subject": "PHYSICS" or "CHEMISTRY" or "MATHEMATICS" or "BIOLOGY",
  "question": "<full question text with LaTeX>",
  "options": ["option A", "option B", "option C", "option D"],
  "answer": "<correct option number as STRING like '1' or '2'>",
  "solution": "<solution text with LaTeX>",
  "exam_name": "<exam name from title>",
  "year": "<year from title>",
  "exam_date": "<date in YYYY-MM-DD>",
  "shift": "Morning" or "Evening",
  "chapter_name": "",
  "topic_name": "",
  "difficulty": "medium",
  "marks_correct": 4,
  "marks_wrong": -1
}}

RULES:
- Return raw JSON ONLY - start response with [ character
- Keep ALL LaTeX exactly as-is: $...$, \\frac{{}}{{}}, \\sqrt{{}}, etc.
- If image reference like \\includegraphics{{image1.png}}, replace with [IMAGE:image1.png]
- Extract question number from \\item or numbering
- For options, look for (1), (2), (3), (4) or (A), (B), (C), (D)
- Answer must be STRING: "1", "2", "3", or "4" (NOT integer)
- Solution is usually after "Sol." or in solution section
- Return [] if no questions found

LaTeX:
{chunk}"""
    
    try:
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        # Debug: Print first 500 chars of response
        print(f"[LLM Parser] Chunk {chunk_num} response preview: {response_text[:500]}...", flush=True)
        
        questions = extract_json(response_text)
        
        if not questions:
            print(f"[LLM Parser] ⚠ Chunk {chunk_num}: No questions extracted from response", flush=True)
            print(f"[LLM Parser] Full response: {response_text[:2000]}", flush=True)
        else:
            print(f"[LLM Parser] ✓ Chunk {chunk_num}: Extracted {len(questions)} questions", flush=True)
        
        return questions
        
    except Exception as e:
        print(f"[LLM Parser] ERROR in chunk {chunk_num}: {e}", flush=True)
        import traceback
        print(f"[LLM Parser] Traceback: {traceback.format_exc()}", flush=True)
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