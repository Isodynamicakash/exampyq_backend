"""
services/llm_parser.py
======================
LaTeX parser with smart chunking for large papers (90+ questions)
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

HAIKU_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 64000  # Haiku maximum
QUESTIONS_PER_CHUNK = 30  # Safe limit for 64K tokens


# ══════════════════════════════════════════════════════════
# MAIN: Parse LaTeX with LLM
# ══════════════════════════════════════════════════════════

async def parse_latex_with_llm(tex: str, api_key: str = None) -> list[dict]:
    """
    Parse LaTeX paper - auto chunks if too large.
    
    Args:
        tex: LaTeX source code
        api_key: Anthropic API key
    
    Returns:
        List of question dicts
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
    if not api_key:
        print("[LLM Parser] ⚠ No API key", flush=True)
        return []
    
    # Clean LaTeX
    tex = _clean_latex(tex)
    
    # Smart chunking based on size
    chunks = _smart_chunk(tex)
    
    if len(chunks) == 1:
        print(f"[LLM Parser] Single chunk: {len(tex)} chars", flush=True)
        return await _parse_chunk(tex, api_key, 1, 1)
    else:
        print(f"[LLM Parser] Processing {len(chunks)} chunks for large paper", flush=True)
        
        # Process chunks concurrently
        tasks = [_parse_chunk(chunk, api_key, i+1, len(chunks)) 
                for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_questions = []
        for chunk_questions in results:
            all_questions.extend(chunk_questions)
        
        print(f"[LLM Parser] ✓ Total: {len(all_questions)} questions from {len(chunks)} chunks", flush=True)
        return all_questions


async def _parse_chunk(tex: str, api_key: str, chunk_num: int, total_chunks: int) -> list[dict]:
    """Parse a single chunk of LaTeX."""
    print(f"[LLM Parser] Chunk {chunk_num}/{total_chunks}: {len(tex)} chars", flush=True)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""You are a PRECISE LaTeX extractor. Your job is to COPY questions EXACTLY as written.

⚠️ CRITICAL: DO NOT reformat, rearrange, or "improve" anything. Extract EXACTLY as-is.

Output: PURE JSON array starting with [ ending with ]
NO ```json fences. NO markdown. NO explanations.

Schema:
{{
  "number": 1,
  "q_type": "MCQ",
  "subject": "PHYSICS",
  "question": "EXACT copy from LaTeX...",
  "options": ["EXACT option 1", "EXACT option 2", "EXACT option 3", "EXACT option 4"],
  "answer": "2",
  "solution": "EXACT copy from solution...",
  "chapter_name": "Mechanics",
  "difficulty": "medium",
  "marks_correct": 4,
  "marks_wrong": -1,
  "q_images": [],
  "sol_images": [],
  "opt_images": {{}}
}}

⚠️ EXTRACTION RULES - FOLLOW EXACTLY:
1. **COPY text character-by-character** - do NOT rephrase or simplify
2. **PRESERVE all spacing, newlines, formatting** exactly as in original
3. **Keep ALL LaTeX commands** exactly: $...$, \\frac{{}}{{}}, \\mathrm{{}}, \\sqrt{{}}, subscripts, superscripts
4. **DO NOT remove or add spaces** between LaTeX expressions
5. **DO NOT merge lines** - if text is on separate lines, keep it separate
6. **DO NOT simplify equations** - copy them EXACTLY including all braces and commands

IMAGE HANDLING:
- If \\includegraphics{{img.png}}: add "img.png" to q_images, replace with [IMAGE:img.png]
- Extract only filename, not path

ANSWER FORMAT:
- answer MUST be STRING: "1", "2", "3", or "4"
- If answer is in format "Sol. (3)", extract just "3"

CHAPTER DETECTION:
- Look at question content and guess chapter (e.g., "Optics", "Thermodynamics")
- If unclear, leave empty ""

⚠️ REMEMBER: Your job is EXTRACTION, not CORRECTION. Copy EXACTLY as written, even if formatting seems odd.

LaTeX:
{tex}"""

    try:
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        # Debug
        print(f"[LLM Parser] Chunk {chunk_num} response: {len(response_text)} chars", flush=True)
        
        # Extract JSON
        questions = _extract_json(response_text)
        
        if questions:
            print(f"[LLM Parser] ✓ Chunk {chunk_num}: {len(questions)} questions", flush=True)
        else:
            print(f"[LLM Parser] ✗ Chunk {chunk_num}: Parse failed", flush=True)
            print(f"[LLM Parser] First 500 chars: {response_text[:500]}", flush=True)
        
        return questions
        
    except Exception as e:
        print(f"[LLM Parser] ERROR chunk {chunk_num}: {e}", flush=True)
        return []


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def _smart_chunk(tex: str) -> list[str]:
    """
    Smart chunking based on LaTeX structure.
    Splits at question boundaries (\\item or question numbers).
    """
    # If small enough for single chunk
    if len(tex) < 50000:  # ~50K chars = safe for single call
        return [tex]
    
    # Find question boundaries
    # Look for \item or numbered questions
    pattern = r'(\\item\s+|\\textbf\{\d+\.\}|^\d+\.|\n\d+\.)'
    splits = list(re.finditer(pattern, tex, re.MULTILINE))
    
    if len(splits) < 2:
        # No clear boundaries, split by character count
        chunk_size = 40000
        return [tex[i:i+chunk_size] for i in range(0, len(tex), chunk_size)]
    
    # Split at question boundaries
    chunks = []
    current_chunk = ""
    question_count = 0
    
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i+1].start() if i+1 < len(splits) else len(tex)
        
        question_text = tex[start:end]
        
        # Add to current chunk
        current_chunk += question_text
        question_count += 1
        
        # Create new chunk after N questions
        if question_count >= QUESTIONS_PER_CHUNK:
            chunks.append(current_chunk)
            current_chunk = ""
            question_count = 0
    
    # Add remaining
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [tex]


def _clean_latex(tex: str) -> str:
    """Remove preamble, keep document body."""
    doc_start = tex.find(r'\begin{document}')
    if doc_start != -1:
        tex = tex[doc_start:]
    
    doc_end = tex.find(r'\end{document}')
    if doc_end != -1:
        tex = tex[:doc_end]
    
    return tex.strip()


def _extract_json(text: str) -> list:
    """Extract JSON array with aggressive cleaning."""
    # Remove markdown fences
    text = re.sub(r'```json|```', '', text).strip()
    
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    
    # Find array boundaries
    start = text.find('[')
    end = text.rfind(']')
    
    if start == -1 or end == -1:
        return []
    
    # Extract and parse
    json_str = text[start:end+1]
    
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError as e:
        print(f"[LLM Parser] Parse error: {e}", flush=True)
        print(f"[LLM Parser] JSON preview: {json_str[:500]}", flush=True)
    
    return []