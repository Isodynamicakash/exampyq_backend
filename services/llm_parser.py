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

🖼️ IMAGE EXTRACTION - EXTREMELY IMPORTANT:
EVERY TIME you see \\includegraphics{{...}} in the LaTeX:

Step 1: Extract the EXACT filename (including extension)
   Example: \\includegraphics{{images/diagrams/fig1.png}} → filename is "fig1.png"
   Example: \\includegraphics{{photo.jpg}} → filename is "photo.jpg"
   Example: \\includegraphics{{circuit}} → filename is "circuit.png" (assume .png if no extension)
   
   ⚠️ CRITICAL: Copy the actual filename from the LaTeX. DO NOT generate UUIDs or random IDs!

Step 2: Determine which section it's in:
   - In question text? → Add to "q_images" array
   - In solution text? → Add to "sol_images" array
   - In option A/B/C/D? → Add to "opt_images" object with key "a"/"b"/"c"/"d"

Step 3: Replace \\includegraphics{{...}} with [IMAGE:filename]
   Example: "See diagram \\includegraphics{{images/fig1.png}} above"
         → "See diagram [IMAGE:fig1.png] above"

Step 4: Verify arrays are populated with ACTUAL FILENAMES
   ✅ CORRECT: {{"q_images": ["fig1.png"], "question": "See [IMAGE:fig1.png]"}}
   ✅ CORRECT: {{"q_images": ["circuit.png"], "question": "...[IMAGE:circuit.png]..."}}
   ❌ WRONG:   {{"q_images": ["a9821fc1-dc76-..."], ...}}  ← NO UUIDs!
   ❌ WRONG:   {{"q_images": [], "question": "See diagram"}}  ← You forgot the image!

📌 COMMON IMAGE PATTERNS TO WATCH FOR:
- "as shown in figure" → Look for \\includegraphics nearby
- "see diagram" → Look for \\includegraphics nearby
- "in the given circuit" → Look for \\includegraphics nearby
- Any \\includegraphics{{...}} MUST become [IMAGE:...] + added to appropriate array

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
        
        # Post-process: Fix newlines
        questions = _fix_newlines(questions)
        
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


def _fix_newlines(questions: list) -> list:
    """Convert escaped newlines to actual newlines for proper rendering."""
    for q in questions:
        # Fix in question text
        if "question" in q and isinstance(q["question"], str):
            q["question"] = q["question"].replace('\\n', '\n')
        
        # Fix in solution
        if "solution" in q and isinstance(q["solution"], str):
            q["solution"] = q["solution"].replace('\\n', '\n')
        
        # Fix in options
        if "options" in q and isinstance(q["options"], list):
            q["options"] = [
                opt.replace('\\n', '\n') if isinstance(opt, str) else opt 
                for opt in q["options"]
            ]
    
    return questions


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