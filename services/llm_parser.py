"""
services/llm_parser.py
======================
LaTeX parser with smart chunking for large papers (90+ questions)

FIXES:
1. ✅ Exam type detection (JEE Main = no Biology!)
2. ✅ Sequential question numbering
3. ✅ Question types: MCQ, MSQ, NUMERICAL only
4. ✅ Marks auto-filled by system (not LLM)
5. ✅ Prompt caching for cost reduction
"""

import os
import re
import json
import asyncio
from typing import Optional
import anthropic

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════

HAIKU_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 64000  # Haiku maximum
QUESTIONS_PER_CHUNK = 30  # Safe limit for 64K tokens


# ══════════════════════════════════════════════════════════════
# MAIN: Parse LaTeX with LLM
# ══════════════════════════════════════════════════════════════

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
    
    # Detect exam type from LaTeX content
    exam_type = _detect_exam_type(tex)
    print(f"[LLM Parser] 📋 Detected exam type: {exam_type}", flush=True)
    
    # Smart chunking based on size
    chunks = _smart_chunk(tex)
    
    if len(chunks) == 1:
        print(f"[LLM Parser] Single chunk: {len(tex)} chars", flush=True)
        questions = await _parse_chunk(tex, api_key, 1, 1, exam_type)
    else:
        print(f"[LLM Parser] Processing {len(chunks)} chunks for large paper", flush=True)
        
        # Process chunks concurrently
        tasks = [_parse_chunk(chunk, api_key, i+1, len(chunks), exam_type) 
                for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        questions = []
        for chunk_questions in results:
            questions.extend(chunk_questions)
        
        print(f"[LLM Parser] ✓ Total: {len(questions)} questions from {len(chunks)} chunks", flush=True)
    
    # Post-process: Sort by question number, add marks, fix images/newlines
    questions = _sort_questions(questions)
    questions = _add_marks(questions, exam_type)
    questions = _fix_newlines(questions)
    
    return questions


# ══════════════════════════════════════════════════════════════
# EXAM TYPE DETECTION
# ══════════════════════════════════════════════════════════════

def _detect_exam_type(tex: str) -> str:
    """
    Detect exam type from LaTeX content.
    
    Returns: "JEE_MAIN", "JEE_ADVANCED", "NEET", or "UNKNOWN"
    """
    tex_lower = tex.lower()
    
    # Check for explicit exam mentions
    if "jee main" in tex_lower or "jee-main" in tex_lower or "jeemain" in tex_lower:
        return "JEE_MAIN"
    if "jee advanced" in tex_lower or "jee-advanced" in tex_lower or "jeeadvanced" in tex_lower:
        return "JEE_ADVANCED"
    if "neet" in tex_lower:
        return "NEET"
    
    # Check subject patterns
    has_biology = any(word in tex_lower for word in [
        "biology", "botany", "zoology", "cell", "photosynthesis", "respiration",
        "genetics", "evolution", "ecology", "reproduction"
    ])
    
    has_chemistry = any(word in tex_lower for word in [
        "chemistry", "chemical", "organic", "inorganic", "physical chemistry"
    ])
    
    has_physics = any(word in tex_lower for word in [
        "physics", "mechanics", "optics", "thermodynamics", "electrostatics"
    ])
    
    has_maths = any(word in tex_lower for word in [
        "mathematics", "calculus", "algebra", "trigonometry", "vectors"
    ])
    
    # NEET has Biology, JEE Main/Advanced don't
    if has_biology:
        return "NEET"
    elif has_chemistry and has_physics and has_maths:
        # Has PCM, no Biology → JEE Main (default)
        return "JEE_MAIN"
    
    return "UNKNOWN"


# ══════════════════════════════════════════════════════════════
# LLM PARSING
# ══════════════════════════════════════════════════════════════

async def _parse_chunk(tex: str, api_key: str, chunk_num: int, total_chunks: int, exam_type: str = "UNKNOWN") -> list[dict]:
    """Parse a single chunk of LaTeX."""
    print(f"[LLM Parser] Chunk {chunk_num}/{total_chunks}: {len(tex)} chars (Exam: {exam_type})", flush=True)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Build subject constraints based on exam type
    if exam_type == "JEE_MAIN" or exam_type == "JEE_ADVANCED":
        subject_instruction = """
⚠️ CRITICAL SUBJECT RULE - THIS IS JEE (NOT NEET):
- ONLY these subjects exist: PHYSICS, CHEMISTRY, MATHEMATICS
- BIOLOGY DOES NOT EXIST in JEE papers
- If a question seems related to biology/life sciences → it's actually CHEMISTRY (Biochemistry/Organic)
- Common mistakes to avoid:
  * Cell biology topics → CHEMISTRY (Biomolecules)
  * Photosynthesis/Respiration → CHEMISTRY (Organic Chemistry)
  * Proteins/Enzymes → CHEMISTRY (Biomolecules)
  * DNA/RNA → CHEMISTRY (Organic Chemistry)
  * Amino acids → CHEMISTRY (Biomolecules)
"""
    elif exam_type == "NEET":
        subject_instruction = """
SUBJECT VALIDATION - THIS IS NEET:
- Valid subjects: PHYSICS, CHEMISTRY, BIOLOGY
- Biology includes: Botany, Zoology, Life Sciences
"""
    else:
        subject_instruction = """
SUBJECT DETECTION:
- Common subjects: PHYSICS, CHEMISTRY, MATHEMATICS, BIOLOGY
- Use UPPERCASE for subject names
"""
    
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
  "chapter_name": "Atoms",
  "topic_name": "Bohr Model",
  "difficulty": "medium"
}}

{subject_instruction}

⚠️ EXTRACTION RULES - FOLLOW EXACTLY:
1. **COPY text character-by-character** - do NOT rephrase or simplify
2. **PRESERVE all spacing, newlines, formatting** exactly as in original
3. **Keep ALL LaTeX commands** exactly: $...$, \\frac{{}}{{}}, \\includegraphics{{}}, \\\\, etc.
4. **DO NOT remove or add spaces** between LaTeX expressions
5. **DO NOT merge lines** - if text is on separate lines, keep it separate
6. **DO NOT simplify equations** - copy them EXACTLY including all braces and commands
7. **Keep \\includegraphics{{...}} EXACTLY as written** - we'll process images later
8. **Keep \\\\ (LaTeX line breaks) EXACTLY as written** - we'll convert them later

QUESTION NUMBERING:
- Extract the EXACT question number from LaTeX (e.g., Q1, Q2, Q3... or 1, 2, 3...)
- Question numbers MUST be sequential: 1, 2, 3, 4, 5...
- If LaTeX shows "Q15", use number: 15

QUESTION TYPE:
- Detect type from LaTeX content
- ONLY use these values: "MCQ", "MSQ", "NUMERICAL"
- MCQ = Single correct (4 options, 1 answer)
- MSQ = Multiple correct (4 options, multiple answers like "1,2")
- NUMERICAL = No options (direct numerical answer)

ANSWER FORMAT:
- For MCQ: answer MUST be STRING: "1", "2", "3", or "4"
- For MSQ: answer MUST be STRING with comma-separated: "1,2", "1,3", "2,4", etc.
- For NUMERICAL: answer is the number as STRING: "5", "2.5", "100"
- If answer is in format "Sol. (3)", extract just "3"

MARKS:
- Do NOT add marks_correct or marks_wrong fields
- System will auto-fill these based on exam type

CHAPTER & TOPIC DETECTION:
- chapter_name: Use standard NCERT chapter names (e.g., "Motion in a Plane", "Hydrocarbons")
- topic_name: Specific topic within chapter (e.g., "Projectile Motion", "Aromaticity")
- If unclear, leave empty ""

⚠️ REMEMBER: Your job is EXTRACTION, not CORRECTION. Copy EXACTLY as written, even if formatting seems odd.

LaTeX:
{tex}"""

    try:
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": f"You are a PRECISE LaTeX extractor for {exam_type} question papers. Extract questions EXACTLY as written.",
                    "cache_control": {"type": "ephemeral"}  # Cache the instructions!
                }
            ],
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


# ══════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════

def _sort_questions(questions: list) -> list:
    """Sort questions by number in ascending order."""
    try:
        return sorted(questions, key=lambda q: int(q.get("number", 0) or 0))
    except:
        return questions


def _add_marks(questions: list, exam_type: str) -> list:
    """
    Add marks_correct and marks_wrong based on exam type and question type.
    System adds these - LLM does NOT.
    """
    for q in questions:
        q_type = q.get("q_type", "MCQ")
        
        if exam_type == "JEE_MAIN":
            # JEE Main marking scheme
            if q_type == "MCQ":
                q["marks_correct"] = 4
                q["marks_wrong"] = -1
            elif q_type == "MSQ":
                q["marks_correct"] = 4
                q["marks_wrong"] = -1
            elif q_type == "NUMERICAL":
                q["marks_correct"] = 4
                q["marks_wrong"] = 0  # No negative in numerical
        
        elif exam_type == "JEE_ADVANCED":
            # JEE Advanced marking scheme (varies by section)
            if q_type == "MCQ":
                q["marks_correct"] = 3
                q["marks_wrong"] = -1
            elif q_type == "MSQ":
                q["marks_correct"] = 4
                q["marks_wrong"] = -2
            elif q_type == "NUMERICAL":
                q["marks_correct"] = 3
                q["marks_wrong"] = 0
        
        elif exam_type == "NEET":
            # NEET marking scheme
            q["marks_correct"] = 4
            q["marks_wrong"] = -1
        
        else:
            # Default marking
            q["marks_correct"] = 4
            q["marks_wrong"] = -1
    
    return questions


def _fix_newlines(questions: list) -> list:
    """
    Post-process questions EXACTLY like parser.py:
    1. Convert LaTeX line breaks (\\) to actual newlines
    2. Extract images from \includegraphics{...} and replace with [IMAGE:...]
    3. Populate q_images and sol_images arrays
    """
    import re
    import os
    
    RE_INCLUDEGFX = re.compile(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}')
    RE_PLACEHOLDER = re.compile(r'\[IMAGE:([^\]]+)\]')
    
    def _unique(lst):
        """Remove duplicates while preserving order."""
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    def _extract_images(text):
        """
        Extract images from \includegraphics{...} and replace with [IMAGE:...]
        Returns: (modified_text, list_of_image_ids)
        """
        if not text:
            return text, []
        
        ids = []
        def _rep(m):
            img_id = os.path.basename(m.group(1).strip())
            ids.append(img_id)
            return f"[IMAGE:{img_id}]"
        
        modified = RE_INCLUDEGFX.sub(_rep, text).strip()
        return modified, ids
    
    for q in questions:
        # Step 1: Convert LaTeX line breaks to newlines
        if "question" in q and isinstance(q["question"], str):
            q["question"] = q["question"].replace('\\\\', '\n')
        
        if "solution" in q and isinstance(q["solution"], str):
            q["solution"] = q["solution"].replace('\\\\', '\n')
        
        if "options" in q and isinstance(q["options"], list):
            q["options"] = [
                opt.replace('\\\\', '\n') if isinstance(opt, str) else opt 
                for opt in q["options"]
            ]
        
        # Step 2: Extract images (EXACTLY like parser.py _postprocess)
        q["question"], qi = _extract_images(q.get("question", ""))
        q["solution"], si = _extract_images(q.get("solution", ""))
        
        # Extract from options
        co = []
        oi = []
        for opt in q.get("options", []):
            c, o = _extract_images(opt)
            co.append(c)
            oi.extend(o)
        q["options"] = co
        
        # Step 3: Collect all image IDs from placeholders
        q_ids = RE_PLACEHOLDER.findall(q.get("question", ""))
        o_ids = []
        for opt in q.get("options", []):
            o_ids.extend(RE_PLACEHOLDER.findall(opt))
        s_ids = RE_PLACEHOLDER.findall(q.get("solution", ""))
        
        # Step 4: Populate q_images and sol_images arrays
        q["q_images"] = _unique(q_ids + o_ids + qi + oi)
        q["sol_images"] = _unique(s_ids + si)
    
    return questions


# Continue in PART 2...

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _smart_chunk(tex: str) -> list[str]:
    """
    Smart chunking based on LaTeX structure.
    Splits at question boundaries (\\item or question numbers).
    """
    # Detect question pattern
    item_pattern = r'\\item'
    num_pattern = r'\n\s*(\d+)\s*\.?\s+'
    
    if item_pattern in tex:
        # Split by \item
        parts = re.split(r'(\\item)', tex)
        chunks = []
        current_chunk = ""
        count = 0
        
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                segment = parts[i] + parts[i+1] + (parts[i+2] if i+2 < len(parts) else "")
                if count >= QUESTIONS_PER_CHUNK and len(current_chunk) > 1000:
                    chunks.append(current_chunk)
                    current_chunk = segment
                    count = 1
                else:
                    current_chunk += segment
                    count += 1
        
        if current_chunk.strip():
            chunks.append(current_chunk)
    
    elif re.search(num_pattern, tex):
        # Split by question numbers
        parts = re.split(num_pattern, tex)
        chunks = []
        current_chunk = parts[0] if parts else ""
        count = 0
        
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                segment = f"\n{parts[i]}. {parts[i+1]}"
                if count >= QUESTIONS_PER_CHUNK and len(current_chunk) > 1000:
                    chunks.append(current_chunk)
                    current_chunk = segment
                    count = 1
                else:
                    current_chunk += segment
                    count += 1
        
        if current_chunk.strip():
            chunks.append(current_chunk)
    
    else:
        # No clear structure - chunk by character count
        chunk_size = 15000  # ~30 questions worth
        chunks = [tex[i:i+chunk_size] for i in range(0, len(tex), chunk_size)]
    
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
        return [data] if isinstance(data, dict) else []
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON array
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    
    # Last resort: try to fix common issues
    try:
        # Remove trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
        # Fix unescaped quotes in strings (simple attempt)
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except:
        pass
    
    return []