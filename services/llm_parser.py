"""
services/llm_parser.py
======================
Simple LaTeX parser using Claude Haiku 4.5
Sends ENTIRE paper in ONE API call - no chunking
"""

import os
import re
import json
from typing import Optional
import anthropic

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

HAIKU_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 16000  # Increased for full paper


# ══════════════════════════════════════════════════════════
# MAIN: Parse LaTeX with LLM
# ══════════════════════════════════════════════════════════

async def parse_latex_with_llm(tex: str, api_key: str = None) -> list[dict]:
    """
    Parse entire LaTeX paper in ONE API call.
    
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
    
    print(f"[LLM Parser] Parsing {len(tex)} chars with Haiku 4.5", flush=True)
    
    # Single API call
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""Extract ALL questions from this LaTeX exam paper.

Return ONLY a valid JSON array. Start with [ and end with ]. NO markdown, NO explanations.

For each question:
{{
  "number": 1,
  "q_type": "MCQ",
  "subject": "PHYSICS",
  "question": "Full question with LaTeX...",
  "options": ["A", "B", "C", "D"],
  "answer": "2",
  "solution": "Solution with LaTeX...",
  "chapter_name": "Mechanics",
  "difficulty": "medium",
  "marks_correct": 4,
  "marks_wrong": -1,
  "q_images": ["image1.png", "image2.png"],
  "sol_images": ["sol1.png"],
  "opt_images": {{"a": "opt_a.png", "b": "opt_b.png"}}
}}

CRITICAL IMAGE HANDLING:
- When you see \\includegraphics{{image1.png}} in question text:
  1. Add "image1.png" to q_images array
  2. Replace it with [IMAGE:image1.png] in the question text
- When you see image in solution: add to sol_images array
- When you see image in options: add to opt_images with key "a", "b", "c", or "d"
- Extract ONLY the filename from \\includegraphics{{path/to/image.png}} → "image.png"

FORMATTING RULES:
- **PRESERVE all newlines from original LaTeX** - use \\n for line breaks
- Keep paragraph breaks as \\n\\n (double newline)
- Maintain spacing between question and options
- Keep solution formatting with proper line breaks
- DO NOT merge everything into one continuous line

OTHER RULES:
- Return ONLY the JSON array - first character must be [
- Keep ALL LaTeX exactly as-is: $...$, \\frac{{}}{{}}, \\sqrt{{}}, etc.
- answer MUST be string: "1", "2", "3", or "4" (NOT number)
- Detect chapter_name from question content (e.g., "Optics", "Thermodynamics")
- If chapter unclear, use best guess or leave empty string

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
        print(f"[LLM Parser] Response length: {len(response_text)} chars", flush=True)
        print(f"[LLM Parser] First 200 chars: {response_text[:200]}", flush=True)
        
        # Extract JSON
        questions = _extract_json(response_text)
        
        # Post-process: Fix LaTeX formatting
        questions = _fix_latex_formatting(questions)
        
        print(f"[LLM Parser] ✓ Extracted {len(questions)} questions", flush=True)
        return questions
        
    except Exception as e:
        print(f"[LLM Parser] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return []


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def _fix_latex_formatting(questions: list) -> list:
    """Fix LaTeX formatting - add proper line breaks."""
    for q in questions:
        # Fix question text
        if "question" in q:
            q["question"] = _add_latex_linebreaks(q["question"])
        
        # Fix solution text
        if "solution" in q:
            q["solution"] = _add_latex_linebreaks(q["solution"])
        
        # Fix options if they're strings
        if "options" in q and isinstance(q["options"], list):
            q["options"] = [_add_latex_linebreaks(opt) if isinstance(opt, str) else opt 
                           for opt in q["options"]]
    
    return questions


def _add_latex_linebreaks(text: str) -> str:
    """Add line breaks at appropriate places in LaTeX."""
    if not text:
        return text
    
    # Add line break before equations on new lines
    text = re.sub(r'(\$\$[^$]+\$\$)', r'\n\1\n', text)
    
    # Add line break after periods followed by capital letter (new sentence)
    text = re.sub(r'\. ([A-Z])', r'.\n\1', text)
    
    # Add line break before "Given:", "Find:", etc.
    text = re.sub(r'(Given:|Find:|Calculate:|Determine:)', r'\n\1', text)
    
    # Clean up multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


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
    original_text = text
    
    # Step 1: Remove ALL markdown fences
    text = re.sub(r'```json|```', '', text)
    
    # Step 2: Strip whitespace
    text = text.strip()
    
    # Step 3: Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            print(f"[LLM Parser] ✓ Parsed {len(data)} items (direct)", flush=True)
            return data
    except json.JSONDecodeError as e:
        print(f"[LLM Parser] Direct parse failed: {e}", flush=True)
    
    # Step 4: Find array boundaries
    start = text.find('[')
    end = text.rfind(']')
    
    if start == -1 or end == -1:
        print(f"[LLM Parser] ✗ No JSON array found", flush=True)
        print(f"[LLM Parser] Response: {original_text[:500]}", flush=True)
        return []
    
    # Step 5: Extract JSON substring
    json_str = text[start:end+1]
    
    # Step 6: Try parsing extracted JSON
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            print(f"[LLM Parser] ✓ Parsed {len(data)} items (extracted)", flush=True)
            return data
    except json.JSONDecodeError as e:
        print(f"[LLM Parser] ✗ Parse error: {e}", flush=True)
        print(f"[LLM Parser] Problematic JSON (first 1000 chars):", flush=True)
        print(json_str[:1000], flush=True)
    
    return []