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
  "marks_wrong": -1
}}

CRITICAL RULES:
- Return ONLY the JSON array - first character must be [
- Keep ALL LaTeX exactly as-is: $...$, \\frac{{}}{{}}, \\sqrt{{}}, etc.
- Replace \\includegraphics{{img.png}} with [IMAGE:img.png]
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