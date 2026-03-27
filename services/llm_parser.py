"""
services/llm_parser.py
======================
LaTeX parser - SINGLE CHUNK ONLY (v2.1)

FEATURES:
✅ Single chunk - full paper in one call
✅ Exam type detection (JEE_MAIN/NEET)
✅ JEE subject validation (no Biology)
✅ Sequential numbering
✅ System marks auto-fill (+4/-1)
✅ Question types: MCQ/MSQ/NUMERICAL
✅ Prompt caching (v2.1)
✅ Image processing
✅ Newline conversion
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
MAX_TOKENS = 64000  # Haiku maximum


# ══════════════════════════════════════════════════════════
# MAIN: Parse LaTeX with LLM
# ══════════════════════════════════════════════════════════

async def parse_latex_with_llm(tex: str, api_key: str = None) -> list[dict]:
    """
    Parse LaTeX paper in SINGLE chunk.
    
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
    
    # Detect exam type
    exam_type = _detect_exam_type(tex)
    print(f"[LLM Parser] 📋 Detected: {exam_type} | Length: {len(tex)} chars", flush=True)
    
    # Parse FULL paper in single call
    questions = _parse_full_paper(tex, api_key, exam_type)
    
    if not questions:
        print(f"[LLM Parser] ✗ Parse failed - no questions extracted", flush=True)
        return []
    
    # Post-process
    questions = _sort_questions(questions)
    questions = _add_marks(questions, exam_type)
    
    print(f"[LLM Parser] ✓ Success: {len(questions)} questions parsed", flush=True)
    return questions


def _detect_exam_type(tex: str) -> str:
    """
    Detect exam type from LaTeX content.
    Returns: JEE_MAIN, JEE_ADVANCED, NEET
    """
    tex_lower = tex.lower()
    
    # Check explicit mentions
    if "jee main" in tex_lower or "jee-main" in tex_lower or "jeemain" in tex_lower:
        return "JEE_MAIN"
    if "jee advanced" in tex_lower or "jee-advanced" in tex_lower or "jeeadvanced" in tex_lower:
        return "JEE_ADVANCED"
    if "neet" in tex_lower:
        return "NEET"
    
    # Check subject patterns
    has_bio = "biology" in tex_lower or "botany" in tex_lower or "zoology" in tex_lower
    has_pcm = all(w in tex_lower for w in ["physics", "chemistry", "math"])
    
    if has_bio:
        return "NEET"
    elif has_pcm:
        return "JEE_MAIN"
    
    return "JEE_MAIN"  # Default


def _parse_full_paper(tex: str, api_key: str, exam_type: str) -> list[dict]:
    """Parse full LaTeX paper in single API call."""
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Build subject validation based on exam type
    if exam_type in ["JEE_MAIN", "JEE_ADVANCED"]:
        subject_rule = """
⚠️ CRITICAL SUBJECT RULE - THIS IS JEE (NOT NEET):
- ONLY these subjects exist: PHYSICS, CHEMISTRY, MATHEMATICS
- BIOLOGY DOES NOT EXIST in JEE papers
- If question seems biology-related → it's CHEMISTRY (Biochemistry/Biomolecules/Organic)

Common mistakes to AVOID:
• Cell biology topics → CHEMISTRY (Biomolecules chapter)
• Photosynthesis/Respiration → CHEMISTRY (Organic Chemistry)
• Proteins/Enzymes/DNA/RNA → CHEMISTRY (Biomolecules/Organic)
• Amino acids → CHEMISTRY (Biomolecules)

Subject must be one of: PHYSICS, CHEMISTRY, MATHEMATICS
"""
    elif exam_type == "NEET":
        subject_rule = """
SUBJECT VALIDATION - THIS IS NEET:
- Valid subjects: PHYSICS, CHEMISTRY, BIOLOGY
- Biology includes Botany and Zoology
"""
    else:
        subject_rule = """
SUBJECT DETECTION:
- Common subjects: PHYSICS, CHEMISTRY, MATHEMATICS, BIOLOGY
- Use UPPERCASE for subject names
"""
    
    prompt = f"""You are a PRECISE LaTeX extractor for competitive exam papers. Your job is to COPY questions EXACTLY as written.

⚠️ CRITICAL: DO NOT reformat, rearrange, or "improve" anything. Extract EXACTLY as-is.

Output: PURE JSON array starting with [ ending with ]
NO ```json fences. NO markdown. NO explanations. NO preamble.

Schema for EACH question:
{{
  "number": 1,
  "q_type": "MCQ",
  "subject": "PHYSICS",
  "question": "EXACT copy from LaTeX...",
  "options": ["EXACT option 1", "EXACT option 2", "EXACT option 3", "EXACT option 4"],
  "answer": "2",
  "solution": "EXACT copy from solution...",
  "chapter_name": "Motion in a Straight Line",
  "topic_name": "Kinematic Equations",
  "difficulty": "medium"
}}

{subject_rule}

⚠️ EXTRACTION RULES - FOLLOW EXACTLY:
1. **COPY text character-by-character** - do NOT rephrase, simplify, or improve
2. **PRESERVE all spacing, newlines, formatting** exactly as in original
3. **Keep ALL LaTeX commands** exactly: $...$, \\frac{{}}{{}}, \\includegraphics{{}}, \\\\, etc.
4. **DO NOT remove or add spaces** between LaTeX expressions
5. **DO NOT merge lines** - if text is on separate lines, keep it separate
6. **DO NOT simplify equations** - copy them EXACTLY including all braces and commands
7. **Keep \\includegraphics{{...}} EXACTLY as written** - we'll process images later
8. **Keep \\\\ (LaTeX line breaks) EXACTLY as written** - we'll convert them later

QUESTION NUMBERING:
- Extract the EXACT question number from LaTeX
- If LaTeX shows "Q1", "Q2", "Q3" → use numbers 1, 2, 3
- If LaTeX shows "1.", "2.", "3." → use numbers 1, 2, 3
- Numbers MUST be sequential: 1, 2, 3, 4, 5, 6...
- DO NOT skip numbers

QUESTION TYPE:
- Detect type from LaTeX structure
- ONLY use these values: "MCQ", "MSQ", "NUMERICAL"
- MCQ = Single correct answer (4 options, 1 correct)
- MSQ = Multiple correct answers (4 options, 2+ correct, answer like "1,2")
- NUMERICAL = No options (direct numerical answer)

ANSWER FORMAT:
- For MCQ: answer MUST be STRING: "1", "2", "3", or "4"
- For MSQ: answer MUST be STRING with comma-separated: "1,2" or "1,3" or "2,4" etc.
- For NUMERICAL: answer is the number as STRING: "5" or "2.5" or "100"
- If answer format is "Sol. (3)" or "Ans. 3", extract just "3"
- If MSQ shows "A, C", convert to "1,3" (A=1, B=2, C=3, D=4)

MARKS FIELDS:
- Do NOT include marks_correct or marks_wrong in output
- System will auto-fill these based on exam type

CHAPTER & TOPIC DETECTION:
- chapter_name: Use standard NCERT chapter names
  Examples: "Motion in a Plane", "Hydrocarbons", "Thermodynamics", "Waves"
- topic_name: Specific topic within chapter
  Examples: "Projectile Motion", "Aromaticity", "First Law", "Doppler Effect"
- If chapter/topic unclear, leave empty ""

SOLUTION:
- Copy ENTIRE solution text EXACTLY as written
- Keep all steps, equations, explanations
- Do NOT summarize or shorten

OPTIONS (for MCQ/MSQ):
- Extract ALL 4 options EXACTLY as written
- Keep options in order: [option1, option2, option3, option4]
- If option is empty, use empty string ""
- Keep all LaTeX formatting in options

⚠️ REMEMBER: 
- Your job is EXTRACTION, not CORRECTION
- Copy EXACTLY as written, even if formatting seems odd
- Extract ALL questions from the paper
- Maintain sequential numbering
- Be precise with subject classification

LaTeX to extract:
{tex}"""

    try:
        print(f"[LLM Parser] Calling API ({exam_type})...", flush=True)
        
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": f"You are a PRECISE LaTeX extractor for {exam_type} question papers (v2.1 - system auto-fills marks). Extract ALL questions EXACTLY as written with accurate subject classification.",
                    "cache_control": {"type": "ephemeral"}  # Cache v2.1
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        print(f"[LLM Parser] Response received: {len(response_text)} chars", flush=True)
        
        # Extract JSON
        questions = _extract_json(response_text)
        
        if not questions:
            print(f"[LLM Parser] ✗ JSON extraction failed", flush=True)
            print(f"[LLM Parser] First 500 chars: {response_text[:500]}", flush=True)
            return []
        
        # Post-process: Fix newlines and images
        questions = _fix_newlines(questions)
        
        print(f"[LLM Parser] ✓ Extracted {len(questions)} questions", flush=True)
        return questions
        
    except Exception as e:
        print(f"[LLM Parser] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return []


# ══════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════

def _sort_questions(questions: list) -> list:
    """Sort questions by number in sequential order."""
    try:
        return sorted(questions, key=lambda q: int(q.get("number", 0) or 0))
    except:
        return questions


def _add_marks(questions: list, exam_type: str) -> list:
    """
    Add marks_correct and marks_wrong based on exam type.
    System adds these - NOT the LLM.
    """
    for q in questions:
        q_type = q.get("q_type", "MCQ")
        
        if exam_type == "JEE_MAIN":
            if q_type == "MCQ":
                q["marks_correct"] = 4
                q["marks_wrong"] = -1
            elif q_type == "MSQ":
                q["marks_correct"] = 4
                q["marks_wrong"] = -1
            elif q_type == "NUMERICAL":
                q["marks_correct"] = 4
                q["marks_wrong"] = 0  # No negative
        
        elif exam_type == "JEE_ADVANCED":
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
            q["marks_correct"] = 4
            q["marks_wrong"] = -1
        
        else:
            # Default
            q["marks_correct"] = 4
            q["marks_wrong"] = -1
    
    return questions


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


def _fix_newlines(questions: list) -> list:
    """
    Post-process questions exactly like parser.py:
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
        # Step 1: Convert LaTeX line breaks (\\) to actual newlines (\n)
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


def _extract_json(text: str) -> list:
    """Extract JSON array with aggressive cleaning."""
    # Remove markdown fences
    text = re.sub(r'```json|```', '', text).strip()
    
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    
    # Find array boundaries
    start = text.find('[')
    end = text.rfind(']')
    
    if start == -1 or end == -1:
        print(f"[LLM Parser] No JSON array found in response", flush=True)
        return []
    
    # Extract and parse
    json_str = text[start:end+1]
    
    # Try parsing
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        else:
            return []
    except json.JSONDecodeError as e:
        print(f"[LLM Parser] JSON parse error: {e}", flush=True)
        print(f"[LLM Parser] JSON preview: {json_str[:300]}", flush=True)
    
    # NUCLEAR OPTION: Protect valid JSON escapes, then escape LaTeX
    # Problem: LLM outputs \section, \frac which are invalid JSON escapes
    # Solution: Protect \n, \t, \\, \" then escape all other backslashes
    
    try:
        # List of VALID JSON escapes we want to preserve
        valid_escapes = ['\\n', '\\t', '\\r', '\\\\', '\\"', '\\/']
        
        # Step 1: Replace valid escapes with placeholders
        protected = json_str
        for i, esc in enumerate(valid_escapes):
            protected = protected.replace(esc, f'<<<VALID{i}>>>')
        
        # Step 2: Now escape ALL remaining backslashes (LaTeX commands like \section, \frac)
        protected = protected.replace('\\', '\\\\')
        
        # Step 3: Restore the valid JSON escapes
        for i, esc in enumerate(valid_escapes):
            protected = protected.replace(f'<<<VALID{i}>>>', esc)
        
        data = json.loads(protected)
        if isinstance(data, list):
            print(f"[LLM Parser] ✓ Fixed LaTeX escapes", flush=True)
            return data
    except Exception as e2:
        print(f"[LLM Parser] Escape fix failed: {e2}", flush=True)
    
    # Last resort: Remove trailing commas
    try:
        fixed = re.sub(r',(\s*[}\]])', r'\1', json_str)
        data = json.loads(fixed)
        if isinstance(data, list):
            print(f"[LLM Parser] ✓ Fixed trailing commas", flush=True)
            return data
    except:
        pass
        
        # Fix 3: Try to salvage partial data
        # Extract valid JSON objects one by one
        try:
            objects = []
            # Find object boundaries
            depth = 0
            obj_start = None
            
            for i, char in enumerate(json_str):
                if char == '{':
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        obj_text = json_str[obj_start:i+1]
                        try:
                            obj = json.loads(obj_text)
                            objects.append(obj)
                        except:
                            # Try fixing this object
                            try:
                                fixed_obj = obj_text.replace('\\', '\\\\')
                                fixed_obj = fixed_obj.replace('\\\\n', '\\n')
                                fixed_obj = fixed_obj.replace('\\\\t', '\\t')
                                obj = json.loads(fixed_obj)
                                objects.append(obj)
                            except:
                                pass
                        obj_start = None
            
            if objects:
                print(f"[LLM Parser] ✓ Salvaged {len(objects)} objects from partial JSON", flush=True)
                return objects
        except Exception as e3:
            print(f"[LLM Parser] Salvage failed: {e3}", flush=True)
    
    return []