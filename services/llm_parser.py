"""
services/llm_parser.py
======================
LaTeX parser - SINGLE CHUNK ONLY (v2.3 - PRECISION UPDATE)

UPDATES:
✅ Literal Copy-Paste Engine - NO \n or \r injection
✅ Enhanced prompt with precision extraction rules
✅ Updated _convert_newlines to remove LLM artifacts
✅ Simplified _clean_backslashes to protect LaTeX commands
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
MUST be valid JSON - all backslashes properly escaped for JSON strings.

Schema for EACH question:
{{
  "number": 1,
  "q_type": "MCQ",
  "subject": "PHYSICS",
  "question": "EXACT copy from LaTeX with JSON escaping...",
  "options": ["EXACT option 1", "EXACT option 2", "EXACT option 3", "EXACT option 4"],
  "answer": "2",
  "solution": "EXACT copy from solution with JSON escaping...",
  "chapter_name": "Motion in a Straight Line",
  "topic_name": "Kinematic Equations",
  "difficulty": "medium"
}}

{subject_rule}

⚠️ EXTRACTION RULES - FOLLOW EXACTLY:
1. **LITERAL EXTRACTION**: You are a copy-paste machine. Do NOT inject the literal string characters `\\n` or `\\r`. If the LaTeX source has a line break, use a single white-space to join the text.
2. **COPY text character-by-character** - do NOT rephrase, simplify, or improve
3. **PRESERVE all LaTeX commands** in JSON-escaped format (see examples below)
4. **Output valid JSON** - all backslashes must be escaped
5. **NO FORMATTING**: Do not summarize, do not fix typos, and do not add your own formatting markers.

⚠️ MANDATORY JSON ENCODING RULE:
You MUST use DOUBLE-backslashes for ALL LaTeX commands so they are valid inside a JSON string.

**CRITICAL**: When you write JSON, you are writing a JSON STRING. In JSON strings, backslashes must be escaped!

Examples of what YOU must type in your JSON output:
- To show `\frac{{1}}{{2}}` in LaTeX → YOU TYPE: `"\\\\frac{{{{1}}}}{{{{2}}}}"`
- To show `\alpha` in LaTeX → YOU TYPE: `"\\\\alpha"`
- To show `\includegraphics{{file}}` → YOU TYPE: `"\\\\includegraphics{{{{file}}}}"`
- To show `$x^2$` → YOU TYPE: `"$x^2$"` (dollar signs don't need escaping)

**DO NOT use literal "\\n" or "\\r" characters.** If the LaTeX has a line break, use a SPACE instead.

✅ CORRECT JSON you must write:
{{
  "question": "Find $\\\\frac{{{{a}}}}{{{{b}}}}$ where $a = 5$",
  "solution": "Step 1: Use \\\\frac{{{{a}}}}{{{{b}}}} then solve"
}}

❌ WRONG JSON (will cause parser to crash):
{{
  "question": "Find $\\frac{{{{a}}}}{{{{b}}}}$",  ← WRONG! Single backslash breaks JSON!
  "solution": "Step 1\\n\\nUse formula"  ← WRONG! Literal \\n breaks everything!
}}

**Remember**: You are writing a JSON file. The JSON parser needs `\\\\` to represent a single `\` character.

⚠️ CRITICAL - IMAGE HANDLING:
- If LaTeX has `\includegraphics{{34599.img}}`, output in JSON: `"\\includegraphics{{34599.img}}"`
- Use SINGLE curly braces around filename: `{{filename}}` NOT `{{{{filename}}}}`
- DO NOT skip `\includegraphics` commands
- DO NOT forget image filenames like 34599.img, 45678.img etc.
- Every image in LaTeX MUST appear in your JSON output with proper escaping

Examples:
✅ CORRECT: `"\\includegraphics{{34599.img}}"`
✅ CORRECT: `"See diagram: \\includegraphics{{image_001.png}}"`
❌ WRONG: `"\\includegraphics{{{{34599.img}}}}"` (too many braces)

⚠️ CRITICAL - QUESTION vs OPTIONS SEPARATION:
- "question" field = ONLY the question statement, NO OPTIONS
- "options" field = ONLY the 4 answer choices

Example of WRONG extraction:
❌ "question": "What is 2+2? (A) 1 (B) 2 (C) 3 (D) 4"
❌ "options": ["(A) 1", "(B) 2", "(C) 3", "(D) 4"]

Example of CORRECT extraction:
✅ "question": "What is 2+2?"
✅ "options": ["1", "2", "3", "4"]

Rules for options:
- Extract ONLY the answer choice text
- DO NOT include (A), (B), (C), (D) labels
- DO NOT include the question in options
- Each option should be concise - just the answer choice

⚠️ CRITICAL - OUTPUT RAW LaTeX:
- Your job is PURE EXTRACTION - output exactly what you see in LaTeX
- Keep LaTeX commands like \\frac{{a}}{{b}}, $x^2$, \\includegraphics{{filename}}
- Keep \\\\ exactly as written (we convert to newlines later)

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
- Keep all \\includegraphics in solution too

OPTIONS (for MCQ/MSQ):
- Extract ALL 4 options EXACTLY as written
- Keep options in order: [option1, option2, option3, option4]
- If option is empty, use empty string ""
- Keep all LaTeX formatting in options (including \\includegraphics)

⚠️ REMEMBER: 
- Your job is EXTRACTION, not CORRECTION
- Copy EXACTLY as written, even if formatting seems odd
- Extract ALL questions from the paper
- Maintain sequential numbering
- Be precise with subject classification
- Keep question and options separate

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
                    "text": f"You are a PRECISE LaTeX extractor for {exam_type} question papers (v2.3 - Literal Copy-Paste Engine). Extract ALL questions with LaTeX commands properly escaped for JSON. DO NOT inject \\n or \\r string literals. CRITICAL: Keep question text and options SEPARATE.",
                    "cache_control": {"type": "ephemeral"}
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
        
        # Post-process: Convert newlines and extract images
        questions = _postprocess_latex(questions)
        
        # Debug: Show first question processing
        if questions and len(questions) > 0:
            q = questions[0]
            print(f"[LLM Parser] Sample Q1 after processing:", flush=True)
            print(f"  Question preview: {q.get('question', '')[:100]}...", flush=True)
            print(f"  Has images: q={len(q.get('q_images', []))}, sol={len(q.get('sol_images', []))}", flush=True)
        
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
# LATEX POST-PROCESSING
# ══════════════════════════════════════════════════════════

def _postprocess_latex(questions: list) -> list:
    r"""
    Post-process LaTeX content:
    1. Remove LLM-generated \n and \r artifacts
    2. Convert \\ (LaTeX line breaks) to actual newlines
    3. Extract images from \includegraphics{...} → [IMAGE:...]
    4. Populate q_images and sol_images arrays
    5. Clean stray backslashes
    """
    import re
    import os
    
    # Regex patterns for image extraction
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
    
    def _convert_newlines(text):
        """
        STEP 1: Remove literal \\n or \\r strings injected by the LLM.
        STEP 2: Collapse all white-space into a single space.
        
        This mimics the clean output of the original regex parser.py
        """
        if not text:
            return text
        
        # 1. Remove literal "\\n" or "\\r" strings injected by the LLM
        # Replacing them with a space ensures LaTeX commands don't get 'stuck' to text
        text = text.replace('\\n', ' ').replace('\\r', ' ')
        
        # 2. Collapse all white-space (newlines, tabs, multiple spaces) into a single space
        # This mimics the clean output of the original regex parser.py
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_images(text):
        r"""
        Extract images from \includegraphics{...} and replace with [IMAGE:...]
        Handles multiple formats:
        - \\includegraphics{file}       (JSON-escaped)
        - \includegraphics{file}        (raw)
        - \\includegraphics[options]{file}
        - \\includegraphics{{file}}     (double braces from LLM JSON)
        Returns: (modified_text, list_of_image_ids)
        """
        if not text:
            return text, []
        
        ids = []
        
        def _rep(m):
            # Extract filename from match
            filename = m.group(1).strip()
            
            # Remove extra curly braces if LLM added them
            # LLM might send: \includegraphics{{filename}} in JSON
            # which Python reads as: \includegraphics{filename}
            # but sometimes the braces are in the filename itself
            filename = filename.strip('{}')
            
            # Get just the basename (remove any path)
            img_id = os.path.basename(filename)
            
            # Clean the image ID - remove any remaining braces
            img_id = img_id.strip('{}')
            
            ids.append(img_id)
            return f"[IMAGE:{img_id}]"
        
        # Replace \includegraphics with placeholder
        # Pattern handles: \includegraphics[options]{file} or \includegraphics{file}
        modified = RE_INCLUDEGFX.sub(_rep, text)
        
        return modified, ids
    
    def _clean_backslashes(text):
        r"""
        ONLY perform basic whitespace cleanup.
        DO NOT use regex to delete backslashes followed by non-letters.
        This protects LaTeX commands from being broken.
        """
        if not text:
            return text
        
        # ONLY remove backslash before space: "is\\ " → "is "
        text = text.replace('\\ ', ' ')
        
        # Collapse multiple spaces into one
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    for q in questions:
        # STEP 1: Convert newlines (removes \\n and \\r artifacts)
        q["question"] = _convert_newlines(q.get("question", ""))
        q["solution"] = _convert_newlines(q.get("solution", ""))
        q["options"] = [_convert_newlines(opt) for opt in q.get("options", [])]
        
        # STEP 2: Extract images from question and solution
        q["question"], q_img_ids = _extract_images(q["question"])
        q["solution"], sol_img_ids = _extract_images(q["solution"])
        
        # STEP 3: Extract images from options
        cleaned_options = []
        opt_img_ids = []
        for opt in q.get("options", []):
            cleaned_opt, opt_ids = _extract_images(opt)
            cleaned_options.append(cleaned_opt)
            opt_img_ids.extend(opt_ids)
        q["options"] = cleaned_options
        
        # STEP 4: Clean stray backslashes (LAST step)
        q["question"] = _clean_backslashes(q["question"])
        q["solution"] = _clean_backslashes(q["solution"])
        q["options"] = [_clean_backslashes(opt) for opt in q["options"]]
        
        # STEP 5: Collect all image IDs from placeholders
        q_placeholder_ids = RE_PLACEHOLDER.findall(q.get("question", ""))
        s_placeholder_ids = RE_PLACEHOLDER.findall(q.get("solution", ""))
        o_placeholder_ids = []
        for opt in q.get("options", []):
            o_placeholder_ids.extend(RE_PLACEHOLDER.findall(opt))
        
        # STEP 6: Populate q_images and sol_images arrays
        q["q_images"] = _unique(q_placeholder_ids + q_img_ids + o_placeholder_ids + opt_img_ids)
        q["sol_images"] = _unique(s_placeholder_ids + sol_img_ids)
    
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


def _extract_json(text: str) -> list:
    """
    Extract JSON array with robust handling of LLM output variations.
    Fixes common JSON escape errors from LLM while preserving literal \n for cleanup.
    """
    import re
    import json
    
    # Step 1: Remove markdown fences (multiple variations)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Step 2: Find JSON array boundaries
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx+1]
    
    # Step 3: Try direct parse first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except json.JSONDecodeError as e:
        print(f"[LLM Parser] Initial JSON parse failed: {e}", flush=True)
        print(f"[LLM Parser] Attempting auto-fix...", flush=True)
        
        # Step 4: Fix common LLM mistakes
        try:
            # The LLM often sends single backslashes like: \frac, \alpha
            # In JSON, these are invalid escapes unless followed by: " \ / b f n r t u
            
            # Strategy: Replace ALL single backslashes with double backslashes,
            # EXCEPT for the valid JSON escapes
            
            # First, protect valid JSON escapes by temporarily replacing them
            protected = text
            protected = protected.replace('\\\\', '\x00DOUBLEBACKSLASH\x00')  # Protect existing \\
            protected = protected.replace('\\"', '\x00QUOTE\x00')
            protected = protected.replace('\\/', '\x00SLASH\x00')
            protected = protected.replace('\\n', '\x00NEWLINE\x00')  # Preserve \n for our cleanup
            protected = protected.replace('\\r', '\x00RETURN\x00')  # Preserve \r for our cleanup
            protected = protected.replace('\\t', '\x00TAB\x00')
            protected = protected.replace('\\b', '\x00BACKSPACE\x00')
            protected = protected.replace('\\f', '\x00FORMFEED\x00')
            
            # Now replace ALL remaining single backslashes with double
            fixed = protected.replace('\\', '\\\\')
            
            # Restore the protected sequences
            fixed = fixed.replace('\x00DOUBLEBACKSLASH\x00', '\\\\')
            fixed = fixed.replace('\x00QUOTE\x00', '\\"')
            fixed = fixed.replace('\x00SLASH\x00', '\\/')
            fixed = fixed.replace('\x00NEWLINE\x00', '\\n')  # Keep as \n for later cleanup
            fixed = fixed.replace('\x00RETURN\x00', '\\r')  # Keep as \r for later cleanup
            fixed = fixed.replace('\x00TAB\x00', '\\t')
            fixed = fixed.replace('\x00BACKSPACE\x00', '\\b')
            fixed = fixed.replace('\x00FORMFEED\x00', '\\f')
            
            # Try parsing the fixed version
            data = json.loads(fixed)
            print(f"[LLM Parser] ✓ Auto-fix successful!", flush=True)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
                
        except json.JSONDecodeError as e2:
            print(f"[LLM Parser] ✗ Auto-fix failed: {e2}", flush=True)
            
            # Show context around error for debugging
            if e2.pos and e2.pos < len(text):
                start = max(0, e2.pos - 100)
                end = min(len(text), e2.pos + 100)
                print(f"[LLM Parser] Error context: ...{text[start:end]}...", flush=True)
            else:
                print(f"[LLM Parser] Response preview: {text[:500]}", flush=True)
            
            return []
    
    return []