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
    """Parse full LaTeX paper using DELIMITER format (NOT JSON)."""
    
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
    
    prompt = f"""You are a PRECISE LaTeX extractor. Your ONLY job is to COPY text EXACTLY from the paper.

⚠️ CRITICAL RULES - FOLLOW EXACTLY:
1. COPY text character-by-character - do NOT invent, guess, or improve anything
2. If answer is NOT in the paper - leave ANSWER field EMPTY (do not guess!)
3. Extract ALL questions from the paper - do NOT skip any
4. Keep LaTeX commands EXACTLY as written: \\frac{{1}}{{2}}, \\alpha, \\includegraphics{{...}}
5. Extract questions in sequential order: 1, 2, 3, 4, 5... up to the last question

OUTPUT FORMAT - Use delimiters for EACH question:

===QUESTION_START===
NUMBER: 1
TYPE: MCQ
SUBJECT: PHYSICS
QUESTION: Copy question text EXACTLY including \\frac{{1}}{{2}} and \\includegraphics{{img.png}}
OPTION_1: Copy first option EXACTLY
OPTION_2: Copy second option EXACTLY  
OPTION_3: Copy third option EXACTLY
OPTION_4: Copy fourth option EXACTLY
ANSWER: Copy answer EXACTLY (if present in paper - otherwise leave EMPTY)
SOLUTION: Copy solution EXACTLY (if present - otherwise leave EMPTY)
CHAPTER: Chapter name (if clear from context)
TOPIC: Topic name (if clear from context)
DIFFICULTY: medium
===QUESTION_END===

ANSWER EXTRACTION RULES:
- ONLY extract answer if it's EXPLICITLY stated in the paper
- Look for patterns like: "Ans. (2)", "Answer: 3", "Sol. (4)", "NTA Ans. (1)"
- For MCQ: answer is "1" or "2" or "3" or "4" ONLY
- For MSQ: answer is "1,2" or "1,3" etc (comma-separated)
- For NUMERICAL: answer is the number like "5" or "2.5"
- If you see "Sol. 15" that means answer is 15 (for NUMERICAL type)
- If you see "Sol. (3)" that means answer is 3 (for MCQ type)
- If NO answer found → leave ANSWER field completely EMPTY
- DO NOT guess or invent answers - ONLY extract what's explicitly there

IMAGE EXTRACTION:
- Keep \\includegraphics{{image_name.png}} EXACTLY as written
- Do NOT remove or modify image references
- Copy the full path/filename EXACTLY

SUBJECT RULES:
{subject_rule}

TYPE DETECTION:
- MCQ: 4 options, single correct answer
- MSQ: 4 options, multiple correct answers (answer like "1,2")
- NUMERICAL: No options, numerical answer

SEQUENTIAL NUMBERING - VERY IMPORTANT:
- Questions are numbered 1, 2, 3, 4... up to the last question
- If paper has 90 questions: extract Q1 through Q90 (all of them!)
- If paper has 75 questions: extract Q1 through Q75 (all of them!)
- DO NOT skip any numbers in the sequence
- Extract EVERY question you find

Extract ALL questions from LaTeX:
{tex}

REMEMBER: Extract EVERYTHING, skip NOTHING, invent NOTHING!"""

    try:
        print(f"[LLM Parser] Calling API ({exam_type})...", flush=True)
        
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": f"You are a PRECISE LaTeX extractor for {exam_type} question papers (v2.4 - delimiter format). Extract ALL questions EXACTLY as written.",
                    "cache_control": {"type": "ephemeral"}  # Cache v2.4
                }
            ],
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        print(f"[LLM Parser] Response received: {len(response_text)} chars", flush=True)
        
        # Parse DELIMITER format (NOT JSON)
        questions = _parse_delimiter_format(response_text)
        
        if not questions:
            print(f"[LLM Parser] ✗ Delimiter parsing failed", flush=True)
            print(f"[LLM Parser] First 500 chars: {response_text[:500]}", flush=True)
            return []
        
        # Post-process: Extract images (like old parser)
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
    Post-process questions EXACTLY like old parser.py:
    1. Extract images from \includegraphics{...} and replace with [IMAGE:...]
    2. Populate q_images and sol_images arrays
    3. Clean stray backslashes (not part of LaTeX commands)
    
    NOTE: Old parser expects LaTeX AS-IS from source.
    No escaping, no unescaping, just image extraction + cleanup.
    """
    import re
    import os
    
    # OLD PARSER regex - SINGLE backslash (from raw LaTeX)
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
        EXACTLY like old parser.py
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
    
    def _clean_backslashes(text):
        """
        Remove stray backslashes that are NOT valid LaTeX commands.
        Keep: \frac, \alpha, \section, \pi, etc. (backslash + letter)
        Remove: "is\ ", "voltage\ ", etc. (backslash + space/punctuation)
        """
        if not text:
            return text
        
        # Remove backslash before space: is\ → is
        text = text.replace('\\ ', ' ')
        
        # Remove backslash before quote: voltage\" → voltage"
        text = text.replace('\\"', '"')
        
        # Remove backslash at end of string
        text = text.rstrip('\\')
        
        # Remove backslash before parentheses: \( → (
        text = text.replace('\\(', '(').replace('\\)', ')')
        
        # Remove backslash before comma/period (but keep LaTeX commands)
        text = re.sub(r'\\([,.])', r'\1', text)
        
        return text
    
    for q in questions:
        # Extract images (EXACTLY like old parser.py _postprocess)
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
        
        # Clean stray backslashes (after image extraction)
        q["question"] = _clean_backslashes(q["question"])
        q["solution"] = _clean_backslashes(q["solution"])
        q["options"] = [_clean_backslashes(opt) for opt in q["options"]]
        
        # Collect all image IDs from placeholders
        q_ids = RE_PLACEHOLDER.findall(q.get("question", ""))
        o_ids = []
        for opt in q.get("options", []):
            o_ids.extend(RE_PLACEHOLDER.findall(opt))
        s_ids = RE_PLACEHOLDER.findall(q.get("solution", ""))
        
        # Populate q_images and sol_images arrays
        q["q_images"] = _unique(q_ids + o_ids + qi + oi)
        q["sol_images"] = _unique(s_ids + si)
    
    return questions
    
    return questions


def _parse_delimiter_format(text: str) -> list:
    """Parse delimiter format into question dicts."""
    questions = []
    
    # Split by question blocks
    blocks = re.split(r'===QUESTION_START===', text)
    
    for block in blocks:
        if '===QUESTION_END===' not in block:
            continue
        
        # Extract content between delimiters
        content = block.split('===QUESTION_END===')[0].strip()
        
        q = {}
        lines = content.split('\n')
        current_field = None
        current_value = []
        
        for line in lines:
            # Check if it's a field header
            if ':' in line and line.split(':')[0].strip().isupper():
                # Save previous field
                if current_field:
                    q[current_field] = '\n'.join(current_value).strip()
                
                # Start new field
                parts = line.split(':', 1)
                current_field = parts[0].strip()
                current_value = [parts[1].strip() if len(parts) > 1 else '']
            else:
                # Continuation of current field
                if current_field:
                    current_value.append(line)
        
        # Save last field
        if current_field:
            q[current_field] = '\n'.join(current_value).strip()
        
        # Convert to standard format
        if q:
            try:
                question_dict = {
                    "number": int(q.get("NUMBER", 0)),
                    "q_type": q.get("TYPE", "MCQ"),
                    "subject": q.get("SUBJECT", "PHYSICS"),
                    "question": q.get("QUESTION", ""),
                    "options": [
                        q.get("OPTION_1", ""),
                        q.get("OPTION_2", ""),
                        q.get("OPTION_3", ""),
                        q.get("OPTION_4", "")
                    ],
                    "answer": q.get("ANSWER", ""),
                    "solution": q.get("SOLUTION", ""),
                    "chapter_name": q.get("CHAPTER", ""),
                    "topic_name": q.get("TOPIC", ""),
                    "difficulty": q.get("DIFFICULTY", "medium")
                }
                questions.append(question_dict)
            except:
                continue
    
    return questions