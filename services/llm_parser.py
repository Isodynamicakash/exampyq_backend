"""
services/llm_parser.py v3.0
============================
DELIMITER FORMAT - No JSON escaping issues!

FEATURES:
✅ Delimiter-based parsing (no JSON!)
✅ LaTeX commands AS-IS
✅ No escaping issues
✅ Reliable & simple
"""

import os
import re
from typing import Optional
import anthropic

HAIKU_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 64000


async def parse_latex_with_llm(tex: str, api_key: str = None) -> list[dict]:
    """Parse LaTeX paper using delimiter format."""
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
    if not api_key:
        print("[LLM Parser] ⚠ No API key", flush=True)
        return []
    
    tex = _clean_latex(tex)
    exam_type = _detect_exam_type(tex)
    print(f"[LLM Parser] 📋 Detected: {exam_type} | Length: {len(tex)} chars", flush=True)
    
    questions = _parse_full_paper(tex, api_key, exam_type)
    
    if not questions:
        print(f"[LLM Parser] ✗ Parse failed", flush=True)
        return []
    
    questions = _sort_questions(questions)
    questions = _add_marks(questions, exam_type)
    questions = _process_images(questions)
    
    print(f"[LLM Parser] ✓ Success: {len(questions)} questions", flush=True)
    return questions


def _detect_exam_type(tex: str) -> str:
    """Detect exam type."""
    tex_lower = tex.lower()
    
    if "jee main" in tex_lower:
        return "JEE_MAIN"
    if "jee advanced" in tex_lower:
        return "JEE_ADVANCED"
    if "neet" in tex_lower:
        return "NEET"
    
    has_bio = "biology" in tex_lower
    has_pcm = all(w in tex_lower for w in ["physics", "chemistry", "math"])
    
    if has_bio:
        return "NEET"
    elif has_pcm:
        return "JEE_MAIN"
    
    return "JEE_MAIN"


def _parse_full_paper(tex: str, api_key: str, exam_type: str) -> list[dict]:
    """Parse using DELIMITER format."""
    client = anthropic.Anthropic(api_key=api_key)
    
    if exam_type in ["JEE_MAIN", "JEE_ADVANCED"]:
        subject_rule = "ONLY subjects: PHYSICS, CHEMISTRY, MATHEMATICS (NO BIOLOGY)"
    elif exam_type == "NEET":
        subject_rule = "Valid subjects: PHYSICS, CHEMISTRY, BIOLOGY"
    else:
        subject_rule = "Subjects: PHYSICS, CHEMISTRY, MATHEMATICS, BIOLOGY"
    
    prompt = f"""You are a PRECISE LaTeX extractor. Your ONLY job is to COPY text EXACTLY from the paper.

⚠️ CRITICAL RULES - FOLLOW EXACTLY:
1. COPY text character-by-character - do NOT invent, guess, or improve anything
2. If answer is NOT in the paper - leave ANSWER field EMPTY (do not guess!)
3. Extract ALL questions from the paper - do NOT skip any
4. Keep LaTeX commands EXACTLY as written: \\frac{{1}}{{2}}, \\alpha, \\includegraphics{{...}}
5. Extract questions in sequential order

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

Extract ALL questions from LaTeX:
{tex}

REMEMBER: Extract EVERYTHING, skip NOTHING, invent NOTHING!"""

    try:
        print(f"[LLM Parser] Calling API...", flush=True)
        
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": f"Extract {exam_type} questions using delimiter format. Copy LaTeX exactly.",
                "cache_control": {"type": "ephemeral"}
            }],
            messages=[{"role": "user", "content": prompt}]
        )
        
        response = message.content[0].text.strip()
        print(f"[LLM Parser] Response: {len(response)} chars", flush=True)
        
        questions = _parse_delimiter(response)
        print(f"[LLM Parser] Parsed: {len(questions)} questions", flush=True)
        
        return questions
        
    except Exception as e:
        print(f"[LLM Parser] ERROR: {e}", flush=True)
        return []


def _parse_delimiter(text: str) -> list[dict]:
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


def _clean_latex(tex: str) -> str:
    """Remove preamble."""
    start = tex.find(r'\begin{document}')
    if start != -1:
        tex = tex[start:]
    
    end = tex.find(r'\end{document}')
    if end != -1:
        tex = tex[:end]
    
    return tex.strip()


def _sort_questions(questions: list) -> list:
    """Sort by number."""
    try:
        return sorted(questions, key=lambda q: int(q.get("number", 0)))
    except:
        return questions


def _add_marks(questions: list, exam_type: str) -> list:
    """Add marks based on exam type."""
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
                q["marks_wrong"] = 0
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
        else:  # NEET or default
            q["marks_correct"] = 4
            q["marks_wrong"] = -1
    
    return questions


def _process_images(questions: list) -> list:
    """
    Extract images from LaTeX - EXACTLY like old parser.py
    NO backslash cleaning, NO validation - just image extraction!
    """
    import os
    
    # OLD PARSER regex - SINGLE backslash (matches raw LaTeX)
    RE_INCLUDEGFX = re.compile(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}')
    RE_PLACEHOLDER = re.compile(r'\[IMAGE:([^\]]+)\]')
    
    def extract_images(text):
        """Extract \includegraphics{...} and replace with [IMAGE:...]"""
        if not text:
            return text, []
        ids = []
        def rep(m):
            img_id = os.path.basename(m.group(1).strip())
            ids.append(img_id)
            return f"[IMAGE:{img_id}]"
        return RE_INCLUDEGFX.sub(rep, text).strip(), ids
    
    def unique(lst):
        """Remove duplicates preserving order"""
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    for q in questions:
        # Extract images from question, solution, options
        q["question"], qi = extract_images(q.get("question", ""))
        q["solution"], si = extract_images(q.get("solution", ""))
        
        co, oi = [], []
        for opt in q.get("options", []):
            c, o = extract_images(opt)
            co.append(c)
            oi.extend(o)
        q["options"] = co
        
        # Populate image arrays
        q_ids = RE_PLACEHOLDER.findall(q.get("question", ""))
        o_ids = []
        for opt in q.get("options", []):
            o_ids.extend(RE_PLACEHOLDER.findall(opt))
        s_ids = RE_PLACEHOLDER.findall(q.get("solution", ""))
        
        q["q_images"] = unique(q_ids + o_ids + qi + oi)
        q["sol_images"] = unique(s_ids + si)
    
    return questions