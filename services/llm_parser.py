"""
services/llm_parser.py v2.5 - BOUNDARY DETECTOR
================================================
LLM JOB: Find boundaries (question start/end, option start/end)
POST-PROCESSING: Everything else (images, newlines, backslash cleanup)
"""

import os
import re
from typing import Optional
import anthropic

HAIKU_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 64000


async def parse_latex_with_llm(tex: str, api_key: str = None) -> list[dict]:
    """Parse LaTeX paper - LLM finds boundaries, post-processing cleans."""
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
    questions = _fix_newlines(questions)  # Post-process AFTER LLM
    
    print(f"[LLM Parser] ✓ Success: {len(questions)} questions", flush=True)
    return questions


def _detect_exam_type(tex: str) -> str:
    """Detect exam type."""
    tex_lower = tex.lower()
    
    if "jee main" in tex_lower: return "JEE_MAIN"
    if "jee advanced" in tex_lower: return "JEE_ADVANCED"
    if "neet" in tex_lower: return "NEET"
    
    has_bio = "biology" in tex_lower
    has_pcm = all(w in tex_lower for w in ["physics", "chemistry", "math"])
    
    if has_bio: return "NEET"
    elif has_pcm: return "JEE_MAIN"
    
    return "JEE_MAIN"


def _parse_full_paper(tex: str, api_key: str, exam_type: str) -> list[dict]:
    """LLM finds boundaries and copies content."""
    client = anthropic.Anthropic(api_key=api_key)
    
    if exam_type in ["JEE_MAIN", "JEE_ADVANCED"]:
        subject_rule = "ONLY subjects: PHYSICS, CHEMISTRY, MATHEMATICS (NO BIOLOGY in JEE)"
    elif exam_type == "NEET":
        subject_rule = "Valid subjects: PHYSICS, CHEMISTRY, BIOLOGY"
    else:
        subject_rule = "Subjects: PHYSICS, CHEMISTRY, MATHEMATICS, BIOLOGY"
    
    prompt = f"""YOU ARE A BOUNDARY DETECTOR for {exam_type} papers.

YOUR ONLY JOB: Find where questions/options/solutions start and end. Copy EVERYTHING between boundaries.

⚠️ YOU ARE NOT:
- A solver (don't solve!)
- A cleaner (don't fix!)
- A processor (don't process LaTeX!)

OUTPUT FORMAT:

===QUESTION_START===
NUMBER: 1
TYPE: MCQ
SUBJECT: PHYSICS
QUESTION: [Copy EVERYTHING from question start to end - keep \\frac{{{{1}}}}{{{{2}}}}, \\includegraphics{{{{img.png}}}}, \\\\, spaces, newlines - ALL!]
OPTION_1: [Copy option 1 EXACTLY]
OPTION_2: [Copy option 2 EXACTLY]
OPTION_3: [Copy option 3 EXACTLY]
OPTION_4: [Copy option 4 EXACTLY]
ANSWER: [Copy ONLY if you see "Ans." or "Answer:" - otherwise LEAVE EMPTY]
SOLUTION: [Copy EVERYTHING from Sol. to next question]
CHAPTER: [Chapter name if mentioned]
TOPIC: [Topic name if mentioned]
DIFFICULTY: medium
===QUESTION_END===

BOUNDARY RULES:

1. QUESTION BOUNDARY:
   - Start: \\item or "1." or "Question 1"
   - End: Next question starts OR options start
   - Copy: EVERYTHING (don't skip spaces/newlines/LaTeX!)

2. OPTION BOUNDARY:
   - Start: (1) or (a) or "a."
   - End: Next option OR answer
   - Copy: EVERYTHING for each option

3. ANSWER BOUNDARY:
   - Look for: "Ans. (2)" "Answer: 3" "Sol. (4)"
   - Copy: Just the number
   - NO "Ans." marker → LEAVE EMPTY (even if you know answer!)

4. SOLUTION BOUNDARY:
   - Start: "Sol." or "Solution:"
   - End: Next question
   - Copy: EVERYTHING

SUBJECT: {subject_rule}

TYPE:
- MCQ: 4 options, single answer
- MSQ: 4 options, multiple answers (1,2)
- NUMERICAL: No options

CHAPTER/TOPIC (Standard NCERT/JEE/NEET):
PHYSICS: Kinematics, Laws of Motion, Work Energy Power, Thermodynamics, Waves, Electrostatics, Current Electricity, Optics, Modern Physics
CHEMISTRY: Atomic Structure, Chemical Bonding, Thermodynamics, Equilibrium, Redox, Organic Basics, Hydrocarbons, Alcohols, Coordination Compounds
MATHEMATICS: Sets Functions, Trigonometry, Complex Numbers, Calculus, Vectors, 3D Geometry, Probability
BIOLOGY: Cell, Genetics, Evolution, Plant Physiology, Human Physiology, Ecology

DIFFICULTY:
- easy: Direct formula, 1 step
- medium: Multiple steps (DEFAULT)
- hard: Complex, multiple concepts

CRITICAL:
⚠️ Copy EVERYTHING: \\frac, \\alpha, \\includegraphics, \\\\, spaces, newlines - ALL AS-IS
⚠️ Don't remove/add/fix anything
⚠️ Sequential: 1,2,3... don't skip
⚠️ Extract ALL questions

LaTeX:
{tex}"""

    try:
        print(f"[LLM Parser] Calling API...", flush=True)
        
        message = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS,
            system=[{
                "type": "text",
                "text": f"Boundary detector for {exam_type} papers (v2.5). Find boundaries, copy content.",
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


def _parse_delimiter(text: str) -> list:
    """Parse delimiter format."""
    questions = []
    blocks = re.split(r'===QUESTION_START===', text)
    
    for block in blocks:
        if '===QUESTION_END===' not in block:
            continue
        
        content = block.split('===QUESTION_END===')[0].strip()
        q = {}
        lines = content.split('\n')
        current_field = None
        current_value = []
        
        for line in lines:
            if ':' in line and line.split(':')[0].strip().isupper():
                if current_field:
                    q[current_field] = '\n'.join(current_value).strip()
                parts = line.split(':', 1)
                current_field = parts[0].strip()
                current_value = [parts[1].strip() if len(parts) > 1 else '']
            else:
                if current_field:
                    current_value.append(line)
        
        if current_field:
            q[current_field] = '\n'.join(current_value).strip()
        
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


def _sort_questions(questions: list) -> list:
    """Sort by number."""
    try:
        return sorted(questions, key=lambda q: int(q.get("number", 0) or 0))
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
        else:  # NEET
            q["marks_correct"] = 4
            q["marks_wrong"] = -1
    
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


def _fix_newlines(questions: list) -> list:
    """
    POST-PROCESSING - Like old parser.py
    Extract images, clean backslashes, populate image arrays.
    """
    import os
    
    RE_INCLUDEGFX = re.compile(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}')
    RE_PLACEHOLDER = re.compile(r'\[IMAGE:([^\]]+)\]')
    
    def extract_images(text):
        if not text:
            return text, []
        ids = []
        def rep(m):
            img_id = os.path.basename(m.group(1).strip())
            ids.append(img_id)
            return f"[IMAGE:{img_id}]"
        return RE_INCLUDEGFX.sub(rep, text).strip(), ids
    
    def clean_backslashes(text):
        if not text:
            return text
        text = text.replace('\\ ', ' ')
        text = text.replace('\\"', '"')
        text = text.rstrip('\\')
        text = text.replace('\\(', '(').replace('\\)', ')')
        text = re.sub(r'\\([,.])', r'\1', text)
        return text
    
    def unique(lst):
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    
    for q in questions:
        # Extract images
        q["question"], qi = extract_images(q.get("question", ""))
        q["solution"], si = extract_images(q.get("solution", ""))
        
        co, oi = [], []
        for opt in q.get("options", []):
            c, o = extract_images(opt)
            co.append(c)
            oi.extend(o)
        q["options"] = co
        
        # Clean backslashes
        q["question"] = clean_backslashes(q["question"])
        q["solution"] = clean_backslashes(q["solution"])
        q["options"] = [clean_backslashes(o) for o in q["options"]]
        
        # Populate image arrays
        q_ids = RE_PLACEHOLDER.findall(q.get("question", ""))
        o_ids = []
        for opt in q.get("options", []):
            o_ids.extend(RE_PLACEHOLDER.findall(opt))
        s_ids = RE_PLACEHOLDER.findall(q.get("solution", ""))
        
        q["q_images"] = unique(q_ids + o_ids + qi + oi)
        q["sol_images"] = unique(s_ids + si)
    
    return questions