"""
services/llm_tagger.py
======================
SINGLE-STEP batch tagger.

ONE API call per subject group:
- Sends full taxonomy (all chapters + all topics) in the system prompt
- Sends all questions in the user message
- LLM returns chapter_id, topic_id, difficulty for every question at once

No 2-step. No guessing. Full context = accurate tagging.

DEPLOY BOTH to Railway:
  services/llm_tagger.py
  services/taxonomy.json
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# ── Load taxonomy from file ───────────────────────────────────────────────────
_TAX_PATH = Path(__file__).parent / "taxonomy.json"
if not _TAX_PATH.exists():
    _TAX_PATH = Path(os.getcwd()) / "taxonomy.json"

try:
    _RAW_TAX = json.loads(_TAX_PATH.read_text(encoding="utf-8"))
except FileNotFoundError:
    raise RuntimeError(f"taxonomy.json not found at {_TAX_PATH}")

TAXONOMY: dict = {
    exam: {
        subj: {
            int(cid): {
                "name": cv["name"],
                "topics": {int(tid): tn for tid, tn in cv["topics"].items()}
            }
            for cid, cv in sv.items()
        }
        for subj, sv in ev.items()
    }
    for exam, ev in _RAW_TAX.items()
}

EXAM_SUBJECTS = {
    "jee-main":     ["Physics", "Chemistry", "Mathematics"],
    "jee-advanced": ["Physics", "Chemistry", "Mathematics"],
    "neet":         ["Physics", "Chemistry", "Biology"],
    "ssc-cgl":      ["Quantitative Aptitude", "General Intelligence and Reasoning",
                     "English Comprehension", "General Awareness"],
    "cuet":         ["Applied Mathematics", "Computer Science", "Accountancy",
                     "Business Studies", "Economics", "Geography", "History",
                     "Political Science", "Psychology", "Sociology", "Philosophy",
                     "Physics", "Chemistry", "Mathematics", "Biology"],
}

_CHAPTER_ID_TO_NAME: dict[int, str] = {}
_TOPIC_ID_TO_NAME:   dict[int, str] = {}
_VALID_CHAPTER_IDS:  set[int] = set()
_CHAPTER_TO_TOPICS:  dict[int, set] = {}

for _e, _ss in TAXONOMY.items():
    for _s, _chs in _ss.items():
        for _cid, _cv in _chs.items():
            _CHAPTER_ID_TO_NAME[_cid] = _cv["name"]
            _VALID_CHAPTER_IDS.add(_cid)
            _CHAPTER_TO_TOPICS[_cid] = set(_cv["topics"].keys())
            for _tid, _tn in _cv["topics"].items():
                _TOPIC_ID_TO_NAME[_tid] = _tn


def _get_exam_slug(exam_name: str, exam_type: str = "") -> str:
    raw = (exam_type or exam_name or "").lower().strip()
    if raw in ("jee-main", "jee-advanced", "neet", "ssc-cgl", "cuet"): return raw
    if "advanced" in raw: return "jee-advanced"
    if "main" in raw or "mains" in raw: return "jee-main"
    if "neet" in raw: return "neet"
    if "ssc" in raw: return "ssc-cgl"
    if "cuet" in raw: return "cuet"
    return "jee-main"


_SUBJECT_ALIASES = {
    "applied maths": "Applied Mathematics",
    "applied math": "Applied Mathematics",
    "comp sci": "Computer Science",
    "cs": "Computer Science",
    "informatics practices": "Computer Science",
    "ip": "Computer Science",
    "accounts": "Accountancy",
    "bst": "Business Studies",
    "eco": "Economics",
    "geo": "Geography",
    "poli sci": "Political Science",
    "political sci": "Political Science",
    "polsci": "Political Science",
    "psych": "Psychology",
    "socio": "Sociology",
    "philo": "Philosophy",
}


def _normalize_subject(subject: str, exam_slug: str) -> str:
    s = subject.strip().title()
    valid = EXAM_SUBJECTS.get(exam_slug, [])
    if s in valid: return s
    sl = s.lower()
    if sl in _SUBJECT_ALIASES and _SUBJECT_ALIASES[sl] in valid:
        return _SUBJECT_ALIASES[sl]
    for v in valid:
        if v.lower() in sl or sl in v.lower(): return v
    return valid[0] if valid else s


def _build_full_taxonomy_text(exam_slug: str, subject_name: str) -> str:
    """Full taxonomy: chapter names + all topic names, with IDs."""
    chapters = TAXONOMY.get(exam_slug, {}).get(subject_name, {})
    lines = []
    for cid in sorted(chapters.keys()):
        cv = chapters[cid]
        lines.append(f"{cid}. {cv['name']}")
        for tid in sorted(cv["topics"].keys()):
            lines.append(f"   {tid}. {cv['topics'][tid]}")
    return "\n".join(lines)


def _build_system_prompt(exam_slug: str, subject_name: str) -> str:
    exam_label = {
        "jee-main": "JEE Main", "jee-advanced": "JEE Advanced",
        "neet": "NEET", "ssc-cgl": "SSC CGL", "cuet": "CUET UG"
    }.get(exam_slug, exam_slug.upper())

    taxonomy_text = _build_full_taxonomy_text(exam_slug, subject_name)

    return f"""You are an expert {exam_label} — {subject_name} question tagger.

For each question, identify the exact chapter and topic from the taxonomy below, and rate its difficulty.

Return ONLY a JSON object:
{{"results": [{{"q": 1, "chapter_id": <int>, "topic_id": <int>, "difficulty": "<easy|medium|hard>"}}, ...]}}

STRICT RULES:
1. chapter_id MUST be one of the chapter IDs in the taxonomy (the non-indented numbers)
2. topic_id MUST be one of the topic IDs under that chapter (the indented numbers)
3. topic_id MUST belong to the chapter_id you chose — do not mix IDs across chapters
4. difficulty MUST be exactly: easy, medium, or hard
5. Return one entry per question — no skipping
6. Spread chapters and topics — do NOT default everything to Algebra or Geometry

DIFFICULTY:
  easy   = direct recall, definition, single formula (e.g. "Find HCF of 12 and 18")
  medium = 2-3 steps, standard application (e.g. "A train covers 360km in 4h, find speed")
  hard   = multi-concept, tricky, long calculation (e.g. "Two trains, relative speed, platform length")

TAXONOMY (chapter_id. Chapter Name → topic_id. Topic Name):
{taxonomy_text}

Example output: {{"results": [{{"q": 1, "chapter_id": 274, "topic_id": 2466, "difficulty": "medium"}}]}}"""


async def _tag_subject_group(
    group: list[tuple[int, dict]],
    exam_slug: str,
    subject_name: str,
    client,
    chunk_size: int = 20,
) -> dict[int, dict]:
    """
    Tag all questions in a subject group.
    Sends full taxonomy + questions in one call per chunk.
    Returns {original_idx: {chapter_id, topic_id, difficulty}}
    """
    chapters = TAXONOMY.get(exam_slug, {}).get(subject_name, {})
    if not chapters:
        log.warning(f"[tagger] No taxonomy for {exam_slug}/{subject_name}")
        return {}

    system_prompt = _build_system_prompt(exam_slug, subject_name)
    results = {}

    # Process in chunks to keep user message manageable
    for chunk_start in range(0, len(group), chunk_size):
        chunk = group[chunk_start:chunk_start + chunk_size]

        q_lines = []
        for local_num, (orig_idx, q) in enumerate(chunk, 1):
            text = (q.get("question") or q.get("question_text") or "").strip()
            if len(text) > 500:
                text = text[:500] + "..."
            q_lines.append(f"Q{local_num}: {text}")
        questions_text = "\n\n".join(q_lines)

        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": questions_text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=len(chunk) * 25 + 100,
                )
                raw  = resp.choices[0].message.content.strip()
                data = json.loads(raw)
                arr  = data.get("results", data if isinstance(data, list) else [])

                chunk_ok = 0
                for entry in arr:
                    local_q  = int(entry.get("q", 0))
                    ch_id    = int(entry.get("chapter_id", 0))
                    tp_id    = int(entry.get("topic_id", 0))
                    diff     = str(entry.get("difficulty", "medium")).lower().strip()

                    if diff not in ("easy", "medium", "hard"):
                        diff = "medium"

                    if not (1 <= local_q <= len(chunk)):
                        continue

                    orig_idx = chunk[local_q - 1][0]

                    # Validate chapter
                    if ch_id not in chapters:
                        log.warning(f"[tagger] invalid chapter_id={ch_id} for Q{local_q}")
                        continue

                    # Validate topic belongs to chapter
                    if tp_id not in chapters[ch_id]["topics"]:
                        # Try to find the topic in the right chapter
                        log.warning(f"[tagger] tp={tp_id} not in ch={ch_id}, skipping Q{local_q}")
                        continue

                    results[orig_idx] = {
                        "chapter_id": ch_id,
                        "topic_id":   tp_id,
                        "difficulty": diff,
                    }
                    chunk_ok += 1

                log.info(f"[tagger] {exam_slug}/{subject_name} chunk {chunk_start//chunk_size+1}: "
                         f"{chunk_ok}/{len(chunk)} tagged")
                break  # success, don't retry

            except Exception as e:
                wait = 2 * (2 ** attempt)
                log.warning(f"[tagger] attempt {attempt+1} failed: {e} — retry {wait}s")
                await asyncio.sleep(wait)

    return results


# ── Main entry point ───────────────────────────────────────────────────────────
async def tag_questions_async(
    questions: list[dict],
    subject: str = "",
    pool=None,
    openai_api_key: str = "",
    exam_type: str = "",
) -> list[dict]:
    if not openai_api_key:
        log.warning("[tagger] No OpenAI key — skipping")
        return questions

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_api_key)
    except ImportError:
        log.warning("[tagger] openai not installed")
        return questions

    # Group by (exam_slug, subject_name)
    groups: dict[tuple, list] = defaultdict(list)
    for i, q in enumerate(questions):
        exam_raw = q.get("exam_name") or "JEE Main"
        slug     = _get_exam_slug(exam_raw, exam_type)
        subj_raw = q.get("subject") or subject or EXAM_SUBJECTS.get(slug, ["Physics"])[0]
        subj     = _normalize_subject(subj_raw, slug)
        groups[(slug, subj)].append((i, q))

    log.info(f"[tagger] {len(questions)} questions in {len(groups)} subject groups")
    for (slug, subj), grp in groups.items():
        log.info(f"  {slug}/{subj}: {len(grp)} questions")

    results = list(questions)

    # Process each subject group
    tasks = [
        _tag_subject_group(grp, slug, subj, client)
        for (slug, subj), grp in groups.items()
    ]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    for (slug, subj), tag_map in zip(groups.keys(), all_results):
        if isinstance(tag_map, Exception):
            log.error(f"[tagger] {slug}/{subj} error: {tag_map}")
            continue
        for orig_idx, tag in tag_map.items():
            q = results[orig_idx]
            q["chapter_id"]   = tag["chapter_id"]
            q["topic_id"]     = tag["topic_id"]
            q["difficulty"]   = tag["difficulty"]
            q["chapter_name"] = _CHAPTER_ID_TO_NAME.get(tag["chapter_id"], "")
            q["topic_name"]   = _TOPIC_ID_TO_NAME.get(tag["topic_id"], "")
            results[orig_idx] = q

    tagged = sum(1 for q in results if isinstance(q.get("chapter_id"), int))
    log.info(f"[tagger] Complete — {tagged}/{len(results)} tagged")
    return results
