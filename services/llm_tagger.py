"""
services/llm_tagger.py
======================
BATCH tagger — loads taxonomy from taxonomy.json (no embedded JSON strings).

DEPLOY BOTH FILES to Railway:
  services/llm_tagger.py   ← this file
  services/taxonomy.json   ← the taxonomy data file

EXAM → SUBJECT mapping is hardcoded.
Parser sets q["subject"] per question — tagger uses that to pick the right taxonomy.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# ── Load taxonomy from file (avoids JSON string escaping issues) ──────────────
_TAX_PATH = Path(__file__).parent / "taxonomy.json"
if not _TAX_PATH.exists():
    _TAX_PATH = Path(os.getcwd()) / "taxonomy.json"

try:
    _RAW_TAX = json.loads(_TAX_PATH.read_text(encoding="utf-8"))
except FileNotFoundError:
    raise RuntimeError(
        f"taxonomy.json not found. Deploy it alongside llm_tagger.py at: {_TAX_PATH}"
    )

TAXONOMY: dict = {
    exam: {
        subj: {int(cid): cv for cid, cv in sv.items()}
        for subj, sv in ev.items()
    }
    for exam, ev in _RAW_TAX.items()
}

# ── Exam → valid subjects ──────────────────────────────────────────────────────
EXAM_SUBJECTS = {
    "jee-main":     ["Physics", "Chemistry", "Mathematics"],
    "jee-advanced": ["Physics", "Chemistry", "Mathematics"],
    "neet":         ["Physics", "Chemistry", "Biology"],
    "ssc-cgl":      ["Quantitative Aptitude", "General Intelligence and Reasoning",
                     "English Comprehension", "General Awareness"],
}

# ── Quick lookup maps ──────────────────────────────────────────────────────────
_CHAPTER_ID_TO_NAME: dict[int, str] = {}
_TOPIC_ID_TO_NAME:   dict[int, str] = {}
for _e, _ss in TAXONOMY.items():
    for _s, _chs in _ss.items():
        for _cid, _cv in _chs.items():
            _CHAPTER_ID_TO_NAME[_cid] = _cv["name"]
            for _tid, _tn in _cv["topics"].items():
                _TOPIC_ID_TO_NAME[_tid] = _tn


def _get_exam_slug(exam_name: str, exam_type: str = "") -> str:
    raw = (exam_type or exam_name or "").lower().strip()
    if raw in ("jee-main", "jee-advanced", "neet", "ssc-cgl"):
        return raw
    if "advanced" in raw: return "jee-advanced"
    if "main" in raw or "mains" in raw: return "jee-main"
    if "neet" in raw: return "neet"
    if "ssc" in raw: return "ssc-cgl"
    return "jee-main"


def _normalize_subject(subject: str, exam_slug: str) -> str:
    s = subject.strip().title()
    valid = EXAM_SUBJECTS.get(exam_slug, [])
    if s in valid: return s
    sl = s.lower()
    for v in valid:
        if v.lower() in sl or sl in v.lower(): return v
    return valid[0] if valid else s


def _chapter_list_text(exam_slug: str, subject_name: str) -> str:
    chapters = TAXONOMY.get(exam_slug, {}).get(subject_name, {})
    return "\n".join(
        f"{cid}. {cv['name']}"
        for cid in sorted(chapters)
        for cv in [chapters[cid]]
    )


def _topic_list_text(chapter_id: int, exam_slug: str, subject_name: str) -> str:
    chapters = TAXONOMY.get(exam_slug, {}).get(subject_name, {})
    topics = chapters.get(chapter_id, {}).get("topics", {})
    return "\n".join(
        f"{tid}. {tn}"
        for tid in sorted(topics)
        for tn in [topics[tid]]
    )


# ── STEP 1 — Batch chapter classification ─────────────────────────────────────
async def _batch_classify_chapters(
    group: list[tuple[int, dict]],
    exam_slug: str,
    subject_name: str,
    client,
) -> dict[int, int]:
    exam_label = {
        "jee-main": "JEE Main", "jee-advanced": "JEE Advanced",
        "neet": "NEET", "ssc-cgl": "SSC CGL"
    }.get(exam_slug, exam_slug.upper())
    chapter_list = _chapter_list_text(exam_slug, subject_name)
    chapters = TAXONOMY.get(exam_slug, {}).get(subject_name, {})
    valid_ch_ids = set(chapters.keys())

    q_lines = []
    for local_num, (orig_idx, q) in enumerate(group, 1):
        text = (q.get("question") or q.get("question_text") or "").strip()
        if len(text) > 400:
            text = text[:400] + "..."
        q_lines.append(f"Q{local_num}: {text}")
    questions_text = "\n\n".join(q_lines)

    system_prompt = (
        f"You are a {exam_label} — {subject_name} question classifier.\n\n"
        f"Classify each question to its chapter. Return ONLY a JSON object:\n"
        f'{{\"results\": [{{\"q\": 1, \"chapter_id\": <int>}}, ...]}}\n\n'
        f"Rules:\n"
        f"- chapter_id MUST be one of the IDs listed below\n"
        f"- Return exactly one entry per question\n\n"
        f"CHAPTERS:\n{chapter_list}"
    )

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
                max_tokens=len(group) * 15 + 100,
            )
            raw  = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            arr  = data.get("results", list(data.values())[0] if data else [])
            if isinstance(data, list):
                arr = data

            result = {}
            for entry in arr:
                local_q = int(entry.get("q", 0))
                ch_id   = int(entry.get("chapter_id", 0))
                if 1 <= local_q <= len(group) and ch_id in valid_ch_ids:
                    orig_idx = group[local_q - 1][0]
                    result[orig_idx] = ch_id
                else:
                    log.warning(f"[tagger] step1 invalid: q={local_q} ch={ch_id}")

            log.info(f"[tagger] Step1 {exam_slug}/{subject_name}: "
                     f"{len(result)}/{len(group)} classified")
            return result

        except Exception as e:
            wait = 2 * (2 ** attempt)
            log.warning(f"[tagger] step1 attempt {attempt+1}: {e} — retry {wait}s")
            await asyncio.sleep(wait)

    return {}


# ── STEP 2 — Batch topic + difficulty per chapter ─────────────────────────────
async def _batch_classify_topics(
    chapter_id: int,
    chapter_group: list[tuple[int, dict]],
    exam_slug: str,
    subject_name: str,
    client,
) -> dict[int, dict]:
    chapter_name = _CHAPTER_ID_TO_NAME.get(chapter_id, "")
    topic_list   = _topic_list_text(chapter_id, exam_slug, subject_name)
    chapters     = TAXONOMY.get(exam_slug, {}).get(subject_name, {})
    valid_tp_ids = set(chapters.get(chapter_id, {}).get("topics", {}).keys())

    q_lines = []
    for local_num, (orig_idx, q) in enumerate(chapter_group, 1):
        text = (q.get("question") or q.get("question_text") or "").strip()
        if len(text) > 400:
            text = text[:400] + "..."
        q_lines.append(f"Q{local_num}: {text}")
    questions_text = "\n\n".join(q_lines)

    system_prompt = (
        f"You are a {subject_name} question classifier for chapter: {chapter_name}\n\n"
        f"For each question return topic_id and difficulty. Return ONLY a JSON object:\n"
        f'{{\"results\": [{{\"q\": 1, \"topic_id\": <int>, \"difficulty\": \"<easy|medium|hard>\"}},...]}}\n\n'
        f"Rules:\n"
        f"- topic_id MUST be one of the IDs below\n"
        f"- difficulty MUST be exactly easy, medium, or hard\n"
        f"- Spread difficulty — not everything is medium\n\n"
        f"DIFFICULTY:\n"
        f"  easy   = direct recall, single formula, definition\n"
        f"  medium = 2-3 steps, standard exam application\n"
        f"  hard   = multi-concept, tricky, deep calculation\n\n"
        f"TOPICS:\n{topic_list}"
    )

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
                max_tokens=len(chapter_group) * 20 + 100,
            )
            raw  = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            arr  = data.get("results", list(data.values())[0] if data else [])
            if isinstance(data, list):
                arr = data

            result = {}
            for entry in arr:
                local_q = int(entry.get("q", 0))
                tp_id   = int(entry.get("topic_id", 0))
                diff    = str(entry.get("difficulty", "medium")).lower().strip()
                if diff not in ("easy", "medium", "hard"):
                    diff = "medium"
                if 1 <= local_q <= len(chapter_group) and tp_id in valid_tp_ids:
                    orig_idx = chapter_group[local_q - 1][0]
                    result[orig_idx] = {"topic_id": tp_id, "difficulty": diff}
                else:
                    log.warning(f"[tagger] step2 invalid: q={local_q} tp={tp_id}")

            log.info(f"[tagger] Step2 ch={chapter_id}({chapter_name}): "
                     f"{len(result)}/{len(chapter_group)} tagged")
            return result

        except Exception as e:
            wait = 2 * (2 ** attempt)
            log.warning(f"[tagger] step2 attempt {attempt+1}: {e} — retry {wait}s")
            await asyncio.sleep(wait)

    return {}


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

    for (exam_slug, subject_name), group in groups.items():
        # Step 1: classify all questions → chapters (1 API call)
        ch_map = await _batch_classify_chapters(group, exam_slug, subject_name, client)
        if not ch_map:
            log.warning(f"[tagger] Step1 empty for {exam_slug}/{subject_name}")
            continue

        # Group by chapter for step 2
        by_chapter: dict[int, list] = defaultdict(list)
        for orig_idx, q in group:
            ch_id = ch_map.get(orig_idx)
            if ch_id:
                by_chapter[ch_id].append((orig_idx, q))

        # Step 2: per chapter, classify topics (1 API call per chapter)
        step2_tasks = [
            _batch_classify_topics(ch_id, ch_group, exam_slug, subject_name, client)
            for ch_id, ch_group in by_chapter.items()
        ]
        step2_results = await asyncio.gather(*step2_tasks, return_exceptions=True)

        for ch_id, step2_result in zip(by_chapter.keys(), step2_results):
            if isinstance(step2_result, Exception):
                log.error(f"[tagger] Step2 ch={ch_id} error: {step2_result}")
                continue
            for orig_idx, tag in step2_result.items():
                q = results[orig_idx]
                q["chapter_id"]   = ch_id
                q["topic_id"]     = tag["topic_id"]
                q["difficulty"]   = tag["difficulty"]
                q["chapter_name"] = _CHAPTER_ID_TO_NAME.get(ch_id, "")
                q["topic_name"]   = _TOPIC_ID_TO_NAME.get(tag["topic_id"], "")
                results[orig_idx] = q

    tagged = sum(1 for q in results if isinstance(q.get("chapter_id"), int))
    log.info(f"[tagger] Complete — {tagged}/{len(results)} tagged")
    return results
