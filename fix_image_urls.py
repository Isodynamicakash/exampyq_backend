"""
fix_image_urls.py
=================
One-time backfill: re-upload any images that are still stored as
[IMAGE:filename.jpg] placeholders (i.e. were never uploaded to R2).

Also fixes questions where the image was uploaded but the placeholder
in question_text / options / solution_text was never replaced.

Run once:
  python fix_image_urls.py

Requirements:
  - .env with DB credentials and R2 credentials
  - The original job folders still exist in JOBS_ROOT
    (they contain the actual image files)
"""

import os, re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from core.database import get_cursor
from services.r2_upload import upload_question_images, _find_image

JOBS_ROOT = Path(os.environ.get("JOBS_ROOT", "/tmp/examside_jobs"))
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", "")

PLACEHOLDER_RE = re.compile(r'\[IMAGE:([^\]]+)\]')


def is_r2_url(val: str) -> bool:
    """True if this is already a full URL (already uploaded)."""
    return val.startswith("http://") or val.startswith("https://")


def find_image_in_jobs(filename: str) -> Path | None:
    """Search all job folders for a specific image file."""
    stem = Path(filename).stem
    for job_dir in JOBS_ROOT.iterdir():
        if not job_dir.is_dir():
            continue
        for images_dir in [job_dir / "images", job_dir]:
            if not images_dir.exists():
                continue
            found = _find_image(images_dir, filename)
            if found:
                return found
    return None


def replace_placeholders(text: str, url_map: dict) -> str:
    if not text:
        return text
    def _sub(m):
        val = m.group(1)
        if is_r2_url(val):
            return m.group(0)  # already replaced
        # try exact, then stem
        if val in url_map:
            return f"[IMAGE:{url_map[val]}]"
        stem = Path(val).stem
        for k, v in url_map.items():
            if Path(k).stem == stem:
                return f"[IMAGE:{v}]"
        return m.group(0)  # couldn't find it
    return PLACEHOLDER_RE.sub(_sub, text)


def main():
    with get_cursor() as cur:
        # Get all questions that still have unresolved [IMAGE:filename] placeholders
        cur.execute("""
            SELECT q.id, q.question_text, q.option_1, q.option_2, q.option_3, q.option_4,
                   a.solution_text
            FROM questions q
            LEFT JOIN answers a ON a.question_id = q.id
            WHERE q.question_text   LIKE '%[IMAGE:%'
               OR q.option_1        LIKE '%[IMAGE:%'
               OR q.option_2        LIKE '%[IMAGE:%'
               OR q.option_3        LIKE '%[IMAGE:%'
               OR q.option_4        LIKE '%[IMAGE:%'
               OR a.solution_text   LIKE '%[IMAGE:%'
        """)
        rows = cur.fetchall()
        print(f"Found {len(rows)} questions with unresolved image placeholders")

        fixed = 0
        for row in rows:
            qid = row["id"]

            # Collect all placeholder filenames from this question
            all_text = " ".join(filter(None, [
                row["question_text"], row["option_1"], row["option_2"],
                row["option_3"], row["option_4"], row["solution_text"]
            ]))
            filenames = [
                m.group(1) for m in PLACEHOLDER_RE.finditer(all_text)
                if not is_r2_url(m.group(1))
            ]

            if not filenames:
                continue  # all placeholders already have URLs

            print(f"\nQ{qid}: needs {len(filenames)} images: {filenames}")

            # Find and upload each image
            url_map = {}
            for fname in filenames:
                img_path = find_image_in_jobs(fname)
                if not img_path:
                    print(f"  ✗ {fname} — not found in any job folder")
                    continue

                from services.r2_upload import upload_image
                try:
                    result = upload_image(img_path, "question", qid)
                    url_map[fname] = result["url"]
                    print(f"  ✓ {fname} → {result['url']}")
                except Exception as e:
                    print(f"  ✗ {fname} — upload failed: {e}")

            if not url_map:
                print(f"  No images uploaded for Q{qid}, skipping DB update")
                continue

            # Update all fields with resolved URLs
            new_qt  = replace_placeholders(row["question_text"],  url_map)
            new_o1  = replace_placeholders(row["option_1"],       url_map)
            new_o2  = replace_placeholders(row["option_2"],       url_map)
            new_o3  = replace_placeholders(row["option_3"],       url_map)
            new_o4  = replace_placeholders(row["option_4"],       url_map)
            new_sol = replace_placeholders(row["solution_text"],  url_map)

            cur.execute("""
                UPDATE questions
                SET question_text = %s,
                    option_1 = %s, option_2 = %s,
                    option_3 = %s, option_4 = %s
                WHERE id = %s
            """, (new_qt, new_o1, new_o2, new_o3, new_o4, qid))

            if new_sol:
                cur.execute("""
                    UPDATE answers SET solution_text = %s WHERE question_id = %s
                """, (new_sol, qid))

            fixed += 1
            print(f"  ✓ Q{qid} updated in DB")

    print(f"\n✓ Done. Fixed {fixed}/{len(rows)} questions.")


if __name__ == "__main__":
    main()