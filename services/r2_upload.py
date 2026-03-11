"""
services/r2_upload.py
=====================
Resize images with Pillow (enforcing the agreed max sizes per position)
then upload to Cloudflare R2.

Position → max size map:
  question   800 × 500
  option_N   400 × 300
  solution   800 × 600
  answer     400 × 300
"""

import os
import io
import boto3
from pathlib import Path
from PIL import Image

# ── Size caps per image position ──────────────────────────────────────────────
MAX_SIZES: dict[str, tuple[int, int]] = {
    "question":  (800, 500),
    "option_1":  (400, 300),
    "option_2":  (400, 300),
    "option_3":  (400, 300),
    "option_4":  (400, 300),
    "solution":  (800, 600),
    "answer":    (400, 300),
}
DEFAULT_MAX = (800, 500)


def _get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url          = os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id     = os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key = os.environ["R2_SECRET_ACCESS_KEY"],
        region_name           = "auto",
    )


def resize_image(img_path: Path, position: str) -> tuple[bytes, int, int]:
    max_w, max_h = MAX_SIZES.get(position, DEFAULT_MAX)
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        final_w, final_h = img.size
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue(), final_w, final_h


def upload_image(
    img_path: Path,
    position: str,
    question_id: int,
    bucket: str | None = None,
    public_url_base: str | None = None,
) -> dict:
    bucket          = bucket or os.environ["R2_BUCKET"]
    public_url_base = public_url_base or os.environ["R2_PUBLIC_URL"]

    jpeg_bytes, w, h = resize_image(img_path, position)

    stem = img_path.stem
    key  = f"questions/{question_id}/{position}_{stem}.jpg"

    client = _get_r2_client()
    client.put_object(
        Bucket      = bucket,
        Key         = key,
        Body        = jpeg_bytes,
        ContentType = "image/jpeg",
        ACL         = "public-read",
    )

    url = f"{public_url_base.rstrip('/')}/{key}"
    return {
        "url":      url,
        "image_id": img_path.name,   # always set here too
        "position": position,
        "width_px": w,
        "height_px": h,
    }


def _find_image(images_dir: Path, image_id: str) -> Path | None:
    """
    Robust image finder — handles all the ways image_id may or may not
    match the actual filename on disk.

    Cases handled:
      1. image_id == exact filename         e.g. "abc.jpg"  → file "abc.jpg"
      2. image_id == stem, file has ext     e.g. "abc"      → file "abc.jpg"
      3. image_id has ext, file is stem     e.g. "abc.jpg"  → file "abc"  (rare)
      4. image_id stem matches file stem    e.g. "abc.png"  → file "abc.jpg"
    """
    if not images_dir or not images_dir.exists():
        return None

    # Build a lookup of all files: both full name and stem → path
    name_map: dict[str, Path] = {}
    stem_map: dict[str, Path] = {}
    for f in images_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            name_map[f.name] = f
            stem_map[f.stem] = f

    # Try in order of specificity
    if image_id in name_map:
        return name_map[image_id]

    stem = Path(image_id).stem
    if stem in stem_map:
        return stem_map[stem]

    # Try without any path prefix (MathPix sometimes emits "images/abc.jpg")
    bare_name = Path(image_id).name
    if bare_name in name_map:
        return name_map[bare_name]
    bare_stem = Path(bare_name).stem
    if bare_stem in stem_map:
        return stem_map[bare_stem]

    return None


def upload_question_images(
    job_dir: Path,           # kept for backward compat but NOT used for lookup
    question_id: int,
    q_image_ids: list[str],
    sol_image_ids: list[str],
    images_dir: Path | None = None,   # ← NEW: pass the real images dir directly
    opt_image_ids: dict | None = None,  # {"a": id, "b": id, "c": id, "d": id}
) -> list[dict]:
    """
    Upload all images for one question to R2.

    images_dir  — the actual folder containing image files (preferred).
                  Falls back to job_dir/images/ for backward compat.

    Returns list of image dicts ready to INSERT into the images table.
    Each dict has: url, image_id, position, width_px, height_px.
    """
    # Resolve which directory to search
    search_dirs: list[Path] = []
    if images_dir and images_dir.exists():
        search_dirs.append(images_dir)
    # Also try job_dir/images as fallback
    fallback = job_dir / "images"
    if fallback.exists() and fallback not in search_dirs:
        search_dirs.append(fallback)
    # And job_dir itself (in case images are at root)
    if job_dir.exists() and job_dir not in search_dirs:
        search_dirs.append(job_dir)

    def _find(image_id: str) -> Path | None:
        for d in search_dirs:
            p = _find_image(d, image_id)
            if p:
                return p
        return None

    uploaded: list[dict] = []
    missing:  list[str]  = []

    for idx, img_id in enumerate(q_image_ids):
        path = _find(img_id)
        if not path:
            missing.append(img_id)
            print(f"[r2_upload] MISSING q_image: {img_id} (searched: {[str(d) for d in search_dirs]})", flush=True)
            continue
        # position: "question" for first, then option_1, option_2 ...
        pos = "question" if idx == 0 else f"option_{idx}"
        try:
            result = upload_image(path, pos, question_id)
            result["image_id"] = img_id   # keep original ID for placeholder replacement
            uploaded.append(result)
            print(f"[r2_upload] ✓ {img_id} → {result['url']}", flush=True)
        except Exception as e:
            print(f"[r2_upload] UPLOAD FAILED {img_id}: {e}", flush=True)

    for img_id in sol_image_ids:
        path = _find(img_id)
        if not path:
            missing.append(img_id)
            print(f"[r2_upload] MISSING sol_image: {img_id} (searched: {[str(d) for d in search_dirs]})", flush=True)
            continue
        try:
            result = upload_image(path, "solution", question_id)
            result["image_id"] = img_id
            uploaded.append(result)
            print(f"[r2_upload] ✓ {img_id} → {result['url']}", flush=True)
        except Exception as e:
            print(f"[r2_upload] UPLOAD FAILED {img_id}: {e}", flush=True)

    # ── Option images: a/b/c/d ────────────────────────────────────────────────
    OPT_POSITION_MAP = {"a": "option_1", "b": "option_2", "c": "option_3", "d": "option_4"}
    for opt_key, img_id in (opt_image_ids or {}).items():
        if not img_id:
            continue
        path = _find(img_id)
        if not path:
            missing.append(img_id)
            print(f"[r2_upload] MISSING opt_{opt_key}_image: {img_id}", flush=True)
            continue
        pos = OPT_POSITION_MAP.get(opt_key.lower(), f"option_{opt_key}")
        try:
            result = upload_image(path, pos, question_id)
            result["image_id"] = img_id
            result["opt_key"]  = opt_key.lower()   # "a" / "b" / "c" / "d"
            uploaded.append(result)
            print(f"[r2_upload] ✓ opt_{opt_key} {img_id} → {result['url']}", flush=True)
        except Exception as e:
            print(f"[r2_upload] UPLOAD FAILED opt_{opt_key} {img_id}: {e}", flush=True)

    if missing:
        print(f"[r2_upload] Q{question_id}: {len(missing)} images not found: {missing}", flush=True)

    return uploaded