"""
services/mathpix.py
===================
Calls the MathPix API to convert a PDF to LaTeX.
Downloads the resulting .tex file and all extracted images to a
temp job directory on disk.

MathPix flow:
  1. POST /v3/pdf  →  { app_id, pdf_id }
  2. Poll GET /v3/pdf/{pdf_id}  until status == "completed"
  3. GET /v3/pdf/{pdf_id}.tex.zip  →  extract .tex + images/ to job_dir

Directory layout on disk:
  /tmp/examside_jobs/{job_id}/
      original.pdf
      output.tex
      images/
          uuid-1_x_y_w_h.jpg
          uuid-2_x_y_w_h.jpg
          ...

CHANGES vs previous version:
  - submit_pdf: added "include_smiles":false and "conversion_formats":{"tex.zip":true}
    "tex.zip" makes MathPix bundle .tex + images/ in one ZIP download
  - download_tex + download_images replaced by download_tex_zip
    OLD: GET /v3/pdf/{pdf_id}.tex          → 404 (wrong endpoint)
    NEW: GET /v3/pdf/{pdf_id}.tex.zip      → ZIP with .tex + images/ inside
  - run_mathpix_pipeline updated to call download_tex_zip instead of both old functions
"""

import asyncio
import io
import os
import uuid
import zipfile
import aiohttp
import aiofiles
from pathlib import Path

MATHPIX_API = "https://api.mathpix.com/v3"
JOBS_ROOT   = Path(os.getenv("JOBS_ROOT", "/tmp/examside_jobs"))


async def _headers() -> dict:
    return {
        "app_id":  os.environ["MATHPIX_APP_ID"],
        "app_key": os.environ["MATHPIX_APP_KEY"],
    }


async def submit_pdf(pdf_bytes: bytes, filename: str) -> str:
    """
    Submit a PDF to MathPix.
    Returns the MathPix pdf_id (used for polling and download).

    "include_smiles": false           → chemistry diagrams stay as images
    "conversion_formats": {"tex.zip"} → MathPix generates a downloadable ZIP
                                        at /v3/pdf/{pdf_id}.tex.zip containing
                                        the .tex file + all images/
    """
    headers = await _headers()
    data    = aiohttp.FormData()
    data.add_field(
        "file",
        pdf_bytes,
        filename    = filename,
        content_type= "application/pdf",
    )
    data.add_field("options_json", '{"math_inline_delimiters":["$","$"],'
                                   '"math_display_delimiters":["$$","$$"],'
                                   '"include_equation_tags":false,'
                                   '"include_smiles":false,'
                                   '"conversion_formats":{"tex.zip":true}}')

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{MATHPIX_API}/pdf",
            headers = headers,
            data    = data,
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()
            return result["pdf_id"]


async def poll_until_done(pdf_id: str, timeout: int = 300, interval: int = 5) -> dict:
    """
    Poll MathPix until the PDF conversion is complete.
    Raises TimeoutError if it takes longer than `timeout` seconds.
    Returns the final status dict.
    """
    headers = await _headers()
    elapsed = 0

    async with aiohttp.ClientSession() as session:
        while elapsed < timeout:
            async with session.get(
                f"{MATHPIX_API}/pdf/{pdf_id}",
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                status = await resp.json()

            state = status.get("status", "")
            if state == "completed":
                return status
            if state == "error":
                raise RuntimeError(f"MathPix error for pdf_id={pdf_id}: {status}")

            await asyncio.sleep(interval)
            elapsed += interval

    raise TimeoutError(f"MathPix did not complete within {timeout}s for pdf_id={pdf_id}")


async def download_tex_zip(pdf_id: str, job_dir: Path) -> Path:
    """
    Download the tex.zip bundle from MathPix and extract it to job_dir.
    The ZIP contains the .tex file + images/ folder.
    Returns path to the extracted .tex file (job_dir/output.tex).

    URL: GET /v3/pdf/{pdf_id}.tex.zip
    (Replaces the old separate download_tex + download_images calls)
    """
    headers = await _headers()
    img_dir = job_dir / "images"
    img_dir.mkdir(exist_ok=True)

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{MATHPIX_API}/pdf/{pdf_id}.tex.zip",
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            zip_bytes = await resp.read()

    tex_path = job_dir / "output.tex"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            data = zf.read(name)
            if name.endswith(".tex"):
                tex_path.write_bytes(data)
            elif any(name.lower().endswith(ext)
                     for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp")):
                # Flatten all images into images/ (strip any subpath from ZIP)
                img_name = Path(name).name
                (img_dir / img_name).write_bytes(data)

    if not tex_path.exists():
        raise RuntimeError(f"No .tex file found inside tex.zip for pdf_id={pdf_id}")

    return tex_path


def make_job_dir(job_id: str) -> Path:
    """Create and return the temp directory for a job."""
    job_dir = JOBS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "images").mkdir(exist_ok=True)
    return job_dir


async def run_mathpix_pipeline(pdf_bytes: bytes, filename: str, job_id: str) -> Path:
    """
    Full MathPix flow for one PDF.
    Returns job_dir Path once .tex and all images are saved to disk.
    """
    job_dir = make_job_dir(job_id)

    # Save original PDF
    async with aiofiles.open(job_dir / "original.pdf", "wb") as f:
        await f.write(pdf_bytes)

    # Submit + poll + download tex.zip (contains .tex + images/ together)
    pdf_id = await submit_pdf(pdf_bytes, filename)
    await poll_until_done(pdf_id)
    await download_tex_zip(pdf_id, job_dir)

    return job_dir