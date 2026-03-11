"""
main.py — ExamSIDE FastAPI application (production)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from routers.admin import router as admin_router
from routers.questions import router as questions_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan: start background cleanup on boot ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from services.job_cleanup import cleanup_loop
    cleanup_task = asyncio.create_task(cleanup_loop())
    logger.info("ExamSIDE API starting up")
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("ExamSIDE API shut down")

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="ExamSIDE API",
    docs_url=None,      # hide in prod
    redoc_url=None,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [
    # Add your specific Vercel deployment URL
    
    "https://pyq-front.vercel.app",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",

] + [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Security headers ──────────────────────────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(admin_router)
app.include_router(questions_router)

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
@limiter.limit("30/minute")
def health(request: Request):
    return {"status": "ok"}