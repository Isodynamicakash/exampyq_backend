"""
core/auth.py -- identify the logged-in ExamsCalendar user.

The frontend sends the Supabase access token as
    Authorization: Bearer <token>
and we ask Supabase Auth who it belongs to. This works with both the old
(shared secret) and new (asymmetric) Supabase JWT signing keys, so there
is no JWT secret to manage here.

Env vars:
    SUPABASE_URL       https://<project>.supabase.co
    SUPABASE_ANON_KEY  the public anon key (same one the frontend uses)
"""

import os
from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import Header, HTTPException


@dataclass
class AuthUser:
    id: str
    email: Optional[str] = None
    name: Optional[str] = None


def get_current_user(authorization: str = Header(default="")) -> AuthUser:
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Please sign in to continue.")
    token = authorization[7:].strip()

    supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    anon_key = os.environ.get("SUPABASE_ANON_KEY", "")
    if not supabase_url or not anon_key:
        raise HTTPException(status_code=500, detail="Server auth is not configured.")

    try:
        res = httpx.get(
            f"{supabase_url}/auth/v1/user",
            headers={"apikey": anon_key, "Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="Couldn't verify your login. Please try again.")

    if res.status_code != 200:
        raise HTTPException(status_code=401, detail="Your session has expired. Please sign in again.")

    data = res.json()
    meta = data.get("user_metadata") or {}
    return AuthUser(
        id=data["id"],
        email=data.get("email"),
        name=meta.get("full_name") or meta.get("name"),
    )
