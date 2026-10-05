"""
routers/payments.py -- Razorpay payments for ExamsCalendar Premium.

Endpoints
    GET  /api/payments/plans          public, plan list for the pricing page
    POST /api/payments/create-order   logged in, creates a Razorpay order
    POST /api/payments/verify         logged in, confirms payment, activates Premium
    POST /api/payments/webhook        Razorpay -> us, backup activation

Env vars
    RAZORPAY_KEY_ID          rzp_test_... (test) or rzp_live_... (live)
    RAZORPAY_KEY_SECRET      from Razorpay Dashboard -> API Keys
    RAZORPAY_WEBHOOK_SECRET  the secret you type when creating the webhook
    SUPABASE_URL, SUPABASE_ANON_KEY  (see core/auth.py)

Security
    * Prices come only from PLANS below; the browser just names a plan.
    * Payments are confirmed by HMAC signature AND by asking Razorpay for
      the payment's status and amount before Premium is granted.
    * Premium is applied inside the database (apply_payment), which is
      idempotent, so verify + webhook can't grant the same order twice.
"""

import hashlib
import hmac
import json
import logging
import os
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from core.auth import AuthUser, get_current_user
from core.database import get_cursor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/payments", tags=["payments"])

RAZORPAY_API = "https://api.razorpay.com/v1"

# ── Plans: the ONLY place prices are defined ─────────────────────────────────
# amount is in paise (₹99 = 9900). Change prices here and redeploy.
PLANS = {
    "monthly": {"label": "Monthly", "amount_paise": 9900, "days": 30},
    "yearly": {"label": "Yearly", "amount_paise": 69900, "days": 365},
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def _keys():
    key_id = os.environ.get("RAZORPAY_KEY_ID", "")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise HTTPException(status_code=503, detail="Payments are not available yet.")
    return key_id, key_secret


def _hmac_sha256(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def _get_order(order_id: str):
    with get_cursor() as cur:
        cur.execute(
            "select order_id, user_id, plan, amount_paise, status from public.payment_orders where order_id = %s",
            (order_id,),
        )
        return cur.fetchone()


def _apply_payment(order_id: str, payment_id: str) -> dict:
    with get_cursor() as cur:
        cur.execute("select public.apply_payment(%s, %s) as result", (order_id, payment_id))
        return cur.fetchone()["result"]


def _fetch_captured_payment(payment_id: str, expected_amount: int):
    """Returns the Razorpay payment, capturing it first if it's only authorized."""
    auth = _keys()
    res = httpx.get(f"{RAZORPAY_API}/payments/{payment_id}", auth=auth, timeout=15)
    if res.status_code != 200:
        logger.error("Razorpay fetch payment failed: %s %s", res.status_code, res.text[:300])
        raise HTTPException(status_code=502, detail="Couldn't confirm the payment with Razorpay.")
    payment = res.json()

    # If auto-capture is off in the Razorpay dashboard, capture it ourselves.
    if payment.get("status") == "authorized":
        cap = httpx.post(
            f"{RAZORPAY_API}/payments/{payment_id}/capture",
            auth=auth,
            json={"amount": expected_amount, "currency": "INR"},
            timeout=15,
        )
        if cap.status_code == 200:
            payment = cap.json()
        else:
            logger.warning("Razorpay capture failed: %s %s", cap.status_code, cap.text[:300])
    return payment


# ── Schemas ─────────────────────────────────────────────────────────────────
class CreateOrderBody(BaseModel):
    plan: str


class VerifyBody(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


# ── Endpoints ───────────────────────────────────────────────────────────────
@router.get("/plans")
def list_plans():
    return {
        "currency": "INR",
        "plans": [
            {"id": plan_id, "label": p["label"], "amount_paise": p["amount_paise"], "days": p["days"]}
            for plan_id, p in PLANS.items()
        ],
    }


@router.post("/create-order")
def create_order(body: CreateOrderBody, user: AuthUser = Depends(get_current_user)):
    plan = PLANS.get(body.plan)
    if not plan:
        raise HTTPException(status_code=400, detail="Unknown plan.")
    key_id, key_secret = _keys()

    receipt = f"ec_{user.id[:8]}_{int(time.time())}"  # Razorpay limit: 40 chars
    res = httpx.post(
        f"{RAZORPAY_API}/orders",
        auth=(key_id, key_secret),
        json={
            "amount": plan["amount_paise"],
            "currency": "INR",
            "receipt": receipt,
            "notes": {"user_id": user.id, "plan": body.plan},
        },
        timeout=15,
    )
    if res.status_code >= 300:
        logger.error("Razorpay create order failed: %s %s", res.status_code, res.text[:300])
        raise HTTPException(status_code=502, detail="Couldn't start the payment. Please try again.")
    order = res.json()

    with get_cursor() as cur:
        cur.execute(
            """insert into public.payment_orders (order_id, user_id, plan, amount_paise, currency, days)
               values (%s, %s, %s, %s, %s, %s)""",
            (order["id"], user.id, body.plan, plan["amount_paise"], "INR", plan["days"]),
        )

    return {
        "order_id": order["id"],
        "amount": order["amount"],
        "currency": order["currency"],
        "key_id": key_id,  # public key, safe to send to the browser
        "plan": body.plan,
        "label": plan["label"],
        "email": user.email,
        "name": user.name,
    }


@router.post("/verify")
def verify_payment(body: VerifyBody, user: AuthUser = Depends(get_current_user)):
    _, key_secret = _keys()

    expected = _hmac_sha256(key_secret, f"{body.razorpay_order_id}|{body.razorpay_payment_id}".encode())
    if not hmac.compare_digest(expected, body.razorpay_signature):
        raise HTTPException(status_code=400, detail="Payment verification failed.")

    order = _get_order(body.razorpay_order_id)
    if not order or str(order["user_id"]) != user.id:
        raise HTTPException(status_code=404, detail="Order not found.")
    if order["status"] == "paid":
        return {"status": "paid", "premium_until": _apply_payment(order["order_id"], body.razorpay_payment_id)["premium_until"]}

    payment = _fetch_captured_payment(body.razorpay_payment_id, order["amount_paise"])
    if payment.get("order_id") != order["order_id"] or int(payment.get("amount", 0)) != order["amount_paise"]:
        logger.error("Payment/order mismatch for %s", order["order_id"])
        raise HTTPException(status_code=400, detail="Payment verification failed.")

    if payment.get("status") != "captured":
        # Rare: capture still pending. The webhook will activate Premium.
        return {"status": "processing"}

    result = _apply_payment(order["order_id"], body.razorpay_payment_id)
    return {"status": "paid", "premium_until": result["premium_until"]}


@router.post("/webhook")
async def razorpay_webhook(request: Request):
    """Backup path: activates Premium even if the user closed the tab before verify ran."""
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="Webhook not configured.")

    raw = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")
    if not hmac.compare_digest(_hmac_sha256(secret, raw), signature):
        raise HTTPException(status_code=400, detail="Invalid signature.")

    event = json.loads(raw)
    if event.get("event") not in ("payment.captured", "order.paid"):
        return {"ok": True}  # other events are acknowledged and ignored

    payment = (event.get("payload", {}).get("payment") or {}).get("entity") or {}
    order_id, payment_id = payment.get("order_id"), payment.get("id")
    if not order_id or not payment_id:
        return {"ok": True}

    order = await run_in_threadpool(_get_order, order_id)
    if not order:
        return {"ok": True}  # not an ExamsCalendar Premium order
    if int(payment.get("amount", 0)) != order["amount_paise"]:
        logger.error("Webhook amount mismatch for %s", order_id)
        return {"ok": True}

    await run_in_threadpool(_apply_payment, order_id, payment_id)
    return {"ok": True}
