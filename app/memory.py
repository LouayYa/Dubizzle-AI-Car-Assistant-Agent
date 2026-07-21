"""Memory.

Short-term: an in-process dict keyed by session_id holding the message history
plus a small context blob {focused_listing_id, last_shown_ids} used to resolve
references like "the first Honda" or "is there a warranty on it?".

Long-term: lives in SQLite (users/preferences/liked_listings/inquiries). This
module builds the concise briefing injected into the system prompt at session
start and refreshes the user's summary string at session end.
"""
from __future__ import annotations

import threading
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from . import db

# session_id -> session state
_SESSIONS: dict[str, dict] = {}
_LOCK = threading.Lock()


def new_session(user_id: Optional[str] = None) -> str:
    session_id = uuid.uuid4().hex
    with _LOCK:
        _SESSIONS[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "history": [],  # {role, content} messages for the LLM
            "context": {"focused_listing_id": None, "last_shown_ids": []},
        }
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    return _SESSIONS.get(session_id)


def require_session(session_id: str) -> dict:
    sess = _SESSIONS.get(session_id)
    if sess is None:
        raise KeyError(f"Unknown session_id: {session_id}")
    return sess


def append_history(session_id: str, message: dict) -> None:
    require_session(session_id)["history"].append(message)


def get_history(session_id: str) -> list[dict]:
    return require_session(session_id)["history"]


def set_last_shown(session_id: str, listing_ids: list[int]) -> None:
    """Record the ids most recently surfaced to the user (ordered)."""
    if not listing_ids:
        return
    ctx = require_session(session_id)["context"]
    ctx["last_shown_ids"] = [int(i) for i in listing_ids]
    # "it" defaults to the first shown car.
    ctx["focused_listing_id"] = int(listing_ids[0])


def set_focus(session_id: str, listing_id: int) -> None:
    require_session(session_id)["context"]["focused_listing_id"] = int(listing_id)


def get_context(session_id: str) -> dict:
    return require_session(session_id)["context"]


# --------------------------------------------------------------------------- #
# Long-term briefing / summary
# --------------------------------------------------------------------------- #
def build_briefing(conn: Session, user_id: str) -> str:
    """Build a concise returning-user briefing string for the system prompt.

    Example: "Returning user Sara; budget 40,000-60,000 AED; likes SUV; prefers
    makes: toyota, nissan; previously viewed listing #12."
    """
    user = db.get_user(conn, user_id)
    if not user:
        return ""
    prefs = user.get("preferences", {}) or {}
    bookings = db.bookings_for_user(conn, user_id)
    has_signal = bool(
        bookings
        or prefs.get("summary")
        or user.get("liked_listings")
        or prefs.get("budget_min")
        or prefs.get("budget_max")
        or prefs.get("preferred_makes")
        or prefs.get("preferred_models")
        or prefs.get("preferred_body_types")
        or prefs.get("fuel_pref")
        or prefs.get("financing_pref")
        or prefs.get("notes")
    )
    if not has_signal:
        return ""

    name = user.get("name") or user_id
    parts: list[str] = [f"Returning user {name}"]

    bmin, bmax = prefs.get("budget_min"), prefs.get("budget_max")
    if bmin and bmax:
        parts.append(f"budget {int(bmin):,}-{int(bmax):,} AED")
    elif bmax:
        parts.append(f"budget up to {int(bmax):,} AED")
    elif bmin:
        parts.append(f"budget from {int(bmin):,} AED")

    if prefs.get("preferred_body_types"):
        parts.append("likes " + ", ".join(prefs["preferred_body_types"]))
    if prefs.get("preferred_makes"):
        parts.append("prefers makes: " + ", ".join(prefs["preferred_makes"]))
    if prefs.get("preferred_models"):
        parts.append("models: " + ", ".join(prefs["preferred_models"]))
    if prefs.get("fuel_pref"):
        parts.append(f"fuel: {prefs['fuel_pref']}")
    if prefs.get("financing_pref"):
        parts.append(f"financing: {prefs['financing_pref']}")
    if user.get("liked_listings"):
        liked = ", ".join(f"#{i}" for i in user["liked_listings"][:5])
        parts.append(f"previously liked {liked}")
    if bookings:
        b = bookings[0]
        car = b.get("car") or {}
        label = f"#{b.get('listing_id')}"
        if car:
            label = f"#{car['listing_id']} {car.get('year')} {car.get('make')} {car.get('model')}".strip()
        extra = f" (+{len(bookings) - 1} more)" if len(bookings) > 1 else ""
        parts.append(f"has a viewing booked: {label} at {b.get('slot_datetime')}{extra}")
    if prefs.get("notes"):
        parts.append(f"notes: {prefs['notes']}")

    # Lead with the hand-written summary, if any.
    summary = prefs.get("summary")
    briefing = "; ".join(parts) + "."
    if summary:
        briefing = f"{summary} ({briefing})"
    return briefing


def refresh_summary(conn: Session, user_id: str) -> str:
    """Recompute a compact summary string from structured prefs (end of session).

    Deterministic and cheap (no LLM) so it always works offline. Stored on the
    preferences row for next-session recall.
    """
    user = db.get_user(conn, user_id)
    if not user:
        return ""
    prefs = user.get("preferences", {}) or {}
    bits: list[str] = []
    bmin, bmax = prefs.get("budget_min"), prefs.get("budget_max")
    if bmin or bmax:
        lo = f"{int(bmin):,}" if bmin else "?"
        hi = f"{int(bmax):,}" if bmax else "?"
        bits.append(f"budget {lo}-{hi} AED")
    if prefs.get("preferred_body_types"):
        bits.append("body: " + "/".join(prefs["preferred_body_types"]))
    if prefs.get("preferred_makes"):
        bits.append("makes: " + "/".join(prefs["preferred_makes"]))
    if user.get("liked_listings"):
        bits.append("liked " + "/".join(f"#{i}" for i in user["liked_listings"][:5]))
    name = user.get("name") or user_id
    summary = f"{name} — " + "; ".join(bits) if bits else f"{name} — no strong prefs yet"
    db.update_preferences(conn, user_id, summary=summary)
    return summary
