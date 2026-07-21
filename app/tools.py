"""Agent tools; in-process Python functions exposed to the model via function
calling (no separate MCP server).

Each tool returns a JSON-serializable dict. Grounding rule: tools are the ONLY
source of car data; they read the real dataset and never invent listings; the
agent surfaces exactly what these return.

Tools that act on "the current user"/"the current session" receive a ToolContext
injected by the agent (conn, session_id, user_id); that context is NOT part of
the schema the model sees, so the model cannot spoof another user.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import config, db, media, retrieval


@dataclass
class ToolContext:
    conn: Session
    session_id: Optional[str] = None
    user_id: Optional[str] = None


# --------------------------------------------------------------------------- #
# Serialization for the client
# --------------------------------------------------------------------------- #
def car_view(listing: dict) -> dict:
    """Shape a listing for API/UI: adds photo_renderable and a price label."""
    photo = media.normalize_photo_url(listing.get("photo_url"))
    renderable = media.is_renderable_photo(photo)

    price = listing.get("price_aed")
    monthly = listing.get("monthly_payment_aed")
    if price is not None:
        price_label = f"AED {int(price):,}"
    elif monthly is not None:
        price_label = f"Finance only — AED {int(monthly):,}/mo"
    else:
        price_label = "Price not listed"

    out = dict(listing)
    out["photo_url"] = photo
    out["photo_renderable"] = renderable
    out["price_label"] = price_label
    return out


def _views(listings: list[dict]) -> list[dict]:
    return [car_view(x) for x in listings]


# --------------------------------------------------------------------------- #
# Inventory tools
# --------------------------------------------------------------------------- #
def search_inventory(
    ctx: ToolContext,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    mileage_max: Optional[int] = None,
    body_type: Optional[str] = None,
    fuel_type: Optional[str] = None,
    spec: Optional[str] = None,
    is_new: Optional[bool] = None,
    sort_by: Optional[str] = None,
    limit: int = 8,
) -> dict:
    """Structured filter over listings. Returns only real matching cars."""
    limit = max(1, min(int(limit or 8), 25))
    cars = db.query_listings(
        ctx.conn,
        make=make,
        model=model,
        year_min=year_min,
        year_max=year_max,
        price_min=price_min,
        price_max=price_max,
        mileage_max=mileage_max,
        body_type=body_type,
        fuel_type=fuel_type,
        spec=spec,
        is_new=is_new,
        sort_by=sort_by,
        limit=limit,
    )
    return {"count": len(cars), "cars": _views(cars)}


def semantic_search(
    ctx: ToolContext,
    query: str,
    limit: int = 8,
    make: Optional[str] = None,
    model: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    mileage_max: Optional[int] = None,
    body_type: Optional[str] = None,
    fuel_type: Optional[str] = None,
    spec: Optional[str] = None,
) -> dict:
    """Semantic (vibe) search over cached embeddings.

    Hybrid: when hard constraints are also given, filter structurally first and
    rank the survivors by semantic similarity to `query`.
    """
    limit = max(1, min(int(limit or 8), 25))
    has_filter = any(
        v is not None
        for v in (make, model, year_min, year_max, price_min, price_max,
                  mileage_max, body_type, fuel_type, spec)
    )
    candidate_ids = None
    if has_filter:
        pre = db.query_listings(
            ctx.conn, make=make, model=model, year_min=year_min, year_max=year_max,
            price_min=price_min, price_max=price_max, mileage_max=mileage_max,
            body_type=body_type, fuel_type=fuel_type, spec=spec,
        )
        candidate_ids = [c["listing_id"] for c in pre]
        if not candidate_ids:
            return {"count": 0, "cars": [], "note": "No cars matched the hard filters."}

    ranked = retrieval.semantic_rank(query, candidate_ids=candidate_ids, limit=limit)
    ids = [lid for lid, _score in ranked]
    cars = db.get_listings_by_ids(ctx.conn, ids)
    scores = dict(ranked)
    for c in cars:
        c["similarity"] = round(scores.get(c["listing_id"], 0.0), 4)
    return {"count": len(cars), "cars": _views(cars)}


def get_listing_details(ctx: ToolContext, listing_id: int) -> dict:
    """Full record for one car (reference-resolution questions)."""
    car = db.get_listing(ctx.conn, int(listing_id))
    if not car:
        return {"found": False, "message": f"No listing with id {listing_id}."}
    return {"found": True, "car": car_view(car)}


# --------------------------------------------------------------------------- #
# Booking
# --------------------------------------------------------------------------- #
_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def parse_slot(slot_datetime: str) -> Optional[datetime]:
    """Parse a slot string; accepts ISO 'YYYY-MM-DDTHH:MM[:SS]' or with a space."""
    s = (slot_datetime or "").strip().replace(" ", "T", 1)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def validate_slot(slot_datetime: str, now: Optional[datetime] = None) -> tuple[bool, str]:
    """Validate against Mon–Sat 08:00–20:00, no past slots. Returns (ok, message)."""
    dt = parse_slot(slot_datetime)
    if dt is None:
        return False, (
            "I couldn't read that date/time. Please use a format like "
            "'2026-07-25 14:30' (24-hour)."
        )
    now = now or datetime.now()
    if dt < now:
        return False, "That time is in the past — please pick a future slot."
    if dt.weekday() == 6:  # Sunday
        return False, "We're closed on Sundays. Viewings run Monday–Saturday, 08:00–20:00."
    after_open = (dt.hour > 8) or (dt.hour == 8 and dt.minute >= 0)
    before_close = (dt.hour < 20) or (dt.hour == 20 and dt.minute == 0)
    if not (after_open and before_close):
        return False, (
            "Viewings are available 08:00–20:00. Please pick a time in that window."
        )
    return True, "ok"


def book_viewing(
    ctx: ToolContext,
    listing_id: int,
    slot_datetime: str,
    user_id: Optional[str] = None,
) -> dict:
    """Validate business rules, enforce the unique-slot constraint, persist."""
    listing_id = int(listing_id)
    car = db.get_listing(ctx.conn, listing_id)
    if not car:
        return {"ok": False, "message": f"No listing with id {listing_id}."}

    ok, msg = validate_slot(slot_datetime)
    if not ok:
        return {"ok": False, "message": msg}

    uid = user_id or ctx.user_id
    if uid:
        # Satisfy the bookings.user_id foreign key (booking implies a user).
        db.create_user(ctx.conn, uid)
    dt = parse_slot(slot_datetime)
    slot_iso = dt.replace(second=0, microsecond=0).isoformat(timespec="minutes")
    try:
        booking_id = db.create_booking(
            ctx.conn,
            listing_id=listing_id,
            user_id=uid,
            session_id=ctx.session_id,
            slot_datetime=slot_iso,
        )
    except IntegrityError as exc:
        if "UNIQUE" in str(exc).upper():
            return {
                "ok": False,
                "message": (
                    f"That exact slot ({slot_iso}) is already booked for this car. "
                    "Please pick a different time."
                ),
            }
        return {"ok": False, "message": f"Could not save the booking: {exc}"}

    # A booking is a strong lead signal — record it to the leads CSV.
    _write_lead_row(ctx, notes=f"Booked viewing of listing #{listing_id} at {slot_iso}")

    when = f"{_WEEKDAY_NAMES[dt.weekday()]} {slot_iso.replace('T', ' ')}"
    return {
        "ok": True,
        "booking_id": booking_id,
        "listing_id": listing_id,
        "slot_datetime": slot_iso,
        "message": (
            f"Booked! Viewing of the {car.get('year')} {car.get('make')} "
            f"{car.get('model')} on {when}."
        ),
    }


# --------------------------------------------------------------------------- #
# Leads
# --------------------------------------------------------------------------- #
_LEAD_HEADER = [
    "timestamp", "session_id", "user_id", "name", "contact",
    "budget_min", "budget_max", "desired_make", "desired_model",
    "desired_body_type", "financing_pref", "timeline", "notes",
]


def _write_lead_row(ctx: ToolContext, **fields: Any) -> str:
    """Append one row to data/leads.csv, backfilling from the user profile."""
    profile = db.get_user(ctx.conn, ctx.user_id) if ctx.user_id else None
    prefs = (profile or {}).get("preferences", {}) if profile else {}

    def pick(key, pref_key=None):
        val = fields.get(key)
        if val in (None, ""):
            val = prefs.get(pref_key) if pref_key else None
        if isinstance(val, (list, tuple)):
            val = ", ".join(str(v) for v in val)
        return "" if val is None else val

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": ctx.session_id or "",
        "user_id": ctx.user_id or "",
        "name": pick("name") or (profile or {}).get("name", ""),
        "contact": pick("contact"),
        "budget_min": pick("budget_min", "budget_min"),
        "budget_max": pick("budget_max", "budget_max"),
        "desired_make": pick("desired_make", "preferred_makes"),
        "desired_model": pick("desired_model", "preferred_models"),
        "desired_body_type": pick("desired_body_type", "preferred_body_types"),
        "financing_pref": pick("financing_pref", "financing_pref"),
        "timeline": pick("timeline"),
        "notes": pick("notes"),
    }

    path = config.LEADS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LEAD_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return str(path)


def save_lead(
    ctx: ToolContext,
    name: Optional[str] = None,
    contact: Optional[str] = None,
    budget_min: Optional[int] = None,
    budget_max: Optional[int] = None,
    desired_make: Optional[str] = None,
    desired_model: Optional[str] = None,
    desired_body_type: Optional[str] = None,
    financing_pref: Optional[str] = None,
    timeline: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    """Persist a qualified lead to data/leads.csv (budget + at least one need)."""
    has_budget = budget_min is not None or budget_max is not None
    has_need = any(
        v for v in (desired_make, desired_model, desired_body_type, notes)
    )
    if not (has_budget or has_need):
        return {
            "ok": False,
            "message": "Not enough to qualify a lead yet (need a budget or a concrete need).",
        }
    path = _write_lead_row(
        ctx,
        name=name, contact=contact, budget_min=budget_min, budget_max=budget_max,
        desired_make=desired_make, desired_model=desired_model,
        desired_body_type=desired_body_type, financing_pref=financing_pref,
        timeline=timeline, notes=notes,
    )
    return {"ok": True, "message": "Lead saved.", "path": path}


# --------------------------------------------------------------------------- #
# User profile (long-term memory)
# --------------------------------------------------------------------------- #
def get_user_profile(ctx: ToolContext, user_id: Optional[str] = None) -> dict:
    uid = user_id or ctx.user_id
    if not uid:
        return {"found": False, "message": "No user is bound to this session."}
    profile = db.get_user(ctx.conn, uid)
    if not profile:
        return {"found": False, "message": f"No profile for user {uid}."}
    return {"found": True, "profile": profile}


def cancel_booking(ctx: ToolContext, booking_id: int, user_id: Optional[str] = None) -> dict:
    """Cancel one of the current user's bookings by its booking_id.

    Ownership-checked: a user can only cancel their own booking. Call
    get_user_bookings first if you need the booking_id.
    """
    uid = user_id or ctx.user_id
    if not uid:
        return {"ok": False, "message": "No user is bound to this session."}
    removed = db.delete_booking(ctx.conn, int(booking_id), user_id=uid)
    if not removed:
        return {
            "ok": False,
            "message": (
                f"No booking #{booking_id} found for you to cancel. "
                "Use get_user_bookings to see your current bookings."
            ),
        }
    return {"ok": True, "booking_id": int(booking_id),
            "message": f"Booking #{booking_id} has been cancelled."}


def get_user_bookings(ctx: ToolContext, user_id: Optional[str] = None) -> dict:
    """List the user's viewing bookings (grounded — reads the bookings table).

    Call this to answer any question about whether/what the user has booked;
    never assert booking status without it.
    """
    uid = user_id or ctx.user_id
    if not uid:
        return {"found": False, "count": 0, "bookings": [],
                "message": "No user is bound to this session."}
    bookings = db.bookings_for_user(ctx.conn, uid)
    return {"found": True, "count": len(bookings), "bookings": bookings}


def update_user_profile(
    ctx: ToolContext,
    user_id: Optional[str] = None,
    budget_min: Optional[int] = None,
    budget_max: Optional[int] = None,
    preferred_makes: Optional[Any] = None,
    preferred_models: Optional[Any] = None,
    preferred_body_types: Optional[Any] = None,
    fuel_pref: Optional[str] = None,
    financing_pref: Optional[str] = None,
    notes: Optional[str] = None,
    liked_listing_id: Optional[int] = None,
) -> dict:
    """Update durable preferences. Call whenever a lasting preference is learned."""
    uid = user_id or ctx.user_id
    if not uid:
        return {"ok": False, "message": "No user is bound to this session."}

    fields = {
        k: v
        for k, v in {
            "budget_min": budget_min,
            "budget_max": budget_max,
            "preferred_makes": preferred_makes,
            "preferred_models": preferred_models,
            "preferred_body_types": preferred_body_types,
            "fuel_pref": fuel_pref,
            "financing_pref": financing_pref,
            "notes": notes,
        }.items()
        if v is not None
    }
    if fields:
        db.update_preferences(ctx.conn, uid, **fields)
    if liked_listing_id is not None:
        db.add_liked_listing(ctx.conn, uid, int(liked_listing_id))

    return {"ok": True, "profile": db.get_user(ctx.conn, uid)}


# --------------------------------------------------------------------------- #
# Registry + schemas exposed to the model
# --------------------------------------------------------------------------- #
TOOL_FUNCTIONS = {
    "search_inventory": search_inventory,
    "semantic_search": semantic_search,
    "get_listing_details": get_listing_details,
    "book_viewing": book_viewing,
    "cancel_booking": cancel_booking,
    "save_lead": save_lead,
    "get_user_profile": get_user_profile,
    "get_user_bookings": get_user_bookings,
    "update_user_profile": update_user_profile,
}

# Tools whose returned cars should refresh short-term "last shown" context.
INVENTORY_TOOLS = {"search_inventory", "semantic_search"}

_SORT_ENUM = [
    "price_asc", "price_desc", "mileage_asc", "mileage_desc", "year_asc", "year_desc",
]

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_inventory",
            "description": (
                "Structured filter over the real car inventory. Use for hard "
                "constraints (make, model, year range, price range in AED, max "
                "mileage in km, body type, fuel type, regional spec). Returns only "
                "real listings. Price filters skip cars with no listed cash price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "make": {"type": "string"},
                    "model": {"type": "string"},
                    "year_min": {"type": "integer"},
                    "year_max": {"type": "integer"},
                    "price_min": {"type": "integer", "description": "Min cash price in AED"},
                    "price_max": {"type": "integer", "description": "Max cash price in AED"},
                    "mileage_max": {"type": "integer", "description": "Max mileage in km"},
                    "body_type": {"type": "string", "description": "SUV, Sedan, Coupe, Hatchback, Pickup, ..."},
                    "fuel_type": {"type": "string", "description": "Petrol, Diesel, Electric, Hybrid"},
                    "spec": {"type": "string", "description": "Regional spec e.g. GCC, US, Japanese"},
                    "is_new": {"type": "boolean", "description": "true for brand-new / 0km cars"},
                    "sort_by": {"type": "string", "enum": _SORT_ENUM},
                    "limit": {"type": "integer", "description": "Max results (default 8)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Semantic/'vibe' search over listing descriptions (e.g. 'sporty', "
                "'family SUV', 'luxury cruiser'). Optionally pass the same hard "
                "filters as search_inventory to filter first, then rank by vibe."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The vibe/intent to match"},
                    "limit": {"type": "integer"},
                    "make": {"type": "string"},
                    "model": {"type": "string"},
                    "year_min": {"type": "integer"},
                    "year_max": {"type": "integer"},
                    "price_min": {"type": "integer"},
                    "price_max": {"type": "integer"},
                    "mileage_max": {"type": "integer"},
                    "body_type": {"type": "string"},
                    "fuel_type": {"type": "string"},
                    "spec": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_listing_details",
            "description": (
                "Fetch the full record for one car by listing_id. Use to answer "
                "follow-ups like 'what's the mileage on it?' or 'is there a warranty?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"listing_id": {"type": "integer"}},
                "required": ["listing_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_viewing",
            "description": (
                "Book a viewing for a car. Slots are Monday–Saturday, 08:00–20:00, "
                "future only. slot_datetime must be 'YYYY-MM-DD HH:MM' (24-hour)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_id": {"type": "integer"},
                    "slot_datetime": {"type": "string", "description": "e.g. '2026-07-25 14:30'"},
                },
                "required": ["listing_id", "slot_datetime"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_lead",
            "description": (
                "Save a qualified sales lead once you know the budget and at least "
                "one concrete need. Fields are optional; unknowns are backfilled "
                "from the user's saved profile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "contact": {"type": "string", "description": "phone or email"},
                    "budget_min": {"type": "integer"},
                    "budget_max": {"type": "integer"},
                    "desired_make": {"type": "string"},
                    "desired_model": {"type": "string"},
                    "desired_body_type": {"type": "string"},
                    "financing_pref": {"type": "string", "description": "cash or finance"},
                    "timeline": {"type": "string", "description": "e.g. 'this month'"},
                    "notes": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Read the current user's saved profile and preferences.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_bookings",
            "description": (
                "List the current user's viewing bookings (with car + slot + "
                "booking_id). ALWAYS call this to answer any question about "
                "whether the user has a booking or what they booked — never guess. "
                "Also call it to find the booking_id before cancelling."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_booking",
            "description": (
                "Cancel one of the current user's viewing bookings by its "
                "booking_id (get it from get_user_bookings first). Only the "
                "user's own bookings can be cancelled."
            ),
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "integer"}},
                "required": ["booking_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": (
                "Persist a durable preference the moment you learn it: budget, "
                "preferred makes/models/body types, fuel or financing preference, "
                "free-form notes, or a liked car (liked_listing_id)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_min": {"type": "integer"},
                    "budget_max": {"type": "integer"},
                    "preferred_makes": {"type": "array", "items": {"type": "string"}},
                    "preferred_models": {"type": "array", "items": {"type": "string"}},
                    "preferred_body_types": {"type": "array", "items": {"type": "string"}},
                    "fuel_pref": {"type": "string"},
                    "financing_pref": {"type": "string"},
                    "notes": {"type": "string"},
                    "liked_listing_id": {"type": "integer"},
                },
            },
        },
    },
]
