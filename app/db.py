"""SQLAlchemy persistence — one SQLite file at data/app.db.

Tables:
- listings: read-mostly reference inventory (the cars)
- users: identity
- preferences: durable per-user memory (budget, favorite makes, etc.)
- liked_listings: user <-> listing many-to-many
- inquiries: log of raw user messages, for cross-session recall
- bookings: viewing bookings, one row per booked slot

Leads are NOT stored here — they are appended to data/leads.csv (see tools.py).
Car data is never duplicated into the bookings table; bookings reference
listing_id only.

Every public function here takes a SQLAlchemy `Session` and returns plain
dicts/lists of dicts (never ORM instances), so callers elsewhere in the app
never need to know this is an ORM — they already treat the session as an
opaque "conn" object and results as plain data.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import Boolean, ForeignKey, Text, UniqueConstraint, create_engine, event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from . import config


# --------------------------------------------------------------------------- #
# ORM models
# --------------------------------------------------------------------------- #
class Base(DeclarativeBase):
    pass


class Listing(Base):
    __tablename__ = "listings"

    listing_id: Mapped[int] = mapped_column(primary_key=True)
    year: Mapped[Optional[int]] = mapped_column()
    make: Mapped[Optional[str]] = mapped_column()
    model: Mapped[Optional[str]] = mapped_column()
    trim: Mapped[Optional[str]] = mapped_column()
    title: Mapped[Optional[str]] = mapped_column()
    description_clean: Mapped[Optional[str]] = mapped_column(Text)
    photo_url: Mapped[Optional[str]] = mapped_column()
    price_aed: Mapped[Optional[int]] = mapped_column()
    monthly_payment_aed: Mapped[Optional[int]] = mapped_column()
    mileage_km: Mapped[Optional[int]] = mapped_column()
    is_new: Mapped[Optional[bool]] = mapped_column(Boolean)
    exterior_color: Mapped[Optional[str]] = mapped_column()
    body_type: Mapped[Optional[str]] = mapped_column()
    transmission: Mapped[Optional[str]] = mapped_column()
    fuel_type: Mapped[Optional[str]] = mapped_column()
    regional_spec: Mapped[Optional[str]] = mapped_column()
    has_warranty: Mapped[Optional[bool]] = mapped_column(Boolean)


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]] = mapped_column()
    created_at: Mapped[str] = mapped_column()


class Preference(Base):
    __tablename__ = "preferences"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), primary_key=True)
    budget_min: Mapped[Optional[int]] = mapped_column()
    budget_max: Mapped[Optional[int]] = mapped_column()
    preferred_makes: Mapped[Optional[str]] = mapped_column(Text)   # JSON list
    preferred_models: Mapped[Optional[str]] = mapped_column(Text)  # JSON list
    preferred_body_types: Mapped[Optional[str]] = mapped_column(Text)  # JSON list
    fuel_pref: Mapped[Optional[str]] = mapped_column()
    financing_pref: Mapped[Optional[str]] = mapped_column()
    notes: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    updated_at: Mapped[Optional[str]] = mapped_column()


class LikedListing(Base):
    __tablename__ = "liked_listings"
    __table_args__ = (UniqueConstraint("user_id", "listing_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"))
    listing_id: Mapped[Optional[int]] = mapped_column(ForeignKey("listings.listing_id"))
    created_at: Mapped[str] = mapped_column()


class Inquiry(Base):
    __tablename__ = "inquiries"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"))
    session_id: Mapped[Optional[str]] = mapped_column()
    text: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column()


class Booking(Base):
    __tablename__ = "bookings"
    __table_args__ = (UniqueConstraint("listing_id", "slot_datetime"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    listing_id: Mapped[Optional[int]] = mapped_column(ForeignKey("listings.listing_id"))
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.user_id"))
    session_id: Mapped[Optional[str]] = mapped_column()
    slot_datetime: Mapped[str] = mapped_column()
    created_at: Mapped[str] = mapped_column()


# JSON-ish preference columns stored as TEXT (JSON) in SQLite.
_LIST_PREF_COLS = (
    "preferred_makes",
    "preferred_models",
    "preferred_body_types",
)
_PREF_COLS = (
    "budget_min",
    "budget_max",
    "preferred_makes",
    "preferred_models",
    "preferred_body_types",
    "fuel_pref",
    "financing_pref",
    "notes",
    "summary",
)

# Columns that make up a full listing row (used for seeding).
LISTING_COLS = (
    "listing_id",
    "year",
    "make",
    "model",
    "trim",
    "title",
    "description_clean",
    "photo_url",
    "price_aed",
    "monthly_payment_aed",
    "mileage_km",
    "is_new",
    "exterior_color",
    "body_type",
    "transmission",
    "fuel_type",
    "regional_spec",
    "has_warranty",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_dict(obj: Any) -> dict:
    return {c.key: getattr(obj, c.key) for c in obj.__table__.columns}


# --------------------------------------------------------------------------- #
# Engine / session
# --------------------------------------------------------------------------- #
def make_engine(db_path: Optional[str] = None):
    """Create a SQLite engine with FK enforcement turned on for every connection."""
    path = str(db_path or config.DB_PATH)
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.close()

    return engine


def get_session(db_path: Optional[str] = None) -> Session:
    """Return a standalone SQLAlchemy session (its own engine) — mainly for tests."""
    engine = make_engine(db_path)
    return sessionmaker(bind=engine, expire_on_commit=True)()


def init_db(session: Session) -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(session.get_bind())


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def _to_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    return bool(v)


def seed_listings_if_empty(session: Session, inventory_path: Optional[str] = None) -> int:
    """Load listings from inventory.json on first startup. Returns rows inserted."""
    count = session.execute(select(func.count()).select_from(Listing)).scalar_one()
    if count > 0:
        return 0

    path = inventory_path or config.INVENTORY_JSON
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)

    inserted = 0
    for r in rows:
        session.merge(
            Listing(
                listing_id=_to_int(r.get("Listing_ID", r.get("listing_id"))),
                year=_to_int(r.get("year")),
                make=r.get("make"),
                model=r.get("model"),
                trim=r.get("trim"),
                title=r.get("title"),
                description_clean=r.get("description_clean") or r.get("description"),
                photo_url=r.get("photo_url"),
                price_aed=_to_int(r.get("price_aed")),
                monthly_payment_aed=_to_int(r.get("monthly_payment_aed")),
                mileage_km=_to_int(r.get("mileage_km")),
                is_new=_to_bool(r.get("is_new")),
                exterior_color=r.get("exterior_color"),
                body_type=r.get("body_type"),
                transmission=r.get("transmission"),
                fuel_type=r.get("fuel_type"),
                regional_spec=r.get("regional_spec"),
                has_warranty=_to_bool(r.get("has_warranty")),
            )
        )
        inserted += 1
    session.commit()
    return inserted


# --------------------------------------------------------------------------- #
# Listing reads
# --------------------------------------------------------------------------- #
def get_listing(session: Session, listing_id: int) -> Optional[dict]:
    obj = session.get(Listing, int(listing_id))
    return _to_dict(obj) if obj else None


def get_listings_by_ids(session: Session, ids: Iterable[int]) -> list[dict]:
    ids = [int(i) for i in ids]
    if not ids:
        return []
    rows = session.execute(select(Listing).where(Listing.listing_id.in_(ids))).scalars().all()
    by_id = {r.listing_id: _to_dict(r) for r in rows}
    # Preserve caller ordering (e.g. semantic ranking).
    return [by_id[i] for i in ids if i in by_id]


def query_listings(
    session: Session,
    *,
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
    ids: Optional[Iterable[int]] = None,
    sort_by: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Structured filter over the listings table. Case-insensitive text matching.

    price_min/price_max only match rows with a known cash price (NULL prices are
    excluded from price-bounded queries, never coerced to 0).
    """
    where = []

    if make:
        where.append(func.lower(Listing.make).like(f"%{make.lower()}%"))
    if model:
        where.append(func.lower(Listing.model).like(f"%{model.lower()}%"))
    if year_min is not None:
        where.append(Listing.year >= year_min)
    if year_max is not None:
        where.append(Listing.year <= year_max)
    if price_min is not None:
        where.append(Listing.price_aed.isnot(None))
        where.append(Listing.price_aed >= price_min)
    if price_max is not None:
        where.append(Listing.price_aed.isnot(None))
        where.append(Listing.price_aed <= price_max)
    if mileage_max is not None:
        where.append(Listing.mileage_km.isnot(None))
        where.append(Listing.mileage_km <= mileage_max)
    if body_type:
        where.append(func.lower(Listing.body_type).like(f"%{body_type.lower()}%"))
    if fuel_type:
        where.append(func.lower(Listing.fuel_type).like(f"%{fuel_type.lower()}%"))
    if spec:
        where.append(func.lower(Listing.regional_spec).like(f"%{spec.lower()}%"))
    if is_new is not None:
        where.append(Listing.is_new == is_new)
    if ids is not None:
        id_list = [int(i) for i in ids]
        if not id_list:
            return []
        where.append(Listing.listing_id.in_(id_list))

    stmt = select(Listing).where(*where)

    # Sorting. NULLS LAST for price/mileage so unknowns don't top the list.
    sort_map = {
        "price_asc": Listing.price_aed.asc().nulls_last(),
        "price_desc": Listing.price_aed.desc().nulls_last(),
        "mileage_asc": Listing.mileage_km.asc().nulls_last(),
        "mileage_desc": Listing.mileage_km.desc().nulls_last(),
        "year_desc": Listing.year.desc(),
        "year_asc": Listing.year.asc(),
    }
    stmt = stmt.order_by(sort_map[sort_by] if sort_by in sort_map else Listing.listing_id.asc())

    if limit is not None:
        stmt = stmt.limit(int(limit))

    rows = session.execute(stmt).scalars().all()
    return [_to_dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Users + preferences (long-term memory)
# --------------------------------------------------------------------------- #
def create_user(session: Session, user_id: str, name: Optional[str] = None) -> dict:
    user = session.get(User, user_id)
    if user is None:
        session.add(User(user_id=user_id, name=name, created_at=_now()))
    elif name and not user.name:
        user.name = name

    if session.get(Preference, user_id) is None:
        session.add(Preference(user_id=user_id, updated_at=_now()))

    session.commit()
    return get_user(session, user_id)


def get_user(session: Session, user_id: str) -> Optional[dict]:
    urow = session.get(User, user_id)
    if not urow:
        return None
    prow = session.get(Preference, user_id)
    prefs = _to_dict(prow) if prow else {}
    prefs.pop("user_id", None)
    for col in _LIST_PREF_COLS:
        raw = prefs.get(col)
        prefs[col] = json.loads(raw) if raw else []
    liked = session.execute(
        select(LikedListing.listing_id).where(LikedListing.user_id == user_id).order_by(LikedListing.id)
    ).scalars().all()
    return {
        "user_id": urow.user_id,
        "name": urow.name,
        "created_at": urow.created_at,
        "preferences": prefs,
        "liked_listings": list(liked),
    }


def update_preferences(session: Session, user_id: str, **fields: Any) -> dict:
    """Upsert preference columns. List-ish fields are JSON-encoded.

    Unknown keys are ignored. Only provided keys are updated.
    """
    create_user(session, user_id)  # ensure rows exist
    pref = session.get(Preference, user_id)
    changed = False
    for key, val in fields.items():
        if key not in _PREF_COLS:
            continue
        if key in _LIST_PREF_COLS:
            # Accept list or comma string; store as JSON list.
            if isinstance(val, str):
                val = [v.strip() for v in val.split(",") if v.strip()]
            val = json.dumps(val or [])
        setattr(pref, key, val)
        changed = True
    if changed:
        pref.updated_at = _now()
        session.commit()
    return get_user(session, user_id)


def add_liked_listing(session: Session, user_id: str, listing_id: int) -> None:
    create_user(session, user_id)
    listing_id = int(listing_id)
    existing = session.execute(
        select(LikedListing).where(LikedListing.user_id == user_id, LikedListing.listing_id == listing_id)
    ).scalar_one_or_none()
    if existing is None:
        session.add(LikedListing(user_id=user_id, listing_id=listing_id, created_at=_now()))
        session.commit()


def add_inquiry(session: Session, user_id: Optional[str], session_id: str, text: str) -> None:
    session.add(Inquiry(user_id=user_id, session_id=session_id, text=text, created_at=_now()))
    session.commit()


def recent_inquiries(session: Session, user_id: str, limit: int = 5) -> list[str]:
    rows = session.execute(
        select(Inquiry.text).where(Inquiry.user_id == user_id).order_by(Inquiry.id.desc()).limit(limit)
    ).scalars().all()
    return list(rows)


# --------------------------------------------------------------------------- #
# Bookings
# --------------------------------------------------------------------------- #
def create_booking(
    session: Session,
    *,
    listing_id: int,
    user_id: Optional[str],
    session_id: Optional[str],
    slot_datetime: str,
) -> int:
    """Insert a booking. Raises sqlalchemy.exc.IntegrityError on duplicate slot.

    Slot/business-hour validation happens in tools.book_viewing before this call.
    """
    booking = Booking(
        listing_id=int(listing_id),
        user_id=user_id,
        session_id=session_id,
        slot_datetime=slot_datetime,
        created_at=_now(),
    )
    session.add(booking)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise
    return int(booking.id)


def bookings_for_listing(session: Session, listing_id: int) -> list[dict]:
    rows = session.execute(
        select(Booking).where(Booking.listing_id == int(listing_id)).order_by(Booking.slot_datetime)
    ).scalars().all()
    return [_to_dict(r) for r in rows]


def delete_booking(session: Session, booking_id: int, user_id: Optional[str] = None) -> bool:
    """Delete a booking. If user_id is given, only delete it when it belongs to
    that user (so one user can't cancel another's). Returns True if a row was
    removed, False if nothing matched.
    """
    booking = session.get(Booking, int(booking_id))
    if booking is None:
        return False
    if user_id is not None and booking.user_id != user_id:
        return False
    session.delete(booking)
    session.commit()
    return True


def bookings_for_user(session: Session, user_id: str) -> list[dict]:
    """All bookings for a user, each enriched with a short car summary.

    Ordered soonest-first so the agent can answer "do I have a booking?".
    """
    rows = session.execute(
        select(Booking).where(Booking.user_id == user_id).order_by(Booking.slot_datetime)
    ).scalars().all()
    out: list[dict] = []
    for b in rows:
        rec = _to_dict(b)
        car = get_listing(session, b.listing_id) if b.listing_id is not None else None
        if car:
            rec["car"] = {
                "listing_id": car["listing_id"],
                "year": car.get("year"),
                "make": car.get("make"),
                "model": car.get("model"),
                "trim": car.get("trim"),
            }
        out.append(rec)
    return out
