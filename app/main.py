"""FastAPI backend.

All LLM/agent/tool logic lives here (and in the sibling modules). The Streamlit
client only makes HTTP calls — no LLM usage on the client side.

Each request gets its own SQLAlchemy session (see db_session below); the
database runs in WAL mode so a long agent turn doesn't block other requests.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from . import agent, config, db, memory, retrieval
from .tools import ToolContext, book_viewing, car_view, cancel_booking, save_lead

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
_engine = db.make_engine()
_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=True)


def db_session():
    """FastAPI dependency: one SQLAlchemy session per request, always closed."""
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    session = _SessionLocal()
    try:
        # WAL lets readers and a writer work concurrently (persisted on the file).
        session.execute(text("PRAGMA journal_mode=WAL"))
        db.init_db(session)
        app.state.seeded = db.seed_listings_if_empty(session)
    finally:
        session.close()

    try:
        retrieval.load_embeddings()
        app.state.embeddings_ok = True
    except Exception as exc:
        app.state.embeddings_ok = False
        app.state.embeddings_error = str(exc)
    yield


app = FastAPI(title="dubizzle Cars AI Assistant", version="0.1.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class SessionCreate(BaseModel):
    user_id: Optional[str] = None
    name: Optional[str] = None


class MessageCreate(BaseModel):
    message: str


class LeadRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    name: Optional[str] = None
    contact: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    desired_make: Optional[str] = None
    desired_model: Optional[str] = None
    desired_body_type: Optional[str] = None
    financing_pref: Optional[str] = None
    timeline: Optional[str] = None
    notes: Optional[str] = None


class BookingRequest(BaseModel):
    listing_id: int
    slot_datetime: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    preferred_makes: Optional[list[str]] = None
    preferred_models: Optional[list[str]] = None
    preferred_body_types: Optional[list[str]] = None
    fuel_pref: Optional[str] = None
    financing_pref: Optional[str] = None
    notes: Optional[str] = None
    summary: Optional[str] = None


# --------------------------------------------------------------------------- #
# Response schemas — declared so /docs documents what each endpoint RETURNS
# (not just what it accepts). extra="allow" on the car/profile shapes means the
# known fields are documented while any additional fields still pass through, so
# adding a column never silently drops from responses.
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    embeddings_loaded: bool
    chat_model: str
    embed_model: str


class SessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    is_returning: bool
    briefing: str
    greeting: str


class EndSessionResponse(BaseModel):
    ok: bool
    summary: str


class CarView(BaseModel):
    model_config = ConfigDict(extra="allow")
    listing_id: int
    year: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    trim: Optional[str] = None
    description_clean: Optional[str] = None
    photo_url: Optional[str] = None
    photo_renderable: bool = False
    price_aed: Optional[int] = None
    monthly_payment_aed: Optional[int] = None
    price_label: str
    mileage_km: Optional[int] = None
    is_new: Optional[bool] = None
    exterior_color: Optional[str] = None
    body_type: Optional[str] = None
    transmission: Optional[str] = None
    fuel_type: Optional[str] = None
    regional_spec: Optional[str] = None
    has_warranty: Optional[bool] = None
    similarity: Optional[float] = None


class ChatResponse(BaseModel):
    reply: str
    cars: list[CarView] = []
    actions: list[dict] = []


class ListingsResponse(BaseModel):
    count: int
    cars: list[CarView]


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="allow")
    user_id: str
    name: Optional[str] = None
    created_at: Optional[str] = None
    preferences: dict = {}
    liked_listings: list[int] = []


class BookingResponse(BaseModel):
    ok: bool
    booking_id: Optional[int] = None
    listing_id: Optional[int] = None
    slot_datetime: Optional[str] = None
    message: str


class LeadResponse(BaseModel):
    ok: bool
    message: str
    path: Optional[str] = None


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "embeddings_loaded": getattr(app.state, "embeddings_ok", False),
        "chat_model": config.CHAT_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
@app.post("/sessions", response_model=SessionResponse, status_code=201)
def create_session(body: SessionCreate, conn: Session = Depends(db_session)):
    """Create a chat session, optionally bound to a (new or returning) user."""
    is_returning = False
    briefing = ""
    if body.user_id:
        db.create_user(conn, body.user_id, body.name)
        briefing = memory.build_briefing(conn, body.user_id)
        is_returning = bool(briefing)
    session_id = memory.new_session(user_id=body.user_id)

    name = body.name or body.user_id
    if is_returning:
        greeting = (
            f"Welcome back{', ' + name if name else ''}! "
            "Want to pick up where you left off, or start a fresh search?"
        )
    elif name:
        greeting = f"Hi {name}, I'm the dubizzle cars assistant. What are you looking for today?"
    else:
        greeting = "Hi! I'm the dubizzle cars assistant. What kind of car can I help you find?"

    return {
        "session_id": session_id,
        "user_id": body.user_id,
        "is_returning": is_returning,
        "briefing": briefing,
        "greeting": greeting,
    }


@app.delete("/sessions/{session_id}", response_model=EndSessionResponse)
def end_session(session_id: str, conn: Session = Depends(db_session)):
    """End a session and refresh the user's long-term summary."""
    sess = memory.get_session(session_id)
    if not sess:
        raise HTTPException(404, "Unknown session")
    summary = ""
    if sess.get("user_id"):
        summary = memory.refresh_summary(conn, sess["user_id"])
    return {"ok": True, "summary": summary}


@app.post("/sessions/{session_id}/messages", response_model=ChatResponse)
def post_message(
    session_id: str,
    body: MessageCreate,
    conn: Session = Depends(db_session),
):
    """Send a message in a session; runs one grounded agent turn."""
    if memory.get_session(session_id) is None:
        raise HTTPException(404, "Unknown session_id — create one via POST /sessions.")
    return agent.run_turn(conn, session_id, body.message)


# --------------------------------------------------------------------------- #
# Listings
# --------------------------------------------------------------------------- #
@app.get("/listings", response_model=ListingsResponse)
def list_listings(
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
    q: Optional[str] = Query(None, description="Optional semantic query"),
    limit: int = 12,
    conn: Session = Depends(db_session),
):
    if q:
        if not getattr(app.state, "embeddings_ok", False):
            raise HTTPException(503, "Semantic search unavailable (embeddings not loaded).")
        has_filter = any(
            v is not None
            for v in (make, model, year_min, year_max, price_min, price_max,
                      mileage_max, body_type, fuel_type, spec)
        )
        cand = None
        if has_filter:
            pre = db.query_listings(
                conn, make=make, model=model, year_min=year_min, year_max=year_max,
                price_min=price_min, price_max=price_max, mileage_max=mileage_max,
                body_type=body_type, fuel_type=fuel_type, spec=spec,
            )
            cand = [x["listing_id"] for x in pre]
        ranked = retrieval.semantic_rank(q, candidate_ids=cand, limit=limit)
        cars = db.get_listings_by_ids(conn, [i for i, _ in ranked])
    else:
        cars = db.query_listings(
            conn, make=make, model=model, year_min=year_min, year_max=year_max,
            price_min=price_min, price_max=price_max, mileage_max=mileage_max,
            body_type=body_type, fuel_type=fuel_type, spec=spec, limit=limit,
        )
    return {"count": len(cars), "cars": [car_view(x) for x in cars]}


@app.get("/listings/{listing_id}", response_model=CarView)
def listing_detail(listing_id: int, conn: Session = Depends(db_session)):
    car = db.get_listing(conn, listing_id)
    if not car:
        raise HTTPException(404, f"No listing with id {listing_id}")
    return car_view(car)


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
@app.get("/users/{user_id}", response_model=UserProfile)
def get_user(user_id: str, conn: Session = Depends(db_session)):
    user = db.get_user(conn, user_id)
    if not user:
        raise HTTPException(404, f"No user {user_id}")
    return user


@app.put("/users/{user_id}", response_model=UserProfile)
def update_user(
    user_id: str,
    body: UserUpdate,
    conn: Session = Depends(db_session),
):
    """Create or update a user's profile and preferences."""
    db.create_user(conn, user_id, body.name)
    fields = {k: v for k, v in body.model_dump().items() if v is not None and k != "name"}
    if fields:
        db.update_preferences(conn, user_id, **fields)
    return db.get_user(conn, user_id)


# --------------------------------------------------------------------------- #
# Leads + bookings
# --------------------------------------------------------------------------- #
@app.post("/leads", response_model=LeadResponse, status_code=201)
def create_lead(body: LeadRequest, conn: Session = Depends(db_session)):
    ctx = ToolContext(conn=conn, session_id=body.session_id, user_id=body.user_id)
    res = save_lead(
        ctx,
        name=body.name, contact=body.contact,
        budget_min=body.budget_min, budget_max=body.budget_max,
        desired_make=body.desired_make, desired_model=body.desired_model,
        desired_body_type=body.desired_body_type, financing_pref=body.financing_pref,
        timeline=body.timeline, notes=body.notes,
    )
    if not res.get("ok"):
        raise HTTPException(422, res.get("message", "Lead not qualified"))
    return res


@app.post("/bookings", response_model=BookingResponse, status_code=201)
def create_booking(body: BookingRequest, conn: Session = Depends(db_session)):
    ctx = ToolContext(conn=conn, session_id=body.session_id, user_id=body.user_id)
    res = book_viewing(ctx, body.listing_id, body.slot_datetime, body.user_id)
    if not res.get("ok"):
        code = 409 if "already booked" in res.get("message", "") else 400
        raise HTTPException(code, res["message"])
    return res


@app.delete("/bookings/{booking_id}", response_model=BookingResponse)
def cancel_booking_endpoint(
    booking_id: int,
    user_id: Optional[str] = Query(None, description="If set, only cancel a booking owned by this user"),
    conn: Session = Depends(db_session),
):
    """Cancel a booking by id. When user_id is given, ownership is enforced."""
    removed = db.delete_booking(conn, booking_id, user_id=user_id)
    if not removed:
        detail = f"No booking #{booking_id} to cancel"
        raise HTTPException(404, detail + (f" for user {user_id}" if user_id else ""))
    return {"ok": True, "booking_id": booking_id, "message": f"Booking #{booking_id} cancelled."}
