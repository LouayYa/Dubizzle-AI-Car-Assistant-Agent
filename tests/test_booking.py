"""Booking rules: Mon–Sat 08:00–20:00, no past, unique (listing, slot)."""
from __future__ import annotations

from datetime import datetime

from app import memory
from app.tools import (
    ToolContext,
    book_viewing,
    cancel_booking,
    get_user_bookings,
    validate_slot,
)
from tests.conftest import next_slot


def test_rejects_sunday(conn):
    ok, msg = validate_slot(next_slot(hour=14, want_sunday=True))
    assert not ok
    assert "Sunday" in msg


def test_rejects_before_open(conn):
    ok, msg = validate_slot(next_slot(hour=7))
    assert not ok
    assert "08:00" in msg


def test_rejects_after_close(conn):
    ok, msg = validate_slot(next_slot(hour=21))
    assert not ok


def test_rejects_past(conn):
    # A Monday well in the past.
    ok, msg = validate_slot("2020-01-06 14:00")
    assert not ok
    assert "past" in msg.lower()


def test_accepts_valid_future_weekday(conn):
    ok, _ = validate_slot(next_slot(hour=10))
    assert ok


def test_book_and_reject_duplicate_slot(conn):
    ctx = ToolContext(conn, session_id="s1", user_id="u1")
    slot = next_slot(hour=15)

    first = book_viewing(ctx, listing_id=1, slot_datetime=slot)
    assert first["ok"] is True
    assert first["booking_id"] >= 1

    # Same car, same slot -> rejected.
    dup = book_viewing(ctx, listing_id=1, slot_datetime=slot)
    assert dup["ok"] is False
    assert "already booked" in dup["message"]


def test_same_car_different_slot_ok(conn):
    ctx = ToolContext(conn, session_id="s1", user_id="u1")
    a = book_viewing(ctx, listing_id=2, slot_datetime=next_slot(hour=9))
    b = book_viewing(ctx, listing_id=2, slot_datetime=next_slot(hour=11))
    assert a["ok"] and b["ok"]


def test_book_unknown_listing(conn):
    ctx = ToolContext(conn)
    res = book_viewing(ctx, listing_id=9999, slot_datetime=next_slot())
    assert res["ok"] is False


def test_get_user_bookings_reflects_a_booking(conn):
    ctx = ToolContext(conn, session_id="s1", user_id="louy")

    # No bookings yet.
    empty = get_user_bookings(ctx)
    assert empty["count"] == 0 and empty["bookings"] == []

    # Book, then it must be readable back (grounded — not guessed).
    book_viewing(ctx, listing_id=1, slot_datetime=next_slot(hour=16))
    res = get_user_bookings(ctx)
    assert res["count"] == 1
    b = res["bookings"][0]
    assert b["listing_id"] == 1
    assert b["car"]["make"] == "honda"


def test_cancel_booking_removes_it(conn):
    ctx = ToolContext(conn, session_id="s1", user_id="louy")
    res = book_viewing(ctx, listing_id=1, slot_datetime=next_slot(hour=16))
    bid = res["booking_id"]

    cancelled = cancel_booking(ctx, booking_id=bid)
    assert cancelled["ok"] is True
    assert get_user_bookings(ctx)["count"] == 0

    # Cancelling again (or an unknown id) fails cleanly, not a crash.
    again = cancel_booking(ctx, booking_id=bid)
    assert again["ok"] is False


def test_cancel_booking_enforces_ownership(conn):
    owner = ToolContext(conn, user_id="owner")
    res = book_viewing(owner, listing_id=1, slot_datetime=next_slot(hour=16))
    bid = res["booking_id"]

    # A different user cannot cancel it.
    attacker = ToolContext(conn, user_id="someone_else")
    blocked = cancel_booking(attacker, booking_id=bid)
    assert blocked["ok"] is False
    assert get_user_bookings(owner)["count"] == 1  # still there


def test_bookings_recalled_in_new_session_briefing(conn):
    # "Session 1": book a viewing.
    ctx = ToolContext(conn, session_id="s1", user_id="louy")
    book_viewing(ctx, listing_id=1, slot_datetime=next_slot(hour=16))

    # "Session 2": a returning-user briefing must mention the booking.
    briefing = memory.build_briefing(conn, "louy")
    assert "viewing booked" in briefing
    assert "#1" in briefing


def test_validate_slot_with_injected_now():
    # Deterministic: 2026-07-25 is a Saturday.
    now = datetime(2026, 7, 20, 12, 0)
    ok, _ = validate_slot("2026-07-25 14:00", now=now)
    assert ok
    # 2026-07-26 is a Sunday.
    ok2, msg2 = validate_slot("2026-07-26 14:00", now=now)
    assert not ok2 and "Sunday" in msg2
