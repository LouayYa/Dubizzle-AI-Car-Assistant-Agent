"""Long-term profile persists across sessions; briefing is built from it."""
from __future__ import annotations

from app import db, memory


def test_profile_persists_across_sessions(conn):
    # "Session 1": learn durable preferences.
    db.create_user(conn, "sara", "Sara")
    db.update_preferences(
        conn, "sara",
        budget_min=40000, budget_max=80000,
        preferred_body_types=["SUV"], preferred_makes=["toyota", "honda"],
    )
    db.add_liked_listing(conn, "sara", 3)

    # "Session 2": a fresh read sees the same persisted data.
    user = db.get_user(conn, "sara")
    prefs = user["preferences"]
    assert prefs["budget_min"] == 40000
    assert prefs["budget_max"] == 80000
    assert prefs["preferred_body_types"] == ["SUV"]
    assert "toyota" in prefs["preferred_makes"]
    assert user["liked_listings"] == [3]


def test_briefing_mentions_saved_preferences(conn):
    db.create_user(conn, "sara", "Sara")
    db.update_preferences(
        conn, "sara", budget_min=40000, budget_max=60000,
        preferred_body_types=["SUV"],
    )
    briefing = memory.build_briefing(conn, "sara")
    assert "Sara" in briefing
    assert "40,000" in briefing and "60,000" in briefing
    assert "SUV" in briefing


def test_new_user_has_empty_briefing_but_is_created(conn):
    db.create_user(conn, "newbie", "Newbie")
    briefing = memory.build_briefing(conn, "newbie")
    # No saved preferences yet -> briefing is empty, so the agent (and the
    # returning-user UI) never falsely treats a brand-new user as returning.
    assert briefing == ""
    assert db.get_user(conn, "newbie") is not None


def test_refresh_summary_persists(conn):
    db.create_user(conn, "sara", "Sara")
    db.update_preferences(conn, "sara", budget_max=80000, preferred_body_types=["SUV"])
    summary = memory.refresh_summary(conn, "sara")
    assert "Sara" in summary
    # Stored on the preferences row for next session.
    assert db.get_user(conn, "sara")["preferences"]["summary"] == summary
