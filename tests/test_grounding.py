"""Grounding: the agent only ever returns cars that came from a tool result.

The LLM is faked (no network). The fake asks the agent to run the *real*
search_inventory tool against the seeded DB, then even tries to name a
non-existent car in prose. We assert the structured `cars` payload contains only
real, tool-sourced listings.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from app import agent, db, memory


def _fn_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _resp(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_agent_returns_only_tool_sourced_cars(conn, monkeypatch):
    scripted = [
        # 1) model asks to search real inventory
        _resp(tool_calls=[_fn_call("c1", "search_inventory", {"make": "honda"})]),
        # 2) model answers, and (mischievously) name-drops a fake car in prose
        _resp(content="Here are two Hondas. You might also like the mythical Honda Zeta 9000!"),
    ]

    def fake_complete(_messages):
        return scripted.pop(0)

    monkeypatch.setattr(agent, "_complete", fake_complete)

    sid = memory.new_session(user_id=None)
    out = agent.run_turn(conn, sid, "show me hondas")

    ids = {c["listing_id"] for c in out["cars"]}
    # Exactly the real Hondas from the tool; the fabricated "Zeta 9000" is absent.
    assert ids == {1, 2}
    for c in out["cars"]:
        assert db.get_listing(conn, c["listing_id"]) is not None
    # Short-term memory recorded what was shown (for later reference resolution).
    assert memory.get_context(sid)["last_shown_ids"] == [1, 2]


def test_multiple_searches_in_one_turn_accumulate(conn, monkeypatch):
    """Two searches in a single turn must BOTH surface.

    Regression test: the payload used to be overwritten by the last tool call,
    so the reply could describe cars the UI never rendered.
    """
    scripted = [
        _resp(tool_calls=[_fn_call("c1", "search_inventory", {"make": "honda"})]),
        _resp(tool_calls=[_fn_call("c2", "search_inventory", {"make": "toyota"})]),
        _resp(content="Here are the Hondas and the Toyota."),
    ]
    monkeypatch.setattr(agent, "_complete", lambda _m: scripted.pop(0))

    sid = memory.new_session(user_id=None)
    out = agent.run_turn(conn, sid, "show me hondas and toyotas")

    ids = [c["listing_id"] for c in out["cars"]]
    assert ids == [1, 2, 3]  # both searches, first-seen order preserved
    assert memory.get_context(sid)["last_shown_ids"] == [1, 2, 3]


def test_repeated_search_does_not_duplicate_cars(conn, monkeypatch):
    scripted = [
        _resp(tool_calls=[_fn_call("c1", "search_inventory", {"make": "honda"})]),
        _resp(tool_calls=[_fn_call("c2", "search_inventory", {"make": "honda"})]),
        _resp(content="Same two Hondas."),
    ]
    monkeypatch.setattr(agent, "_complete", lambda _m: scripted.pop(0))

    sid = memory.new_session(user_id=None)
    out = agent.run_turn(conn, sid, "hondas again")
    assert [c["listing_id"] for c in out["cars"]] == [1, 2]  # deduped


def test_reference_resolution_via_get_details(conn, monkeypatch):
    # Turn 1: show Hondas. Turn 2: "warranty on the first one?" -> details of #1.
    scripted = [
        _resp(tool_calls=[_fn_call("c1", "search_inventory", {"make": "honda"})]),
        _resp(content="Found them."),
        _resp(tool_calls=[_fn_call("c2", "get_listing_details", {"listing_id": 1})]),
        _resp(content="Yes, the CR-V has warranty."),
    ]
    monkeypatch.setattr(agent, "_complete", lambda _m: scripted.pop(0))

    sid = memory.new_session(user_id=None)
    agent.run_turn(conn, sid, "show me hondas")

    out2 = agent.run_turn(conn, sid, "is there a warranty on the first one?")
    assert out2["cars"][0]["listing_id"] == 1
    assert out2["cars"][0]["has_warranty"] is True
    assert memory.get_context(sid)["focused_listing_id"] == 1
