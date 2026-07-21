"""Guardrails: decline non-automotive + competitor mentions; allow car chat.

The competitor/off-domain fast-path is deterministic (no LLM), so the agent
short-circuits before ever calling the model.
"""
from __future__ import annotations

import pytest

from app import agent, memory


def test_competitor_mentions_declined():
    for text in ["What about Cars24?", "Is OpenSooq cheaper?", "compare with CarSwitch",
                 "have you heard of hatla2ee", "YallaMotor has this car"]:
        decline = agent.guardrail_check(text)
        assert decline is not None
        assert "other marketplaces" in decline


def test_non_automotive_declined():
    for text in ["Write me a Python script to sort a list",
                 "write a poem about the sea",
                 "what is the capital of France",
                 "can you help with my homework"]:
        decline = agent.guardrail_check(text)
        assert decline is not None


def test_car_chat_not_falsely_blocked():
    for text in ["Show me a white Toyota SUV under 80k",
                 "Does it have a warranty?",
                 "Book a viewing for the first one on Saturday",
                 "I want a sporty coupe"]:
        assert agent.guardrail_check(text) is None


def test_run_turn_short_circuits_without_llm(conn, monkeypatch):
    # If the guardrail fires, the model must never be called.
    def boom(_messages):
        raise AssertionError("LLM should not be called for a blocked message")

    monkeypatch.setattr(agent, "_complete", boom)
    sid = memory.new_session(user_id=None)
    out = agent.run_turn(conn, sid, "Should I use Cars24 instead?")
    assert out["cars"] == []
    assert out["actions"] == [{"type": "guardrail_decline"}]
    assert "other marketplaces" in out["reply"]


def test_check_output_scrubs_competitor_from_reply():
    # The model volunteers a competitor name the user never typed.
    reply = "This is a great deal — much cheaper than you'd find on CarSwitch!"
    clean, blocked = agent.check_output(reply)
    assert blocked is True
    assert "carswitch" not in clean.lower()
    assert "other marketplaces" in clean


def test_check_output_passes_clean_reply():
    reply = "The 2019 Jaguar F-Pace is AED 70,000 with a warranty until 2027."
    clean, blocked = agent.check_output(reply)
    assert blocked is False
    assert clean == reply


def test_output_guardrail_fires_when_model_names_competitor(conn, monkeypatch):
    # Input passes the pre-check (user never names a competitor), but the model
    # does — the output-side guardrail must catch it and drop the cars.
    from types import SimpleNamespace

    def fake_complete(_messages):
        msg = SimpleNamespace(
            content="Sure! You could also try SellAnyCar for a quick sale.",
            tool_calls=None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr(agent, "_complete", fake_complete)
    sid = memory.new_session(user_id=None)
    out = agent.run_turn(conn, sid, "where can I sell my car quickly?")
    assert "sellanycar" not in out["reply"].lower()
    assert "other marketplaces" in out["reply"]
    assert out["cars"] == []
    assert {"type": "guardrail_output_blocked"} in out["actions"]
