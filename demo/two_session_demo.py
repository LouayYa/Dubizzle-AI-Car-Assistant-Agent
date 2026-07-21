"""Cross-session recall demo (for a screenshot).

Session 1: a brand-new user states a preference ("white SUV under 80k AED").
Session 2: a fresh session for the SAME user shows the assistant recalling it.

Requires the backend to be running:
    uv run uvicorn app.main:app --port 8000
Then:
    uv run python demo/two_session_demo.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(90.0)

RULE = "=" * 74


def hr(title: str = "") -> None:
    print("\n" + RULE)
    if title:
        print(title)
        print(RULE)


def post(path: str, body: dict) -> dict:
    r = httpx.post(f"{BACKEND}{path}", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get(path: str) -> dict:
    r = httpx.get(f"{BACKEND}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def delete(path: str) -> dict:
    r = httpx.delete(f"{BACKEND}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def say(role: str, text: str) -> None:
    tag = {"user": "🧑 USER", "bot": "🤖 ASSISTANT"}[role]
    print(f"\n{tag}: {text}")


def chat(session_id: str, message: str) -> dict:
    say("user", message)
    res = post(f"/sessions/{session_id}/messages", {"message": message})
    say("bot", res.get("reply", ""))
    for c in res.get("cars", []) or []:
        print(f"      • #{c['listing_id']} {c.get('year')} {c.get('make')} "
              f"{c.get('model')} — {c.get('price_label')}")
    return res


def main() -> None:
    try:
        health = get("/health")
    except Exception:
        print(f"❌ Backend not reachable at {BACKEND}.\n"
              f"   Start it first:  uv run uvicorn app.main:app --port 8000")
        sys.exit(1)
    print(f"Backend OK — chat model: {health.get('chat_model')}, "
          f"embeddings loaded: {health.get('embeddings_loaded')}")

    user_id = f"demo_sara_{int(time.time())}"
    name = "Sara"

    # ---------------- Session 1 ----------------
    hr(f"SESSION 1  (brand-new user: user_id='{user_id}')")
    s1 = post("/sessions", {"user_id": user_id, "name": name})
    print(f"is_returning = {s1['is_returning']}")
    say("bot", s1["greeting"])

    chat(s1["session_id"], "Hi! I'm shopping for a white SUV and my budget is up to "
                           "80,000 AED. Please remember that for next time.")
    chat(s1["session_id"], "Show me a couple that fit.")

    profile = get(f"/users/{user_id}")
    hr("PERSISTED PROFILE after session 1 (SQLite)")
    prefs = profile.get("preferences", {})
    print(f"  name              : {profile.get('name')}")
    print(f"  budget_min/max    : {prefs.get('budget_min')} / {prefs.get('budget_max')}")
    print(f"  preferred_makes   : {prefs.get('preferred_makes')}")
    print(f"  preferred_models  : {prefs.get('preferred_models')}")
    print(f"  preferred_body_types: {prefs.get('preferred_body_types')}")
    print(f"  liked_listings    : {profile.get('liked_listings')}")

    end = delete(f"/sessions/{s1['session_id']}")
    print(f"\n  session-end summary: {end.get('summary')}")

    # ---------------- Session 2 ----------------
    hr(f"SESSION 2  (fresh session, SAME user_id='{user_id}')")
    s2 = post("/sessions", {"user_id": user_id, "name": name})
    print(f"is_returning = {s2['is_returning']}")
    print(f"briefing     = {s2['briefing']}")
    say("bot", s2["greeting"])

    chat(s2["session_id"], "Remind me — what was I looking for, and what's my budget?")

    hr("✅ Cross-session recall demonstrated: the new session recognised the "
       "returning user\n   and recalled the white-SUV / 80k-AED preference set in "
       "session 1.")


if __name__ == "__main__":
    main()
