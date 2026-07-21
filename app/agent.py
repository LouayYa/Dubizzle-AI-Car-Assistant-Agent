"""The agent: system prompt (persona + guardrails + injected memory), a thin
tool-calling loop over LiteLLM/Gemini, and a lightweight input pre-check.

Grounding is structural: the only car data returned to the client comes from
tool results (see tools.py), so the model cannot surface a car that isn't real.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Optional

from . import config, db, memory
from .tools import (
    INVENTORY_TOOLS,
    TOOL_FUNCTIONS,
    TOOL_SCHEMAS,
    ToolContext,
)

MAX_STEPS = 6            # max tool-call round-trips per turn
MAX_HISTORY = 20         # trailing messages sent to the LLM
MAX_CARS_PER_TURN = 24   # cap on cars surfaced to the client in one turn
_LLM_RETRIES = 4         # retry transient 429s (free-tier RPM is bursty)

# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
# Never name, compare to, or recommend these competitor platforms.
COMPETITORS = [
    "carabia", "opensooq", "yallamotor", "dubaicars", "dubicars",
    "carswitch", "cars24", "autozel", "sellanycar", "hatla2ee",
]

# Back up the system prompt's guardrail with a regex-based fast-path for
# obviously non-automotive requests.

_NONAUTO_PATTERNS = [
    r"\b(write|generate|debug|fix|refactor|create)\b[^.?!]*\b(code|script|program|function|app|website|webpage|regex|sql|python|javascript|typescript|java|html|css)\b",
    r"\bwrite\b[^.?!]*\b(poem|essay|story|song|joke|haiku|lyrics|email|cover letter)\b",
    r"\b(capital of|who (is|was|are) the|when did|history of|translate this|meaning of life)\b",
    r"\b(recipe|weather forecast|stock price|my homework|solve this equation)\b",
]

_COMPETITOR_RE = re.compile(
    r"(?<![a-z0-9])(" + "|".join(re.escape(c) for c in COMPETITORS) + r")(?![a-z0-9])",
    re.IGNORECASE,
)
_NONAUTO_RES = [re.compile(p, re.IGNORECASE) for p in _NONAUTO_PATTERNS]

_DECLINE_COMPETITOR = (
    "I can only help with cars here on dubizzle, so I can't get into other "
    "marketplaces; but I'm happy to! What kind of car are you after?"
)
_DECLINE_NONAUTO = (
    "That's a bit outside my lane; I'm the dubizzle cars assistant, so I stick "
    "to helping you find, compare, and book cars. Want me to line up some "
    "listings for you?"
)


def guardrail_check(message: str) -> Optional[str]:
    """Input-side guardrail: block the user's message before calling the model.

    Returns a warm decline string if the message must be blocked, else None.
    """
    text = message or ""
    if _COMPETITOR_RE.search(text):
        return _DECLINE_COMPETITOR
    for rx in _NONAUTO_RES:
        if rx.search(text):
            return _DECLINE_NONAUTO
    return None


def check_output(reply: str) -> tuple[str, bool]:
    """Output-side guardrail: scan the MODEL's reply before it reaches the user.

    Defense in depth — the system prompt tells the model never to name a
    competitor, but that's a nudge, not a guarantee. If a competitor slips into
    the reply anyway (the model volunteers it, or echoes it from listing text),
    replace the whole reply with the safe redirect. Returns (reply, was_blocked).
    """
    if reply and _COMPETITOR_RE.search(reply):
        return _DECLINE_COMPETITOR, True
    return reply, False


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #
_BASE_PERSONA = """You are the dubizzle Cars assistant; a warm, concise, and helpful guide for the dubizzle used-cars marketplace in the UAE. You help people browse real inventory, compare cars, answer questions about specific listings, capture their needs, and book viewings.

GROUNDING (non-negotiable):
- You may ONLY talk about specific cars that came back from a tool call. Never invent or guess a listing, price, mileage, color, spec, or any detail.
- If a field is unknown/null in the data, say so plainly (e.g. "the cash price isn't listed — it's advertised on finance only"). Do NOT derive a cash price from a monthly payment.
- Prices are in AED. Some listings have no cash price (finance only or not listed) and some have no mileage, mention that honestly.

TOOLS:
- search_inventory: hard filters (make, model, year/price/mileage ranges, body type, fuel, spec). Use for concrete constraints.
- semantic_search: vibe/intent queries ("sporty", "family SUV", "something luxurious"). You can combine it with hard filters.
- get_listing_details: full record for one car. Use for follow-up questions about a specific listing.
- book_viewing: viewings are Monday to Saturday, 08:00 to 20:00, future times only; the same car can't be double-booked for the same slot.
- get_user_bookings: the user's existing viewing bookings (with booking_id). ALWAYS call this before answering any question about whether they have a booking or what they booked — never guess "you have no bookings" from memory.
- cancel_booking: cancel one of the user's bookings by booking_id. Call get_user_bookings first to get the id, and confirm which one they mean before cancelling. Only offer to cancel if you can actually do it with this tool.
- save_lead: once you know the budget AND at least one concrete need, save the lead.
- get_user_profile / update_user_profile: whenever you learn a DURABLE preference (budget, favored make/model/body type, fuel or financing preference, or that they liked a specific car), call update_user_profile immediately so it's remembered next time.

STYLE:
- Keep replies short and skimmable. When you show cars, briefly say why they fit; the UI renders the full cards, so don't dump every spec as text.
- Ask one clarifying question at a time when you need it. Be proactive about booking a viewing or saving their needs once they show intent.

GUARDRAILS:
- You ONLY handle cars and this marketplace. Politely decline anything non-automotive (coding, trivia, homework, general chit-chat beyond a friendly hello) and steer back to cars.
- NEVER mention, compare to, endorse, or recommend any competitor platform (including Carabia, OpenSooq, YallaMotor, Dubaicars, CarSwitch, Cars24, Autozel, SellAnyCar, Hatla2ee, etc.). If asked, gently redirect to how you can help here."""


def build_system_prompt(
    conn,
    *,
    user_id: Optional[str],
    context: dict,
) -> str:
    parts = [_BASE_PERSONA]

    now = datetime.now()
    parts.append(
        f"\nCONTEXT:\n- Today is {now:%A, %Y-%m-%d %H:%M}. When booking, only offer "
        f"future Mon to Sat 08:00–20:00 slots."
    )

    if user_id:
        briefing = memory.build_briefing(conn, user_id)
        if briefing:
            parts.append(f"- Returning-user briefing: {briefing}")
        else:
            parts.append(f"- The user is signed in as '{user_id}' (new — no saved preferences yet).")

    shown = context.get("last_shown_ids") or []
    if shown:
        labels = []
        for lid in shown:
            car = db.get_listing(conn, lid)
            if car:
                labels.append(
                    f"[{lid}] {car.get('year')} {car.get('make')} {car.get('model')} {car.get('trim') or ''}".strip()
                )
        if labels:
            parts.append(
                "- Cars most recently shown (in order — 'the first one' = position 1, "
                "'it'/'that one' defaults to the focused car):\n  "
                + "\n  ".join(labels)
            )
    focused = context.get("focused_listing_id")
    if focused:
        parts.append(f"- Focused car (resolve 'it'/'this car' to this): listing #{focused}")

    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# LLM plumbing
# --------------------------------------------------------------------------- #
def _assistant_msg_to_dict(msg: Any) -> dict:
    tool_calls = []
    for tc in (msg.tool_calls or []):
        tool_calls.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
        )
    out: dict = {"role": "assistant", "content": msg.content or ""}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def _complete(messages: list[dict]):
    """Call the chat model with tools, retrying transient rate-limit errors."""
    import litellm

    config.require_api_key()
    last_err = None
    for attempt in range(_LLM_RETRIES):
        try:
            return litellm.completion(
                model=config.CHAT_MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                # Low temperature: keeps answers factual and on-topic, not made up.
                temperature=0.1,
            )
        except Exception as exc:  # normalize provider errors (rate limit, network, parse)
            last_err = exc
            name = type(exc).__name__.lower()
            if "ratelimit" in name or "429" in str(exc) or "resource_exhausted" in str(exc).lower():
                time.sleep(min(4 * (attempt + 1), 15))
                continue
            raise
    raise last_err  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Turn execution
# --------------------------------------------------------------------------- #
def run_turn(conn, session_id: str, message: str) -> dict:
    """Run one conversational turn. Returns {reply, cars, actions}."""
    sess = memory.require_session(session_id)
    user_id = sess.get("user_id")
    ctx = ToolContext(conn=conn, session_id=session_id, user_id=user_id)

    # Log the raw inquiry for cross-session recall (best-effort).
    try:
        db.add_inquiry(conn, user_id, session_id, message)
    except Exception:
        pass

    # 1) Deterministic guardrail fast-path.
    decline = guardrail_check(message)
    if decline is not None:
        memory.append_history(session_id, {"role": "user", "content": message})
        memory.append_history(session_id, {"role": "assistant", "content": decline})
        return {"reply": decline, "cars": [], "actions": [{"type": "guardrail_decline"}]}

    # 2) Build messages: system + trailing history + this user turn.
    memory.append_history(session_id, {"role": "user", "content": message})
    system = build_system_prompt(conn, user_id=user_id, context=memory.get_context(session_id))
    history = memory.get_history(session_id)[-MAX_HISTORY:]
    messages: list[dict] = [{"role": "system", "content": system}] + history

    # Cars surfaced to the client. Accumulated across every car-returning tool
    # call in this turn (deduped, first-seen order) so the rendered cards always
    # cover everything the reply talks about — one turn may run more than one
    # search, and overwriting would leave the prose referencing cars the UI
    # never showed.
    cars: list[dict] = []
    seen_ids: set[int] = set()

    def add_cars(new_cars: list[dict]) -> None:
        for car in new_cars:
            lid = car.get("listing_id")
            if lid is not None and lid not in seen_ids:
                seen_ids.add(lid)
                cars.append(car)
        del cars[MAX_CARS_PER_TURN:]

    actions: list[dict] = []
    final_reply = ""

    # 3) Tool-calling loop.
    for _step in range(MAX_STEPS):
        try:
            resp = _complete(messages)
        except Exception as exc:
            final_reply = (
                "Sorry, I'm having trouble reaching the assistant service right now "
                "(the free-tier quota may be exhausted). Please try again in a moment."
            )
            actions.append({"type": "error", "detail": str(exc)[:200]})
            break

        msg = resp.choices[0].message
        if not msg.tool_calls:
            final_reply = msg.content or ""
            break

        messages.append(_assistant_msg_to_dict(msg))
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = TOOL_FUNCTIONS.get(name)
            if fn is None:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = fn(ctx, **args)
                except TypeError as exc:
                    result = {"error": f"bad arguments for {name}: {exc}"}
                except Exception as exc:                   
                    result = {"error": f"{name} failed: {exc}"}

            # Surface grounded cars + update short-term memory.
            if name in INVENTORY_TOOLS and isinstance(result, dict) and result.get("cars"):
                add_cars(result["cars"])
                memory.set_last_shown(session_id, [c["listing_id"] for c in cars])
            elif name == "get_listing_details" and isinstance(result, dict) and result.get("car"):
                add_cars([result["car"]])
                memory.set_focus(session_id, result["car"]["listing_id"])
            elif name == "book_viewing" and isinstance(result, dict) and result.get("ok"):
                actions.append({"type": "booking", **{k: result[k] for k in ("booking_id", "listing_id", "slot_datetime")}})
            elif name == "cancel_booking" and isinstance(result, dict) and result.get("ok"):
                actions.append({"type": "booking_cancelled", "booking_id": result.get("booking_id")})
            elif name == "save_lead" and isinstance(result, dict) and result.get("ok"):
                actions.append({"type": "lead_saved"})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result, default=str),
                }
            )
    else:
        # Loop exhausted without a plain-text answer.
        if not final_reply:
            final_reply = "Here's what I found." if cars else "Could you rephrase that?"

    # Output-side guardrail: last line of defense before the reply leaves.
    final_reply, blocked = check_output(final_reply)
    if blocked:
        cars = []  # the reply was replaced with a redirect; cards would be orphaned
        actions.append({"type": "guardrail_output_blocked"})

    memory.append_history(session_id, {"role": "assistant", "content": final_reply})
    return {"reply": final_reply, "cars": cars, "actions": actions}
