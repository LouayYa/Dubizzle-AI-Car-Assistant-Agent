"""Streamlit chat client.

A thin HTTP client for the FastAPI backend — it makes NO LLM calls of its own.
Prompts for a name/user_id to trigger long-term recall, streams the chat, and
renders returned cars as cards.

Run:  uv run streamlit run client/app.py
"""
from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(90.0)

st.set_page_config(page_title="dubizzle Cars Assistant", page_icon="🚗", layout="centered")


# --------------------------------------------------------------------------- #
# Backend helpers
# --------------------------------------------------------------------------- #
def api_post(path: str, json: dict) -> dict:
    r = httpx.post(f"{BACKEND}{path}", json=json, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_delete(path: str) -> dict:
    r = httpx.delete(f"{BACKEND}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def backend_online() -> bool:
    try:
        httpx.get(f"{BACKEND}/health", timeout=5.0).raise_for_status()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Card rendering
# --------------------------------------------------------------------------- #
def render_car(car: dict) -> None:
    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            if car.get("photo_renderable") and car.get("photo_url"):
                st.image(car["photo_url"], use_container_width=True)
            elif car.get("photo_url"):
                st.markdown(f"🖼️ [View photo]({car['photo_url']})")
                st.caption("(.heic — not shown inline)")
            else:
                st.caption("No photo")
        with cols[1]:
            title = f"{car.get('year','')} {(car.get('make') or '').title()} {(car.get('model') or '').title()}"
            trim = car.get("trim")
            if trim and str(trim).lower() not in ("other", "nan", "none"):
                title += f" · {trim}"
            st.markdown(f"**{title}**  \n`#{car.get('listing_id')}`")

            bits = [f"💰 {car.get('price_label', 'Price not listed')}"]
            if car.get("mileage_km") is not None:
                bits.append(f"🛣️ {int(car['mileage_km']):,} km")
            elif car.get("is_new"):
                bits.append("🛣️ Brand new (0 km)")
            else:
                bits.append("🛣️ Mileage not listed")
            if car.get("body_type"):
                bits.append(f"🚙 {car['body_type']}")
            if car.get("fuel_type"):
                bits.append(f"⛽ {car['fuel_type']}")
            if car.get("regional_spec"):
                bits.append(f"🌍 {car['regional_spec']}")
            if car.get("has_warranty"):
                bits.append("🛡️ Warranty")
            st.markdown("  ·  ".join(bits))

            desc = car.get("description_clean")
            if desc:
                st.caption(desc[:200])


# --------------------------------------------------------------------------- #
# Sidebar — identity + session
# --------------------------------------------------------------------------- #
st.sidebar.title("🚗 dubizzle Cars")
st.sidebar.caption(f"Backend: {BACKEND}")

if not backend_online():
    st.sidebar.error("Backend not reachable. Start it with:\n\n`uv run uvicorn app.main:app --port 8000`")

with st.sidebar.form("identity"):
    st.markdown("**Sign in** (enables cross-session recall)")
    user_id = st.text_input("User ID", value=st.session_state.get("user_id", ""),
                            placeholder="e.g. sara")
    name = st.text_input("Name", value=st.session_state.get("name", ""),
                         placeholder="e.g. Sara")
    start = st.form_submit_button("Start / restart session")

if start:
    payload = {}
    if user_id.strip():
        payload["user_id"] = user_id.strip()
    if name.strip():
        payload["name"] = name.strip()
    try:
        res = api_post("/sessions", payload)
        st.session_state.session_id = res["session_id"]
        st.session_state.user_id = user_id.strip()
        st.session_state.name = name.strip()
        st.session_state.messages = [{"role": "assistant", "content": res["greeting"], "cars": []}]
        if res.get("briefing"):
            st.sidebar.success(f"Recalled: {res['briefing']}")
    except Exception as exc:
        st.sidebar.error(f"Could not start session: {exc}")

if st.session_state.get("user_id"):
    if st.sidebar.button("End session (save summary)"):
        try:
            out = api_delete(f"/sessions/{st.session_state.session_id}")
            st.sidebar.info(f"Saved summary: {out.get('summary','')}")
        except Exception as exc:
            st.sidebar.warning(f"Could not save summary: {exc}")

st.sidebar.divider()
st.sidebar.caption(
    "Ask for cars by make, budget, body type, or vibe (\"sporty\", \"family SUV\"). "
    "I can pull up details, book a viewing (Mon–Sat 08:00–20:00), and remember "
    "your preferences."
)


# --------------------------------------------------------------------------- #
# Main chat
# --------------------------------------------------------------------------- #
st.title("dubizzle Cars Assistant")

if "session_id" not in st.session_state:
    st.info("👈 Enter a User ID (and name) in the sidebar and click **Start session** to begin.")
    st.stop()

for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for car in msg.get("cars", []) or []:
            render_car(car)

prompt = st.chat_input("Ask about cars…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "cars": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                res = api_post(
                    f"/sessions/{st.session_state.session_id}/messages",
                    {"message": prompt},
                )
                reply = res.get("reply", "")
                cars = res.get("cars", []) or []
                actions = res.get("actions", []) or []
            except Exception as exc:
                reply, cars, actions = f"⚠️ Error talking to backend: {exc}", [], []
        st.markdown(reply)
        for car in cars:
            render_car(car)
        for a in actions:
            if a.get("type") == "booking":
                st.success(f"✅ Booking #{a.get('booking_id')} confirmed for {a.get('slot_datetime')}.")
            elif a.get("type") == "lead_saved":
                st.info("📝 Your details were saved for our sales team.")

    st.session_state.messages.append({"role": "assistant", "content": reply, "cars": cars})
