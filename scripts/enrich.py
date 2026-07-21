"""Phase 1 — offline enrichment (run ONCE; outputs are committed).

Reads the `cleaned dataset` sheet, cleans the free text, extracts structured
fields (LLM pass with a regex fallback for price/mileage), writes
data/inventory.json, and precomputes per-listing embeddings to
data/embeddings.npy (+ data/embeddings_ids.json).

Idempotent + resumable: per-listing LLM extractions and embeddings are cached to
hidden files in data/, so a re-run after a rate-limit interruption picks up where
it left off and costs nothing for already-processed rows. Use --force to redo.

Usage:
    uv run python scripts/enrich.py            # normal (resumes from cache)
    uv run python scripts/enrich.py --force    # ignore caches, redo everything
    uv run python scripts/enrich.py --no-llm   # regex-only extraction (no API)
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.media import normalize_photo_url

ENRICH_CACHE = config.DATA_DIR / ".enrich_cache.json"
EMBED_CACHE = config.DATA_DIR / ".embed_cache.json"

FIELDS = [
    "price_aed", "monthly_payment_aed", "mileage_km", "is_new", "exterior_color",
    "body_type", "transmission", "fuel_type", "regional_spec", "has_warranty",
    "description_clean",
]


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_RE = re.compile(r"#\S+")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
_WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Strip HTML/entities, drop URLs/phones/hashtags/mojibake, collapse space."""
    if raw is None:
        return ""
    s = str(raw)
    s = s.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ")
    s = _TAG_RE.sub(" ", s)
    s = html.unescape(s)
    s = s.replace("�", " ")          # replacement char (lost bullets/emoji)
    s = _URL_RE.sub(" ", s)
    s = _HASHTAG_RE.sub(" ", s)
    s = _PHONE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# --------------------------------------------------------------------------- #
# Regex fallback extractors (price + mileage)
# --------------------------------------------------------------------------- #
_NUM = r"([0-9][0-9,\.]{2,})"


def _to_int(num_str: str) -> int | None:
    try:
        return int(round(float(num_str.replace(",", ""))))
    except (ValueError, AttributeError):
        return None


def regex_price(text: str) -> tuple[int | None, int | None]:
    """Return (cash_price_aed, monthly_payment_aed) using surrounding context."""
    low = text.lower()
    cash = None
    monthly = None

    # Monthly finance: "AED 2,111.00 monthly", "2,111 / month", "AED 1,876 per month"
    for m in re.finditer(r"(?:aed\s*)?" + _NUM + r"\s*(?:aed\s*)?(?:/|per\s+)?\s*month", low):
        val = _to_int(m.group(1))
        if val and (monthly is None or val < monthly):
            monthly = val

    # Cash price: number near the word "cash".
    for m in re.finditer(_NUM + r"[^0-9]{0,15}cash", low):
        val = _to_int(m.group(1))
        if val and val >= 1000:
            cash = val if cash is None else max(cash, val)
    if cash is None:
        for m in re.finditer(r"cash[^0-9]{0,15}(?:aed\s*)?" + _NUM, low):
            val = _to_int(m.group(1))
            if val and val >= 1000:
                cash = val if cash is None else max(cash, val)

    # Bare "AED 119,750" not tied to 'month' → treat as cash if large.
    if cash is None:
        for m in re.finditer(r"aed\s*" + _NUM, low):
            span_end = m.end()
            trailing = low[span_end:span_end + 12]
            if "month" in trailing:
                continue
            val = _to_int(m.group(1))
            if val and val >= 10000:
                cash = val if cash is None else max(cash, val)
    return cash, monthly


def regex_mileage(text: str) -> tuple[int | None, bool | None]:
    """Return (mileage_km, is_new)."""
    low = text.lower()
    if re.search(r"\b(brand[\s-]?new)\b", low) or re.search(r"\b0\s*km\b", low):
        return 0, True
    # "Mileage: 68,000 km", "68000 kms", "68,000 kilometers"
    best = None
    for m in re.finditer(_NUM + r"\s*(?:km|kms|kilometer|kilometre)", low):
        val = _to_int(m.group(1))
        if val is not None and 0 <= val <= 1_000_000:
            best = val if best is None else best
            break
    if best is not None:
        return best, (best == 0)
    return None, None


# --------------------------------------------------------------------------- #
# LLM extraction
# --------------------------------------------------------------------------- #
_EXTRACT_SYS = """You extract structured car-listing fields from messy marketplace text (some Arabic, some HTML, dealer spam). Return STRICT JSON only — no prose, no code fences.

Rules:
- price_aed: integer cash price in AED. NULL if only financing/monthly is given or if no price appears. NEVER derive a cash price from a monthly payment.
- monthly_payment_aed: integer monthly finance amount if present, else null.
- mileage_km: integer km. "0km"/"brand new" -> 0. null if absent.
- is_new: true only for brand-new / 0km cars, else false.
- exterior_color: string or null.
- body_type: one of SUV, Sedan, Coupe, Hatchback, Pickup, Convertible, Wagon, Van, or null.
- transmission: "Automatic", "Manual", or null.
- fuel_type: "Petrol", "Diesel", "Electric", "Hybrid", or null.
- regional_spec: "GCC", "US", "Japanese", "European", "Canadian", "other", or null.
- has_warranty: true/false/null.
- description_clean: a concise (<=220 char) English-readable summary of the car. If the source is Arabic, summarize it in English. No phone numbers, URLs, or hashtags.

Output keys EXACTLY: price_aed, monthly_payment_aed, mileage_km, is_new, exterior_color, body_type, transmission, fuel_type, regional_spec, has_warranty, description_clean."""


def _parse_json(txt: str) -> dict:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*", "", txt).strip().rstrip("`").strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def llm_extract(make, model, trim, year, title, desc, retries: int = 5) -> dict | None:
    import litellm

    litellm.suppress_debug_info = True
    user = (
        f"make={make} | model={model} | trim={trim} | year={year}\n"
        f"title: {title}\n"
        f"description: {desc}"
    )
    delay = 3.0
    last = None
    for _ in range(retries):
        try:
            resp = litellm.completion(
                model=config.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": _EXTRACT_SYS},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return _parse_json(resp.choices[0].message.content or "{}")
        except Exception as exc:
            last = exc
            if "429" in str(exc) or "ratelimit" in type(exc).__name__.lower():
                print(f"    rate-limited, backing off {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 1.6, 40)
                continue
            print(f"    LLM extract error: {exc}")
            break
    if last:
        print(f"    giving up on LLM for this row: {last}")
    return None


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def embed_text(text: str, retries: int = 6) -> list[float]:
    import litellm

    litellm.suppress_debug_info = True
    delay = 3.0
    last = None
    for _ in range(retries):
        try:
            resp = litellm.embedding(model=config.EMBED_MODEL, input=[text])
            return [float(x) for x in resp["data"][0]["embedding"]]
        except Exception as exc:
            last = exc
            if "429" in str(exc) or "ratelimit" in type(exc).__name__.lower():
                print(f"    embed rate-limited, backing off {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 1.6, 40)
                continue
            raise
    raise last  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def merge_fields(llm: dict, rx_price, rx_monthly, rx_mileage, rx_isnew, clean_desc) -> dict:
    def g(key, default=None):
        v = llm.get(key, default)
        return v if v not in ("", "null", "None") else default

    price = g("price_aed")
    if price is None:
        price = rx_price
    monthly = g("monthly_payment_aed")
    if monthly is None:
        monthly = rx_monthly
    mileage = g("mileage_km")
    if mileage is None:
        mileage = rx_mileage
    is_new = g("is_new")
    if is_new is None:
        is_new = rx_isnew if rx_isnew is not None else (mileage == 0 if mileage is not None else False)

    desc_clean = g("description_clean") or clean_desc

    return {
        "price_aed": _to_int(str(price)) if price is not None else None,
        "monthly_payment_aed": _to_int(str(monthly)) if monthly is not None else None,
        "mileage_km": _to_int(str(mileage)) if mileage is not None else None,
        "is_new": bool(is_new) if is_new is not None else None,
        "exterior_color": g("exterior_color"),
        "body_type": g("body_type"),
        "transmission": g("transmission"),
        "fuel_type": g("fuel_type"),
        "regional_spec": g("regional_spec"),
        "has_warranty": g("has_warranty"),
        "description_clean": desc_clean,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore caches; redo all")
    ap.add_argument("--no-llm", action="store_true", help="regex-only extraction")
    args = ap.parse_args()

    if not args.no_llm:
        config.require_api_key()

    print(f"Loading {config.XLSX_PATH} (sheet: {config.SHEET_NAME})")
    df = pd.read_excel(config.XLSX_PATH, sheet_name=config.SHEET_NAME)
    print(f"  {len(df)} rows")

    # Normalize photo URLs up front (excel -> df): dubizzle CDN .heic links are
    # transcoded to WebP and render fine; ensure the transcoding param is present.
    df["photo_url"] = df["photo_url"].map(normalize_photo_url)
    heic = df["photo_url"].astype(str).str.contains(".heic", case=False).sum()
    print(f"  photo URLs normalized ({heic} CDN .heic links kept as renderable WebP)")

    enrich_cache = {} if args.force else _load_cache(ENRICH_CACHE)
    embed_cache = {} if args.force else _load_cache(EMBED_CACHE)

    records: list[dict] = []
    for i, row in df.iterrows():
        lid = int(row["Listing_ID"])
        title_raw = str(row["title"])
        desc_raw = str(row["description"])
        clean_desc = clean_text(desc_raw)
        clean_title = clean_text(title_raw)
        combined = f"{clean_title}. {clean_desc}"

        rx_price, rx_monthly = regex_price(combined)
        rx_mileage, rx_isnew = regex_mileage(combined)

        key = str(lid)
        if args.no_llm:
            llm = {}
        elif key in enrich_cache:
            llm = enrich_cache[key]
        else:
            print(f"  [{i+1}/{len(df)}] LLM extract listing #{lid} "
                  f"({row['make']} {row['model']} {row['year']})")
            result = llm_extract(row["make"], row["model"], row["trim"], row["year"],
                                  clean_title, clean_desc)
            # Only cache on success; a failed extraction (None) gets retried
            # on the next run instead of being stuck as empty forever.
            llm = result or {}
            if result is not None:
                enrich_cache[key] = result
                _save_cache(ENRICH_CACHE, enrich_cache)

        fields = merge_fields(llm, rx_price, rx_monthly, rx_mileage, rx_isnew, clean_desc)

        rec = {
            "Listing_ID": lid,
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "make": row["make"],
            "model": row["model"],
            "trim": row["trim"],
            "title": title_raw,
            "description": desc_raw,
            "photo_url": row["photo_url"],
            **fields,
        }
        records.append(rec)

    # Write inventory.json
    config.INVENTORY_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    priced = sum(1 for r in records if r["price_aed"] is not None)
    mileaged = sum(1 for r in records if r["mileage_km"] is not None)
    print(f"\nWrote {config.INVENTORY_JSON}  "
          f"({len(records)} rows; {priced} with cash price, {mileaged} with mileage)")

    # Embeddings over "make model trim year body_type description_clean"
    print("\nComputing embeddings...")
    vectors: list[list[float]] = []
    ids: list[int] = []
    for i, rec in enumerate(records):
        lid = rec["Listing_ID"]
        embed_src = " ".join(str(x) for x in [
            rec["make"], rec["model"], rec["trim"], rec["year"],
            rec.get("body_type") or "", rec.get("description_clean") or "",
        ]).strip()
        key = str(lid)
        if key in embed_cache:
            vec = embed_cache[key]
        else:
            print(f"  [{i+1}/{len(records)}] embedding listing #{lid}")
            vec = embed_text(embed_src)
            embed_cache[key] = vec
            if (i + 1) % 5 == 0:
                _save_cache(EMBED_CACHE, embed_cache)
        vectors.append(vec)
        ids.append(lid)
    _save_cache(EMBED_CACHE, embed_cache)

    mat = np.array(vectors, dtype=np.float32)
    np.save(config.EMBEDDINGS_NPY, mat)
    config.EMBEDDINGS_IDS_JSON.write_text(json.dumps(ids), encoding="utf-8")
    print(f"Wrote {config.EMBEDDINGS_NPY}  shape={mat.shape}")
    print(f"Wrote {config.EMBEDDINGS_IDS_JSON}  ({len(ids)} ids)")
    print("\nEnrichment complete.")


if __name__ == "__main__":
    main()
