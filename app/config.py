"""Central configuration and paths.

Loads .env once and exposes settings + filesystem paths used across the backend,
enrichment script, demo, and tests. LiteLLM reads GEMINI_API_KEY from the
environment, so we make sure it is exported here.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of the app/ package directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Load .env from the project root (no-op if absent; real env vars still win).
load_dotenv(ROOT / ".env")

# --- Models -----------------------------------------------------------------
# Using gemini-flash-lite-latest instead of gemini-2.0-flash (fast, free-tier works).
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini/gemini-flash-lite-latest")

# Using gemini-embedding-2 for free-tier keys.
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini/gemini-embedding-2")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# LiteLLM's Gemini provider looks up GEMINI_API_KEY; ensure it is present.
if GEMINI_API_KEY:
    os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- Filesystem paths -------------------------------------------------------
XLSX_PATH = DATA_DIR / "Copy_of_sample_cars_dataset.xlsx"
INVENTORY_JSON = DATA_DIR / "inventory.json"
EMBEDDINGS_NPY = DATA_DIR / "embeddings.npy"
EMBEDDINGS_IDS_JSON = DATA_DIR / "embeddings_ids.json"
DB_PATH = DATA_DIR / "app.db"
LEADS_CSV = DATA_DIR / "leads.csv"

# Sheet in the workbook that carries the Listing_ID primary key.
SHEET_NAME = "cleaned dataset"

# Structured fields produced by enrichment (used for schema + serialization).
ENRICHED_FIELDS = [
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
    "description_clean",
]


def require_api_key() -> str:
    """Return the Gemini API key or raise a clear error (used before LLM calls)."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and paste your "
            "Google AI Studio key, or export GEMINI_API_KEY."
        )
    return GEMINI_API_KEY
