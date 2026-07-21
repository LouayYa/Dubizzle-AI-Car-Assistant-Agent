"""Shared pytest fixtures.

Tests run fully offline: a temporary SQLite DB is seeded with a handful of
hand-crafted listings, so nothing depends on the enrichment output or a live
API. No test makes a real LLM/embedding call.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app import config, db


@pytest.fixture(autouse=True)
def _isolate_side_effects(tmp_path, monkeypatch):
    """Redirect file side-effects (leads CSV) into tmp so tests never touch data/."""
    monkeypatch.setattr(config, "LEADS_CSV", tmp_path / "leads.csv")

# (listing_id, year, make, model, trim, title, description_clean, photo_url,
#  price_aed, monthly_payment_aed, mileage_km, is_new, exterior_color,
#  body_type, transmission, fuel_type, regional_spec, has_warranty)
_SEED = [
    (1, 2020, "honda", "cr-v", "EX", "Honda CR-V 2020", "White family SUV, GCC.",
     "http://img/1.jpg", 70000, None, 40000, 0, "white", "SUV", "Automatic", "Petrol", "GCC", 1),
    (2, 2019, "honda", "civic", "LX", "Honda Civic 2019", "Sporty sedan, finance only.",
     "http://img/2.jpg", None, 1500, 60000, 0, "black", "Sedan", "Automatic", "Petrol", "US", 0),
    (3, 2021, "toyota", "land cruiser", "GXR", "Toyota Land Cruiser 2021", "Big luxury SUV.",
     "http://img/3.heic", 250000, None, 20000, 0, "white", "SUV", "Automatic", "Petrol", "GCC", 1),
    (4, 2018, "nissan", "patrol", "SE", "Nissan Patrol 2018", "Rugged SUV.",
     "http://img/4.jpg", 90000, None, 120000, 0, "silver", "SUV", "Automatic", "Petrol", "GCC", 0),
    (5, 2022, "bmw", "320i", "M-Sport", "BMW 320i 2022", "Sharp sport sedan.",
     "http://img/5.jpg", 130000, None, 15000, 0, "blue", "Sedan", "Automatic", "Petrol", "European", 1),
    (6, 2023, "tesla", "model 3", "LR", "Tesla Model 3 2023", "Brand new electric sedan.",
     "http://img/6.jpg", 160000, None, 0, 1, "red", "Sedan", "Automatic", "Electric", "US", 1),
]


@pytest.fixture()
def conn(tmp_path):
    s = db.get_session(str(tmp_path / "test.db"))
    db.init_db(s)
    s.add_all(db.Listing(**dict(zip(db.LISTING_COLS, row))) for row in _SEED)
    s.commit()
    yield s
    s.close()


def next_slot(hour: int = 14, want_sunday: bool = False) -> str:
    """A future slot string 'YYYY-MM-DD HH:MM'. Skips or targets Sunday."""
    d = datetime.now() + timedelta(days=2)
    if want_sunday:
        while d.weekday() != 6:
            d += timedelta(days=1)
    else:
        while d.weekday() == 6:
            d += timedelta(days=1)
    d = d.replace(hour=hour, minute=0, second=0, microsecond=0)
    return d.strftime("%Y-%m-%d %H:%M")
