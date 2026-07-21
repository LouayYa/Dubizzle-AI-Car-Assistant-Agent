"""Structured search returns only real listings; null prices handled correctly."""
from __future__ import annotations

from app import db
from app.tools import ToolContext, search_inventory


def _ids(cars):
    return {c["listing_id"] for c in cars}


def test_search_returns_only_real_listings(conn):
    res = search_inventory(ToolContext(conn), make="honda")
    ids = _ids(res["cars"])
    assert ids == {1, 2}  # only the two real Hondas, nothing invented
    # every returned car exists in the DB
    for c in res["cars"]:
        assert db.get_listing(conn, c["listing_id"]) is not None


def test_body_type_filter(conn):
    res = search_inventory(ToolContext(conn), body_type="SUV")
    assert _ids(res["cars"]) == {1, 3, 4}


def test_price_range_excludes_null_prices(conn):
    # Honda Civic (#2) has price_aed = NULL (finance only) and must NOT appear.
    res = search_inventory(ToolContext(conn), price_min=50000, price_max=100000)
    ids = _ids(res["cars"])
    assert 2 not in ids
    assert ids == {1, 4}  # 70k and 90k; 250k/130k/160k out of range, null excluded


def test_mileage_max_filter(conn):
    res = search_inventory(ToolContext(conn), mileage_max=30000)
    # #3 (20k), #5 (15k), #6 (0km). #1(40k),#2(60k),#4(120k) excluded.
    assert _ids(res["cars"]) == {3, 5, 6}


def test_sort_by_price_asc_puts_nulls_last(conn):
    res = search_inventory(ToolContext(conn), sort_by="price_asc", limit=25)
    prices = [c["price_aed"] for c in res["cars"]]
    # Known prices ascending first, the single NULL price last.
    known = [p for p in prices if p is not None]
    assert known == sorted(known)
    assert prices[-1] is None


def test_price_label_for_finance_only(conn):
    res = search_inventory(ToolContext(conn), make="honda", model="civic")
    car = res["cars"][0]
    assert car["price_aed"] is None
    assert "Finance only" in car["price_label"]


def test_new_filter(conn):
    res = search_inventory(ToolContext(conn), is_new=True)
    assert _ids(res["cars"]) == {6}
