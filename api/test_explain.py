"""Run from this folder:  python -m pytest -q test_explain.py
Needs outputs\explain_*.csv (from 10_explain.py); skipped if missing."""
import pytest
from fastapi.testclient import TestClient
import settings
from app import app

pytestmark = pytest.mark.skipif(not (settings.OUT_DIR / settings.RESULT_FILES["explain"]).exists(), reason="no explain_*.csv yet")


def test_weeks_and_default_week_add_up():
    with TestClient(app) as c:
        w = c.get("/explain/weeks").json()
        assert {x["kind"] for x in w} == {"forecast", "fitted_to_history"} and [x["week_start"] for x in w] == sorted(x["week_start"] for x in w)
        j = c.get("/explain").json()
        assert j["kind"] == "forecast" and j["adds_up"] and j["headline"].startswith("Week of") and j["drivers"]
        assert abs(j["normal_week_units"] + sum(d["units"] for d in j["drivers"]) - j["expected_units"]) <= 1.0 + 0.002 * j["expected_units"]


def test_every_level_adds_up():
    with TestClient(app) as c:
        cat = c.get("/series", params={"grain": "national"}).json()[0]
        for params in ({"grain": "national"}, {"grain": "kolkata"}, {"category": cat["category"]}, {"product_id": cat["product_id"]},
                       {"grain": "kolkata", "depot_id": "bar"}, {"series_id": cat["series_id"]}):
            j = c.get("/explain", params=params).json(); assert j["adds_up"], params


def test_errors():
    with TestClient(app) as c:
        assert c.get("/explain", params={"week_start": "Oct 2026"}).status_code == 422
        assert c.get("/explain", params={"week_start": "1999-01-04"}).status_code == 404
        assert c.get("/explain", params={"product_id": "nope"}).status_code == 404
        assert c.get("/explain", params={"depot_id": "bar"}).status_code == 404       # National has no depots
