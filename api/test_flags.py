"""Run from this folder:  python -m pytest -q test_flags.py
Needs outputs\challenge_flags.csv (from 08_challenge_flags.py); skipped if it is missing."""
import pytest
from fastapi.testclient import TestClient
import settings
from app import app

pytestmark = pytest.mark.skipif(not (settings.OUT_DIR / settings.RESULT_FILES["flags"]).exists(), reason="no challenge_flags.csv yet")


def test_summary_adds_up():
    with TestClient(app) as c:
        j = c.get("/flags/summary").json()
        assert {(s["grain"], s["version"]) for s in j["summary"]} == {(g, v) for g in ("national", "kolkata") for v in ("M1", "M2", "M3")}
        for s in j["summary"]:
            assert s["green"] + s["amber"] + s["red"] == s["rows"]
            assert abs(s["green_pct"] + s["amber_pct"] + s["red_pct"] - 100) <= 0.3


def test_rows_filters_and_paging():
    with TestClient(app) as c:
        r = c.get("/flags", params={"severity": "Red", "limit": 5}).json()
        assert len(r["rows"]) <= 5 and all(x["severity"] == "Red" and x["n_flags"] >= 2 and x["reason"] for x in r["rows"])
        weeks = [x["week_start"] for x in r["rows"]]; assert weeks == sorted(weeks, reverse=True)
        nxt = c.get("/flags", params={"severity": "Red", "limit": 5, "offset": 5}).json()
        assert nxt["total"] == r["total"] and not ({x["week_start"] + x["series_id"] + x["version"] for x in r["rows"]} & {x["week_start"] + x["series_id"] + x["version"] for x in nxt["rows"]})
        flagged = c.get("/flags", params={"flagged_only": True, "limit": 50}).json()
        assert all(x["severity"] in ("Amber", "Red") for x in flagged["rows"])


def test_top_products_and_errors():
    with TestClient(app) as c:
        t = c.get("/flags/top-products", params={"limit": 3}).json()
        assert 1 <= len(t) <= 3 and t == sorted(t, key=lambda x: (-x["flagged"], -x["red"]))
        assert c.get("/flags", params={"from_week": "Oct 2024"}).status_code == 422
        assert c.get("/flags", params={"severity": "Blue"}).status_code == 422
        assert c.get("/flags", params={"product_id": "nope"}).status_code == 404
