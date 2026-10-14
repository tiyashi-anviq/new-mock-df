"""Run from this folder:  python -m pytest -q test_accuracy.py
Needs outputs\rolling_months_by_version.csv (from 05_rolling_backtest.py); skipped if it is missing."""
import pytest
from fastapi.testclient import TestClient
import settings
from app import app

pytestmark = pytest.mark.skipif(not (settings.OUT_DIR / settings.RESULT_FILES["accuracy"]).exists(), reason="no rolling_months_by_version.csv yet")


def test_default_shape_and_identities():
    with TestClient(app) as c:
        j = c.get("/accuracy").json()
        assert j["from_month"] == "2025-10" and j["months_scored"] >= 1
        assert {s["version"] for s in j["summary"]} == {"M1", "M2", "M3"} and {s["forecast"] for s in j["summary"]} >= {"model", "sales_team"}
        for s in j["summary"]:
            assert s["accuracy"] <= 100
            assert abs(s["bias_pct"] - (s["over_pct"] - s["under_pct"])) <= 0.2          # bias = over - under
            assert abs(s["accuracy"] - (100 - s["under_pct"] - s["over_pct"])) <= 0.2    # accuracy = 100 - under - over


def test_later_start_scores_fewer_months_and_one_version():
    with TestClient(app) as c:
        a = c.get("/accuracy", params={"version": "M3"}).json()
        b = c.get("/accuracy", params={"version": "M3", "from_month": "2026-03"}).json()
        assert {s["version"] for s in a["summary"]} == {"M3"} and b["months_scored"] < a["months_scored"]


def test_kolkata_filter_and_errors():
    with TestClient(app) as c:
        assert c.get("/accuracy", params={"grain": "kolkata", "depot_id": "bar"}).status_code == 200
        assert c.get("/accuracy", params={"from_month": "2025/10"}).status_code == 422
        assert c.get("/accuracy", params={"version": "M4"}).status_code == 422
        assert c.get("/accuracy", params={"product_id": "nope"}).status_code == 404
        assert c.get("/accuracy", params={"from_month": "2031-01"}).status_code == 404
