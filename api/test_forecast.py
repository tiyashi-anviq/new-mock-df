"""Run from this folder:  python -m pytest -q test_forecast.py
Needs outputs\forecast_history.csv (from 03_final_forecast.py); skipped if it is missing."""
import pytest
from fastapi.testclient import TestClient
import settings
from app import app

pytestmark = pytest.mark.skipif(not (settings.OUT_DIR / settings.RESULT_FILES["forecast"]).exists(), reason="no forecast_history.csv yet")


def test_runs_and_latest_series():
    with TestClient(app) as c:
        r = c.get("/forecast/runs").json(); assert r["latest"] == r["runs"][-1]
        nat = c.get("/series", params={"grain": "national"}).json()[0]["series_id"]
        f = c.get("/forecast", params={"series_id": nat}).json()
        p = f["points"]; assert f["run_date"] == r["latest"] and f["grain"] == "national" and len(p) >= 26
        assert [x["weeks_ahead"] for x in p] == sorted(x["weeks_ahead"] for x in p)
        assert all(x["p10"] <= x["p50"] <= x["p90"] for x in p)       # the range never crosses


def test_all_and_filters():
    with TestClient(app) as c:
        nat = c.get("/forecast/all", params={"grain": "national"}).json()
        kol = c.get("/forecast/all", params={"grain": "kolkata"}).json()
        assert nat["series"] and kol["series"] and len(kol["series"]) > len(nat["series"])
        assert all(s["series_id"].startswith("nat|ALL|") for s in nat["series"])
        both = c.get("/forecast/all").json(); assert len(both["series"]) == len(nat["series"]) + len(kol["series"])


def test_errors():
    with TestClient(app) as c:
        assert c.get("/forecast", params={"series_id": "nope"}).status_code == 404
        assert c.get("/forecast").status_code == 422
        sid = c.get("/series").json()[0]["series_id"]
        assert c.get("/forecast", params={"series_id": sid, "run_date": "1999-01-04"}).status_code == 404
        assert c.get("/forecast", params={"series_id": sid, "run_date": "not-a-date"}).status_code == 404
