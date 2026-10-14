"""Run from this folder:  python -m pytest -q test_api.py"""
from fastapi.testclient import TestClient
from app import app


def client():
    return TestClient(app)


def test_health():
    with client() as c:
        r = c.get("/health"); assert r.status_code == 200
        j = r.json(); assert j["status"] == "ok" and j["national_series"] > 0 and j["kolkata_series"] > j["national_series"]
        assert set(j["results_available"]) == {"forecast", "accuracy", "flags", "explain"}


def test_series_and_filters():
    with client() as c:
        allr = c.get("/series").json()
        nat = c.get("/series", params={"grain": "national"}).json()
        kol = c.get("/series", params={"grain": "kolkata", "depot_id": "bar"}).json()
        assert len(allr) > len(nat) > 0 and all(s["depot_id"] == "ALL" for s in nat)
        assert all(s["depot_id"] == "bar" and s["series_id"].startswith("kol|bar|") for s in kol)
        ids = [s["series_id"] for s in allr]; assert len(ids) == len(set(ids))


def test_series_errors():
    with client() as c:
        assert c.get("/series", params={"grain": "world"}).status_code == 422
        assert c.get("/series", params={"product_id": "nope"}).status_code == 404
