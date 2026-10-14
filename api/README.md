# API (FastAPI)

Step 1: `/health` and `/series`. More endpoints are added one at a time.

## Run (Windows PowerShell, from the project folder that holds `data\` and `outputs\`)

```
.\.venv\Scripts\Activate.ps1
pip install -r api\requirements.txt
cd api
python -m uvicorn app:app --reload --port 4004
```

Open http://127.0.0.1:4004/docs . Test: `python -m pytest -q test_api.py`.

The API reads `data\eveready_mock_data.csv` and (from later steps) the files in `outputs\`. It never trains a model.

## With uv instead of pip

From this `api` folder: `uv sync` (creates `api\.venv` and installs everything), then `uv run uvicorn app:app --reload --port 4004` and `uv run pytest -q`.
`requirements.txt` is kept for the pip route; keep both dependency lists in step.

## Endpoints so far

- `GET /health`, `GET /series`
- `GET /forecast/runs` - the weekly forecast runs stored in `outputs\forecast_history.csv`
- `GET /forecast?series_id=nat|ALL|bt20` - 26 weeks of expected demand with P10 / P50 / P90 for one series (latest run, or add `&run_date=YYYY-MM-DD`)
- `GET /forecast/all?grain=national` - every series of a grain
- `GET /accuracy?grain=national&version=M3&from_month=2025-10` - rolling accuracy and bias (under and over separate) of the model, the Sales team and the Budget, whole window and month by month. Same maths as `06_rescore.py`; filters: `series_id`, `product_id`, `depot_id`, `category`.
- `GET /flags/summary` - Green / Amber / Red split of the Sales team forecasts per grain and version, with the Sales team's average miss for each colour (do the flags catch bad forecasts?). Starts 2024-10-07 by default.
- `GET /flags/top-products` - products with the most Amber and Red forecasts
- `GET /flags` - the forecasts themselves with their reasons, newest week first. Filters: `severity`, `flagged_only`, `version`, `product_id`, `depot_id`, `category`, `from_week`, `to_week`; paging with `limit` and `offset`
- `GET /explain/weeks` - the weeks that can be explained (last 26 fitted weeks + forecast weeks), with the expected demand and how far it is from a normal week
- `GET /explain?week_start=2026-10-12` - why the forecast is what it is: the normal week plus each driver (units and % of a normal week), a plain-words headline, and a check that it adds up. Filters: `grain`, `category`, `product_id`, `depot_id`, `series_id`. Reads `outputs\explain_*.csv` from `10_explain.py`.
