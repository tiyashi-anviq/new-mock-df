"""Eveready demand forecasting API (steps 1-5: health, series list, forecasts, accuracy, challenge flags, explanation).
Run from this folder:   python -m uvicorn app:app --reload --port 4004
Then open  http://127.0.0.1:4004/docs  to try every endpoint in the browser."""
from contextlib import asynccontextmanager
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import settings
from catalogue import Catalogue, load_catalogue
from forecast_store import ForecastStore, points
from accuracy_store import AccuracyStore
from flags_store import FlagStore
from explain_store import ExplainStore, headline

FLAGS = FlagStore(settings.OUT_DIR / settings.RESULT_FILES["flags"])
EXPLAIN = ExplainStore(settings.OUT_DIR)
STATE: dict = {}
FORECASTS = ForecastStore(settings.OUT_DIR / settings.RESULT_FILES["forecast"])
ACCURACY = AccuracyStore(settings.OUT_DIR / settings.RESULT_FILES["accuracy"], settings.DATA_FILE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.DATA_FILE.exists():
        raise RuntimeError(f"Data file not found: {settings.DATA_FILE}. Put eveready_mock_data.csv in the data folder, or set DF_ROOT.")
    STATE["cat"] = load_catalogue(settings.DATA_FILE)
    yield
    STATE.clear()


app = FastAPI(title="Eveready demand forecasting API", version="0.5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


class Health(BaseModel):
    status: Literal["ok"]
    series: int
    national_series: int
    kolkata_series: int
    first_week: str
    last_history_week: str
    weeks: int
    results_available: dict[str, bool]


class Series(BaseModel):
    series_id: str
    grain: Literal["national", "kolkata"]
    product_id: str
    product_name: str
    category: str
    segment: str
    depot_id: str
    depot_name: str
    is_new_product: bool
    pack_format: str | None = None
    technology_level: str | None = None
    first_week: str
    last_week: str
    weeks: int


def _cat() -> Catalogue:
    return STATE["cat"]


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.get("/health", response_model=Health, summary="Is the API up, and which result files exist?")
def health():
    c = _cat()
    nat = sum(s["grain"] == "national" for s in c.series)
    return Health(status="ok", series=len(c.series), national_series=nat, kolkata_series=len(c.series) - nat,
                  first_week=c.first_week, last_history_week=c.last_week, weeks=c.n_weeks,
                  results_available={k: (settings.OUT_DIR / f).exists() for k, f in settings.RESULT_FILES.items()})


@app.get("/series", response_model=list[Series], summary="All forecastable series (30 National + Kolkata product x depot)")
def series(grain: Literal["national", "kolkata"] | None = None, category: str | None = None,
           product_id: str | None = None, depot_id: str | None = Query(None, description="tar, bar, dan, bkp or bpr")):
    out = _cat().series
    if grain: out = [s for s in out if s["grain"] == grain]
    if category: out = [s for s in out if s["category"] == category]
    if product_id: out = [s for s in out if s["product_id"] == product_id]
    if depot_id: out = [s for s in out if s["depot_id"] == depot_id]
    if not out:
        raise HTTPException(404, "No series match these filters. Call /series with no filters to see what exists.")
    return out


# ---------------------------------------------------------------- forecasts (step 2)
class Point(BaseModel):
    week_start: str
    weeks_ahead: int
    expected: float      # expected demand (the mean), units
    p10: float           # low case, units
    p50: float           # middle case, units
    p90: float           # high case, units


class SeriesForecast(BaseModel):
    series_id: str
    grain: Literal["national", "kolkata"]
    run_date: str        # the Monday of data this forecast was made from
    points: list[Point]


class ForecastAll(BaseModel):
    run_date: str
    grain: Literal["national", "kolkata"] | None
    series: list[SeriesForecast]


class Runs(BaseModel):
    latest: str
    runs: list[str]


def _select(run_date: str | None):
    try:
        return FORECASTS.select(run_date)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except KeyError as e:
        raise HTTPException(404, e.args[0])


@app.get("/forecast/runs", response_model=Runs, summary="Dates of the weekly forecast runs that are stored")
def forecast_runs():
    try:
        r = FORECASTS.runs()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return Runs(latest=r[-1], runs=r)


@app.get("/forecast", response_model=SeriesForecast, summary="Forecast for one series (26 weeks ahead)")
def forecast(series_id: str = Query(..., description="e.g. nat|ALL|bt20 or kol|bar|bt20 (see /series)"),
             run_date: str | None = Query(None, description="YYYY-MM-DD of a weekly run. Default: the latest run.")):
    meta = next((s for s in _cat().series if s["series_id"] == series_id), None)
    if meta is None:
        raise HTTPException(404, f"Unknown series_id '{series_id}'. Call /series to see the valid ids.")
    run, rows = _select(run_date)
    rows = rows[rows.api_id == series_id]
    if rows.empty:
        raise HTTPException(404, f"No forecast for {series_id} in the run of {run}.")
    return SeriesForecast(series_id=series_id, grain=meta["grain"], run_date=run, points=points(rows))


@app.get("/forecast/all", response_model=ForecastAll, summary="Forecast for every series of one grain (or both)")
def forecast_all(grain: Literal["national", "kolkata"] | None = None,
                 run_date: str | None = Query(None, description="YYYY-MM-DD of a weekly run. Default: the latest run.")):
    run, rows = _select(run_date)
    if grain:
        rows = rows[rows.grain == grain]
    if rows.empty:
        raise HTTPException(404, f"No forecast rows for grain '{grain}' in the run of {run}.")
    out = [SeriesForecast(series_id=sid, grain=g.grain.iloc[0], run_date=run, points=points(g))
           for sid, g in rows.groupby("api_id", sort=True)]
    return ForecastAll(run_date=run, grain=grain, series=out)


# ---------------------------------------------------------------- accuracy (step 3)
Version = Literal["M1", "M2", "M3"]


class Score(BaseModel):
    forecast: Literal["model", "p50", "sales_team", "budget"]   # model = the model's expected demand
    accuracy: float | None       # 100 - WAPE, percent
    bias_pct: float | None       # over - under, percent of actual. Negative = forecast too low
    under_pct: float | None      # share of actual demand the forecast missed by forecasting too low
    over_pct: float | None       # share of actual demand the forecast overshot
    actual_units: float
    forecast_units: float
    rows: int


class SummaryRow(Score):
    version: Version
    months: int


class MonthRow(Score):
    version: Version
    target_month: str


class AccuracyResponse(BaseModel):
    grain: Literal["national", "kolkata"]
    from_month: str
    to_month: str | None
    available_from: str          # first and last target month in the saved rolling test
    available_to: str
    months_scored: int
    scored_at: str
    filters: dict[str, str]
    summary: list[SummaryRow]    # whole window, per version and forecast
    monthly: list[MonthRow]      # the same, month by month


@app.get("/accuracy", response_model=AccuracyResponse,
         summary="Rolling accuracy of the model, the Sales team and the Budget, for M1, M2 and M3")
def accuracy(grain: Literal["national", "kolkata"] = "national",
             version: Version | None = Query(None, description="M1, M2 or M3. Default: all three."),
             from_month: str = Query("2025-10", pattern=r"^\d{4}-\d{2}$", description="First target month scored. Default Oct 2025: the model then had a full year of history."),
             to_month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"),
             series_id: str | None = None, product_id: str | None = None, depot_id: str | None = None, category: str | None = None):
    filters = {k: v for k, v in dict(series_id=series_id, product_id=product_id, depot_id=depot_id, category=category).items() if v}
    ids = None
    if filters:
        sel = [s for s in _cat().series if s["grain"] == grain]
        for k, v in filters.items():
            sel = [s for s in sel if s["series_id" if k == "series_id" else k] == v]
        if not sel:
            raise HTTPException(404, f"No {grain} series match {filters}. Call /series to see valid values.")
        ids = {s["series_id"] for s in sel}
    try:
        lo, hi = ACCURACY.month_range()
        res = ACCURACY.score(grain, [version] if version else ["M3", "M2", "M1"], from_month, to_month, ids)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    if res is None:
        raise HTTPException(404, f"No scored months between {from_month} and {to_month or hi} for these filters. The saved test covers {lo} to {hi}.")
    summary, monthly = res
    return AccuracyResponse(
        grain=grain, from_month=from_month, to_month=to_month, available_from=lo, available_to=hi,
        months_scored=len({m["target_month"] for m in monthly}),
        scored_at="product x month" if grain == "national" else "product x depot x month",
        filters=filters, summary=summary, monthly=monthly)


# ---------------------------------------------------------------- challenge flags (step 4)
FLAGS_FROM = "2024-10-07"       # earlier rows had under a year of history behind them
Severity = Literal["Green", "Amber", "Red"]
_WEEK = r"^\d{4}-\d{2}-\d{2}$"


class FlagRow(BaseModel):
    week_start: str
    series_id: str
    grain: Literal["national", "kolkata"]
    product_id: str
    depot_id: str
    version: Version
    severity: Severity
    n_flags: int
    vs_budget: bool          # check 1: gap to budget unusual for this product
    vs_last_year: bool       # check 2: gap to trend-adjusted last year unusual
    vs_stock: bool           # check 3: forecast cut with stock short, or raised with stock long
    reason: str | None       # plain words, only when flagged
    sales_forecast: float | None
    budget: float | None
    last_year_ref: float | None
    stock_cover_weeks: float | None
    actual: float | None     # empty for weeks whose actuals are not known
    sales_team_error_pct: float | None


class FlagsPage(BaseModel):
    total: int
    limit: int
    offset: int
    rows: list[FlagRow]


class FlagSummary(BaseModel):
    grain: Literal["national", "kolkata"]
    version: Version
    rows: int
    green: int
    amber: int
    red: int
    green_pct: float
    amber_pct: float
    red_pct: float
    green_avg_error_pct: float | None    # average miss of the Sales team forecast in that colour: do the flags catch bad forecasts?
    amber_avg_error_pct: float | None
    red_avg_error_pct: float | None
    flag_vs_budget_pct: float            # share of forecasts failing each check
    flag_vs_last_year_pct: float
    flag_vs_stock_pct: float


class FlagSummaryResponse(BaseModel):
    from_week: str
    to_week: str | None
    summary: list[FlagSummary]


class TopProduct(BaseModel):
    product_id: str
    product_name: str
    rows: int
    amber: int
    red: int
    flagged: int
    flagged_pct: float


def _flag_window(grain, version, from_week, to_week, series_id, product_id, depot_id, category):
    filters = {k: v for k, v in dict(series_id=series_id, product_id=product_id, depot_id=depot_id, category=category).items() if v}
    ids = None
    if filters:
        sel = [s for s in _cat().series if s["grain"] == grain]
        for k, v in filters.items():
            sel = [s for s in sel if s[k] == v]
        if not sel:
            raise HTTPException(404, f"No {grain} series match {filters}. Call /series to see valid values.")
        ids = {s["series_id"] for s in sel}
    try:
        return FLAGS.window(grain, version, from_week, to_week, ids)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/flags/summary", response_model=FlagSummaryResponse, summary="Green / Amber / Red split of the Sales team forecasts")
def flags_summary(grain: Literal["national", "kolkata"] | None = None, version: Version | None = None,
                  from_week: str = Query(FLAGS_FROM, pattern=_WEEK), to_week: str | None = Query(None, pattern=_WEEK),
                  product_id: str | None = None, depot_id: str | None = None, category: str | None = None):
    parts = []
    for g in ([grain] if grain else ["national", "kolkata"]):
        try:
            d = _flag_window(g, version, from_week, to_week, None, product_id, depot_id, category)
        except HTTPException as e:
            if e.status_code == 404 and "match" in str(e.detail):
                continue                   # a filter that matches nothing in this grain (e.g. a depot for National): skip that grain
            raise
        if not d.empty:
            parts.append(FlagStore.summary(d))
    summary = [r for p in parts for r in p]
    if not summary:
        raise HTTPException(404, "No flagged forecasts in this window and selection.")
    return FlagSummaryResponse(from_week=from_week, to_week=to_week, summary=summary)


@app.get("/flags/top-products", response_model=list[TopProduct], summary="Products with the most Amber and Red forecasts: where to look first")
def flags_top_products(grain: Literal["national", "kolkata"] = "national", version: Version | None = None,
                       from_week: str = Query(FLAGS_FROM, pattern=_WEEK), to_week: str | None = Query(None, pattern=_WEEK),
                       limit: int = Query(10, ge=1, le=50)):
    d = _flag_window(grain, version, from_week, to_week, None, None, None, None)
    if d.empty:
        raise HTTPException(404, "No flagged forecasts in this window.")
    names = {s["product_id"]: s["product_name"] for s in _cat().series}
    return [TopProduct(product_name=names.get(r["product_id"], r["product_id"]), **r) for r in FlagStore.top_products(d, limit)]


@app.get("/flags", response_model=FlagsPage, summary="Sales team forecasts with their flags, newest week first")
def flags(grain: Literal["national", "kolkata"] = "national", version: Version | None = None, severity: Severity | None = None,
          flagged_only: bool = Query(False, description="Only Amber and Red"),
          from_week: str = Query(FLAGS_FROM, pattern=_WEEK), to_week: str | None = Query(None, pattern=_WEEK),
          series_id: str | None = None, product_id: str | None = None, depot_id: str | None = None, category: str | None = None,
          limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    d = _flag_window(grain, version, from_week, to_week, series_id, product_id, depot_id, category)
    if severity:
        d = d[d.severity == severity]
    elif flagged_only:
        d = d[d.severity != "Green"]
    return FlagsPage(total=int(len(d)), limit=limit, offset=offset, rows=FlagStore.rows(d, limit, offset))


# ---------------------------------------------------------------- explanation: "why is the forecast what it is?" (step 5)
EXPLAIN_NOTES = [
    "Each driver is what the model learned that group of inputs does to demand, applied one after the other. They add up exactly to the forecast.",
    "Drivers are patterns the model found in your history, not proof of cause.",
    "Weather shows normal seasonal weather only. A storm that has not happened yet is not in the forecast.",
]


class Driver(BaseModel):
    driver: str
    units: float
    pct_of_normal: float | None
    material: bool            # 1.5% of a normal week or more: the ones worth showing


class ExplainWeek(BaseModel):
    week_start: str
    kind: Literal["forecast", "fitted_to_history"]
    normal_week_units: float
    expected_units: float
    vs_normal_pct: float | None


class ExplainResponse(BaseModel):
    grain: Literal["national", "kolkata"]
    scope: str
    week_start: str
    kind: Literal["forecast", "fitted_to_history"]
    normal_week_units: float      # the product's own normal week (outside festival peaks), summed over the selection
    expected_units: float
    vs_normal_pct: float | None
    drivers: list[Driver]
    adds_up: bool                 # normal week + all drivers = expected demand (within rounding)
    headline: str
    notes: list[str]


def _explain_frame(grain, category, product_id, depot_id, series_id):
    sel = [s for s in _cat().series if s["grain"] == grain]
    for k, v in dict(category=category, product_id=product_id, depot_id=depot_id, series_id=series_id).items():
        if v:
            sel = [s for s in sel if s[k] == v]
    if not sel:
        raise HTTPException(404, f"No {grain} series match this selection. Call /series to see valid values.")
    try:
        frame, drivers = EXPLAIN.query(grain, category, product_id, depot_id, series_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    if frame is None:
        raise HTTPException(404, "No explanation rows for this selection.")
    names = sorted({s["product_name"] for s in sel})
    scope = "National" if grain == "national" else "Kolkata depots"
    if depot_id: scope = sel[0]["depot_name"]
    if category and not product_id and not series_id: scope += f", {category}"
    elif len(names) == 1: scope += f", {names[0]}"
    elif not category: scope += ", all products"
    return frame, drivers, scope


@app.get("/explain/weeks", response_model=list[ExplainWeek],
         summary="Every week that can be explained: the last 26 fitted weeks and the forecast weeks")
def explain_weeks(grain: Literal["national", "kolkata"] = "national", category: str | None = None, product_id: str | None = None,
                  depot_id: str | None = None, series_id: str | None = None):
    frame, _, _ = _explain_frame(grain, category, product_id, depot_id, series_id)
    return [ExplainWeek(week_start=r.target_week.date().isoformat(), kind=r.kind, normal_week_units=round(float(r.usual_week_units), 1),
                        expected_units=round(float(r.expected_units), 1),
                        vs_normal_pct=round(100 * (r.expected_units / r.usual_week_units - 1), 1) if r.usual_week_units else None)
            for r in frame.itertuples(index=False)]


@app.get("/explain", response_model=ExplainResponse, summary="Why is the forecast for this week what it is? (drivers vs a normal week)")
def explain(grain: Literal["national", "kolkata"] = "national", week_start: str | None = Query(None, pattern=_WEEK, description="Monday, YYYY-MM-DD. Default: the first forecast week."),
            category: str | None = None, product_id: str | None = None, depot_id: str | None = None, series_id: str | None = None):
    frame, names, scope = _explain_frame(grain, category, product_id, depot_id, series_id)
    if week_start is None:
        row = frame[frame.kind == "forecast"].iloc[0] if (frame.kind == "forecast").any() else frame.iloc[-1]
    else:
        hit = frame[frame.target_week == pd.Timestamp(week_start)]
        if hit.empty:
            raise HTTPException(404, f"No explanation for the week of {week_start}. Call /explain/weeks to see the weeks available.")
        row = hit.iloc[0]
    normal, expected = float(row["usual_week_units"]), float(row["expected_units"])
    drivers = []
    for n in names:
        u = float(row["effect__" + n]); pct = 100 * u / normal if normal else None
        drivers.append(dict(driver=n, units=round(u, 1), pct_of_normal=None if pct is None else round(pct, 1),
                            material=bool(pct is not None and abs(pct) >= 1.5)))
    adds_up = abs(normal + sum(float(row["effect__" + n]) for n in names) - expected) <= 1.0 + 0.001 * abs(expected)
    week = row["target_week"]
    return ExplainResponse(
        grain=grain, scope=scope, week_start=week.date().isoformat(), kind=row["kind"], normal_week_units=round(normal, 1),
        expected_units=round(expected, 1), vs_normal_pct=round(100 * (expected / normal - 1), 1) if normal else None,
        drivers=drivers, adds_up=adds_up, headline=headline(f"{week:%d %b}", expected, normal, drivers), notes=EXPLAIN_NOTES)



if __name__ == "__main__":
    # `python app.py` starts the API on port 4004 with auto-reload.
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=4004, reload=True)