"""Re-scores the rolling test from the saved file - NO retraining, takes seconds.
Why: the earliest cutoffs (Jan-Aug 2025) had under a year of history, so the model had never seen a full seasonal cycle or 'same week last year'.
This prints the scorecard for any start month, so you can see both the full period and the period after that warm-up.
Run:  python 06_rescore.py              (from Oct 2025 onwards)
      python 06_rescore.py 2025-02      (everything, same as 05)
      python 06_rescore.py 2026-01      (last 9 months only)"""
import sys, numpy as np, pandas as pd
from config import *

start = sys.argv[1] if len(sys.argv) > 1 else "2025-10"
mo = pd.read_csv(OUT_DIR / "rolling_months_by_version.csv")
# Re-read the Sales team's forecasts from the data file, so any correction made to the data is picked up without re-running 05
ref = pd.read_csv(EVEREADY, parse_dates=["week_beginning"], usecols=["week_beginning", "reporting_level", "depot_code", "product_code",
        "sales_forecast_1m_ahead_units", "sales_forecast_2m_ahead_units", "sales_forecast_3m_ahead_units"])
ref["grain"] = np.where(ref.reporting_level == "National", "national", "kolkata")
ref["series_id"] = np.where(ref.grain == "national", ref.product_code, ref.product_code + "|" + ref.depot_code)
ref["target_month"] = ref.week_beginning.dt.to_period("M").astype(str)
sf = ref.groupby(["grain", "series_id", "target_month"], as_index=False)[["sales_forecast_1m_ahead_units", "sales_forecast_2m_ahead_units", "sales_forecast_3m_ahead_units"]].sum()
mo = mo.drop(columns=["sales_team"]).merge(sf, on=["grain", "series_id", "target_month"], how="left")
mo["sales_team"] = np.select([mo.version == "M1", mo.version == "M2", mo.version == "M3"],
                             [mo.sales_forecast_1m_ahead_units, mo.sales_forecast_2m_ahead_units, mo.sales_forecast_3m_ahead_units])
mo = mo[(mo.target_month >= start) & mo.sales_team.notna()]

def sc(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    return dict(accuracy=100 * (1 - np.abs(f - a).sum() / a.sum()), bias_pct=100 * (f.sum() - a.sum()) / a.sum(),
                under_pct=100 * np.clip(a - f, 0, None).sum() / a.sum(), over_pct=100 * np.clip(f - a, 0, None).sum() / a.sum())
rows = []
for (g, v), d in mo.groupby(["grain", "version"]):
    for name, col in (("Expected demand (model)", "expected"), ("P50 (model)", "p50"), ("Sales team", "sales_team"), ("Budget", "budget")):
        s = sc(d.actual, d[col]); s.update(grain=g, version=v, forecast=name, months=d.target_month.nunique(), n=len(d)); rows.append(s)
card = pd.DataFrame(rows)[["grain", "version", "forecast", "accuracy", "bias_pct", "under_pct", "over_pct", "months", "n"]].round(1)
print(f"Months scored: {mo.target_month.min()} to {mo.target_month.max()}")
print(card.to_string(index=False))
card.to_csv(OUT_DIR / f"rolling_scorecard_from_{start}.csv", index=False)
