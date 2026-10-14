"""Builds the CFO view: for every target month, the forecast as made 3, 2 and 1 months earlier (M3, M2, M1), next to the actual.
Version rule: M1 = first weekly run in the month before the target month, M2 = two months before, M3 = three months before.
Sources: outputs/forecast_history.csv (weekly runs, grows each week) and outputs/weekly_snapshots_jul_sep.csv (the Jul-Sep test).
Only 'expected demand' and P50 are summed to months. P10/P90 are not additive, so ranges stay weekly.
Run:  python 04_versions.py"""
import numpy as np, pandas as pd
from config import *

parts = []
for f, rc in ((OUT_DIR / "forecast_history.csv", "run_date"), (OUT_DIR / "weekly_snapshots_jul_sep.csv", "origin")):
    if f.exists():
        d = pd.read_csv(f, parse_dates=["origin", "target_week"] + (["run_date"] if rc == "run_date" else []))
        d["run_date"] = d[rc]; d["source"] = f.stem; parts.append(d[["grain", "series_id", "run_date", "target_week", "expected", "p50", "source"]])
if not parts: raise SystemExit("No forecast files yet - run 03_final_forecast.py or 01_weekly_test.py first.")
w = pd.concat(parts, ignore_index=True)
w["run_m"] = w.run_date.dt.to_period("M"); w["target_m"] = w.target_week.dt.to_period("M")
w["lag"] = (w.target_m - w.run_m).apply(lambda x: x.n)
w = w[w.lag.isin([1, 2, 3])]
first_run = w.groupby(["run_m"]).run_date.min().rename("first_run")                # first weekly run in each month
w = w.join(first_run, on="run_m"); w = w[w.run_date == w.first_run]
m = (w.groupby(["grain", "series_id", "target_m", "lag"], as_index=False)
       .agg(expected=("expected", "sum"), p50=("p50", "sum"), run_date=("run_date", "first"), weeks=("target_week", "nunique")))
m = m[m.weeks >= 4]                                                                  # only months with full weekly coverage
m["version"] = "M" + m.lag.astype(str)

e = pd.read_csv(EVEREADY, parse_dates=["week_beginning"], usecols=["week_beginning", "reporting_level", "depot_code", "product_code", TARGET])
e["grain"] = np.where(e.reporting_level == "National", "national", "kolkata")
e["series_id"] = np.where(e.grain == "national", e.product_code, e.product_code + "|" + e.depot_code)
e["target_m"] = e.week_beginning.dt.to_period("M")
act = e.groupby(["grain", "series_id", "target_m"], as_index=False)[TARGET].sum().rename(columns={TARGET: "actual"})
m = m.merge(act, on=["grain", "series_id", "target_m"], how="left")
m["target_month"] = m.target_m.astype(str); m = m.drop(columns=["target_m", "lag"])
m.assign(expected=m.expected.round(1), p50=m.p50.round(1)).to_csv(OUT_DIR / "versions_by_month.csv", index=False)

def score(d, col):
    d = d.dropna(subset=["actual"]); a, f = d.actual.values, d[col].values
    return pd.Series(dict(accuracy=100 * (1 - np.abs(f - a).sum() / a.sum()), bias_pct=100 * (f.sum() - a.sum()) / a.sum(),
                          under_pct=100 * np.clip(a - f, 0, None).sum() / a.sum(), over_pct=100 * np.clip(f - a, 0, None).sum() / a.sum(), n=len(a)))
rows = []
for (g, v), d in m.groupby(["grain", "version"]):
    s = score(d, "expected"); s["grain"] = g; s["version"] = v; rows.append(s)
sc = pd.DataFrame(rows)[["grain", "version", "accuracy", "bias_pct", "under_pct", "over_pct", "n"]].sort_values(["grain", "version"]).round(1)
sc.to_csv(OUT_DIR / "scorecard_by_version.csv", index=False)
print("Monthly accuracy by version (expected demand, months that have actuals):"); print(sc.to_string(index=False))
print("\nSaved outputs/versions_by_month.csv and outputs/scorecard_by_version.csv")
