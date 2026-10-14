"""Rolling test over the whole history (the CFO's 'prove it on 36 months').
At the first Monday of every month from Jan 2025 to Aug 2026 the four models are retrained on ALL data up to that day (so the model sees the
full past, never the future) and forecast the next weeks. For each target month:  M1 = forecast made 1 month earlier, M2 = 2 months, M3 = 3 months.
Months are scored against what really happened, next to the Sales team's M1/M2/M3 and the Budget.
(The first 12 months of history cannot be scored: the model needs a year behind it to know 'same week last year'.)
Run:  python 05_rolling_backtest.py          (about 20-35 min)
      python 05_rolling_backtest.py quick    (4 products, a few minutes)"""
import sys, time, numpy as np, pandas as pd
from config import *
import data_prep as dp
from models import fit_four, predict_four

ROLL_START = pd.Timestamp("2025-01-01")
quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
t_all = time.time()
e, x = dp.load_inputs()
products = sorted(e.product_code.unique())[:4] if quick else None
bf = dp.read_budget_future()
last = e.week_beginning.max()
mon = pd.date_range("2024-09-30", last, freq="7D")
cut = pd.Series(mon).groupby(mon.to_period("M")).min()
cut = [d for p, d in cut.items() if d >= ROLL_START and p < last.to_period("M")]
print(f"{len(cut)} monthly retrains: {cut[0].date()} ... {cut[-1].date()}", flush=True)

weekly = []
for g in GRAINS:
    S = dp.build_table(e, x, g, products=products, budget_future=bf)
    for k, c in enumerate(cut):
        m = fit_four(dp.split_train(S, c), f"{g} {k+1}/{len(cut)} cutoff {c.date()}")
        r = predict_four(m, S[S.origin == c]); r["run_month"] = c.to_period("M")
        r["target_m"] = r.target_week.dt.to_period("M"); r["lag"] = (r.target_m - r.run_month).apply(lambda z: z.n)
        weekly.append(r[(r.lag.isin([1, 2, 3])) & (r.target_week <= last) & r.actual.notna()])
w = pd.concat(weekly, ignore_index=True)

ref = pd.read_csv(EVEREADY, parse_dates=["week_beginning"],
                  usecols=["week_beginning", "reporting_level", "depot_code", "product_code", "budget_units",
                           "sales_forecast_1m_ahead_units", "sales_forecast_2m_ahead_units", "sales_forecast_3m_ahead_units"])
ref["grain"] = np.where(ref.reporting_level == "National", "national", "kolkata")
ref["series_id"] = np.where(ref.grain == "national", ref.product_code, ref.product_code + "|" + ref.depot_code)
ref = ref.rename(columns={"week_beginning": "target_week"}).drop(columns=["reporting_level", "depot_code", "product_code"])
w = w.merge(ref, on=["grain", "series_id", "target_week"], how="left")
w["sales_team"] = np.select([w.lag == 1, w.lag == 2, w.lag == 3],
                            [w.sales_forecast_1m_ahead_units, w.sales_forecast_2m_ahead_units, w.sales_forecast_3m_ahead_units])
w["n"] = 1
mo = w.groupby(["grain", "series_id", "target_m", "lag"], as_index=False).agg(
    expected=("expected", "sum"), p50=("p50", "sum"), actual=("actual", "sum"), sales_team=("sales_team", "sum"),
    budget=("budget_units", "sum"), weeks=("n", "sum"), no_sales=("sales_team", lambda s: s.isna().any()))
mo = mo[mo.weeks >= 4].copy(); mo["version"] = "M" + mo.lag.astype(str); mo["target_month"] = mo.target_m.astype(str)
mo.loc[mo.no_sales, "sales_team"] = np.nan
mo.drop(columns=["target_m", "no_sales"]).to_csv(OUT_DIR / "rolling_months_by_version.csv", index=False)

def sc(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float); ok = ~np.isnan(a) & ~np.isnan(f); a, f = a[ok], f[ok]
    return dict(accuracy=100 * (1 - np.abs(f - a).sum() / a.sum()), bias_pct=100 * (f.sum() - a.sum()) / a.sum(),
                under_pct=100 * np.clip(a - f, 0, None).sum() / a.sum(), over_pct=100 * np.clip(f - a, 0, None).sum() / a.sum(), n=int(ok.sum()))
rows = []
for (g, v), d in mo.groupby(["grain", "version"]):
    d = d.dropna(subset=["sales_team"])                                  # same rows for every forecaster
    for name, col in (("Expected demand (model)", "expected"), ("P50 (model)", "p50"), ("Sales team", "sales_team"), ("Budget", "budget")):
        s = sc(d.actual, d[col]); s.update(grain=g, version=v, forecast=name); rows.append(s)
card = pd.DataFrame(rows)[["grain", "version", "forecast", "accuracy", "bias_pct", "under_pct", "over_pct", "n"]].round(1)
card.to_csv(OUT_DIR / "rolling_scorecard.csv", index=False)
print("\nRolling scorecard - monthly, every month from Feb 2025 to Sep 2026 that can be checked (CFO targets: ~80% at M3, ~90% at M1):")
print(card.to_string(index=False))

# by month, so you can see whether the festive peaks and winter/summer dips are caught
bm = []
for (g, tm, v), d in mo.groupby(["grain", "target_month", "version"]):
    s = sc(d.actual, d.expected); s.update(grain=g, target_month=tm, version=v); bm.append(s)
bm = pd.DataFrame(bm)
piv = bm.pivot_table(index=["grain", "target_month"], columns="version", values=["accuracy", "bias_pct"]).round(1)
piv.to_csv(OUT_DIR / "rolling_accuracy_by_month.csv")
print("\nNational expected-demand accuracy and bias by target month:")
print(piv.loc["national"].to_string() if "national" in piv.index.get_level_values(0) else piv.to_string())
print(f"\nSaved outputs/rolling_*.csv   (total {(time.time()-t_all)/60:.1f} min)")
