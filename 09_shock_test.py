"""Short-cycle shock test (CFO demand item 5): how does the forecast behave when a weather shock hits, and while demand recovers?
Event: the late-May 2026 cyclone (week of 25 May 2026: 170 mm rain vs ~16 normal, 17 hours of power cuts vs ~7).
For every week from 4 May to 13 Jul 2026 the expected-demand model is RETRAINED using only data up to that week (the weekly retrain), then
forecasts the next 8 weeks. We then compare, for the categories that react to storms, with what actually happened and with two simple rivals:
   'Run rate'  = the last 4 weeks held flat      'Last year' = same weeks last year scaled by recent growth
Run:  python 09_shock_test.py     (about 5 minutes)"""
import time, numpy as np, pandas as pd
from config import *
import data_prep as dp
from models import fit_mean

ONSET = pd.Timestamp("2026-05-25")
ORIGINS = pd.date_range("2026-05-04", "2026-07-13", freq="7D")
HMAX = 8
AFFECTED = ["Emergency & rechargeable LED", "Flashlights — conventional", "Flashlights — rechargeable", "Batteries — carbon-zinc"]
t0 = time.time()
e, x = dp.load_inputs()
parts = []
for g in GRAINS:
    S = dp.build_table(e, x, g, budget_future=dp.read_budget_future())
    for k, c in enumerate(ORIGINS):
        m = fit_mean(PARAMS, dp.split_train(S, c))
        R = S[(S.origin == c) & (S.h <= HMAX) & S.y_tw.notna()].copy()
        R["model"] = np.clip(m.predict(R[dp.FEATURES]), 0, None) * R.scale.values
        R["run_rate"] = R.m4
        R["last_year"] = (R.ly_same * R.yoy_13.clip(0.6, 1.6).fillna(1.0)).fillna(R.m13)
        R["grain"] = g; R["category"] = R.product_category.astype(str)
        parts.append(R[["grain", "series_id", "category", "origin", "tw", "h", "y_tw", "model", "run_rate", "last_year"]])
        print(f"   [{g}] origin {c.date()} done ({time.time()-t0:.0f}s)", flush=True)
W = pd.concat(parts, ignore_index=True).rename(columns={"y_tw": "actual"})
W.to_csv(OUT_DIR / "shock_test_series_weekly.csv", index=False)
A = W[W.category.isin(AFFECTED)].groupby(["grain", "category", "tw", "h"], as_index=False)[["actual", "model", "run_rate", "last_year"]].sum()
A.to_csv(OUT_DIR / "shock_test_category_weekly.csv", index=False)

def acc(d, col):
    return 100 * (1 - (d[col] - d.actual).abs().sum() / d.actual.sum()), 100 * (d[col].sum() - d.actual.sum()) / d.actual.sum()
phase = lambda tw: np.where(tw < ONSET, "1 Before (11-18 May)", np.where(tw <= ONSET + pd.Timedelta(weeks=6), "2 Shock + 6 weeks (25 May - 6 Jul)", "3 After (13 Jul on)"))
W["phase"] = phase(W.tw); Wa = W[W.category.isin(AFFECTED)]
rows = []
for (g, h, ph), d in Wa[Wa.h.isin([1, 4])].groupby(["grain", "h", "phase"]):
    r = dict(grain=g, weeks_ahead=h, phase=ph, n=len(d))
    for col in ("model", "run_rate", "last_year"):
        a, b = acc(d, col); r[f"{col}_acc"] = round(a, 1); r[f"{col}_bias"] = round(b, 1)
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(OUT_DIR / "shock_test_scorecard.csv", index=False)
print("\nAccuracy (100 - WAPE) and bias % for storm-sensitive categories (emergency lights, both flashlight types, carbon-zinc batteries).")
print("weeks_ahead = how far ahead the forecast was made; a forecast made BEFORE the storm cannot see it.")
print(T.to_string(index=False))
print("\nNational, Emergency & rechargeable LED and Flashlights - conventional: actual vs forecast made 1 and 4 weeks earlier")
for cat in ("Emergency & rechargeable LED", "Flashlights — conventional"):
    p = A[(A.grain == "national") & (A.category == cat) & A.h.isin([1, 4])].pivot(index="tw", columns="h", values=["actual", "model", "run_rate"])
    out = pd.DataFrame({"actual": p["actual"][1], "model_1wk": p["model"][1], "model_4wk": p["model"][4], "runrate_1wk": p["run_rate"][1]}).dropna(how="all").round(0)
    print("\n" + cat); print(out.to_string())
print(f"\nSaved outputs/shock_test_*.csv  (total {(time.time()-t0)/60:.1f} min)")
