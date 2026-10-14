"""Retrain on ALL data to Sep 2026 and forecast the next 26 weeks (Oct 2026 - Mar 2027): expected demand, P10, P50, P90.
Also writes the past 26 weeks (Apr-Sep 2026) as fitted values for the chart. Run weekly after new data arrives."""
import time, pandas as pd
from config import *
import data_prep as dp
from models import fit_four, predict_four

t0 = time.time()
e, x = dp.load_inputs()
bf = dp.read_budget_future()
last = e.week_beginning.max()
out = []
for g in GRAINS:
    S = dp.build_table(e, x, g, budget_future=bf)
    m = fit_four(dp.split_train(S, last), f"{g} final")
    fut = S[(S.origin == last) & (S.tw > last)]
    r = predict_four(m, fut); r["kind"] = "forecast"; out.append(r)
    # fitted past: 1-week-ahead prediction for each of the last 26 weeks (in-sample, label 'fitted to history')
    past = S[(S.h == 1) & (S.tw > last - pd.Timedelta(weeks=26)) & (S.tw <= last)]
    p = predict_four(m, past); p["kind"] = "fitted_to_history"; out.append(p)
res = pd.concat(out, ignore_index=True)
res["months_ahead_band"] = pd.cut(res.h, [0, 4, 13, 26], labels=["weeks 1-4", "weeks 5-13", "weeks 14-26"])
res["run_date"] = last                                    # week of data this forecast was made from
res.to_csv(OUT_DIR / f"forecast_{last.date()}.csv", index=False)
# keep every weekly forecast so M3/M2/M1 versions can be built later (re-running the same week replaces it)
hist_path = OUT_DIR / "forecast_history.csv"
fc = res[res.kind == "forecast"]
if hist_path.exists():
    h = pd.read_csv(hist_path, parse_dates=["origin", "target_week", "run_date"])
    h = h[h.run_date != last]; fc = pd.concat([h, fc], ignore_index=True)
fc.to_csv(hist_path, index=False)
print(f"forecast_history.csv now holds {fc.run_date.nunique()} weekly run(s)")
print(f"Saved outputs/forecast_{last.date()}.csv  ({len(res):,} rows, {(time.time()-t0)/60:.1f} min)")
nat = res[(res.grain == "national") & (res.kind == "forecast")].groupby("target_week")[["expected"]].sum().round(0)
print("\nNational expected demand, all products (P10/P50/P90 are per product and are not added up):")
print(nat.to_string())
