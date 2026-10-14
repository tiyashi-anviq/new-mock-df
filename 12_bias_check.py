"""Is the model's forecast LOW or HIGH on average?  - from the saved rolling test, no retraining, seconds.
Shows bias (forecast - actual) / actual for the model's expected demand, by grain, M1/M2/M3, and by calendar month (to see winter / March).
Negative = the model forecasts too low.  Run:  python 12_bias_check.py"""
import numpy as np, pandas as pd
from config import *
mo = pd.read_csv(OUT_DIR / "rolling_months_by_version.csv"); mo = mo[mo.target_month >= "2025-10"]
mo["cal_month"] = mo.target_month.str[5:7]
def bias(d): return round(100 * (d.expected.sum() - d.actual.sum()) / d.actual.sum(), 1)
for g in GRAINS:
    d = mo[mo.grain == g]
    print(f"\n[{g}]  bias % of the model's expected demand, Oct 2025 onwards (negative = forecast too low)")
    t = d.groupby(["cal_month", "version"]).apply(bias, include_groups=False).unstack("version"); t["ALL"] = d.groupby("cal_month").apply(bias, include_groups=False)
    t.loc["ALL MONTHS"] = [bias(d[d.version == v]) for v in t.columns[:-1]] + [bias(d)]
    print(t.to_string())
