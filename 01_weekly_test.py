"""Jul-Sep 2026 test. Trains on Oct 2023-Jun 2026, then retrains every week (13 retrains) and saves each week's forecast.
Run:  python 01_weekly_test.py            (all products)
      python 01_weekly_test.py quick      (4 products, for a fast first check)"""
import sys, time, pandas as pd
from config import *
import data_prep as dp
from models import fit_four, predict_four

quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
t_all = time.time()
print("Loading data ...", flush=True)
e, x = dp.load_inputs()
products = sorted(e.product_code.unique())[:4] if quick else None
bf = dp.read_budget_future()
cutoffs = pd.date_range(TRAIN_END, TEST_LAST - pd.Timedelta(weeks=1), freq="7D")   # 13 weekly retrains
snaps = []
for g in GRAINS:
    t0 = time.time()
    S = dp.build_table(e, x, g, products=products, budget_future=bf)
    print(f"[{g}] table built: {len(S):,} rows in {time.time()-t0:.0f}s", flush=True)
    for k, c in enumerate(cutoffs):
        m = fit_four(dp.split_train(S, c), f"{g} retrain {k+1}/{len(cutoffs)} cutoff {c.date()}")
        rows = S[(S.tw >= TEST_FIRST) & (S.tw <= TEST_LAST) & ((S.origin == c) | ((S.origin < c) & (k == 0)))]
        rows = rows[rows.tw > rows.origin]
        r = predict_four(m, rows); r["model_cutoff"] = c
        snaps.append(r)
snap = pd.concat(snaps, ignore_index=True)
snap.to_csv(OUT_DIR / "weekly_snapshots_jul_sep.csv", index=False)
print(f"Saved {len(snap):,} rows -> outputs/weekly_snapshots_jul_sep.csv   (total {(time.time()-t_all)/60:.1f} min)")
