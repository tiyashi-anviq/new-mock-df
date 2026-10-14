"""National only. Tries two SIMPLER versions of the expected-demand model (less chance to overfit 30 product series), a simple last-year
baseline, and blends of each model with that baseline. Same rolling windows and same months as 05 (target months Oct 2025 - Sep 2026 are scored).
Blend weights are chosen walk-forward: for each month the weight is picked using only months whose actuals were already known when the
forecast was made, so nothing is tuned on the months being scored.   Run:  python 07_national_tune.py   (about 5-8 min)"""
import time, numpy as np, pandas as pd
from config import *
import data_prep as dp
from models import fit_mean

SCORE_FROM = pd.Period("2025-10")
ROLL_START = pd.Timestamp("2025-01-01")
base = dict(random_state=SEED, n_jobs=-1, verbose=-1, subsample_freq=1)
VARIANTS = {
    "A current":  PARAMS,
    "B simpler":  dict(n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=300, subsample=0.7, colsample_bytree=0.6, reg_lambda=10.0, **base),
    "C simplest": dict(n_estimators=150, learning_rate=0.05, num_leaves=4, min_child_samples=500, subsample=0.7, colsample_bytree=0.5, reg_lambda=20.0, **base),
}
t_all = time.time()
e, x = dp.load_inputs(); last = e.week_beginning.max()
mon = pd.date_range("2024-09-30", last, freq="7D"); cut = pd.Series(mon).groupby(mon.to_period("M")).min()
cuts = [d for p, d in cut.items() if d >= ROLL_START and p < last.to_period("M")]
S = dp.build_table(e, x, "national", budget_future=dp.read_budget_future())
print(f"National table {len(S):,} rows; {len(cuts)} monthly cutoffs; {len(VARIANTS)} model variants", flush=True)

rows = []
for k, c in enumerate(cuts):
    tr = dp.split_train(S, c); R = S[(S.origin == c) & S.y_tw.notna()].copy()
    R["target_m"] = R.tw.dt.to_period("M"); R["lag"] = (R.target_m - c.to_period("M")).apply(lambda z: z.n)
    R = R[R.lag.isin([1, 2, 3])].copy()
    for name, p in VARIANTS.items():
        R[name] = np.clip(fit_mean(p, tr).predict(R[dp.FEATURES]), 0, None) * R.scale.values
    R["last-year baseline"] = (R.ly_same * R.yoy_13.clip(0.6, 1.6).fillna(1.0)).fillna(R.m13)
    rows.append(R); print(f"   window {k+1}/{len(cuts)} cutoff {c.date()} done ({time.time()-t_all:.0f}s)", flush=True)
W = pd.concat(rows, ignore_index=True)
mcols = list(VARIANTS) + ["last-year baseline"]
mo = W.groupby(["series_id", "target_m", "lag"], as_index=False).agg(actual=("y_tw", "sum"), n=("y_tw", "size"), **{c: (c, "sum") for c in mcols})
mo = mo[mo.n >= 4]

ref = pd.read_csv(EVEREADY, parse_dates=["week_beginning"], usecols=["week_beginning", "reporting_level", "product_code",
        "sales_forecast_1m_ahead_units", "sales_forecast_2m_ahead_units", "sales_forecast_3m_ahead_units"])
ref = ref[ref.reporting_level == "National"].rename(columns={"product_code": "series_id"})
ref["target_m"] = ref.week_beginning.dt.to_period("M")
sl = ref.groupby(["series_id", "target_m"], as_index=False)[["sales_forecast_1m_ahead_units", "sales_forecast_2m_ahead_units", "sales_forecast_3m_ahead_units"]].sum()
mo = mo.merge(sl, on=["series_id", "target_m"], how="left")
mo["Sales team"] = np.select([mo.lag == 1, mo.lag == 2, mo.lag == 3], [mo.sales_forecast_1m_ahead_units, mo.sales_forecast_2m_ahead_units, mo.sales_forecast_3m_ahead_units])

grid = np.round(np.arange(0, 1.01, 0.1), 1)
def walk_forward(d, mcol):
    out, ws = [], []
    for T in sorted(d.target_m.unique()):
        lag = int(d.lag.iloc[0]); hist = d[d.target_m < T - lag]
        w = 0.5
        if hist.target_m.nunique() >= 3:
            w = grid[int(np.argmin([np.abs(g * hist[mcol] + (1 - g) * hist["last-year baseline"] - hist.actual).sum() for g in grid]))]
        t = d[d.target_m == T].copy(); t["blend"] = w * t[mcol] + (1 - w) * t["last-year baseline"]; t["w"] = w; out.append(t)
    return pd.concat(out)

def sc(d, col):
    a, f = d.actual.values, d[col].values
    return dict(accuracy=100 * (1 - np.abs(f - a).sum() / a.sum()), bias_pct=100 * (f.sum() - a.sum()) / a.sum(),
                under_pct=100 * np.clip(a - f, 0, None).sum() / a.sum(), over_pct=100 * np.clip(f - a, 0, None).sum() / a.sum())
res = []
for lag, d in mo.groupby("lag"):
    d = d.sort_values("target_m"); sd = d[(d.target_m >= SCORE_FROM) & d["Sales team"].notna()]
    for col in mcols + ["Sales team"]:
        s = sc(sd, col); s.update(version=f"M{lag}", forecast=col, blend_weight_on_model=np.nan); res.append(s)
    for v in VARIANTS:
        wf = walk_forward(d, v); wf = wf[(wf.target_m >= SCORE_FROM) & wf["Sales team"].notna()]
        s = sc(wf, "blend"); s.update(version=f"M{lag}", forecast=f"Blend: {v} + baseline", blend_weight_on_model=wf.w.mean()); res.append(s)
card = pd.DataFrame(res)[["version", "forecast", "accuracy", "bias_pct", "under_pct", "over_pct", "blend_weight_on_model"]].round(2).sort_values(["version", "accuracy"], ascending=[True, False])
card.to_csv(OUT_DIR / "national_tuning_scorecard.csv", index=False)
mo.to_csv(OUT_DIR / "national_tuning_months.csv", index=False)
print(f"\nNational, target months {SCORE_FROM} to {last.to_period('M')} (best first within each lead). 'blend_weight_on_model' = average share given to the model, the rest goes to the last-year baseline.")
print(card.round(1).to_string(index=False))
print(f"\nSaved outputs/national_tuning_*.csv   (total {(time.time()-t_all)/60:.1f} min)")
