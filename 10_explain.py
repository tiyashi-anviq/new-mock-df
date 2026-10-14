"""WHY is the forecast what it is?  (CFO demand item 5 - 'explain' half)
For every forecast week this splits the expected demand into plain-language DRIVERS, in units and as % of a normal week:
   Usual week  ->  + Recent momentum  + Last year's pattern  + Season & festivals  + Weather  + Price & trade scheme
               + Stock & supply  + Market conditions  + Launch stage  + How far ahead   =  Expected demand
'Normal week' = the product's own normal weekly level (the product's identity - what it is and where it sells - is built into it). Each driver's step is what the model learned that factor does to demand, one after the other
(so the steps always add up exactly to the forecast).  They describe patterns the model found - not proof of cause.
Future weather is only the normal (average) weather, so the Weather driver shows seasonal weather, not a storm that has not happened yet.
Uses the same expected-demand (mean) model as 03_final_forecast.py, retrained on all data. Run:  python 10_explain.py   (about 1-2 min)"""
import time, numpy as np, pandas as pd
from config import *
import data_prep as dp
from models import fit_mean

GROUPS = {  # feature -> driver a planner would recognise
    "Recent momentum": ["last_r", "m4_r", "m8_r", "m13_r", "m26_r", "sd13_r", "trend_4v13"],
    "Last year's pattern": ["ly_same_r", "yoy_13"],
    "Season & festivals": ["iso_week", "month", "woy_sin", "woy_cos", "f_durga", "f_diwali", "f_lakshmi", "f_bhai", "f_xmas", "f_any", "sellin_durga", "sellin_diwali",
                  "weeks_to_durga", "weeks_since_durga", "weeks_to_diwali", "weeks_since_diwali"],
    "Weather": ["exp_temp", "exp_rain", "exp_power_cut", "outage_idx_4w"],
    "Price & trade scheme": ["price_at_origin", "scheme_at_origin"],
    "Stock & supply": ["fill_rate_4w", "stock_cover_r"],
    "Market conditions": ["zinc_price", "rural_income", "led_price_index", "dengue_index"],
    "Product & depot profile": ["product_code", "product_category", "business_segment", "pack_format", "technology_level", "depot_code", "log_scale", "weeks_on_sale"],   # size and age describe WHAT the product is
    "Launch stage": ["is_new_product", "weeks_since_launch"],
    "How far ahead": ["h"],
}
if USE_BUDGET: GROUPS["Budget"] = ["budget_r"]
ORDER = [d for d in GROUPS if d != "Product & depot profile"]      # the profile defines the usual week, so it is not shown as a weekly driver
assert sorted(sum(GROUPS.values(), [])) == sorted(dp.FEATURES), "every model input must belong to exactly one driver"

t0 = time.time()
e, x = dp.load_inputs(); last = e.week_beginning.max()
cat = e.drop_duplicates("product_code").set_index("product_code").product_category
long = []
for g in GRAINS:
    S = dp.build_table(e, x, g, budget_future=dp.read_budget_future())
    m = fit_mean(PARAMS, dp.split_train(S, last))
    R = pd.concat([S[(S.origin == last) & (S.tw > last)].assign(kind="forecast"),
                   S[(S.h == 1) & (S.tw > last - pd.Timedelta(weeks=26)) & (S.tw <= last)].assign(kind="fitted_to_history")])
    X = R[dp.FEATURES]
    raw = m.predict(X, pred_contrib=True)                       # log-scale contributions, last column = model baseline
    base, C = raw[:, -1], raw[:, :-1]
    pred = np.clip(m.predict(X), 0, None)
    assert np.allclose(np.exp(raw.sum(axis=1)), m.predict(X), rtol=1e-4), "contributions must add up to the prediction"
    cols = {d: C[:, [dp.FEATURES.index(f) for f in fs]].sum(axis=1) for d, fs in GROUPS.items()}
    scale = R.scale.values; prof = cols["Product & depot profile"]
    level = np.exp(base + prof) * scale; usual = level.copy(); prev = level      # usual week = this product's own normal level
    out = R[["series_id", "tw", "kind", "product_code", "depot_code"]].copy().reset_index(drop=True)
    out["grain"] = g; out["category"] = out.product_code.astype(str).map(cat); out["usual_week_units"] = usual
    cum = prof.copy()
    for d in ORDER:
        cum = cum + cols[d]; cur = np.exp(base + cum) * scale
        out["effect__" + d] = cur - prev; prev = cur
    assert np.allclose(prev, pred * scale, rtol=1e-4), "usual week + all driver steps must equal the expected demand"
    out["expected_units"] = prev
    long.append(out)
    print(f"   [{g}] explained {len(out):,} rows  ({time.time()-t0:.0f}s)", flush=True)
W = pd.concat(long, ignore_index=True).rename(columns={"tw": "target_week"})
W.to_csv(OUT_DIR / "explain_series_weekly.csv", index=False, float_format="%.1f")
eff = ["effect__" + d for d in ORDER]
for key, name in ((["grain", "target_week", "kind"], "explain_grain_weekly.csv"), (["grain", "category", "target_week", "kind"], "explain_category_weekly.csv")):
    A = W.groupby(key, as_index=False)[["usual_week_units", "expected_units"] + eff].sum(); A.to_csv(OUT_DIR / name, index=False, float_format="%.1f")
G = W[W.grain == "national"].groupby(["target_week", "kind"], as_index=False)[["usual_week_units", "expected_units"] + eff].sum()

def sentence(r):
    pct = {d: 100 * r["effect__" + d] / r.usual_week_units for d in ORDER}
    ups = sorted([(v, d) for d, v in pct.items() if v >= 1.5], reverse=True)[:2]; dns = sorted([(v, d) for d, v in pct.items() if v <= -1.5])[:2]
    tot = 100 * (r.expected_units / r.usual_week_units - 1)
    s = f"Week of {r.target_week:%d %b}: {r.expected_units/1e6:.1f}M units, {abs(tot):.0f}% {'above' if tot >= 0 else 'below'} a normal week."
    if ups: s += " Pushing up: " + ", ".join(f"{d} {v:+.0f}%" for v, d in ups) + "."
    if dns: s += " Pulling down: " + ", ".join(f"{d} {v:+.0f}%" for v, d in dns) + "."
    return s
G["headline"] = G.apply(sentence, axis=1); G.to_csv(OUT_DIR / "explain_national_headlines.csv", index=False, float_format="%.1f")
F = G[G.kind == "forecast"].sort_values("target_week")
print("\nNational, all products - what drives each forecast week (% of a normal week):\n")
for s in F.headline: print("  " + s)
M = F.assign(month=F.target_week.dt.to_period("M")).groupby("month")[["usual_week_units"] + eff].sum()
print("\nWhat drives each month (% of a normal month; negative = pulls demand down):")
print((100 * M[eff].div(M.usual_week_units, axis=0)).rename(columns=lambda c: c.replace("effect__", "")).round(1).to_string())
print(f"\nSaved outputs/explain_*.csv  (total {(time.time()-t0)/60:.1f} min)")
