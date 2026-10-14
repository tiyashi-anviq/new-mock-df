"""WHY does 'Recent momentum' pull the forecast down by 5-10%?  (diagnostic only - no retraining of your saved models)
Splits the old 'Recent momentum' driver into its separate inputs and shows, month by month, what each one does to the National forecast
(% of expected demand, units-weighted).  Two of the inputs (log_scale, weeks_on_sale) describe how BIG / OLD a product is, not how it is selling now.
Run:  python 11_momentum_check.py   (about 1 minute)"""
import numpy as np, pandas as pd
from config import *
import data_prep as dp
from models import fit_mean

PARTS = {"Size of product (log_scale)": ["log_scale"], "Age on sale (weeks_on_sale)": ["weeks_on_sale"],
         "Last week's sales": ["last_r"], "4-8 week run rate": ["m4_r", "m8_r"], "13-26 week run rate": ["m13_r", "m26_r"],
         "Volatility": ["sd13_r"], "Short vs long trend": ["trend_4v13"]}
e, x = dp.load_inputs(); last = e.week_beginning.max()
for g in GRAINS:
    S = dp.build_table(e, x, g, budget_future=dp.read_budget_future())
    m = fit_mean(PARAMS, dp.split_train(S, last))
    R = S[(S.origin == last) & (S.tw > last)]; X = R[dp.FEATURES]
    raw = m.predict(X, pred_contrib=True); C = raw[:, :-1]
    w = m.predict(X) * R.scale.values; w = w / w.sum()
    rows = {}
    for name, fs in PARTS.items():
        c = C[:, [dp.FEATURES.index(f) for f in fs]].sum(axis=1)
        rows[name] = 100 * (np.exp(c) - 1)
    T = pd.DataFrame(rows, index=R.tw.values); T["month"] = pd.to_datetime(R.tw.values).to_period("M"); T["w"] = w
    out = T.groupby("month").apply(lambda d: pd.Series({k: np.average(d[k], weights=d.w) for k in rows}), include_groups=False).round(1)
    out["TOTAL"] = out.sum(axis=1).round(1)
    print(f"\n[{g}]  effect of each 'Recent momentum' input on the forecast (%, units-weighted; each input alone vs the model baseline):\n")
    print(out.to_string())
    # same-weeks-last-year level for context
    d = e.groupby("week_beginning").orders_received_units.sum().sort_index()
    print(f"\n   Context (all products, units): last 13 weeks {d[-13:].mean()/1e6:.1f}M vs same 13 weeks a year ago {d[-65:-52].mean()/1e6:.1f}M  ({100*(d[-13:].mean()/d[-65:-52].mean()-1):+.0f}%)") if g == GRAINS[0] else None
