"""Four models per grain: expected demand (mean) + P10/P50/P90. Target is demand / series 52-week average."""
import time, numpy as np, pandas as pd, lightgbm as lgb
from config import *
from data_prep import FEATURES


def fit_four(train, label=""):
    t0 = time.time()
    X, y = train[FEATURES], train["ratio"].clip(lower=0)
    out = {}
    out["mean"] = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER, **PARAMS).fit(X, y)
    for name, a in QUANTILES.items():
        out[name] = lgb.LGBMRegressor(objective="quantile", alpha=a, **PARAMS).fit(X, y)
    print(f"   [{label}] 4 models fitted on {len(train):,} rows in {time.time()-t0:.0f}s", flush=True)
    return out


def predict_four(models, rows):
    X = rows[FEATURES]
    p = {k: np.clip(m.predict(X), 0, None) for k, m in models.items()}
    q = np.sort(np.vstack([p["p10"], p["p50"], p["p90"]]), axis=0)      # stop quantiles crossing
    res = rows[["grain", "series_id", "origin", "tw", "h", "scale", "y_tw"]].copy()
    res["expected"] = p["mean"] * rows.scale.values
    res["p10"], res["p50"], res["p90"] = (q[0] * rows.scale.values, q[1] * rows.scale.values, q[2] * rows.scale.values)
    res = res.rename(columns={"tw": "target_week", "y_tw": "actual"})
    return res


def fit_mean(params, train):
    """One expected-demand (mean) model with its own settings - used by 07_national_tune.py to compare simpler versions."""
    X, y = train[FEATURES], train["ratio"].clip(lower=0)
    return lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=TWEEDIE_POWER, **params).fit(X, y)
