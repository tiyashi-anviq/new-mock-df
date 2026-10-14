"""Builds the supervised table: one row per (series, origin week, weeks ahead).
Rules that keep it honest:
  * features about the past use only weeks <= the origin week
  * features about the target week are things known in advance (festival calendar, expected weather, launch date)
  * the sales-team forecasts, the consensus forecast and override reasons are NEVER read (they are not inputs)
"""
import numpy as np, pandas as pd
from config import *

KEEP = ["reporting_level", "depot_code", "product_code", "product_category", "business_segment", "pack_format",
        "technology_level", "is_new_product", "launch_date", "week_beginning", "orders_received_units",
        "primary_sales_units", "closing_stock_units", "average_selling_price_inr", "trade_scheme_discount_pct", "budget_units"]
STATIC_CAT = ["product_code", "product_category", "business_segment", "pack_format", "technology_level", "depot_code"]

FEATURES = (
    # about the past, at the origin week (ratios to the series' own 52-week average)
    ["last_r", "m4_r", "m8_r", "m13_r", "m26_r", "sd13_r", "trend_4v13", "yoy_13", "log_scale", "weeks_on_sale",
     "fill_rate_4w", "stock_cover_r", "price_at_origin", "scheme_at_origin", "outage_idx_4w",
     "zinc_price", "rural_income", "led_price_index", "dengue_index"]
    # about the target week, known in advance
    + ["h", "iso_week", "month", "woy_sin", "woy_cos", "f_durga", "f_diwali", "f_lakshmi", "f_bhai", "f_xmas", "f_any",
       "sellin_durga", "sellin_diwali", "weeks_to_durga", "weeks_since_durga", "weeks_to_diwali", "weeks_since_diwali",
       "exp_temp", "exp_rain", "exp_power_cut", "ly_same_r", "weeks_since_launch", "is_new_product"]
    + STATIC_CAT
)
if USE_BUDGET:
    FEATURES = FEATURES + ["budget_r"]


def load_inputs():
    e = pd.read_csv(EVEREADY, usecols=KEEP, parse_dates=["week_beginning", "launch_date"])
    x = pd.read_csv(EXTERNAL, parse_dates=["week_beginning"])
    return e, x


def _next_prev(weeks, flag_weeks):
    """weeks until the next flagged week and weeks since the previous one (NaN if none in the calendar)."""
    fw = np.array(sorted(flag_weeks), dtype="datetime64[ns]")
    w = np.array(weeks, dtype="datetime64[ns]")
    nxt, prv = [], []
    for d in w:
        i = np.searchsorted(fw, d, side="left")
        nxt.append((fw[i] - d) / np.timedelta64(7, "D") if i < len(fw) else np.nan)
        j = np.searchsorted(fw, d, side="right") - 1
        prv.append((d - fw[j]) / np.timedelta64(7, "D") if j >= 0 else np.nan)
    return np.array(nxt, float), np.array(prv, float)


def calendar_table(x):
    c = x[["week_beginning"]].copy().rename(columns={"week_beginning": "tw"})
    fest = x.festival.fillna("")
    sell = x.festival_sell_in_period.fillna("")
    c["iso_week"] = c.tw.dt.isocalendar().week.astype(int); c["month"] = c.tw.dt.month
    c["woy_sin"] = np.sin(2 * np.pi * c.iso_week / 52.18); c["woy_cos"] = np.cos(2 * np.pi * c.iso_week / 52.18)
    c["f_durga"] = fest.str.contains("Durga Puja").astype(int).values
    c["f_diwali"] = fest.str.contains("Kali Puja|Diwali").astype(int).values
    c["f_lakshmi"] = fest.str.contains("Lakshmi Puja").astype(int).values
    c["f_bhai"] = fest.str.contains("Bhai Phonta").astype(int).values
    c["f_xmas"] = fest.str.contains("Christmas").astype(int).values
    c["f_any"] = (fest != "").astype(int).values
    c["sellin_durga"] = sell.str.contains("Durga").astype(int).values
    c["sellin_diwali"] = sell.str.contains("Kali|Diwali").astype(int).values
    for name, flag in (("durga", c.f_durga), ("diwali", c.f_diwali)):
        nxt, prv = _next_prev(c.tw, c.tw[flag == 1])
        # cap so training weeks and the Oct-Mar forecast weeks look alike (the 2027 festivals are not in the calendar)
        c[f"weeks_to_{name}"] = np.where(np.isnan(nxt) | (nxt > 10), 10.0, nxt)
        c[f"weeks_since_{name}"] = np.where(np.isnan(prv) | (prv > 30), 30.0, prv)
    c["exp_temp"] = x.expected_temp_max_c.values; c["exp_rain"] = x.expected_rainfall_mm.values
    c["exp_power_cut"] = x.expected_power_cut_hours.values
    return c


def make_panel(e, grain, products=None):
    if grain == "national":
        d = e[e.reporting_level == "National"].copy(); d["series_id"] = d.product_code
    else:
        d = e[e.reporting_level == "Kolkata depot"].copy(); d["series_id"] = d.product_code + "|" + d.depot_code
    if products:
        d = d[d.product_code.isin(products)]
    pre = d.launch_date.notna() & (d.week_beginning < d.launch_date)       # not on sale yet -> not demand
    d.loc[pre, TARGET] = np.nan
    return d.rename(columns={"week_beginning": "week"})


def _wide(d, col, weeks):
    return d.pivot(index="week", columns="series_id", values=col).reindex(weeks)


def _long(w, name):
    out = w.copy(); out.index.name = "week"
    return out.reset_index().melt(id_vars="week", var_name="series_id", value_name=name)


def build_table(e, x, grain, hmax=FORECAST_WEEKS, products=None, budget_future=None):
    d = make_panel(e, grain, products)
    last = d.week.max()
    weeks = pd.date_range(d.week.min(), last + pd.Timedelta(weeks=hmax), freq="7D")
    Y = _wide(d, TARGET, weeks)
    P = _wide(d, "primary_sales_units", weeks); STK = _wide(d, "closing_stock_units", weeks)
    PRICE = _wide(d, "average_selling_price_inr", weeks).ffill(); SCH = _wide(d, "trade_scheme_discount_pct", weeks).ffill()
    obs = pd.Series(weeks <= last, index=weeks)

    scale = Y.rolling(52, min_periods=MIN_HISTORY_WEEKS).mean()
    feats = {
        "scale": scale, "last": Y,
        "m4": Y.rolling(4, min_periods=2).mean(), "m8": Y.rolling(8, min_periods=4).mean(),
        "m13": Y.rolling(13, min_periods=6).mean(), "m26": Y.rolling(26, min_periods=10).mean(),
        "sd13": Y.rolling(13, min_periods=6).std(),
        "m13_ly": Y.shift(52).rolling(13, min_periods=6).mean(),
        "n_on_sale": Y.notna().cumsum().astype(float),
        "fill4": P.rolling(4, min_periods=1).sum() / Y.rolling(4, min_periods=1).sum(),
        "stock": STK, "price_at_origin": PRICE, "scheme_at_origin": SCH}
    O = None
    for k, w in feats.items():
        l = _long(w, k); O = l if O is None else O.merge(l, on=["week", "series_id"])
    O = O[O.week <= last].rename(columns={"week": "origin"})
    O = O[O.scale.notna() & (O.scale > 0) & (O.n_on_sale >= MIN_HISTORY_WEEKS)].copy()
    for k in ("last", "m4", "m8", "m13", "m26", "sd13"):
        O[f"{k}_r"] = O[k] / O.scale
    O["trend_4v13"] = O.m4 / O.m13; O["yoy_13"] = O.m13 / O.m13_ly
    O["log_scale"] = np.log(O.scale); O["weeks_on_sale"] = O.n_on_sale
    O["fill_rate_4w"] = O.fill4; O["stock_cover_r"] = O.stock / O.scale

    xs = x.set_index("week_beginning")
    ext = pd.DataFrame({"origin": xs.index, "outage_idx_4w": xs.outage_pressure_index.rolling(4, min_periods=1).mean().values,
                        "zinc_price": xs.zinc_price_inr_per_kg.values, "rural_income": xs.rural_income_index.values,
                        "led_price_index": xs.led_price_index.values, "dengue_index": xs.dengue_index.values})
    O = O.merge(ext, on="origin", how="left")

    # target-week features (known in advance)
    ly = Y.shift(52).rolling(3, center=True, min_periods=2).mean()
    T = _long(ly, "ly_same").rename(columns={"week": "tw"})
    T = T.merge(_long(Y, "y_tw").rename(columns={"week": "tw"}), on=["tw", "series_id"])
    static = d.drop_duplicates("series_id").set_index("series_id")[["launch_date", "is_new_product"] + STATIC_CAT]
    T = T.merge(calendar_table(x), on="tw", how="left")
    T = T.join(static, on="series_id")
    T["weeks_since_launch"] = np.where(T.launch_date.notna(), (T.tw - T.launch_date).dt.days / 7.0, np.nan)
    T = T.drop(columns=["launch_date"])
    if USE_BUDGET:
        B = d.pivot(index="week", columns="series_id", values="budget_units").reindex(weeks)
        if budget_future is not None:
            B = B.combine_first(budget_future.reindex(weeks))
        T = T.merge(_long(B, "budget_tw").rename(columns={"week": "tw"}), on=["tw", "series_id"])

    H = pd.DataFrame({"h": np.arange(1, hmax + 1)})
    S = O.merge(H, how="cross")
    S["tw"] = S.origin + pd.to_timedelta(7 * S.h, unit="D")
    S = S.merge(T, on=["tw", "series_id"], how="inner")
    S["ratio"] = S.y_tw / S.scale                                  # target: demand as a multiple of the series' 52-week average
    S["ly_same_r"] = S.ly_same / S.scale
    if USE_BUDGET:
        S["budget_r"] = S.budget_tw / S.scale
    for c in STATIC_CAT:
        S[c] = S[c].astype("category")
    S["grain"] = grain
    return S


def split_train(S, cutoff):
    """rows whose answer was known by the cutoff week"""
    return S[(S.tw <= cutoff) & S.ratio.notna() & (S.origin <= cutoff)]


def read_budget_future():
    if USE_BUDGET and BUDGET_FUTURE.exists():
        b = pd.read_csv(BUDGET_FUTURE, parse_dates=["week_beginning"])
        return b
    return None
