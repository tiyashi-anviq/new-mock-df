"""Sales-side 'challenge' flags (CFO demand item 2). Every Sales team forecast at M3, M2 and M1 is checked against three references and marked
Green / Amber / Red. No model and no training - it only uses the data file.
 Check 1  vs BUDGET      : gap between forecast and budget is unusual for THIS product (z-score above Z_LIMIT)
 Check 2  vs LAST YEAR   : gap to the same week last year (scaled by the recent growth trend, kept between 0.7x and 1.3x) is unusual (z-score above Z_LIMIT)
 Check 3  vs DEPOT STOCK : stock cover under 1 week and forecast cut over 15%, or cover over 3 weeks and forecast raised over 15% (vs the recent run rate)
 Severity: 0 checks failed = Green, 1 = Amber, 2 or more = Red.
'Unusual' is judged against the product's own earlier gaps, using only what was known when the forecast was made.
Run:  python 08_challenge_flags.py"""
import numpy as np, pandas as pd
from config import *

Z_LIMIT = 2.0                 # how unusual a gap must be (in standard deviations of that product's own gaps)
STOCK_LOW_WEEKS, STOCK_HIGH_WEEKS, CHANGE_LIMIT = 1.0, 3.0, 0.15
MIN_HISTORY = 26              # weeks of earlier gaps needed before a gap can be called unusual
USE = ["week_beginning", "reporting_level", "depot_code", "product_code", "orders_received_units", "primary_sales_units",
       "closing_stock_units", "budget_units", "sales_forecast_1m_ahead_units", "sales_forecast_2m_ahead_units", "sales_forecast_3m_ahead_units"]
e = pd.read_csv(EVEREADY, usecols=USE, parse_dates=["week_beginning"])
e["series"] = e.reporting_level + "|" + e.depot_code.fillna("") + "|" + e.product_code
out = []
for sid, d in e.groupby("series"):
    d = d.sort_values("week_beginning").set_index("week_beginning").asfreq("7D")
    y, ps, stk, bud = d.orders_received_units, d.primary_sales_units, d.closing_stock_units, d.budget_units
    ly = y.shift(52).rolling(3, center=True, min_periods=2).mean()
    m13 = y.rolling(13, min_periods=6).mean()
    trend = (m13 / m13.shift(52)).clip(0.7, 1.3).fillna(1.0)
    run4 = ps.rolling(4, min_periods=2).mean()
    for lag, col in (("M1", "sales_forecast_1m_ahead_units"), ("M2", "sales_forecast_2m_ahead_units"), ("M3", "sales_forecast_3m_ahead_units")):
        lead = LEADS[lag]; f = d[col]
        t = pd.DataFrame({"forecast": f, "budget": bud, "actual": y})
        t["last_year_ref"] = ly * trend.shift(lead)                        # trend known at the time the forecast was made
        t["stock_cover_weeks"] = (stk / run4).shift(lead)
        t["run_rate"] = run4.shift(lead)
        t["gap_budget"] = f / bud - 1
        t["gap_last_year"] = f / t.last_year_ref - 1
        t["change_vs_run_rate"] = f / t.run_rate - 1
        for g in ("gap_budget", "gap_last_year"):
            s = t[g].replace([np.inf, -np.inf], np.nan)
            mu = s.expanding(MIN_HISTORY).mean().shift(lead); sd = s.expanding(MIN_HISTORY).std().shift(lead)   # only gaps known by the origin week
            t["z_" + g[4:]] = (s - mu) / sd
        t["version"] = lag; t["series"] = sid; out.append(t.reset_index())
F = pd.concat(out, ignore_index=True)
F[["reporting_level", "depot_code", "product_code"]] = F.series.str.split("|", expand=True)
F["flag_vs_budget"] = (F.z_budget.abs() > Z_LIMIT).astype(int)
F["flag_vs_last_year"] = (F.z_last_year.abs() > Z_LIMIT).astype(int)
F["flag_vs_stock"] = (((F.stock_cover_weeks < STOCK_LOW_WEEKS) & (F.change_vs_run_rate < -CHANGE_LIMIT)) |
                      ((F.stock_cover_weeks > STOCK_HIGH_WEEKS) & (F.change_vs_run_rate > CHANGE_LIMIT))).astype(int)
F["n_flags"] = F.flag_vs_budget + F.flag_vs_last_year + F.flag_vs_stock
F["severity"] = np.select([F["n_flags"] >= 2, F["n_flags"] == 1], ["Red", "Amber"], "Green")
def why(r):
    p = []
    if r.flag_vs_budget: p.append(f"{r.gap_budget:+.0%} vs budget")
    if r.flag_vs_last_year: p.append(f"{r.gap_last_year:+.0%} vs last year (trend-adjusted)")
    if r.flag_vs_stock: p.append(f"stock cover {r.stock_cover_weeks:.1f} wks, forecast {r.change_vs_run_rate:+.0%} vs run rate")
    return "; ".join(p)
F["reason"] = ""; m = F["n_flags"] > 0; F.loc[m, "reason"] = F[m].apply(why, axis=1)
F = F[F.forecast.notna()].rename(columns={"week_beginning": "target_week"})
F["sales_team_error_pct"] = np.where(F.actual > 0, 100 * (F.forecast - F.actual).abs() / F.actual, np.nan)
cols = ["target_week", "reporting_level", "depot_code", "product_code", "version", "forecast", "budget", "last_year_ref", "stock_cover_weeks",
        "gap_budget", "gap_last_year", "change_vs_run_rate", "z_budget", "z_last_year", "flag_vs_budget", "flag_vs_last_year", "flag_vs_stock",
        "n_flags", "severity", "reason", "actual", "sales_team_error_pct"]
F[cols].to_csv(OUT_DIR / "challenge_flags.csv", index=False, float_format="%.3f")

G = F[F.target_week >= "2024-10-07"]                          # rows that had a year of history behind them
s = (G.groupby(["reporting_level", "version", "severity"]).size().unstack(fill_value=0).reindex(columns=["Green", "Amber", "Red"], fill_value=0))
share = (100 * s.div(s.sum(axis=1), axis=0)).round(1)
share.to_csv(OUT_DIR / "challenge_flag_summary.csv")
print("Share of Sales team forecasts by colour (%), from Oct 2024:"); print(share.to_string())
chk = G.dropna(subset=["sales_team_error_pct"]).groupby(["reporting_level", "version", "severity"]).sales_team_error_pct.mean().unstack().reindex(columns=["Green", "Amber", "Red"]).round(1)
print("\nDo the flags catch bad forecasts?  Average weekly error of the Sales team forecast (%), by colour:"); print(chk.to_string())
print("\nWhich checks fire (share of forecasts, %):"); print((100 * G.groupby(["reporting_level", "version"])[["flag_vs_budget", "flag_vs_last_year", "flag_vs_stock"]].mean()).round(1).to_string())
print("\nExamples of Red forecasts:"); print(G[G.severity == "Red"].sort_values("target_week").tail(8)[["target_week", "reporting_level", "depot_code", "product_code", "version", "reason"]].to_string(index=False))
print("\nSaved outputs/challenge_flags.csv and outputs/challenge_flag_summary.csv")
