"""Scores the saved Jul-Sep snapshots at 1, 2 and 3 months ahead and compares with the sales team and the consensus.
Accuracy = 100 - WAPE.  Bias = (forecast - actual) / actual (negative = under-forecast)."""
import numpy as np, pandas as pd
from config import *

snap = pd.read_csv(OUT_DIR / "weekly_snapshots_jul_sep.csv", parse_dates=["origin", "target_week", "model_cutoff"])
e = pd.read_csv(EVEREADY, parse_dates=["week_beginning"],
                usecols=["week_beginning", "reporting_level", "depot_code", "product_code", "orders_received_units",
                         "sales_forecast_3m_ahead_units", "sales_forecast_2m_ahead_units", "sales_forecast_1m_ahead_units",
                         "consensus_forecast_1m_ahead_units"])
e["grain"] = np.where(e.reporting_level == "National", "national", "kolkata")
e["series_id"] = np.where(e.grain == "national", e.product_code, e.product_code + "|" + e.depot_code)
e = e.rename(columns={"week_beginning": "target_week"})

def metrics(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    ok = ~np.isnan(a) & ~np.isnan(f)
    a, f = a[ok], f[ok]
    wape = np.abs(f - a).sum() / a.sum()
    return dict(accuracy=100 * (1 - wape), bias_pct=100 * (f.sum() - a.sum()) / a.sum(),
                under_pct=100 * np.clip(a - f, 0, None).sum() / a.sum(), over_pct=100 * np.clip(f - a, 0, None).sum() / a.sum(),
                coverage_p10_p90=np.nan, n=len(a))

rows = []
for lead_name, lead in LEADS.items():
    s = snap[snap.h == lead].merge(e.drop(columns=["reporting_level", "depot_code", "product_code"]),
                                   on=["grain", "series_id", "target_week"], how="left")
    sales_col = {"M1": "sales_forecast_1m_ahead_units", "M2": "sales_forecast_2m_ahead_units", "M3": "sales_forecast_3m_ahead_units"}[lead_name]
    for g, d in s.groupby("grain"):
        for name, col in (("Expected demand (model)", "expected"), ("P50 (model)", "p50"), ("Sales team", sales_col)):
            m = metrics(d.orders_received_units, d[col]); m.update(grain=g, lead=lead_name, forecast=name); rows.append(m)
        if lead_name == "M1":
            m = metrics(d.orders_received_units, d.consensus_forecast_1m_ahead_units); m.update(grain=g, lead=lead_name, forecast="Consensus"); rows.append(m)
        a = d.orders_received_units
        cov = ((a >= d.p10) & (a <= d.p90)).mean() * 100          # should be close to 80
        rows.append(dict(grain=g, lead=lead_name, forecast="P10-P90 range", accuracy=np.nan, bias_pct=np.nan, under_pct=np.nan,
                         over_pct=np.nan, coverage_p10_p90=cov, n=len(d)))
res = pd.DataFrame(rows)[["grain", "lead", "forecast", "accuracy", "bias_pct", "under_pct", "over_pct", "coverage_p10_p90", "n"]].round(1)
res.to_csv(OUT_DIR / "scorecard_jul_sep.csv", index=False)
print(res.to_string(index=False))
