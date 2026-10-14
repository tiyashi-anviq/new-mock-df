"""Scores the saved rolling test (outputs\\rolling_months_by_version.csv, made by 05_rolling_backtest.py) - the same maths as 06_rescore.py.
Each row there is one series, one target month and one version (M1 / M2 / M3 = forecast made 1 / 2 / 3 months ahead), with the model's
expected demand and P50, the Sales team forecast, the Budget and the actual.
As in 06_rescore.py, the Sales team numbers are re-read from the data file, so a correction to the data is picked up without re-running 05.
Accuracy = 100 - WAPE, pooled over all rows scored.  Under / over-forecast are reported separately:  bias = over - under."""
from pathlib import Path
import numpy as np
import pandas as pd
from forecast_store import api_series_id

SALES_COLS = {"M1": "sales_forecast_1m_ahead_units", "M2": "sales_forecast_2m_ahead_units", "M3": "sales_forecast_3m_ahead_units"}
FORECASTS = {"model": "expected", "p50": "p50", "sales_team": "sales_team", "budget": "budget"}   # api key -> column in the rolling file
KEYS = ["grain", "series_id", "target_month"]


def _score(actual, fc) -> dict:
    a, f = np.asarray(actual, float), np.asarray(fc, float)
    tot = a.sum()
    if tot <= 0:
        return dict(accuracy=None, bias_pct=None, under_pct=None, over_pct=None, actual_units=round(float(tot), 1), forecast_units=round(float(f.sum()), 1))
    under = np.clip(a - f, 0, None).sum() / tot * 100
    over = np.clip(f - a, 0, None).sum() / tot * 100
    return dict(accuracy=round(100 - under - over, 1), bias_pct=round(over - under, 1), under_pct=round(under, 1), over_pct=round(over, 1),
                actual_units=round(float(tot), 1), forecast_units=round(float(f.sum()), 1))


class AccuracyStore:
    def __init__(self, rolling_path: Path, data_path: Path):
        self.rolling, self.data = Path(rolling_path), Path(data_path)
        self._stamp = None
        self._df = None

    def _frame(self) -> pd.DataFrame:
        if not self.rolling.exists():
            raise FileNotFoundError(f"{self.rolling.name} not found in the outputs folder. Run 05_rolling_backtest.py first.")
        stamp = (self.rolling.stat().st_mtime, self.data.stat().st_mtime)
        if stamp != self._stamp:
            mo = pd.read_csv(self.rolling)
            ref = pd.read_csv(self.data, parse_dates=["week_beginning"],
                              usecols=["week_beginning", "reporting_level", "depot_code", "product_code", *SALES_COLS.values()])
            ref["grain"] = np.where(ref.reporting_level == "National", "national", "kolkata")
            ref["series_id"] = np.where(ref.grain == "national", ref.product_code, ref.product_code + "|" + ref.depot_code)
            ref["target_month"] = ref.week_beginning.dt.to_period("M").astype(str)
            sf = ref.groupby(KEYS, as_index=False)[list(SALES_COLS.values())].sum()
            mo = mo.drop(columns=["sales_team"]).merge(sf, on=KEYS, how="left")
            mo["sales_team"] = np.select([mo.version == v for v in SALES_COLS], [mo[c] for c in SALES_COLS.values()])
            mo["api_id"] = [api_series_id(g, s) for g, s in zip(mo.grain, mo.series_id)]
            self._df, self._stamp = mo, stamp
        return self._df

    def month_range(self) -> tuple[str, str]:
        d = self._frame()
        return d.target_month.min(), d.target_month.max()

    def score(self, grain: str, versions: list[str], from_month: str, to_month: str | None, ids: set[str] | None):
        """Summary (whole window) and month-by-month scores for each version x forecast. None if no rows match."""
        d = self._frame()
        d = d[(d.grain == grain) & (d.target_month >= from_month)]
        if to_month:
            d = d[d.target_month <= to_month]
        if ids is not None:
            d = d[d.api_id.isin(ids)]
        if d.empty:
            return None
        summary, monthly = [], []
        for v in versions:
            dv = d[d.version == v]
            for key, col in FORECASTS.items():
                x = dv[dv[col].notna() & dv.actual.notna()]
                if x.empty:
                    continue
                summary.append(dict(version=v, forecast=key, months=int(x.target_month.nunique()), rows=int(len(x)), **_score(x.actual, x[col])))
                for m, xm in x.groupby("target_month"):
                    monthly.append(dict(version=v, forecast=key, target_month=m, rows=int(len(xm)), **_score(xm.actual, xm[col])))
        return (summary, monthly) if summary else None
