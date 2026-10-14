"""Reads outputs\\challenge_flags.csv (written by 08_challenge_flags.py).
Each row is one Sales team forecast (one series, one target week, one version M1 / M2 / M3) checked against three references:
budget, last year (trend-adjusted) and depot stock.  0 checks failed = Green, 1 = Amber, 2 or more = Red.
Rows before 2024-10-07 had under a year of history behind them, so the API starts there by default (as 08 does for its summary)."""
from pathlib import Path
import numpy as np
import pandas as pd

SEVERITIES = ("Green", "Amber", "Red")
CHECKS = ("flag_vs_budget", "flag_vs_last_year", "flag_vs_stock")


class FlagStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._mtime = None
        self._df = None

    def _frame(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path.name} not found in the outputs folder. Run 08_challenge_flags.py first.")
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            d = pd.read_csv(self.path, parse_dates=["target_week"])
            d["grain"] = np.where(d.reporting_level == "National", "national", "kolkata")
            d["api_id"] = np.where(d.grain == "national", "nat|ALL|" + d.product_code, "kol|" + d.depot_code + "|" + d.product_code)
            self._df, self._mtime = d, mtime
        return self._df

    def window(self, grain: str, version: str | None, from_week: str, to_week: str | None, ids: set[str] | None) -> pd.DataFrame:
        d = self._frame()
        d = d[(d.grain == grain) & (d.target_week >= pd.Timestamp(from_week))]
        if to_week:
            d = d[d.target_week <= pd.Timestamp(to_week)]
        if version:
            d = d[d.version == version]
        if ids is not None:
            d = d[d.api_id.isin(ids)]
        return d

    @staticmethod
    def summary(d: pd.DataFrame) -> list[dict]:
        """Colour split per grain and version, plus how big the Sales team's miss was for each colour - do the flags catch bad forecasts?"""
        out = []
        for (g, v), x in d.groupby(["grain", "version"]):
            n = len(x)
            row = dict(grain=g, version=v, rows=int(n))
            for sev in SEVERITIES:
                xs = x[x.severity == sev]
                err = xs.sales_team_error_pct.dropna()
                k = sev.lower()
                row[k] = int(len(xs)); row[k + "_pct"] = round(100 * len(xs) / n, 1)
                row[k + "_avg_error_pct"] = round(float(err.mean()), 1) if len(err) else None
            for c in CHECKS:
                row[c + "_pct"] = round(100 * float(x[c].mean()), 1)
            out.append(row)
        return out

    @staticmethod
    def top_products(d: pd.DataFrame, limit: int) -> list[dict]:
        """Products with the most Amber and Red forecasts: where to look first."""
        g = d.groupby("product_code").agg(rows=("severity", "size"), amber=("severity", lambda s: int((s == "Amber").sum())),
                                          red=("severity", lambda s: int((s == "Red").sum()))).reset_index()
        g["flagged"] = g.amber + g.red
        g["flagged_pct"] = (100 * g.flagged / g.rows).round(1)
        g = g.sort_values(["flagged", "red"], ascending=False).head(limit)
        return [dict(product_id=r.product_code, rows=int(r.rows), amber=int(r.amber), red=int(r.red), flagged=int(r.flagged),
                     flagged_pct=float(r.flagged_pct)) for r in g.itertuples(index=False)]

    @staticmethod
    def rows(d: pd.DataFrame, limit: int, offset: int) -> list[dict]:
        """Newest week first. NaN becomes None so the JSON is valid."""
        page = d.sort_values(["target_week", "n_flags", "api_id"], ascending=[False, False, True]).iloc[offset:offset + limit]
        out = []
        for r in page.itertuples(index=False):
            def num(x):
                return None if pd.isna(x) else float(x)
            out.append(dict(
                week_start=r.target_week.date().isoformat(), series_id=r.api_id, grain=r.grain, product_id=r.product_code,
                depot_id=r.depot_code, version=r.version, severity=r.severity, n_flags=int(r.n_flags),
                vs_budget=bool(r.flag_vs_budget), vs_last_year=bool(r.flag_vs_last_year), vs_stock=bool(r.flag_vs_stock),
                reason=None if pd.isna(r.reason) else str(r.reason), sales_forecast=num(r.forecast), budget=num(r.budget),
                last_year_ref=num(r.last_year_ref), stock_cover_weeks=num(r.stock_cover_weeks), actual=num(r.actual),
                sales_team_error_pct=num(r.sales_team_error_pct)))
        return out
