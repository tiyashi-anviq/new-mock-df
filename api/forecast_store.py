"""Reads forecast_history.csv (written by 03_final_forecast.py) and hands out forecasts by series and by weekly run.
The file is re-read automatically when it changes, so a new Monday run shows up without restarting the API.
Each row is one series, one forecast week, one weekly run:  expected demand plus P10 / P50 / P90 (units, per series - never summed here)."""
from pathlib import Path
import pandas as pd


def api_series_id(grain: str, training_id: str) -> str:
    """Training scripts say 'bt20' (National) and 'bt20|bar' (Kolkata). The API says nat|ALL|bt20 and kol|bar|bt20."""
    if grain == "national":
        return f"nat|ALL|{training_id}"
    product, depot = training_id.split("|")
    return f"kol|{depot}|{product}"


class ForecastStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._mtime = None
        self._df = None

    def _frame(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path.name} not found in the outputs folder. Run 03_final_forecast.py first.")
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            d = pd.read_csv(self.path, parse_dates=["run_date", "target_week"])
            d["api_id"] = [api_series_id(g, s) for g, s in zip(d["grain"], d["series_id"])]
            self._df = d.sort_values(["run_date", "api_id", "target_week"]).reset_index(drop=True)
            self._mtime = mtime
        return self._df

    def runs(self) -> list[str]:
        """Every weekly run in the file, oldest first (ISO dates)."""
        return [x.date().isoformat() for x in sorted(self._frame().run_date.unique())]

    def select(self, run_date: str | None) -> tuple[str, pd.DataFrame]:
        """The rows of one run. No run_date means the latest run."""
        d = self._frame()
        if run_date is None:
            chosen = d.run_date.max()
        else:
            try:
                chosen = pd.Timestamp(run_date)
            except ValueError:
                raise KeyError(f"'{run_date}' is not a date. Use YYYY-MM-DD, for example 2026-09-28.")
            if chosen not in set(d.run_date.unique()):
                raise KeyError(f"No forecast run on {run_date}. Call /forecast/runs to see the available dates.")
        return chosen.date().isoformat(), d[d.run_date == chosen]


def points(rows: pd.DataFrame) -> list[dict]:
    """One series' rows -> the list of weekly points the UI draws."""
    return [dict(week_start=r.target_week.date().isoformat(), weeks_ahead=int(r.h), expected=round(float(r.expected), 1),
                 p10=round(float(r.p10), 1), p50=round(float(r.p50), 1), p90=round(float(r.p90), 1))
            for r in rows.sort_values("target_week").itertuples(index=False)]
