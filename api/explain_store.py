"""Reads the driver explanation written by 10_explain.py:  explain_grain_weekly.csv, explain_category_weekly.csv, explain_series_weekly.csv.
Each row is one week (a forecast week, or one of the last 26 weeks fitted to history) split into:
   normal week  +  one step per driver (effect__<driver>, in units)  =  expected demand.
The driver columns are read from the file itself, so a change to the driver list in 10_explain.py flows through without a code change here."""
from pathlib import Path
import pandas as pd
from forecast_store import api_series_id

FILES = {"grain": "explain_grain_weekly.csv", "category": "explain_category_weekly.csv", "series": "explain_series_weekly.csv"}
PREFIX = "effect__"


class ExplainStore:
    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir)
        self._cache: dict = {}

    def _frame(self, level: str) -> pd.DataFrame:
        p = self.dir / FILES[level]
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found in the outputs folder. Run 10_explain.py first.")
        mtime = p.stat().st_mtime
        c = self._cache.get(level)
        if c is None or c[0] != mtime:
            d = pd.read_csv(p, parse_dates=["target_week"])
            if level == "series":
                d["api_id"] = [api_series_id(g, s) for g, s in zip(d["grain"], d["series_id"])]
            self._cache[level] = (mtime, d)
        return self._cache[level][1]

    def query(self, grain: str, category: str | None, product_id: str | None, depot_id: str | None, series_id: str | None):
        """One row per week (summed over the selected series) and the list of driver names. (None, []) if nothing matches."""
        if series_id or product_id or depot_id:
            d = self._frame("series"); d = d[d["grain"] == grain]
            if series_id: d = d[d["api_id"] == series_id]
            if product_id: d = d[d["product_code"] == product_id]
            if depot_id: d = d[d["depot_code"] == depot_id]
            if category: d = d[d["category"] == category]
        elif category:
            d = self._frame("category"); d = d[(d["grain"] == grain) & (d["category"] == category)]
        else:
            d = self._frame("grain"); d = d[d["grain"] == grain]
        if d.empty:
            return None, []
        eff = [c for c in d.columns if c.startswith(PREFIX)]
        num = ["usual_week_units", "expected_units", *eff]
        out = d.groupby(["target_week", "kind"], as_index=False)[num].sum().sort_values("target_week").reset_index(drop=True)
        return out, [c[len(PREFIX):] for c in eff]


def fmt_units(x: float) -> str:
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


def headline(week_label: str, expected: float, normal: float, drivers: list[dict], limit_pct: float = 1.5) -> str:
    """'Week of 12 Oct: 18.0M units, 30% above a normal week. Pushing up: ... Pulling down: ...' - drivers under 1.5% are left out."""
    tot = 100 * (expected / normal - 1) if normal else 0.0
    s = f"Week of {week_label}: {fmt_units(expected)} units, {abs(tot):.0f}% {'above' if tot >= 0 else 'below'} a normal week."
    ups = sorted([d for d in drivers if d["pct_of_normal"] is not None and d["pct_of_normal"] >= limit_pct], key=lambda d: -d["pct_of_normal"])[:2]
    dns = sorted([d for d in drivers if d["pct_of_normal"] is not None and d["pct_of_normal"] <= -limit_pct], key=lambda d: d["pct_of_normal"])[:2]
    if ups: s += " Pushing up: " + ", ".join(f"{d['driver']} {d['pct_of_normal']:+.0f}%" for d in ups) + "."
    if dns: s += " Pulling down: " + ", ".join(f"{d['driver']} {d['pct_of_normal']:+.0f}%" for d in dns) + "."
    return s
