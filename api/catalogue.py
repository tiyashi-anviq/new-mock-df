"""The list of forecastable series, read once from the data file when the API starts.
Series ids follow the module convention:  nat|ALL|<product>   and   kol|<depot>|<product>.
(Your training scripts use 'product' for National and 'product|depot' for Kolkata - same series, different spelling.)"""
from dataclasses import dataclass
import pandas as pd

COLS = ["week_beginning", "reporting_level", "depot_code", "depot_name", "product_code", "product_name",
        "product_category", "business_segment", "is_new_product", "pack_format", "technology_level"]


@dataclass(frozen=True)
class Catalogue:
    series: list[dict]
    first_week: str
    last_week: str
    n_weeks: int


def series_id(grain: str, depot: str, product: str) -> str:
    return f"{'nat' if grain == 'national' else 'kol'}|{depot}|{product}"


def load_catalogue(path) -> Catalogue:
    d = pd.read_csv(path, usecols=COLS, parse_dates=["week_beginning"])
    d["grain"] = d.reporting_level.map({"National": "national", "Kolkata depot": "kolkata"})
    if d.grain.isna().any():
        raise ValueError("reporting_level has values other than 'National' and 'Kolkata depot'")
    rows = []
    for (grain, depot, prod), g in d.groupby(["grain", "depot_code", "product_code"], sort=True):
        last = g.iloc[-1]
        rows.append(dict(
            series_id=series_id(grain, depot, prod), grain=grain, product_id=prod, product_name=last.product_name,
            category=last.product_category, segment=last.business_segment, depot_id=depot, depot_name=last.depot_name,
            is_new_product=bool(g.is_new_product.max()), pack_format=last.pack_format, technology_level=last.technology_level,
            first_week=g.week_beginning.min().date().isoformat(), last_week=g.week_beginning.max().date().isoformat(), weeks=int(len(g))))
    return Catalogue(rows, d.week_beginning.min().date().isoformat(), d.week_beginning.max().date().isoformat(), int(d.week_beginning.nunique()))
