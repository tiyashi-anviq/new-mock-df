"""Settings shared by every script. Edit paths and dates here only."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                      # put the three CSV files here
EVEREADY = DATA_DIR / "eveready_mock_data.csv"
EXTERNAL = DATA_DIR / "external_factors_data.csv"
BUDGET_FUTURE = DATA_DIR / "budget_future.csv"   # optional: week_beginning, product_code, depot_code, budget_units for Oct-Mar
OUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
for p in (OUT_DIR, MODEL_DIR):
    p.mkdir(exist_ok=True)

SEED = 20261013
TARGET = "orders_received_units"              # demand = orders received (not dispatches)

# --- split ---
TRAIN_END = pd.Timestamp("2026-06-29")        # last week whose actual may be used to train (frozen test)
TEST_FIRST = pd.Timestamp("2026-07-06")
TEST_LAST = pd.Timestamp("2026-09-28")
FORECAST_WEEKS = 26                           # Oct 2026 - Mar 2027
LEADS = {"M1": 5, "M2": 10, "M3": 14}         # weeks between last observed week and target week (same as the sales-team snapshots)

# --- the budget (units) is only used if you supply Oct-Mar budget rows (see README) ---
USE_BUDGET = False

# --- the four models per grain ---
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=SEED, n_jobs=-1, verbose=-1)
TWEEDIE_POWER = 1.2                           # expected-demand model (average outcome, never negative)
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
MIN_HISTORY_WEEKS = 8                         # a series needs this many weeks of sales before it is forecast
GRAINS = ("national", "kolkata")              # National, and the five Kolkata depots
