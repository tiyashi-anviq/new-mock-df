"""Where the API finds things. Everything is relative to the project folder (the one that holds data\\ and outputs\\),
so the same code runs on your laptop and later in a container.  Override with the environment variable DF_ROOT."""
import os
from pathlib import Path

ROOT_DIR = Path(os.environ.get("DF_ROOT", Path(__file__).resolve().parents[1]))
DATA_FILE = ROOT_DIR / "data" / "eveready_mock_data.csv"
OUT_DIR = ROOT_DIR / "outputs"

# Result files the weekly run writes. /health reports which of them exist, so you can see what each later endpoint can serve.
RESULT_FILES = {
    "forecast": "forecast_history.csv",
    "accuracy": "rolling_months_by_version.csv",
    "flags": "challenge_flags.csv",
    "explain": "explain_series_weekly.csv",
}
# Browser origins allowed in development (the React dev server).  Override: EVEREADY_CORS="http://a,http://b"
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "EVEREADY_CORS", "http://localhost:3004,http://127.0.0.1:3004,http://localhost:5173,http://localhost:4173").split(",") if o.strip()]
