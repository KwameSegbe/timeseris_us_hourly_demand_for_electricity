import pandas as pd
import numpy as np
import requests 
import os 
import mlforecast as mlf
from datetime import datetime, timedelta
from utilsforecast.plotting import plot_series
from statsforecast import StatsForecast
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*to_pydatetime.*"
)


from statsforecast.models import(
    AutoARIMA, 
    MSTL, 
    Naive, 
    SeasonalNaive, 
    Theta,
    DynamicOptimizedTheta,
    AutoARIMA,
    HoltWinters)


# # EIA API configuration
# api_key = os.getenv("EIA_API_KEY")
# if not api_key:
#     raise RuntimeError("EIA_API_KEY not set in environment variables")


# api_url = "https://api.eia.gov/v2/"
# api_path = "electricity/rto/region-data/data"

# # Ensure data directory exists
# DATA_DIR = r"C:\Users\HP\Documents\timeseries_analysis\data"
# os.makedirs(DATA_DIR, exist_ok=True)

# params = {
#     "api_key": api_key,
#     "data[0]": "value"
# }
from config import api_key, api_url, api_path, DATA_DIR, params

response = requests.get(api_url + api_path, params=params)
data = response.json()
print(data)


# Check for expected structure
data = response.json()
if "response" not in data or "data" not in data["response"]:
    raise ValueError("Unexpected API response structure")

df = pd.DataFrame(data["response"]["data"])


# Writing to CSV
file_path = os.path.join(DATA_DIR, "eia_region_data.csv")
df.to_csv(file_path, index=False)

# Load data for analysis
df = pd.read_csv(file_path)


# --- Data preparation ---
ts = df[["period", "value"]].copy()
ts["period"] = pd.to_datetime(ts["period"], errors="coerce")
ts = ts.sort_values("period")

ts = ts.rename(columns={"period": "ds", "value": "y"})
ts["unique_id"] = 1
ts = ts[["unique_id", "ds", "y"]].sort_values("ds")
# Final column order    
ts = ts[["unique_id", "ds", "y"]].reset_index(drop=True)
ts.to_csv(os.path.join(DATA_DIR, "prepared_data.csv"), index=False)


#--- Evaluation Metrics ---
def mape(y, yhat, eps=1e-8):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.mean(np.abs(y - yhat) / np.maximum(np.abs(y), eps))

def rmse(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.sqrt(np.mean((y - yhat) ** 2))

def coverage(y, lower, upper):
    y = np.asarray(y)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return np.mean((y >= lower) & (y <= upper))


# Identify model point-forecast columns (exclude metadata + interval columns)
ignore = {"unique_id", "ds"}
model_cols = [c for c in forecast_stats.columns if c not in ignore and "-lo-" not in c and "-hi-" not in c]

rows = []
for col in model_cols:
    y = fc["y"].values
    yhat = fc[col].values

    # interval columns exist only if you passed level=[95]
    lo_col = f"{col}-lo-95"
    hi_col = f"{col}-hi-95"

    row = {
        "model": col,
        "mape": mape(y, yhat),
        "rmse": rmse(y, yhat),
    }

    if lo_col in fc.columns and hi_col in fc.columns:
        row["coverage_95"] = coverage(y, fc[lo_col].values, fc[hi_col].values)
    else:
        row["coverage_95"] = np.nan

    rows.append(row)

fc_performance = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
fc_performance


