import pandas as pd
import numpy as np
import requests 
import os 
import mlforecast as mlf
from datetime import datetime, timedelta
from utilsforecast.plotting import plot_series
from statsforecast import StatsForecast
from mlforecast import MLForecast


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


# Configuration import
from config import api_key, api_url, api_path, DATA_DIR, params

# Function to fetch data from EIA API
def fetch_eia_df(api_url, api_path, params):
    response = requests.get(api_url + api_path, params=params)
    data = response.json()
    print(data)

    # Check for expected structure
    data = response.json()
    if "response" not in data or "data" not in data["response"]:
        raise ValueError("Unexpected API response structure")

    df = pd.DataFrame(data["response"]["data"])
    return df

#Call the function to fetch data
df = fetch_eia_df(api_url, api_path, params)



def prepare_eia_ts(df):
    ts = df[["period", "value"]].copy()
    ts["period"] = pd.to_datetime(ts["period"], errors="coerce")
    ts = ts.sort_values("period")

    ts = ts.rename(columns={"period": "ds", "value": "y"})
    ts["unique_id"] = 1
    ts = ts[["unique_id", "ds", "y"]].sort_values("ds")

    # Final column order
    ts = ts[["unique_id", "ds", "y"]].reset_index(drop=True)
    return ts


# Writing to CSV
file_path = os.path.join(DATA_DIR, "eia_region_data.csv")
df.to_csv(file_path, index=False)

# Load data for analysis
df = pd.read_csv(file_path)

# Prepare the time series data
ts = prepare_eia_ts(df)
ts.to_csv(os.path.join(DATA_DIR, "prepared_data.csv"), index=False)
