import pandas as pd
import numpy as np
import os
import requests
import json
import datetime 
from statistics import mean, median

from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals
from window_ops.expanding import expanding_mean
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from utilsforecast.plotting import plot_series

# --- Data preparation ---
ts = pd.read_csv(r"C:\Users\HP\Documents\timeseries_analysis\data\tunnel.csv", encoding="utf-8")
ts = ts.rename(columns={
    "Day": "ds",
    "NumVehicles": "y"
})

ts["unique_id"] = "tunnel_traffic"
ts = ts[["unique_id", "ds", "y"]]
ts["ds"] = pd.to_datetime(ts["ds"])
ts = ts.sort_values("ds")

end = ts["ds"].max()
start = end - datetime.timedelta(hours=24 * 31 * 25)
ts = ts[ts["ds"] >= start]
print(ts.head())
os.environ["NIXTLA_ID_AS_COL"] = "1"


# --- Define the forecasting models ---
ml_models = {
    "LightGBM": LGBMRegressor(n_estimators=500, verbosity=-1),
    "xgboost": XGBRegressor(),
    "linear_regression": LinearRegression(),
    "lasso": Lasso(),
    "ridge": Ridge(),
}

mlf = MLForecast(
    models=ml_models,
    freq="D",
    lags=list(range(1, 15)),
    date_features=["month", "day", "dayofweek", "week"],
)


# --- Set the backtesting parameters ---

# Window Settings
h = 7
partitions = 10
step_size = h


# Prediction Intervals Settings
n_windows = 3
method = "conformal_distribution"
pi = PredictionIntervals(h=h, n_windows=n_windows, method=method)
levels = [95]


# --- Training models with backtesting ---

bkt_df = mlf.cross_validation(
    df=ts,
    h=h,
    step_size=step_size,
    n_windows=n_windows,
    prediction_intervals=pi,
    level=levels,
)


print(bkt_df.head())



# Create partition mapping from cutoff dates
cutoff = bkt_df["cutoff"].unique()

partitions_mapping = pd.DataFrame({
    "cutoff": cutoff,
    "partition": range(1, len(cutoff) + 1)
})

print(partitions_mapping)

model_label = [
    "LightGBM",
    "xgboost",
    "linear_regression",
    "lasso",
    "ridge"
]

model_name = [
    "LGBMRegressor",
    "XGBRegressor",
    "LinearRegression",
    "Lasso",
    "Ridge"
]

models_mapping = pd.DataFrame({
    "model_label": model_label,
    "model_name": model_name
})

print(models_mapping)


bkt_long = pd.melt(
    bkt_df,
    id_vars=["unique_id", "ds", "cutoff", "y"],
    value_vars=(
        model_label
        + [f"{model}-lo-95" for model in model_label]
        + [f"{model}-hi-95" for model in model_label]
    ),
    var_name="model_label",
    value_name="value",
)


def split_model_confidence(model_name):
    if "-lo-95" in model_name:
        return model_name.replace("-lo-95", ""), "lower"
    elif "-hi-95" in model_name:
        return model_name.replace("-hi-95", ""), "upper"
    else:
        return model_name, "forecast"


bkt_long["model_label"], bkt_long["type"] = zip(
    *bkt_long["model_label"].map(split_model_confidence)
)



bkt = (
    bkt_long
    .merge(partitions_mapping, how="left", on=["cutoff"])
    .pivot(
        index=["unique_id", "ds", "model_label", "partition", "y"],
        columns="type",
        values="value",
    )
    .reset_index()
    .merge(models_mapping, how="left", on=["model_label"])
)


def mape(y, yhat):
    return (abs(y - yhat) / y).mean()


def rmse(y, yhat):
    return ((y - yhat) ** 2).mean() ** 0.5


def coverage(y, lower, upper):
    return ((y >= lower) & (y <= upper)).sum() / len(y)


def score(df):
    mape_score = mape(y=df["y"], yhat=df["forecast"])
    rmse_score = rmse(y=df["y"], yhat=df["forecast"])
    coverage_score = coverage(
        y=df["y"],
        lower=df["lower"],
        upper=df["upper"],
    )

    return pd.Series(
        [mape_score, rmse_score, coverage_score],
        index=["mape", "rmse", "coverage"],
    )


score_df = (
    bkt
    .groupby(
        ["unique_id", "model_label", "model_name", "partition"]
    )[["y", "forecast", "lower", "upper"]]
    .apply(score)
    .reset_index()
)

print(score_df.head())

# print(score_df.shape)

