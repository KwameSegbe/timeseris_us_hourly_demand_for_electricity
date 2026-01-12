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



def prepare_tunnel_ts(ts:pd.DataFrame)->pd.DataFrame:
    """
    Prepare tunnel time series data for forecasting.

    This function:
    - Adds a constant `unique_id` column required by Nixtla / MLForecast
    - Keeps only the required columns: unique_id, ds (timestamp), y (target)
    - Ensures the timestamp column is in datetime format
    - Sorts the data chronologically
    - Restricts the dataset to a recent rolling window (last ~25 months)
    - Sets the Nixtla environment variable to treat `unique_id` as a column

    Parameters
    ----------
    ts : pd.DataFrame
        Input time series data containing at least:
        - `ds`: timestamp column
        - `y`: target variable column

    Returns
    -------
    pd.DataFrame
        Cleaned and filtered time series ready for backtesting and forecasting."""
        
    ts["unique_id"] = "tunnel_traffic"
    ts = ts[["unique_id", "ds", "y"]]
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.sort_values("ds")

    end = ts["ds"].max()
    start = end - datetime.timedelta(hours=24 * 31 * 25)
    ts = ts[ts["ds"] >= start]

    print(ts.head())

    os.environ["NIXTLA_ID_AS_COL"] = "1"

    return ts

#Call the function to prepare the data
ts = prepare_tunnel_ts(ts)


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


def split_model_confidence(model_name:str)->tuple:
    """
    Split a model output column name into its base model label and value type.

    This function is used when reshaping backtesting results that contain
    point forecasts and prediction interval bounds in column names.

    Examples
    --------
    "ridge"          -> ("ridge", "forecast")
    "ridge-lo-95"    -> ("ridge", "lower")
    "ridge-hi-95"    -> ("ridge", "upper")

    Parameters
    ----------
    model_name : str
        Column name from the backtesting output. This may represent:
        - a point forecast (e.g., "ridge")
        - a lower prediction bound (e.g., "ridge-lo-95")
        - an upper prediction bound (e.g., "ridge-hi-95")

    Returns
    -------
    tuple (str, str)
        - Base model label (e.g., "ridge")
        - Type of value:
            * "forecast" for point predictions
            * "lower" for lower interval bounds
            * "upper" for upper interval bounds
    """
    if "-lo-95" in model_name:
        return model_name.replace("-lo-95", ""), "lower"
    elif "-hi-95" in model_name:
        return model_name.replace("-hi-95", ""), "upper"
    else:
        return model_name, "forecast"


def build_backtest_tidy_table(
    bkt_long: pd.DataFrame,
    partitions_mapping: pd.DataFrame,
    models_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert long-form backtest output into a tidy table with forecast + interval columns.

    - Applies split_model_confidence to label each row as forecast/lower/upper
    - Adds partition numbers via cutoff mapping
    - Pivots values into columns: forecast, lower, upper
    - Adds model metadata (model_name)
    """
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

    return bkt

#call the function to build tidy backtest table
bkt = build_backtest_tidy_table(bkt_long, partitions_mapping, models_mapping)


# def mape(y, yhat):
#     return (abs(y - yhat) / y).mean()
def mape(y, yhat):
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Measures the average absolute error as a percentage of the actual values.
    Lower values indicate better forecasting accuracy.

    Parameters
    ----------
    y : array-like
        True observed values.
    yhat : array-like
        Predicted values.

    Returns
    -------
    float
        Mean Absolute Percentage Error.
    """
    return (abs(y - yhat) / y).mean()


def rmse(y, yhat):
    """
    Compute Root Mean Squared Error (RMSE).

    Penalizes larger errors more heavily than MAE and is expressed
    in the same units as the target variable.

    Parameters
    ----------
    y : array-like
        True observed values.
    yhat : array-like
        Predicted values.

    Returns
    -------
    float
        Root Mean Squared Error.
    """
    return ((y - yhat) ** 2).mean() ** 0.5


def coverage(y, lower, upper):
    """
    Compute empirical coverage of prediction intervals.

    Measures the proportion of true values that fall within the
    predicted lower and upper bounds.

    Parameters
    ----------
    y : array-like
        True observed values.
    lower : array-like
        Lower bound of the prediction interval.
    upper : array-like
        Upper bound of the prediction interval.

    Returns
    -------
    float
        Coverage ratio (between 0 and 1).
    """
    return ((y >= lower) & (y <= upper)).sum() / len(y)


def score(df):
    """
    Compute forecasting performance metrics for a backtest partition.

    Aggregates point forecast accuracy and interval quality metrics
    into a single Series for evaluation.

    Metrics computed:
    - MAPE: point forecast accuracy
    - RMSE: scale-dependent error
    - Coverage: prediction interval calibration

    Parameters
    ----------
    df : pd.DataFrame
        Backtesting results for a single model and partition.
        Expected columns:
        - 'y': true values
        - 'forecast': point predictions
        - 'lower': lower interval bound
        - 'upper': upper interval bound

    Returns
    -------
    pd.Series
        Series containing MAPE, RMSE, and coverage metrics.
    """
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
    
    
# Compute scores for each model and partition
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

