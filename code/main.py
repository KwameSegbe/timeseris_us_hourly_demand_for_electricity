import pandas as pd
import requests 
import os 
import statsforecast as sf
import mlforecast as mlf
from datetime import datetime, timedelta
from utilsforecast.plotting import plot_series
from statsforecast import StatsForecast

from statsforecast.models import(
    AutoARIMA, 
    MSTL, 
    Naive, 
    SeasonalNaive, 
    Theta,
    DynamicOptimizedTheta,
    AutoARIMA,
    HoltWinters)


# EIA API configuration
api_key = os.getenv("EIA_API_KEY")
if not api_key:
    raise RuntimeError("EIA_API_KEY not set in environment variables")


api_url = "https://api.eia.gov/v2/"
api_path = "electricity/rto/region-data/data"

# Ensure data directory exists
DATA_DIR = r"C:\Users\HP\Documents\timeseries_analysis\data"
os.makedirs(DATA_DIR, exist_ok=True)

params = {
    "api_key": api_key,
    "data[0]": "value"
}


response = requests.get(api_url + api_path, params=params)
data = response.json()
print(data)

df = pd.DataFrame(data['response']['data'])
# print(df.head(5))

#Validating resposnse
if "response" not in data:
    raise ValueError("Response key not found in the API response")


# Writing to CSV
file_path = os.path.join(DATA_DIR, "eia_region_data.csv")
df.to_csv(file_path, index=False)

# Load data for analysis
df = pd.read_csv(file_path)
# df.head()


#Data Preparation and Preprocessing
ts = df[["period", "value"]]
print(ts.head())
ts['period'] = pd.to_datetime(ts['period'])
ts = ts.sort_values('period')
# print(ts.info())
end = ts['period'].max()

ts = ts.rename(columns={"period":"ds", "value":"y"})    
ts['unique_id'] = 1  # Single time series identifier
ts = ts[["unique_id", "ds", "y"]]


ts["ds"] = pd.to_datetime(ts["ds"])
ts = ts.sort_values("ds")
ts = ts[["unique_id", "ds", "y"]]

# Tell Nixtla / utilsforecast that unique_id is a column (not an index)
os.environ["NIXTLA_ID_AS_COL"] = "1"

# print(ts.head())

# Leave last 72 hours as test data
test_length = 24

# Define end of series
end = ts["ds"].max()

# Compute train cutoff
train_end = end - pd.Timedelta(hours=test_length)

# Split data
train = ts[ts["ds"] <= train_end]
test  = ts[ts["ds"] > train_end]

# print("Train rows:", len(train))
# print("Test rows:", len(test))

# plot_series(train, engine="plotly")
# plot_series(test, engine="plotly")

# Model Training and Forecasting

auto_arima = AutoARIMA(season_length=24)
s_naive = SeasonalNaive(season_length=24)
theta   = Theta(season_length=24)

mstl = MSTL(season_length=[24,168],
            trend_forecaster=AutoARIMA(),
            alias="MSTL_AutoARIMA")

mstl2 = MSTL(season_length=[24,168],
            trend_forecaster=HoltWinters(),
            alias="MSTL_HoltWinters")

# Initialize StatsForecast with models
statmodels = [auto_arima, s_naive, theta, mstl, mstl2]

# Instantiate StatsForecast
sf = StatsForecast(
    models=statmodels,
    freq="h",
    n_jobs=-1,
    fallback_model=AutoARIMA()
)

#create mlforecast forecaster
forecast_stats = sf.forecast(df=train, h=test_length, level =[95])

p = plot_series(test, forecast_stats, engine="plotly", level=[95])
p.update_layout(height=400)