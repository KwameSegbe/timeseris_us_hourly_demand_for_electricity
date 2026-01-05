import pandas as pd
import requests 
import os 
import statsforecast as sf
import mlforecast as mlf

api_key = os.getenv("EIA_API_KEY")
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
# print(data)

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

ts = df[["period", "value"]]
print(ts.head())
ts['period'] = pd.to_datetime(ts['period'])
ts = ts.sort_values('period')
print(ts.info())