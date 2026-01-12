import os
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
