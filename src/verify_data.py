import pandas as pd
import json

# Verify taxi data
print("=" * 60)
print("NYC Yellow Taxi Data - Jan 2023")
print("=" * 60)
df = pd.read_parquet(r"C:\Users\32722\Desktop\数据挖掘大作业\data\yellow_tripdata_2023-01.parquet")
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nSample rows:")
print(df.head(3)[["tpep_pickup_datetime", "passenger_count", "trip_distance"]].to_string())
print(f"\nPickup datetime range: {df['tpep_pickup_datetime'].min()} to {df['tpep_pickup_datetime'].max()}")

# Verify weather data
print("\n" + "=" * 60)
print("Weather Data")
print("=" * 60)
with open(r"C:\Users\32722\Desktop\数据挖掘大作业\data\nyc_weather_2023-01.json", "r") as f:
    weather = json.load(f)
print(f"Keys: {list(weather.keys())}")
print(f"Hourly data keys: {list(weather.get('hourly', {}).keys())}")
print(f"Number of hourly records: {len(weather.get('hourly', {}).get('time', []))}")
print(f"Sample times: {weather['hourly']['time'][:3]}")
print(f"Sample temps: {weather['hourly']['temperature_2m'][:3]}")
print(f"Sample precip: {weather['hourly']['precipitation'][:3]}")
