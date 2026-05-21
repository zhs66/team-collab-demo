"""演示：如何把 LocationID 映射为经纬度"""
import pandas as pd

data_dir = r"C:\Users\32722\Desktop\数据挖掘大作业\data"

# 1. 读数据
df = pd.read_parquet(f"{data_dir}/yellow_tripdata_2023-01.parquet")
centroids = pd.read_csv(f"{data_dir}/taxi_zone_centroids.csv")

print("=" * 60)
print("出租车原始数据（前3条）")
print("=" * 60)
cols = ["tpep_pickup_datetime", "PULocationID", "DOLocationID",
        "passenger_count", "trip_distance"]
print(df[cols].head(3).to_string())

print(f"\n当前格式：只知道区域编号（PULocationID={df['PULocationID'].iloc[0]}），不知道经纬度")

# 2. 把 centroids 的列改个名字，避免 JOIN 后列名冲突
centroids_pu = centroids.rename(columns={"lon": "pickup_lon", "lat": "pickup_lat"})
centroids_do = centroids.rename(columns={"lon": "dropoff_lon", "lat": "dropoff_lat"})

# 3. 两次 LEFT JOIN：上车位置 + 下车位置
df = df.merge(centroids_pu, left_on="PULocationID", right_on="LocationID", how="left")
df = df.merge(centroids_do, left_on="DOLocationID", right_on="LocationID", how="left",
              suffixes=("_pu", "_do"))

print("\n" + "=" * 60)
print("关联后的数据（有了经纬度！）")
print("=" * 60)
print(df[["tpep_pickup_datetime", "PULocationID", "pickup_lon", "pickup_lat",
          "DOLocationID", "dropoff_lon", "dropoff_lat"]].head(3).to_string())

# 4. 筛选曼哈顿范围
manhattan_lon = (-74.02, -73.92)
manhattan_lat = (40.70, 40.88)

mask = (
    (df["pickup_lon"].between(*manhattan_lon)) &
    (df["pickup_lat"].between(*manhattan_lat))
)
df_manhattan = df[mask]

print(f"\n原始记录数: {len(df):,}")
print(f"曼哈顿范围内上车记录: {len(df_manhattan):,}")
print(f"筛选后保留比例: {len(df_manhattan)/len(df)*100:.1f}%")
