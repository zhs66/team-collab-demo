"""
成员A：数据预处理脚本
NYC Yellow Taxi 2023年1月 -> train/val/test .npy 文件
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. 配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR  = os.path.join(DATA_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

LON_MIN, LON_MAX = -74.02, -73.92
LAT_MIN, LAT_MAX =  40.70,  40.88
GRID_SIZE = 20
HOURS_IN_MONTH = 31 * 24
RECENT_HOURS = 3
DAILY_DAYS = 3

# ============================================================
# 1. 读取原始数据
# ============================================================
print("[1/7] 读取数据...")
df = pd.read_parquet(os.path.join(DATA_DIR, "raw", "yellow_tripdata_2023-01.parquet"))
centroids = pd.read_csv(os.path.join(DATA_DIR, "raw", "taxi_zone_centroids.csv"))
with open(os.path.join(DATA_DIR, "raw", "nyc_weather_2023-01.json"), "r") as f:
    weather_raw = json.load(f)

print(f"  出租车: {len(df):,} 条")
print(f"  天气:   {len(weather_raw['hourly']['time'])} 小时")

# ============================================================
# 2. 关联经纬度、筛选曼哈顿、网格化
# ============================================================
print("\n[2/7] 关联经纬度并筛选曼哈顿范围...")

centroids_pu = centroids.rename(columns={"lon": "pickup_lon", "lat": "pickup_lat"})
centroids_do = centroids.rename(columns={"lon": "dropoff_lon", "lat": "dropoff_lat"})
df = df.merge(centroids_pu[["LocationID", "pickup_lon", "pickup_lat"]],
              left_on="PULocationID", right_on="LocationID", how="left")
df = df.merge(centroids_do[["LocationID", "dropoff_lon", "dropoff_lat"]],
              left_on="DOLocationID", right_on="LocationID", how="left")

mask = (df["pickup_lon"].between(LON_MIN, LON_MAX) &
        df["pickup_lat"].between(LAT_MIN, LAT_MAX))
df = df[mask].copy()
print(f"  曼哈顿范围内: {len(df):,} 条")

def lonlat_to_grid(lon, lat):
    gx = np.floor((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_SIZE).astype(int).clip(0, GRID_SIZE - 1)
    gy = np.floor((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_SIZE).astype(int).clip(0, GRID_SIZE - 1)
    return gx, gy

df["pickup_time"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["hour_slot"] = df["pickup_time"].dt.strftime("%Y-%m-%d %H:00")
df["hour_idx"] = ((df["pickup_time"] - pd.Timestamp("2023-01-01")).dt.total_seconds() // 3600).astype(int)

df["gx_pu"], df["gy_pu"] = lonlat_to_grid(df["pickup_lon"].values, df["pickup_lat"].values)
df["gx_do"], df["gy_do"] = lonlat_to_grid(df["dropoff_lon"].values, df["dropoff_lat"].values)

# ============================================================
# 3. 聚合为热力图 (744, 20, 20, 2)
# ============================================================
print("\n[3/7] 聚合为小时级热力图...")

heatmap = np.zeros((HOURS_IN_MONTH, GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)

pu_counts = df.groupby(["hour_idx", "gx_pu", "gy_pu"]).size()
for (h, gx, gy), cnt in pu_counts.items():
    if 0 <= h < HOURS_IN_MONTH:
        heatmap[h, gx, gy, 0] = cnt

do_counts = df.groupby(["hour_idx", "gx_do", "gy_do"]).size()
for (h, gx, gy), cnt in do_counts.items():
    if 0 <= h < HOURS_IN_MONTH:
        heatmap[h, gx, gy, 1] = cnt

print(f"  热力图形状: {heatmap.shape}")
print(f"  非零网格比例: {(heatmap.sum(axis=-1) > 0).mean()*100:.1f}%")

# ============================================================
# 4. 构造外部特征
# ============================================================
print("\n[4/7] 构造外部特征...")

weather_times = pd.to_datetime(weather_raw["hourly"]["time"])
weather_df = pd.DataFrame({
    "time": weather_times,
    "temperature": weather_raw["hourly"]["temperature_2m"],
    "precipitation": weather_raw["hourly"]["precipitation"],
})
weather_df["hour_idx"] = ((weather_df["time"] - pd.Timestamp("2023-01-01")).dt.total_seconds() // 3600).astype(int)

ext_features = np.zeros((HOURS_IN_MONTH, 5), dtype=np.float32)
for h in range(HOURS_IN_MONTH):
    ts = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h)
    w = weather_df[weather_df["hour_idx"] == h]
    ext_features[h, 0] = w["temperature"].values[0] if len(w) > 0 else ext_features[h - 1, 0]
    ext_features[h, 1] = w["precipitation"].values[0] if len(w) > 0 else 0.0
    ext_features[h, 2] = ts.hour
    ext_features[h, 3] = ts.dayofweek
    ext_features[h, 4] = 1 if ts.month == 1 and ts.day == 1 else 0

print(f"  外部特征形状: {ext_features.shape}")
print(f"  温度范围: {ext_features[:, 0].min():.1f} ~ {ext_features[:, 0].max():.1f} C")
print(f"  降水量范围: {ext_features[:, 1].min():.1f} ~ {ext_features[:, 1].max():.1f} mm")

# ============================================================
# 5. 滑动窗口构造样本
# ============================================================
print("\n[5/7] 滑动窗口构造样本...")

MIN_HISTORY = DAILY_DAYS * 24
samples = []

for t in range(MIN_HISTORY, HOURS_IN_MONTH):
    recent = np.concatenate([
        heatmap[t-3, :, :, :],
        heatmap[t-2, :, :, :],
        heatmap[t-1, :, :, :],
    ], axis=-1)

    daily = np.concatenate([
        heatmap[t - 24, :, :, :],
        heatmap[t - 48, :, :, :],
        heatmap[t - 72, :, :, :],
    ], axis=-1)

    samples.append({
        "hour_idx": t,
        "X_recent": recent,
        "X_daily": daily,
        "X_ext": ext_features[t],
        "y": heatmap[t, :, :, :],
    })

print(f"  总样本数: {len(samples)}")

# ============================================================
# 6. 数据划分 + 标准化
# ============================================================
print("\n[6/7] 数据划分与标准化...")

TRAIN_END = 22 * 24
VAL_END   = 26 * 24

train_samples = [s for s in samples if s["hour_idx"] < TRAIN_END]
val_samples   = [s for s in samples if TRAIN_END <= s["hour_idx"] < VAL_END]
test_samples  = [s for s in samples if s["hour_idx"] >= VAL_END]

def to_array(sample_list):
    return {
        "X_recent": np.stack([s["X_recent"] for s in sample_list]),
        "X_daily":  np.stack([s["X_daily"]  for s in sample_list]),
        "X_ext":    np.stack([s["X_ext"]    for s in sample_list]),
        "y":        np.stack([s["y"]        for s in sample_list]),
    }

train_data = to_array(train_samples)
val_data   = to_array(val_samples)
test_data  = to_array(test_samples)

print(f"  训练集: {len(train_samples)} 样本")
print(f"  验证集: {len(val_samples)} 样本")
print(f"  测试集: {len(test_samples)} 样本")

# 标准化：log(1+x) -> Z-score（基于训练集统计）
print("\n  对需求值做 log(1+x) 变换 + Z-score 标准化...")

def log1p_normalize(data, stats=None):
    log_data = np.log1p(data)
    if stats is None:
        mean = log_data.mean()
        std = log_data.std()
        stats = {"mean": mean, "std": std}
    normalized = (log_data - stats["mean"]) / (stats["std"] + 1e-8)
    return normalized, stats

all_vals = np.concatenate([
    train_data["y"].ravel(),
    train_data["X_recent"].ravel(),
    train_data["X_daily"].ravel()
])
all_log = np.log1p(all_vals)
y_stats = {"mean": float(all_log.mean()), "std": float(all_log.std())}
print(f"  全局 log(1+x) 均值: {y_stats['mean']:.4f}, 标准差: {y_stats['std']:.4f}")

train_data["y"], _        = log1p_normalize(train_data["y"], y_stats)
train_data["X_recent"], _ = log1p_normalize(train_data["X_recent"], y_stats)
train_data["X_daily"], _  = log1p_normalize(train_data["X_daily"], y_stats)

val_data["y"], _   = log1p_normalize(val_data["y"], y_stats)
val_data["X_recent"], _ = log1p_normalize(val_data["X_recent"], y_stats)
val_data["X_daily"], _  = log1p_normalize(val_data["X_daily"], y_stats)

test_data["y"], _   = log1p_normalize(test_data["y"], y_stats)
test_data["X_recent"], _ = log1p_normalize(test_data["X_recent"], y_stats)
test_data["X_daily"], _  = log1p_normalize(test_data["X_daily"], y_stats)

# ============================================================
# 7. 输出 .npy 文件
# ============================================================
print("\n[7/7] 保存 .npy 文件...")

for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
    np.save(os.path.join(OUT_DIR, f"X_recent_{name}.npy"), data["X_recent"])
    np.save(os.path.join(OUT_DIR, f"X_daily_{name}.npy"),  data["X_daily"])
    np.save(os.path.join(OUT_DIR, f"X_ext_{name}.npy"),    data["X_ext"])
    np.save(os.path.join(OUT_DIR, f"y_{name}.npy"),        data["y"])
    print(f"  {name}: X_recent {data['X_recent'].shape}, X_daily {data['X_daily'].shape}, "
          f"X_ext {data['X_ext'].shape}, y {data['y'].shape}")

np.savez(os.path.join(OUT_DIR, "scaler.npz"),
         log_mean=y_stats["mean"], log_std=y_stats["std"])

with open(os.path.join(OUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
    f.write(f"""预处理数据说明
==============
成员A：张皓淞  学号：202411079482

数据来源：NYC Yellow Taxi 2023年1月 + Open-Meteo 天气数据

文件列表：
  X_recent_{{split}}.npy  - 近期分支 (N, 20, 20, 6)
  X_daily_{{split}}.npy   - 日周期分支 (N, 20, 20, 6)
  X_ext_{{split}}.npy     - 外部特征 (N, 5)
  y_{{split}}.npy         - 预测目标 (N, 20, 20, 2)
  scaler.npz              - 标准化参数

数据划分：前22天训练 / 中4天验证 / 后5天测试
样本数：训练 {len(train_samples)} / 验证 {len(val_samples)} / 测试 {len(test_samples)}
""")

print("\n" + "=" * 60)
print("Done! Output:", OUT_DIR)
print("=" * 60)
