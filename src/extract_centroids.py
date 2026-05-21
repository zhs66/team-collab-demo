import geopandas as gpd
import pandas as pd
import numpy as np
import zipfile
import os

data_dir = r"C:\Users\32722\Desktop\数据挖掘大作业\data"

# Extract shapefile
zip_path = os.path.join(data_dir, "taxi_zones.zip")
extract_dir = os.path.join(data_dir, "taxi_zones_shapefile")
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(extract_dir)

# Find the .shp file (recursive search)
shp_files = []
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        if f.endswith('.shp'):
            shp_files.append(os.path.join(root, f))
print(f"Found .shp files: {shp_files}")
shp_path = shp_files[0]

# Read shapefile
gdf = gpd.read_file(shp_path)
print(f"\nShapefile columns: {list(gdf.columns)}")
print(f"CRS: {gdf.crs}")

# Project to WGS84 if needed
if gdf.crs and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

# Calculate centroids
gdf['centroid'] = gdf.geometry.centroid
gdf['lon'] = gdf['centroid'].x
gdf['lat'] = gdf['centroid'].y

# Show Manhattan zones
manhattan_zones = gdf[gdf['borough'].str.contains('Manhattan', case=False)]
print(f"\nManhattan zones: {len(manhattan_zones)}")
print(manhattan_zones[['LocationID', 'zone', 'borough', 'lon', 'lat']].head(15).to_string())

# Save centroids
centroids = gdf[['LocationID', 'lon', 'lat']].copy()
centroids.to_csv(os.path.join(data_dir, "taxi_zone_centroids.csv"), index=False)
print(f"\nCentroids saved. Total zones: {len(centroids)}")
print(f"Manhattan lon range: {manhattan_zones['lon'].min():.4f} to {manhattan_zones['lon'].max():.4f}")
print(f"Manhattan lat range: {manhattan_zones['lat'].min():.4f} to {manhattan_zones['lat'].max():.4f}")
