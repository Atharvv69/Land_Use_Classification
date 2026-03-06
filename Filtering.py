import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import shutil
import os
from shapely.geometry import Point

Delhi_NCR = gpd.read_file("Datasets/delhi_ncr_region.geojson")
Sentinel_RGB = "Datasets/rgb"

File_Names =[
    f for f in os.listdir(Sentinel_RGB)
    if f.endswith('.png')
]

Data = []

for f in File_Names:
    name = f.replace(".png","")
    lat, lon = map(float,name.split("_"))

    Data.append({
        "filename" : f,
        "latitude": lat,
        "longitude": lon
    })
Geometry = [
    Point(lon, lat)
    for lat, lon in zip(
        [dict["latitude"] for dict in Data],
        [dict["longitude"] for dict in Data]
    )
]

points = gpd.GeoDataFrame(Data, geometry=Geometry, crs=Delhi_NCR.crs)

Filtered = gpd.sjoin(
    points,
    Delhi_NCR,
    predicate="within",
    how="inner"
)

Filtered_Images = Filtered["filename"].tolist()

Sentinel_RGB = "Datasets/rgb"
valid_folder = "Datasets/Valid_Images"
invalid_folder = "Datasets/Invalid_Images"

valid_count = 0
invalid_count = 0

for img in os.listdir(Sentinel_RGB):
    if not img.endswith(".png"):
        continue
    src = os.path.join(Sentinel_RGB, img)

    if img in Filtered_Images:
        dst = os.path.join(valid_folder, img)
        valid_count+=1
    else:
        dst = os.path.join(invalid_folder, img)
        invalid_count+=1

    shutil.copy(src, dst)

print("Total Images: ", valid_count+invalid_count)
print("Valid Images: ", valid_count)
print("Invalid Images: ", invalid_count)


