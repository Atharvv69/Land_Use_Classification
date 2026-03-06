import os
import numpy as np
import rasterio
from collections import Counter
from rasterio.windows import Window

raster_path = "Datasets/worldcover_bbox_delhi_ncr_2021.tif"
valid_folder = "Datasets/Valid_Images"

labels = []

with rasterio.open(raster_path) as src:

    height = src.height
    width = src.width

    for img in os.listdir(valid_folder):

        if not img.endswith(".png"):
            continue

        name = img.replace(".png","")
        lat, lon = map(float, name.split("_"))

        row, col = src.index(lon, lat)

        
        if row-64 < 0 or col-64 < 0 or row+64 >= height or col+64 >= width:
            continue

        window = Window(col-64, row-64, 128, 128)
        patch = src.read(1, window=window)

        values = patch.flatten()
        values = values[values != 0]   

        if len(values) == 0:
            continue

        dominant_class = Counter(values).most_common(1)[0][0]

        labels.append((img, dominant_class))

if __name__ == "__main__":

    print("Total labeled images:", len(labels))
    class_counts = Counter([label for _, label in labels])
    print(class_counts)