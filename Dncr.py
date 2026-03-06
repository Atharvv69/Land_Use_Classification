import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box

Delhi_NCR = gpd.read_file("Datasets/delhi_ncr_region.geojson")
print("Old CRS: ",Delhi_NCR.crs)
proj_Delhi_NCR = Delhi_NCR.to_crs(epsg=32643)
print("New CRS: ",proj_Delhi_NCR.crs)

fig, region = plt.subplots(figsize = (10,10), dpi = 150)
proj_Delhi_NCR.plot(ax=region, color='lightblue')
proj_Delhi_NCR.boundary.plot(ax=region, color='black', linewidth = 0.5)
region.set_aspect('equal', adjustable='box')

xmin, ymin, xmax, ymax = proj_Delhi_NCR.total_bounds
grid_size = 60000

grid_cells = []

for x in np.arange(xmin, xmax, grid_size):
    for y in np.arange(ymin, ymax, grid_size):
        cell = box(
            x,y,
            x+grid_size,
            y+grid_size
        )
        grid_cells.append(cell)

Grid = gpd.GeoDataFrame(
    {'geometry': grid_cells},
    crs = proj_Delhi_NCR.crs
)

region.set_title("Delhi NCR – 60 km Grid Map", fontsize=14, pad=15)
region.set_xlabel("Easting (km)")
region.set_ylabel("Northing (km)")

x_ticks = np.arange(xmin, xmax+grid_size, grid_size)
y_ticks = np.arange(ymin, ymax+grid_size, grid_size)
region.set_xticks(x_ticks)
region.set_yticks(y_ticks)
region.set_xticklabels((x_ticks/1000).astype(int))
region.set_yticklabels((y_ticks/1000).astype(int))

grid = gpd.overlay(Grid, proj_Delhi_NCR, how='union')
grid.boundary.plot(ax=region, color='black',linewidth = 0.25, linestyle = '--', alpha = 0.3)
plt.show()

