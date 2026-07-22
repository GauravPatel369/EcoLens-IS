import rasterio
from rasterio.merge import merge
import glob

tiles = glob.glob("geo_data/dem_tiles/*.tif")
print(f"found {len(tiles)} tiles")

srcs = [rasterio.open(t) for t in tiles]
mosaic_arr, mosaic_transform = merge(srcs)

profile = srcs[0].profile.copy()
profile.update(height=mosaic_arr.shape[1], width=mosaic_arr.shape[2], transform=mosaic_transform)

with rasterio.open("geo_data/dem_mosaic.vrt", "w", **profile) as dst:
    dst.write(mosaic_arr)

for s in srcs:
    s.close()
print("done")