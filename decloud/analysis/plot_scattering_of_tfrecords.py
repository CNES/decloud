"""
Copyright (c) 2020-2022 INRAE

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import os
import sys
import numpy as np
import json
from osgeo import gdal, osr
import osgeo

tfrecords_dir = sys.argv[1]

# X and Y coordinates of upper left coordinates of patches
xs = []
ys = []

# Fill xs and ys
for root, subdirs, files in os.walk(tfrecords_dir):
    for file in files:
        if file.endswith('geojson'):
            with open(os.path.join(root, file)) as f:
                data = json.load(f)

            # reading all the patches position of the TFRecord
            for x in data["features"]:
                ULX, ULY = x['geometry']['coordinates'][0][0]
                xs.append(ULX)
                ys.append(ULY)

# Set up the coordinate reference system, WGS84
crs_4326 = osr.SpatialReference()
if int(osgeo.__version__[0]) >= 3:
    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
    crs_4326.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
crs_4326.ImportFromEPSG(4326)

# Target CRS
target_crs = osr.SpatialReference()
target_crs.ImportFromEPSG(32630)

# create a transform object to convert between coordinate systems
transform = osr.CoordinateTransformation(crs_4326, target_crs)

# Transform the coordinates from lat long
coordinates_projected = [transform.TransformPoint(*coordinates)[:2] for coordinates in zip(xs, ys)]
x_projected = [coord[0] for coord in coordinates_projected]
y_projected = [coord[1] for coord in coordinates_projected]

output_resolution = 256 * 10  # size of the patches in meters

upper_left_x = min(x_projected)
upper_left_y = max(y_projected)
# coordinates correspond to the upper left of the patch, hence adding the size of the patch
lower_right_x = max(x_projected) + output_resolution
lower_right_y = min(y_projected) - output_resolution
print(upper_left_x, upper_left_y, lower_right_x, lower_right_y)

ncols = int(abs(upper_left_x - lower_right_x) / output_resolution)
nrows = int(abs(upper_left_y - lower_right_y) / output_resolution)
print(ncols, nrows)

heatmap, xedges, yedges = np.histogram2d(x_projected, y_projected, bins=(ncols, nrows))

heatmap = heatmap.T
# Histogram is generated using a positive y-spacing whereas for CRS it is negative, thus we flip the array
heatmap = np.flipud(heatmap)

geotransform = (upper_left_x, output_resolution, 0, upper_left_y, 0, -output_resolution)

output_raster = gdal.GetDriverByName('GTiff').Create(
    '/tmp/histogram_2d_{}.tif'.format(os.path.basename(os.path.normpath(tfrecords_dir))), ncols, nrows, 1,
    gdal.GDT_Float32)
output_raster.SetGeoTransform(geotransform)
output_raster.SetProjection(target_crs.ExportToWkt())
# writing the raster without CRS
output_raster.GetRasterBand(1).WriteArray(heatmap)  # Writes my array to the raster

output_raster.FlushCache()
