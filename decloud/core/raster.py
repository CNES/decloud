# -*- coding: utf-8 -*-
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
"""Classes for basic raster and array manipulation"""
import osgeo
from osgeo import gdal, osr
import numpy as np

# ---------------------------------------------------- Helpers ---------------------------------------------------------


def gdal_open(filename):
    """
    Read an image as numpy array
    """
    gdal_ds = gdal.Open(filename)
    if gdal_ds is None:
        raise Exception("Unable to open file {}".format(filename))

    return gdal_ds


def read_as_np(filename):
    """
    Read a raster image as numpy array
    """
    gdal_ds = gdal_open(filename)

    return gdal_ds.ReadAsArray()


def set_gdal_cachemax(gdal_cachemax):
    """
    Set GDAL_CACHEMAX
    """
    gdal.SetConfigOption("GDAL_CACHEMAX", gdal_cachemax)


def get_sub_arr(np_arr, patch_location, patch_size, ref_patch_size):
    """
    Get the np.array
    :param np_arr: the numpy array, either edge_stats_fn or clouds_stats_fn
    :param patch_location: patch location (new_elem, y)
    :param patch_size: patch size (single value) it will set the size of the returned sub array, wrt the ref_patch_size
    :param ref_patch_size: reference patch size (single value)
    :return: the sub array
    """
    scale = int(patch_size / ref_patch_size)
    return np_arr[scale * patch_location[1]:scale * (patch_location[1] + 1),
                  scale * patch_location[0]:scale * (patch_location[0] + 1)]


def save_numpy_array_as_raster(ref_fn, np_arr, out_fn, scale=1.0):
    """
    Save a numpy array into a raster file
    :param ref_fn: reference raster (output will have the same proj, geotransform, size)
    :param np_arr: numpy array
    :param out_fn: output filename for the raster
    :param scale: output pixel spacing scaling
    """
    gdal.AllRegister()
    in_ds = gdal_open(ref_fn)
    driver = in_ds.GetDriver()
    out_ds = driver.Create(out_fn, np_arr.shape[1], np_arr.shape[0], 1, gdal.GDT_Int32)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np.transpose(np_arr), 0, 0)
    out_band.FlushCache()

    geotransform = in_ds.GetGeoTransform()

    # Update if we want to change pixel spacing
    if scale != 1.0:
        geotransform = list(geotransform)
        geotransform[1] *= scale
        geotransform[5] *= scale
        geotransform = tuple(geotransform)

    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(in_ds.GetProjection())


def convert_to_4326(coordinates, gdal_ds):
    """
    Convert some coordinates to 4326
    :param coordinates: (x, y) coordinates, expressed in the coordinate system of the dataset `gdal_ds`
    :param gdal_ds: gdal dataset
    :return:
    """
    # Get the coordinate system of the dataset
    initial_crs = osr.SpatialReference()
    initial_crs.ImportFromWkt(gdal_ds.GetProjectionRef())

    # Set up the coordinate reference system, WGS84
    crs_4326 = osr.SpatialReference()
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        crs_4326.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    crs_4326.ImportFromEPSG(4326)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(initial_crs, crs_4326)

    # Transform the coordinates in lat long
    lon, lat, _ = transform.TransformPoint(*coordinates)

    return lon, lat
