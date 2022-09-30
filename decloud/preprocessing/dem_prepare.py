#!/usr/bin/env python3
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
"""Prepare the DEM image"""
import argparse
import logging
import elevation
from osgeo import gdal
import decloud.preprocessing.constants as constants
import otbApplication as otb
from decloud.core import system


def clean(params):
    """
    Remove temporary files
    """
    elevation.datasource.distclean(cache_dir=params.tmp)


def get_bounds(raster):
    """
    Return the bounding box in physical units
    """

    ulx, xres, _, uly, _, yres = raster.GetGeoTransform()
    lrx = ulx + (raster.RasterXSize * xres)
    lry = uly + (raster.RasterYSize * yres)

    return [ulx, lry, lrx, uly]


def get_bounds_wgs84(params):
    """
    Return the bounding box in WGS84 CRS
    """

    option_clip = gdal.WarpOptions(format="Mem", dstSRS='EPSG:4326')

    raster_reproj = gdal.Warp("", params.reference, options=option_clip)

    bounds = get_bounds(raster_reproj)
    logging.info("Bounds: %s", bounds)
    return elevation.datasource.build_bounds(bounds=bounds, margin=params.margin)


def download_dem(params, output):
    """
    Download the DEM
    :param params: parameters
    :param output: output
    """
    bounds = get_bounds_wgs84(params)
    logging.info("Seed")
    datasource_root = elevation.seed(cache_dir=params.tmp, product=elevation.DEFAULT_PRODUCT, bounds=bounds)
    logging.info("Clip")
    elevation.datasource.do_clip(path=datasource_root, bounds=bounds, output=output)


def superimpose(params, tmp_filename):
    """
    Re-sample the DEM
    :param params: parameters
    :param tmp_filename: filename
    """

    logging.info("Superimpose")

    out_fn = params.output
    out_fn += "?&gdal:co:COMPRESS=DEFLATE"
    out_fn += "&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue={}".format(4 * params.tilesize)
    out_fn += "&gdal:co:TILED=YES&gdal:co:BLOCKXSIZE={}&gdal:co:BLOCKYSIZE={}".format(params.tilesize, params.tilesize)

    app = otb.Registry.CreateApplication("Superimpose")
    app.SetParameterString("inr", params.reference)
    app.SetParameterString("inm", tmp_filename)
    app.SetParameterString("out", out_fn)
    app.SetParameterOutputImagePixelType("out", otb.ImagePixelType_int16)
    app.ExecuteAndWriteOutput()


def main(args):
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Download and pre-process DEM files")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True, help="Output geotiff file")
    parser.add_argument("--tmp", required=True, help="Temporary directory to download/pre-process the files")
    parser.add_argument("--tilesize", type=int, default=constants.PATCHSIZE_REF)
    parser.add_argument("--margin", type=str, default="0")
    parser.add_argument("--clean", action='store_false')
    params = parser.parse_args(args)

    elevation.CACHE_DIR = params.tmp

    if params.clean:
        logging.info("Cleaning cache")
        clean(params)

    tmp_filename = "{}/{}.tif".format(params.tmp, system.new_bname(params.output, "elev_tmp"))
    download_dem(params, output=tmp_filename)
    superimpose(params, tmp_filename=tmp_filename)


if __name__ == "__main__":
    system.run_and_terminate(main)
