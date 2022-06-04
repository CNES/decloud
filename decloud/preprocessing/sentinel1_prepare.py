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
"""Pre-process one Sentinel-1 image"""
import os
import logging
import argparse
import otbApplication
from decloud.core import system, tile_io
import decloud.preprocessing.constants as constants

system.basic_logging_init()


def _bm(input_fn):
    """ BandMath for SAR normalization """
    bm = otbApplication.Registry.CreateApplication("BandMath")
    bm.SetParameterStringList("il", [input_fn])
    bm.SetParameterString("exp", constants.S1_NORMALIZATION_BM_EXPR)
    bm.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_uint16)
    bm.Execute()
    return bm


# Input filenames
def _check_file_exists(fn):
    if not system.file_exists(fn):
        logging.fatal("File %s not found!", fn)
        system.terminate()


def main(args):
    """ Main """
    # Arguments
    parser = argparse.ArgumentParser(description="S1 to tiled geotiff",
                                     usage="The program stacks Sentinel-1 VH/VV channels, performs linear to dB, then "
                                           "performs a linear stretching to fit the uint16 encoding used for the "
                                           "output raster")
    parser.add_argument("--input_s1_vh", required=True,
                        help="Input Sentinel-1 VH channel (in sigma nought, linear scale)")
    parser.add_argument("--out_s1_dir", required=True, help="Directory for processed Sentinel-1 tile")
    params = parser.parse_args(args)

    logging.info("Create or use the following output directory: %s", params.out_s1_dir)
    system.mkdir(params.out_s1_dir)

    vh_fn = params.input_s1_vh
    vv_fn = vh_fn.replace("_vh_", "_vv_")
    _check_file_exists(vh_fn)
    _check_file_exists(vv_fn)

    # Output filename
    out_fn = vh_fn
    out_fn = system.basename(out_fn)
    out_fn = out_fn.replace("_vh_", "_vvvh_")
    out_fn = "{}_{}".format(out_fn[:out_fn.rfind(".")], constants.SUFFIX_S1)
    out_fn += ".tif?&gdal:co:COMPRESS=DEFLATE&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue={" \
              "}&gdal:co:TILED=YES&gdal:co:BLOCKXSIZE={}&gdal:co:BLOCKYSIZE={}".format(4 * constants.PATCHSIZE_REF,
                                                                                       constants.PATCHSIZE_REF,
                                                                                       constants.PATCHSIZE_REF)

    # Calibration + concatenation + tiling/compression
    out_fn = os.path.join(params.out_s1_dir, out_fn)
    if system.is_complete(out_fn):
        logging.info("File %s already exists. Skipping.", system.remove_ext_filename(out_fn))
    else:
        bm_vv = _bm(vv_fn)
        bm_vh = _bm(vh_fn)
        conc = otbApplication.Registry.CreateApplication("ConcatenateImages")
        conc.AddImageToParameterInputImageList("il", bm_vv.GetParameterOutputImage("out"))
        conc.AddImageToParameterInputImageList("il", bm_vh.GetParameterOutputImage("out"))
        conc.SetParameterString("out", out_fn)
        conc.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_uint16)
        conc.ExecuteAndWriteOutput()
        system.declare_complete(out_fn)

    # Generate ancillary files
    tile_io.create_s1_image(vvvh_gtiff=system.remove_ext_filename(out_fn), ref_patchsize=constants.PATCHSIZE_REF,
                            patchsize_10m=256)


if __name__ == "__main__":
    system.run_and_terminate(main)
