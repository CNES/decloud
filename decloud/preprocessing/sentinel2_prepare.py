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
"""Pre-process one Sentinel-2 image"""
import logging
import argparse
import otbApplication
from decloud.core import system, tile_io
import decloud.preprocessing.constants as constants

system.basic_logging_init()


def fconc(il, suffix, tilesize, out_tile_dir, pixel_type=otbApplication.ImagePixelType_int16):
    """
    Performs the concatenation of bands + tiling + compression
    :param il: input images list
    :param suffix: suffix
    :param tilesize: tile size
    :param out_tile_dir: Output repository
    :param pixel_type: otb pixel type
    """
    logging.info("Concatenate + Tile + Compress GeoTiff for files: %s", "".join(il))
    out_fn = system.basename(il[0])
    out_fn = out_fn[:out_fn.rfind("_")]
    out_fn = os.path.join(out_tile_dir, out_fn + "_" + suffix)
    out_fn += ".tif?&gdal:co:COMPRESS=DEFLATE"
    out_fn += "&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue={}".format(4 * tilesize)
    out_fn += "&gdal:co:TILED=YES&gdal:co:BLOCKXSIZE={ts}&gdal:co:BLOCKYSIZE={ts}".format(ts=tilesize)
    if system.is_complete(out_fn):
        logging.info("File %s already existing. Skipping.", system.remove_ext_filename(out_fn))
        return
    logging.info("Output: %s Tile size: %s", out_fn, tilesize)
    conc = otbApplication.Registry.CreateApplication("ConcatenateImages")
    conc.SetParameterStringList("il", il)
    conc.SetParameterString("out", out_fn)
    conc.SetParameterOutputImagePixelType("out", pixel_type)
    conc.ExecuteAndWriteOutput()
    system.declare_complete(out_fn)


def main(args):
    """Main"""
    # Arguments
    parser = argparse.ArgumentParser(description="S2 to tiled geotiff",
                                     usage="The input (\"--in_image parameter\") can be a .zip file, or a directory "
                                           "containing the product. "
                                           "The \"--out_s2_dir\" parameter is for the root directory for all "
                                           "Sentinel-2 tiles sub-folders, that will be automatically created if they "
                                           "don't exist. No need to create the tiles folders like \"T31TEJ\"... just "
                                           "provide the root directory that will host all the tiles using the "
                                           "\"--out_s2_dir\" parameter (e.g. \"--out_s2_dir /path/to/S2_PREPARE/\"). "
                                           "When the output files already exist, they are skipped.")
    parser.add_argument("--in_image", required=True, help="Input Sentinel-2 image (dir or .zip file)")
    parser.add_argument("--out_s2_dir", required=True, help="Directory for processed Sentinel-2 images")
    parser.add_argument("--tilesize", type=int, default=constants.PATCHSIZE_REF)
    params = parser.parse_args(args)

    is_zip = params.in_image.lower().endswith(".zip")
    if is_zip:
        logging.info("Input type is a .zip archive")
        files = system.list_files_in_zip(params.in_image)
    else:
        logging.info("Input type is a directory")
        files = system.get_files(params.in_image, ".tif")

    edg_mask = None
    cld_mask = None
    channels_10m = []
    channels_20m = []

    for file in files:
        if is_zip:
            file = system.to_vsizip(params.in_image, file)

        if "EDG_R1" in file:
            edg_mask = file
        if "CLM_R1" in file:
            cld_mask = file
        if any(bn in file for bn in ["FRE_B2.", "FRE_B3.", "FRE_B4.", "FRE_B8."]):
            channels_10m.append(file)
        if any(bn in file for bn in ["FRE_B5.", "FRE_B6.", "FRE_B7.", "FRE_B8A.", "FRE_B11.", "FRE_B12."]):
            channels_20m.append(file)

    channels_10m.sort()
    channels_20m.sort()

    assert edg_mask is not None
    assert cld_mask is not None
    assert len(channels_10m) == 4
    assert len(channels_20m) == 6

    # Tile name
    first_ch_fn = channels_10m[0]
    tile_name_matches = constants.SENTINEL_TILE.search(first_ch_fn)
    if tile_name_matches is None:
        logging.fatal("ERROR: unable to detect tile name! (from file %s)", first_ch_fn)
        system.terminate()
    tile_name = tile_name_matches[0]
    logging.info("Tile name is %s", tile_name)

    # Product name
    product_name = system.basename(params.in_image)
    if is_zip:
        product_name = product_name[:product_name.rfind(".")]
    if not product_name.startswith("SENTINEL2"):
        logging.fatal(
            "ERROR: Wrong product name: %s (Matched from input path: %s)", product_name, params.in_image)
        system.terminate()
    logging.info("Product name is %s", product_name)

    # Out directory
    out_tile_dir = os.path.join(params.out_s2_dir, tile_name, product_name)
    logging.info("Create or use the following output directory: %s", out_tile_dir)
    system.mkdir(out_tile_dir)

    # Concatenate + Tile + Compress
    fconc(channels_10m, "10m", int(1.0 * params.tilesize), out_tile_dir)
    fconc(channels_20m, "20m", int(0.5 * params.tilesize), out_tile_dir)
    fconc([edg_mask], "R1", int(1.0 * params.tilesize), out_tile_dir, otbApplication.ImagePixelType_uint8)
    fconc([cld_mask], "R1", int(1.0 * params.tilesize), out_tile_dir, otbApplication.ImagePixelType_uint8)

    # Generate ancillary files
    tile_io.create_s2_image_from_dir(s2_product_dir=out_tile_dir, ref_patchsize=constants.PATCHSIZE_REF,
                                     patchsize_10m=256,
                                     with_cld_mask=True, with_20m_bands=True)


if __name__ == "__main__":
    system.run_and_terminate(main)
