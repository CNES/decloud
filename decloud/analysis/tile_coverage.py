#!/usr/bin/python3
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
"""
Compute cloud coverage and pixel validity from an input set of tiles
"""
import argparse
import logging
import numpy as np
from decloud.core import system, tile_io, raster

# Application parameters
parser = argparse.ArgumentParser(description="Compute cloud coverage and pixel validity from an input set of tiles")
parser.add_argument("--tiles", required=True, help="Tiles (.json file)")
parser.add_argument("--out_dir", required=True, help="Output directory for generated stats")
params = parser.parse_args()

# Logging init
system.basic_logging_init()


def compute_stats(tile_name, tile_handler):
    """
    Compute statistics.
    :param tile_name: Name of the tile to process
    :param tile_handler: Tile handler instance
    """
    ref_fn = tile_handler.s2_images[0].clouds_stats_fn
    out_prefix = system.pathify(params.out_dir) + tile_name

    # Statistics
    cloud_cov = np.sum(np.multiply(tile_handler.s2_images_validity, tile_handler.s2_images_cloud_coverage), axis=0)
    cloud_cov = np.divide(cloud_cov, np.sum(tile_handler.s2_images_validity, axis=0))
    nb_pix_s1 = np.sum(tile_handler.s1_images_validity, axis=0)
    nb_pix_s2 = np.sum(tile_handler.s2_images_validity, axis=0)

    # Save
    raster.save_numpy_array_as_raster(ref_fn=ref_fn, np_arr=cloud_cov, out_fn=out_prefix + "_cloud_cov.tif")
    raster.save_numpy_array_as_raster(ref_fn=ref_fn, np_arr=nb_pix_s1, out_fn=out_prefix + "_nb_pix_s1.tif")
    raster.save_numpy_array_as_raster(ref_fn=ref_fn, np_arr=nb_pix_s2, out_fn=out_prefix + "_nb_pix_s2.tif")


if __name__ == "__main__":

    tiles = tile_io.TilesLoader(params.tiles, patchsize_10m=64)

    # Compute stats for each tile
    for name, th in tiles.items():
        logging.info("Computing stats for tile %s", name)
        compute_stats(tile_name=name, tile_handler=th)
