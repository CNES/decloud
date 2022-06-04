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
"""
Analyze the S1 and S2 orbits
"""
import os
import argparse
import numpy as np
import logging
from decloud.core import system
from decloud.core.tile_io import TilesLoader
from decloud.preprocessing import constants
import pyotb

system.basic_logging_init()

parser = argparse.ArgumentParser(description="S1/S2 orbits analysis. The program writes the closest gap between "
                                             "the S1 and S2 images for each patches, in the form of an histogram "
                                             "stored in the channel of the output geotiff image (each pixel "
                                             "corresponds to a patch).")
parser.add_argument("--tiles", required=True, help="Path to tile handler file (.json)")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--patchsize", type=int, default=256)
parser.add_argument("--nbins", type=int, default=8, help="Number of bins in histogram")
parser.add_argument("--quant", type=int, default=12, help="Histogram bins quantization (hours)")
params = parser.parse_args()

# Tiles handlers
th = TilesLoader(params.tiles, patchsize_10m=params.patchsize)

# Histogram bins
max_n_bins = params.nbins
bins_quant = params.quant
exp = "{" + (max_n_bins * "0;")[:-1] + "}"  # equals "{0;...;0}"

# Compute
for tile_name, tile_handler in th.items():

    logging.info("Processing tile %s", tile_name)

    # Fill this raster with zeros
    zeros_raster = pyotb.BandMathX(il=tile_handler.s2_images[0].clouds_stats_fn, exp=exp)
    scale = constants.PATCHSIZE_REF / params.patchsize
    initialized_raster = pyotb.RigidTransformResample({"in": zeros_raster, "interpolator": "nn",
                                                       "transform.type.id.scalex": scale,
                                                       "transform.type.id.scaley": scale})

    # Histogram of S1/S2 temporal gap
    histo_array = np.zeros(shape=initialized_raster.shape)

    def _count_gaps(pos):
        for s2_idx, _ in enumerate(tile_handler.s2_images):
            closest_s1 = tile_handler.closest_s1[pos]
            if s2_idx in closest_s1:
                gap = closest_s1[s2_idx].distance
                bin = min(max_n_bins - 1, int(gap / (bins_quant * 3600)))
                histo_array[pos[1]][pos[0]][bin] += 1

    tile_handler.for_each_pos(_count_gaps)

    # Export with pyotb
    out = np.add(initialized_raster, histo_array)  # this is a pyotb object
    out_fn = os.path.join(params.out_dir, f"{tile_name}_s1s2gap_hist.tif")
    out.write(out_fn)
