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
This scripts summarizes the number of samples that we can get from an AcquisitionsLayout
suited for single optical image reconstruction from date SAR/optical pair, for different
parameters of the AcquisitionsLayout
"""
import os
import argparse
import logging
from decloud.acquisitions.sensing_layout import AcquisitionsLayout, S1Acquisition, S2Acquisition
import numpy as np
from decloud.core import system, raster
from decloud.core.tile_io import TilesLoader
import decloud.preprocessing.constants as constants
system.basic_logging_init()

parser = argparse.ArgumentParser(description="Acquisition layout analysis")
parser.add_argument("--tiles", required=True, help="Path to tile handler file (.json)")
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--int_radius_list", nargs='+', type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                    help="Values to test for interior radius (days)")
parser.add_argument("--ext_radius_list", nargs='+', type=int, default=[15, 20],
                    help="Values to test for exterior radius (days)")
parser.add_argument("--maxgaps1s2_list", nargs='+', type=int, default=[24, 36, 48, 72],
                    help="Values to test for maximum temporal gap between S1/S2 acquisitions (hours)")
parser.add_argument('--al', default='os1', const='os1', nargs='?', choices=['os1', 'os2'], help='AL choice')
params = parser.parse_args()


def create_os1_al(max_s1s2_gap_hours, int_radius, ext_radius):
    """
    Create an AcquisitionsLayout suited for OS1 models
    """
    new_al = AcquisitionsLayout()
    new_al.new_acquisition("t",
                           s1_acquisition=S1Acquisition(),
                           s2_acquisition=S2Acquisition(min_cloud_percent=0, max_cloud_percent=0),
                           max_s1s2_gap_hours=max_s1s2_gap_hours,
                           timeframe_origin=True)
    new_al.new_acquisition("t-1",
                           s1_acquisition=S1Acquisition(),
                           s2_acquisition=S2Acquisition(min_cloud_percent=0, max_cloud_percent=0),
                           max_s1s2_gap_hours=max_s1s2_gap_hours,
                           timeframe_start_hours=-24 * ext_radius,
                           timeframe_end_hours=-24 * int_radius)
    new_al.new_acquisition("t+1",
                           s1_acquisition=S1Acquisition(),
                           s2_acquisition=S2Acquisition(min_cloud_percent=0, max_cloud_percent=0),
                           max_s1s2_gap_hours=max_s1s2_gap_hours,
                           timeframe_start_hours=24 * int_radius,
                           timeframe_end_hours=24 * ext_radius)
    return new_al


def create_os2_al(max_s1s2_gap_hours, int_radius, ext_radius):
    """
    Create an AcquisitionsLayout suited for OS2 models
    """
    new_al = AcquisitionsLayout()
    new_al.new_acquisition("t",
                           s1_acquisition=S1Acquisition(),
                           s2_acquisition=S2Acquisition(min_cloud_percent=1, max_cloud_percent=100),
                           max_s1s2_gap_hours=max_s1s2_gap_hours,
                           timeframe_origin=True)
    new_al.new_acquisition("target",
                           s1_acquisition=S1Acquisition(),
                           s2_acquisition=S2Acquisition(min_cloud_percent=0, max_cloud_percent=0),
                           max_s1s2_gap_hours=max_s1s2_gap_hours,
                           timeframe_start_hours=-24 * int_radius,
                           timeframe_end_hours=24 * int_radius)
    new_al.new_acquisition("t-1",
                           s1_acquisition=S1Acquisition(),
                           s2_acquisition=S2Acquisition(min_cloud_percent=0, max_cloud_percent=100),
                           max_s1s2_gap_hours=max_s1s2_gap_hours,
                           timeframe_start_hours=-24 * ext_radius,
                           timeframe_end_hours=-24 * int_radius)
    new_al.new_acquisition("t+1",
                           s1_acquisition=S1Acquisition(),
                           s2_acquisition=S2Acquisition(min_cloud_percent=0, max_cloud_percent=100),
                           max_s1s2_gap_hours=max_s1s2_gap_hours,
                           timeframe_start_hours=24 * int_radius,
                           timeframe_end_hours=24 * ext_radius)
    return new_al


# Tiles handlers
th = TilesLoader(params.tiles, patchsize_10m=params.patch_size)

# Compute
scale = float(params.patch_size) / float(constants.PATCHSIZE_REF)
res = []
create_al_fn = create_os1_al if params.al == "os1" else create_os2_al
for max_s1s2_gap_hours in params.maxgaps1s2_list:
    for ext_radius in params.ext_radius_list:
        for int_radius in params.int_radius_list:
            if int_radius >= ext_radius:
                break

            al = create_al_fn(max_s1s2_gap_hours=max_s1s2_gap_hours, int_radius=int_radius, ext_radius=ext_radius)
            total = 0
            for tile_name, tile_handler in th.items():
                # Reference raster grid
                ref_fn = tile_handler.s2_images[0].clouds_stats_fn

                # Accumulators
                ref_np_arr = tile_handler.s2_images_validity[0, :, :]
                np_counts = np.zeros_like(ref_np_arr, dtype=np.float32)

                # Search patches
                tuple_search = tile_handler.tuple_search(acquisitions_layout=al, roi=None)

                # Number of patches
                logging.info("Counting patches...")
                for pos, pos_list in tuple_search.items():
                    nb_samples_in_patch = len(pos_list)
                    total += nb_samples_in_patch
                    np_counts[pos[0], pos[1]] += nb_samples_in_patch

                # Export
                out_fn = f"count_gap{max_s1s2_gap_hours}_range{int_radius}-{ext_radius}_{tile_name}.tif"
                out_fn = os.path.join(params.out_dir, out_fn)
                logging.info("Saving %s", out_fn)
                raster.save_numpy_array_as_raster(ref_fn=ref_fn, np_arr=np_counts, out_fn=out_fn, scale=scale)

            res.append([max_s1s2_gap_hours, int_radius, ext_radius, total])

# Display
ncols = 25
for s in ["S1/S2 Max. Gap (h)", "Int. radius (jours)", "Ext. radius (jours)", "Nb. of samples"]:
    print("| {}".format(s).ljust(ncols), end="")
print("|")
for s in res[0]:
    print("| ".ljust(ncols, '-'), end="")
print("|")
for r in res:
    for s in r:
        print("| {}".format(s).ljust(ncols), end="")
    print("|")
