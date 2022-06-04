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
Compute the number of S1 and S2 images used for each patch.
"""
import os
import argparse
import logging
import numpy as np
from decloud.core import system, raster
from decloud.acquisitions.acquisition_factory import AcquisitionFactory
from decloud.core.tile_io import TilesLoader
from decloud.core.dataset import RoisLoader
import decloud.preprocessing.constants as constants


parser = argparse.ArgumentParser(description="Patches coverage analysis")
parser.add_argument("--acquisition_layouts", nargs='+', required=True, help="Path to acquisition layout file (.json)")
parser.add_argument("--tiles", required=True, help="Path to tile handler file (.json)")
parser.add_argument("--out_dir", required=True, help="Output directory for generated stats files")
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--rois", help="Path to a .json file used to instantiate the RoisLoader")
parser.add_argument("--roi_key", help="ROI key (roi_valid or roi_train)", default="roi_train")
params = parser.parse_args()

system.basic_logging_init()

# Acquisitions layout
als = [(system.basename(fn)[:-5], AcquisitionFactory.get_acquisition(fn)) for fn in params.acquisition_layouts]

# Tiles handlers
th = TilesLoader(params.tiles, patchsize_10m=params.patch_size)

# Tiles ROIs
rois = RoisLoader(the_json=params.rois) if params.rois else None

# Compute
scale = float(params.patch_size) / float(constants.PATCHSIZE_REF)
for al_bname, al in als:
    for tile_name, tile_handler in th.items():
        # Output files prefix
        out_prefix = os.join(params.out_dir, tile_name + "_" + al_bname)

        # Reference raster grid
        ref_fn = tile_handler.s2_images[0].clouds_stats_fn

        # Accumulators
        ref_np_arr = tile_handler.s2_images_validity[0, :, :]
        np_counts = {key: np.zeros_like(ref_np_arr, dtype=np.float32) for key in ["s1", "s2"]}
        count_map = dict()

        # Search patches
        roi = rois[params.roi_key][tile_name] if rois else None
        tuple_search = tile_handler.tuple_search(acquisitions_layout=al, roi=roi)

        # Number of patches
        logging.info("Counting patches...")
        for pos, pos_list in tuple_search.items():
            # pos: the location of the patch. e.g. (12, 1)
            for sample_dict in pos_list:
                def _store_index(_key, idx):
                    np_counts[key][pos[0], pos[1]] += 1
                    if pos not in count_map:
                        count_map[pos] = dict()
                    if _key in count_map[pos]:
                        count_map[pos][_key].append(idx)
                    else:
                        count_map[pos][key] = [idx]
                for acq_key, acq_dict in sample_dict.items():
                    # acq_key: the acquisition key. e.g. "s1_tm1", "s2_target", etc
                    for key in ["s1", "s2"]:
                        if key in acq_dict:
                            _store_index(key, acq_dict[key])

        # Export
        for key in ["s1", "s2"]:
            out_fn = f"{out_prefix}_{key}_freq.tif"
            logging.info("Saving %s", out_fn)
            raster.save_numpy_array_as_raster(ref_fn=ref_fn, np_arr=np_counts[key], out_fn=out_fn, scale=scale)
