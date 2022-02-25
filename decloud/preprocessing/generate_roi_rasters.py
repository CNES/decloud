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
"""Create ROI binary mask rasters"""
import argparse
import json
import os
import logging
import numpy as np
import pyotb

from decloud.core import system
from decloud.preprocessing import constants


def main(args):
    parser = argparse.ArgumentParser(description="Creating TFRecords")
    parser.add_argument("--tiles", "-t", required=True, help="Path to a .json file used to instantiate the TilesLoader")
    parser.add_argument("--output_dir", required=True, help="Output directory to store generated datasets")
    parser.add_argument("--datasets", required=True, nargs='+',
                        help="Datasets to be considered for the generation of ROIs, e.g. train, valid, test...")
    parser.add_argument('--patchsize', '-p', type=int, default=256,
                        help="Patch size (in pixels) for 10m rasters. "
                             "Must be a multiple of {}".format(constants.PATCHSIZE_REF))
    parser.add_argument('--props', required=True, nargs='+', type=float,
                        help="Proportions of samples (one value for each dataset)")
    parser.add_argument('--rois', nargs='+', default=[],
                        help="Paths to vector files specifying ROIs "
                             "(one path for each dataset: use empty paths for datasets with not ROI)")

    params = parser.parse_args(args)

    system.basic_logging_init()

    # Load the TileHandlers descriptors
    # Note that here we don't use any TileHandler but just the paths to the tiles
    with open(params.tiles) as f:
        tiles = json.load(f)

    # Check number of params
    nb_datasets = len(params.datasets)
    assert len(params.props) == nb_datasets
    assert not params.rois or len(params.rois) == nb_datasets
    assert params.patchsize % constants.PATCHSIZE_REF == 0

    # normalize the number samples to percentages whose sum makes 100%
    props = [x / sum(params.props) for x in params.props]

    for tile in tiles['TILES']:
        # Get an arbitrary raster corresponding to the extent of this tile. We take the first CLM_stats of S2_
        matches = [os.path.join(tiles['S2_ROOT_DIR'], tile, x) for x in
                   os.listdir(os.path.join(tiles['S2_ROOT_DIR'], tile)) if
                   os.path.isdir(os.path.join(tiles['S2_ROOT_DIR'], tile, x))]
        # Throw an error if no Sentinel-2 tile has been found
        if len(matches) == 0:
            logging.fatal("No Sentinel-2 tile found in %s. Please check the tiles descriptor (%s)",
                          tiles['S2_ROOT_DIR'], params.tiles)
            system.terminate()
        first_s2_dir = matches[0]
        candidates = [os.path.join(first_s2_dir, x) for x in os.listdir(first_s2_dir) if x.endswith('CLM_R1_stats.tif')]
        # Throw an error if no cloud coverage stats has been found
        if len(candidates) == 0:
            logging.fatal("No Sentinel-2 auxiliary file for cloud coverage statistics found in %s", first_s2_dir)
            system.terminate()
        first_clm_stats = candidates[0]

        # Fill this raster with zeros
        # This raster pixel spacing is 10m * constants.PATCHSIZE_REF (i.e. likely 640m)
        initialized_raster = pyotb.BandMath(il=first_clm_stats, exp="1==1 ? 0 : 0")

        # Create a undersampled raster where one pixel is equivalent to one patch
        scale = constants.PATCHSIZE_REF / params.patchsize
        undersampled = pyotb.RigidTransformResample({"in": initialized_raster, "interpolator": "nn",
                                                     "transform.type.id.scalex": scale,
                                                     "transform.type.id.scaley": scale})

        fg_val = 1
        rois_arrays = [np.asarray(pyotb.Rasterization({'in': roi, 'im': undersampled,
                                                       'mode.binary.foreground': fg_val})) if roi else None for roi
                       in params.rois]

        # Random selection of zones for the specified datasets, fills the array with 0 for the 1st dataset,
        # 1 for 2nd dataset etc...
        random_patches = np.random.choice(np.arange(0, len(params.datasets)), p=props,
                                          size=undersampled.shape)

        # Using a specific ROI for each dataset
        # Patches under a polygon are selected, and this prevails on the random selection.
        # Warning: if ROI of different datasets overlap together, resulting datasets can have a non-null intersection.
        for dataset_id, rois_array in enumerate(rois_arrays):
            if rois_array:
                random_patches = np.where(rois_array == fg_val, dataset_id, random_patches)

        for dataset_id, dataset in enumerate(params.datasets):
            patches = np.add(undersampled, (random_patches == dataset_id).astype(int))  # this is a pyotb object

            # Resample to the needed spacing
            final_roi = pyotb.Superimpose(inm=patches, inr=initialized_raster, interpolator='nn')
            final_roi.write(os.path.join(params.output_dir, '{}_{}.tif'.format(tile, dataset)))


if __name__ == "__main__":
    system.run_and_terminate(main)
