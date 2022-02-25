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
import argparse
import os

from decloud.acquisitions.acquisition_factory import AcquisitionFactory
from decloud.core.tile_io import TilesLoader
from decloud.core.dataset import RandomIterator, ConstantIterator, RoisLoader
from decloud.core import system

system.basic_logging_init()

parser = argparse.ArgumentParser(description="Acquisition layout analysis")
parser.add_argument("--tiles", required=True, help="Path to tile handler file (.json)")
parser.add_argument("--rois", required=True, help="Path to roi file (.json)")
parser.add_argument("--al_root_dir", required=True, help="Path to the folder containing all the AL (.json)")
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument('--constant', dest='constant', action='store_true',
                    help="Use the constant iterator")
parser.set_defaults(constant=False)
params = parser.parse_args()

# Tiles handlers
th = TilesLoader(params.tiles, patchsize_10m=params.patch_size)

# Tiles ROIs
rois = RoisLoader(the_json=params.rois)

# Compute
ItClass = ConstantIterator if params.constant else RandomIterator
for file in os.listdir(params.al_root_dir):
    if file.endswith('.json'):
        acquisition_layout = os.path.join(params.al_root_dir, file)
        al = AcquisitionFactory.get_acquisition(acquisition_layout)
        train_it = ItClass(tile_handlers=th, acquisitions_layout=al, tile_rois=rois['roi_train'])
        print('#### ', file, ' ####')
        print('train: ', train_it.nb_of_tuples)

        valid_it = ItClass(tile_handlers=th, acquisitions_layout=al, tile_rois=rois['roi_valid'])
        print('valid: ', valid_it.nb_of_tuples)
        print('###########################')
