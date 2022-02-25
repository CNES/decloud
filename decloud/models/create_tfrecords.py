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
"""Create some TFRecords from a decloud.dataset"""
import argparse
import sys
import logging
from decloud.core import system
from decloud.acquisitions.acquisition_factory import AcquisitionFactory
from decloud.core.dataset import RoisLoader, Dataset, RandomIterator, OversamplingIterator, ConstantIterator
from decloud.models.tfrecord import TFRecords
from decloud.core.tile_io import TilesLoader
from functools import partial


def main(args):
    parser = argparse.ArgumentParser(description="Creating TFRecords")
    parser.add_argument("--acquisition_train", help="Path to training acquisition file")
    parser.add_argument("--output_train", help="Output directory for training TFRecord shards")
    parser.add_argument("--acquisitions_valid", nargs='+', help="Paths to validation acquisition files")
    parser.add_argument("--outputs_valid", nargs='+', help="Output directories for valid TFRecord shards")
    parser.add_argument("--tiles", "-t", required=True, help="Path to a .json file used to instantiate the TilesLoader")
    parser.add_argument("--rois", required=True, help="Path to a .json file used to instantiate the RoisLoader")
    parser.add_argument('--patchsize', '-p', type=int, default=256, help="Patch size")
    parser.add_argument('--maxntrain', type=int, default=None, help="Max number of training samples")
    parser.add_argument('--maxnvalid', type=int, default=None, help="Max number of validation samples")
    parser.add_argument('--n_samples_per_shard', '-s', type=int, default=100, help="Number of samples per shard")
    parser.add_argument('--oversampling', dest='oversampling', action='store_true',
                        help="Performs validation on oversampled dataset")
    parser.add_argument('--constant', dest='constant', action='store_true',
                        help="Create datasets with spatially constant number of samples")
    parser.add_argument('--constant_nb', type=int, default=10, help="Max number of samples per patch, in constant mode")
    parser.add_argument('--with_20m_bands', dest='with_20m_bands', action='store_true',
                        help="Generate TFRecords containing 10m & 20m bands")
    parser.add_argument('--drop_remainder', dest='drop_remainder', action='store_true',
                        help="Generate TFRecords exclusively of `n_samples_per_shard` samples. Additional samples "
                             "will be dropped. Advisable if using multiworkers training")
    parser.set_defaults(oversampling=False)
    parser.set_defaults(with_20m_bands=False)
    parser.set_defaults(constant=False)
    parser.set_defaults(drop_remainder=True)

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    params = parser.parse_args(args)

    system.basic_logging_init()

    # Tiles handlers
    th = TilesLoader(params.tiles, patchsize_10m=params.patchsize, with_20m_bands=params.with_20m_bands)

    # Tiles ROIs
    rois = RoisLoader(the_json=params.rois)

    def _ds2tfrecord(acquisition_layout, output_dir, max_nb_of_samples, iterator_class, roi_key="roi_valid"):
        acquisitions = AcquisitionFactory.get_acquisition(acquisition_layout)
        ds = Dataset(acquisitions_layout=acquisitions, tile_handlers=th, tile_rois=rois[roi_key],
                     iterator_class=iterator_class, max_nb_of_samples=max_nb_of_samples)
        tfrecord = TFRecords(output_dir)
        tfrecord.ds2tfrecord(ds, n_samples_per_shard=params.n_samples_per_shard, drop_remainder=params.drop_remainder)

    # iterator
    iterator = RandomIterator
    if params.constant:
        logging.info("Using constant iterator")
        iterator = partial(ConstantIterator, nbsample_max=params.constant_nb)

    if params.acquisition_train is not None:
        _ds2tfrecord(acquisition_layout=params.acquisition_train, output_dir=params.output_train, roi_key="roi_train",
                     max_nb_of_samples=params.maxntrain, iterator_class=iterator)

    if params.acquisitions_valid is not None:
        for acquisition_valid, output_dir in list(zip(params.acquisitions_valid, params.outputs_valid)):
            _ds2tfrecord(acquisition_layout=acquisition_valid, output_dir=output_dir,
                         iterator_class=iterator if not params.oversampling else OversamplingIterator,
                         max_nb_of_samples=params.maxnvalid)


if __name__ == "__main__":
    system.run_and_terminate(main)
