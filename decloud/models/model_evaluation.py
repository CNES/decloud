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
"""Perform the evaluation of models from TFRecords"""
import argparse
import logging
import sys

import tensorflow as tf
from decloud.core import system
from decloud.models.model_factory import ModelFactory
from decloud.models.tfrecord import TFRecords
from decloud.models import metrics
from decloud.models.utils import get_available_gpus


def main(args):
    # Application parameters parsing
    parser = argparse.ArgumentParser(description="Saved model evaluation")
    parser.add_argument("--savedmodel", help="SavedModel path. Mandatory for trained deep learning models.")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--test_records", nargs='+', default=[], help="Set of folders containing shards and .pkl files")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--strategy', default='mirrored',
                        const='mirrored',
                        nargs='?',
                        choices=['mirrored', 'singlecpu'],
                        help='tf.distribute strategy')

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    params = parser.parse_args(args)

    # Logging
    system.basic_logging_init()

    # Check SavedModel
    if not params.savedmodel:
        logging.warning("No SavedModel provided! Are you using a deterministic model?")
    elif not system.is_dir(params.savedmodel):
        logging.fatal("SavedModel directory %s doesn't exist, exiting.", params.savedmodel)
        system.terminate()

    # Strategy
    # For model evaluation we restrain strategies to "singlecpu" and "mirrored"
    n_workers = 1
    if params.strategy == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
        # Get number of GPUs
        n_workers = len(get_available_gpus())
        if n_workers == 0:
            logging.error("No GPU device found. At least one GPU is required! "
                          "Did you set correctly the CUDA_VISIBLE_DEVICES environment variable?")
            system.terminate()
    elif params.strategy == "singlecpu":
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        logging.error("Please provide a supported tf.distribute strategy.")
        system.terminate()

    # Datasets
    if not params.test_records:
        logging.error("Please provide at least one directory containing TFRecords files.")
        system.terminate()
    tfrecord_test_array = [TFRecords(rep) for rep in params.test_records]

    # Shape of the first dataset
    dataset_shapes = tfrecord_test_array[0].output_shape

    # Model
    model = ModelFactory.get_model(params.model, dataset_shapes=dataset_shapes)

    # List of tf.dataset
    tf_ds_test = [tfrecord.read(batch_size=params.batch_size,
                                target_keys=model.model_output_keys,
                                n_workers=n_workers,
                                drop_remainder=False) for tfrecord in tfrecord_test_array]

    with strategy.scope():
        # Create the model
        model.create_network()
        if params.savedmodel:
            # Load the SavedModel if provided (the model can be deterministic e.g. gapfilling)
            logging.info("Loading model weight from \"{}\"".format(params.savedmodel))
            model.load_weights(params.savedmodel)

        # Metrics
        metrics_list = [metrics.MeanSquaredError(), metrics.StructuralSimilarity(), metrics.PSNR(),
                        metrics.SpectralAngle()]
        model.compile(metrics={out_key: metrics_list for out_key in model.model_output_keys})
        model.summary()

        # Validation on multiple datasets
        for tf_ds in tf_ds_test:
            model.evaluate(tf_ds)


if __name__ == "__main__":
    system.run_and_terminate(main)
