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
"""Run prediction of a model on some TFRecords"""
import argparse
import logging
import sys
import os
import tensorflow as tf
import numpy as np
from decloud.models.model_factory import ModelFactory
from decloud.core import system
from decloud.models.tfrecord import TFRecords
from decloud.preprocessing.normalization import denormalize_s2


def main(args):
    """
    Run the prediction
    """
    # Application parameters parsing
    parser = argparse.ArgumentParser(description="Prediction on TFRecords")
    parser.add_argument("--valid_records", nargs='+', required=True,
                        help="TFRecords, can be a file, a list of files or a directory. Additional json files are "
                             "required to be in the same directory as TFRecords")
    parser.add_argument("--savedmodel", help="Path to the saved model (directory containing the .pb)")
    parser.add_argument("--model", help="Name of the deterministic model")
    parser.add_argument("--outdir", required=True, help="Directory to write tensorboard summaries")
    parser.add_argument("--model_output_key", default='s2_target', help="Key of the output tensor")
    parser.add_argument('-bv', '--batch_size_valid', type=int, default=16)

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    params = parser.parse_args(args)

    # Logging
    system.basic_logging_init()

    if not params.savedmodel and not params.model:
        logging.error("Please provide a path to a saved model or a deterministic model name")
        system.terminate()

    # Datasets
    tfrecord_valid_array = [TFRecords(rep) for rep in params.valid_records]

    # Model instantiation
    if params.model:
        logging.info('Creating deterministic model {}'.format(params.model))
        model = ModelFactory.get_model(params.model, dataset_shapes=tfrecord_valid_array[0].output_shape)
        model.create_network()
        if params.savedmodel:
            model.load_weights(params.savedmodel)
    elif params.savedmodel:
        if not system.is_dir(params.savedmodel):
            logging.error("Please provide a directory for the SavedModel.")
            system.terminate()
        logging.info('Loading model from {}'.format(params.savedmodel))
        model = tf.keras.models.load_model(params.savedmodel, compile=False)

    # TF.dataset-s instantiation
    tf_ds_valid = [tfrecord.read(batch_size=params.batch_size_valid,
                                 target_keys=[params.model_output_key]) for tfrecord in tfrecord_valid_array]

    def postprocess(x, key, denormalize=True):
        x = tf.cast(x, tf.float32)
        # Convert from float range to int16 range
        if denormalize:
            x = denormalize_s2(x)

        # Convert to uint8 range
        if key.startswith('s2'):
            # remove 4th band and arange in RGB mode
            x = tf.stack([x[:, :, 2], x[:, :, 1], x[:, :, 0]], axis=2)
            # For value < 2000 in int16, linearly scale between 0 and 230.
            # For value between 2000 and 10000, linearly scale between 230 et 255
            x = np.where(x < 2000, x * 230 / 2000, x * (255 - 230) / (10000 - 2000) + 0.25 * (230 * 10000 / 2000 - 255))
        if key.startswith('s1'):
            # single band
            x = tf.expand_dims(x[:, :, 0], axis=2)
            # Stretching
            x = ((255 - 0) / (65535 - 10000)) * x - 10000 * (255 - 0) / (65535 - 10000)

        x = tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8)
        return x

    for dataset_name, dataset in zip([system.basename(x) for x in params.valid_records], tf_ds_valid):
        # Save to image the data contained in the samples
        for i, element in enumerate(dataset.unbatch()):
            # Inputs (first element of the tuple)
            for key, x in element[0].items():
                if len(x.shape) > 2:  # and key.startswith('s2'):
                    tf.keras.preprocessing.image.save_img(
                        os.path.join(params.outdir, '{}_{}_{}.png'.format(dataset_name, i, key)),
                        postprocess(x, key, denormalize=False), scale=False)  # inputs data are not normalized...

            tf.keras.preprocessing.image.save_img(
                os.path.join(params.outdir, '{}_{}_{}.png'.format(dataset_name, i, params.model_output_key)),
                postprocess(element[1][params.model_output_key], 's2'))  # ... whereas target data are normalized

        # Predict the outputs from model of all samples
        out = model.predict(dataset)

        # Save all predictions to image
        out_arrays = out[params.model_output_key]
        for i, out_array in enumerate(out_arrays):
            tf.keras.preprocessing.image.save_img(
                os.path.join(params.outdir, '{}_{}_reconstructed.png'.format(dataset_name, i)),
                postprocess(out_array, 's2'))


if __name__ == "__main__":
    system.run_and_terminate(main)
