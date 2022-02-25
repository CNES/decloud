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
"""Train some model from TFRecords"""
import argparse
import logging
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
from decloud.core import system
from decloud.models.model_factory import ModelFactory
from decloud.models.tfrecord import TFRecords
from decloud.models import metrics
from decloud.models.callbacks import AdditionalValidationSets, ArchiveCheckpoint
from decloud.core.summary import PreviewsCallback
from decloud.models.utils import get_available_gpus
from decloud.models.utils import _is_chief


def main(args):
    """
    Run the training and validation process
    """
    # Application parameters parsing
    parser = argparse.ArgumentParser(description="Network training from TFRecords")
    parser.add_argument("--training_record", help="Folder containing shards and .json files")
    parser.add_argument("--valid_records", nargs='+', default=[], help="Folders containing shards and .json files")
    parser.add_argument("-m", "--model", required=True, help="Model name")
    parser.add_argument("--logdir", help="Directory to write tensorboard summaries")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0002)
    parser.add_argument('-bt', '--batch_size_train', type=int, default=4)
    parser.add_argument('-bv', '--batch_size_valid', type=int, default=4)
    parser.add_argument('-e', '--epochs', type=int, default=300,
                        help="Nb of epochs. If set to zero, only performs the model saving")
    parser.add_argument('--ckpt_dir', help="Directory to save & load model checkpoints")
    parser.add_argument('--out_savedmodel', help="Parent directory for output SavedModel")
    parser.add_argument('--save_best', dest='save_best', action='store_true',
                        help="SavedModel is written when the metric specified with \"save_best_ref\" is the lowest")
    parser.set_defaults(save_best=False)
    parser.add_argument('--save_best_ref', help="Name of the scalar metric to save the best model", default="val_loss")
    parser.add_argument('--all_metrics', dest='all_metrics', action='store_true',
                        help="Performs validation using all metrics")
    parser.set_defaults(all_metrics=False)
    parser.add_argument('--previews', dest='previews', action='store_true',
                        help="Enable images summary (from validation datasets)")
    parser.set_defaults(previews=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Enable full Keras verbosity, can be useful for debug")
    parser.set_defaults(verbose=False)
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true',
                        help="Stops the training if the loss doesn't improve during several epochs")
    parser.set_defaults(early_stopping=False)
    parser.add_argument('--profiling', default=0, help="Batch number (e.g. 45), or range of batches (e.g. "
                                                       "(start, end)) to profile. Default is off")
    parser.add_argument('--strategy', default='mirrored',
                        const='mirrored',
                        nargs='?',
                        choices=['mirrored', 'multiworker', 'singlecpu'],
                        help='tf.distribute strategy')
    parser.add_argument('--plot_model', dest='plot_model', action='store_true',
                        help="Whether we want to plot the model architecture. Requires additional libraries")
    parser.add_argument('--shuffle_buffer_size', type=int, default=5000,
                        help="Shuffle buffer size. To be decreased if low RAM is available.")
    parser.set_defaults(plot_model=False)

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    params = parser.parse_args(args)

    # Logging
    system.basic_logging_init()

    # Check that we have at least one training dataset
    if not params.training_record:
        logging.error("Please provide at least one training dataset.")
        system.terminate()

    # Check that we have a SavedModel path if save_best is true
    if params.save_best and not params.out_savedmodel:
        logging.error("Please provide a path for the output SavedModel.")
        system.terminate()

    # Strategy
    if params.strategy == "multiworker":
        # Srategy cf http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-tf-multi.html
        # build multi-worker environment from Slurm variables
        cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=13565)  # On Jean-Zay cluster
        # use NCCL communication protocol
        implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
        communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)
        # declare distribution strategy
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver,
                                                             communication_options=communication_options)
        # get total number of workers
        n_workers = int(os.environ['SLURM_NTASKS'])
    elif params.strategy == "mirrored":
        strategy = tf.distribute.MirroredStrategy()
        # Get number of GPUs
        n_workers = len(get_available_gpus())
    elif params.strategy == "singlecpu":
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        n_workers = 0
    else:
        logging.error("Please provide a supported tf.distribute strategy.")
        system.terminate()

    # CPU or GPU
    if n_workers == 0:
        logging.info('No GPU found, using CPU')
        n_workers = 1
        suffix = "_cpu"
    else:
        logging.info('Number of available GPUs: %s', n_workers)
        suffix = "_{}gpus".format(n_workers)

    # Name of the experiment
    expe_name = "{}".format(params.model)
    expe_name += "_{}".format(system.get_commit_hash())
    expe_name += "_bt{}".format(params.batch_size_train)
    expe_name += "_bv{}".format(params.batch_size_valid)
    expe_name += "_lr{}".format(params.learning_rate)
    expe_name += "_e{}".format(params.epochs)
    expe_name += suffix

    if True:  # TODO: detete, just used for review
        # Date tag
        date_tag = time.strftime("%d-%m-%y-%H%M%S")

        # adding the info to the SavedModel path
        out_savedmodel = None if params.out_savedmodel is None else \
            system.pathify(params.out_savedmodel) + expe_name + date_tag

        # Scaling batch size and learning rate accordingly to number of workers
        batch_size_train = params.batch_size_train * n_workers
        batch_size_valid = params.batch_size_valid * n_workers
        learning_rate = params.learning_rate * n_workers

        logging.info("Learning rate was scaled to %s, effective batch size is %s (%s workers)",
                     learning_rate, batch_size_train, n_workers)

        # Datasets
        tfrecord_train = TFRecords(params.training_record) if params.training_record else None
        tfrecord_valid_array = [TFRecords(rep) for rep in params.valid_records]

        # Model instantiation
        model = ModelFactory.get_model(params.model, dataset_shapes=tfrecord_train.output_shape)

        # TF.dataset-s instantiation
        tf_ds_train = tfrecord_train.read(batch_size=batch_size_train,
                                          target_keys=model.model_output_keys,
                                          n_workers=n_workers,
                                          shuffle_buffer_size=params.shuffle_buffer_size) if tfrecord_train else None
        tf_ds_valid = [tfrecord.read(batch_size=batch_size_valid,
                                     target_keys=model.model_output_keys,
                                     n_workers=n_workers) for tfrecord in tfrecord_valid_array]

        with strategy.scope():
            # Creating the Keras network corresponding to the model
            model.create_network()

            # Metrics
            metrics_list = [metrics.MeanSquaredError(), metrics.PSNR()]
            if params.all_metrics:
                metrics_list += [metrics.StructuralSimilarity(), metrics.SpectralAngle()]  # A bit slow to compute

            # Creating the model or loading it from checkpoints
            logging.info("Loading model \"%s\"", params.model)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss=model.get_loss(),
                          metrics={out_key: metrics_list for out_key in model.model_output_keys})
            model.summary(strategy)

            if params.plot_model:
                model.plot('/tmp/model_architecture_{}.png'.format(model.__class__.__name__), strategy)

            callbacks = []
            # Define the checkpoint callback
            if params.ckpt_dir:
                if params.strategy == 'singlecpu':
                    logging.warning('Checkpoints can not be saved while using singlecpu option. Discarding checkpoints')
                else:
                    # Create a backup
                    backup_dir = system.pathify(params.ckpt_dir) + params.model
                    callbacks.append(keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir))

                    # Save the checkpoint to a persistent location
                    callbacks.append(ArchiveCheckpoint(backup_dir, strategy))

            # Define the Keras TensorBoard callback.
            logdir = None
            if params.logdir:
                logdir = system.pathify(params.logdir) + "{}_{}".format(date_tag, expe_name)
                tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,
                                                                   profile_batch=params.profiling)
                callbacks.append(tensorboard_callback)

                # Define the previews callback
                if params.previews:
                    # We run the preview on an arbitrary sample of the validation dataset
                    sample = tfrecord_valid_array[0].read_one_sample(target_keys=model.model_output_keys)
                    previews_callback = PreviewsCallback(sample, logdir, input_keys=model.dataset_input_keys,
                                                         target_keys=model.model_output_keys)
                    callbacks.append(previews_callback)

            # Validation on multiple datasets
            if tf_ds_valid:
                additional_validation_callback = AdditionalValidationSets(tf_ds_valid[1:], logdir)
                callbacks.append(additional_validation_callback)

            # Save best checkpoint only
            if params.save_best:
                callbacks.append(keras.callbacks.ModelCheckpoint(params.out_savedmodel, save_best_only=True,
                                                                 monitor=params.save_best_ref, mode='min'))

            # Early stopping if the training stops improving
            if params.early_stopping:
                callbacks.append(keras.callbacks.EarlyStopping(monitor=params.save_best_ref, min_delta=0.0001,
                                                               patience=10, mode='min'))

            # Training
            model.fit(tf_ds_train,
                      epochs=params.epochs,
                      validation_data=tf_ds_valid[0] if tf_ds_valid else None,
                      callbacks=callbacks,
                      verbose=1 if params.verbose else 2)

            # Multiworker training tries to save the model multiple times and this can create corrupted models
            # Thus we save the model at the final path only for the 'chief' worker
            if params.strategy != 'singlecpu':
                if not _is_chief(strategy):
                    out_savedmodel = None

            # Export SavedModel
            if out_savedmodel and not params.save_best:
                logging.info("Saving SavedModel in %s", out_savedmodel)
                model.save(out_savedmodel)


if __name__ == "__main__":
    system.run_and_terminate(main)
