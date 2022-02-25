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
"""Helpers to handle checkpoints"""
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from decloud.core import system
from decloud.models.utils import _is_chief

# Callbacks being called at the end of each epoch during training


class ArchiveCheckpoint(keras.callbacks.Callback):
    """
    This callback save checkpoints to a persistent folder. Useful to easily continue the training after a training has
    completed and the model has been saved.
    """

    def __init__(self, backup_dir, strategy):
        """
        :param backup_dir: directory used by BackupAndRestore callback for checkpoints
        """
        super().__init__()
        self.backup_dir = backup_dir
        self.strategy = strategy

    def on_epoch_end(self, epoch, logs=None):
        """
        At the end of each epoch, we save the directory of BackupAndRestore to a different name for archiving
        """
        try:
            if _is_chief(self.strategy):
                chief_backup = os.path.join(self.backup_dir, 'chief')
                if os.path.isdir(chief_backup) and os.listdir(chief_backup):  # directory not empty
                    archived_dir = self.backup_dir + '_save'
                    chief_archived_dir = os.path.join(archived_dir, 'chief')
                    if not os.path.isdir(archived_dir):  # the root directory of dst need to exist
                        os.makedirs(archived_dir)
                    if os.path.exists(chief_archived_dir):  # the dst directory must not exist
                        shutil.rmtree(chief_archived_dir)
                    shutil.copytree(chief_backup, chief_archived_dir)
        except Exception as e:
            print('There was an error in ArchiveCheckpoint')
            print(e)


class AdditionalValidationSets(keras.callbacks.Callback):
    """
    This callback performs validation on some datasets.
    """

    def __init__(self, validation_datasets, logdir=None, validation_steps=None):
        """
        :param validation_datasets: a list of tf.Dataset, each yielding a tuple (features, target)
        :param logdir: tensorboard logdir
        :param validation_steps: how many batches of samples to evaluate. Optional, if not specified, the whole dataset
        will be evaluated
        """
        super().__init__()
        self.validation_datasets = validation_datasets
        self.logdir = logdir
        self.validation_steps = validation_steps

    def on_epoch_end(self, epoch, logs=None):
        """
        For each validation dataset, this function evaluate the loss/metrics and writes the results to Tensorboard
        under the name validation_{i}
        """
        for i, dataset in enumerate(self.validation_datasets):
            results = self.model.evaluate(dataset, steps=self.validation_steps, verbose=1)

            for metric, result in zip(self.model.metrics_names, results):
                if self.logdir:
                    writer = tf.summary.create_file_writer(system.pathify(self.logdir) + 'validation_{}'.format(i + 1))
                    with writer.as_default():
                        tf.summary.scalar('epoch_' + metric, result, step=epoch)  # tensorboard adds an 'epoch_' prefix
                else:
                    print('validation_{}_{}'.format(i + 1, metric), ':', result)
