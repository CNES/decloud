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
"""Classes for Tensorflow summaries"""
import tensorflow as tf
from tensorflow import keras
from decloud.core import system
from decloud.preprocessing import constants


# --------------------------------------- Some images summary helpers --------------------------------------------------
def get_preview_fn(key):
    """
    Return Input preview function
    :return: function
    """
    func = None
    if key.startswith("s1"):
        func = s1_image_preview_fn
    elif key.startswith("s2"):
        func = s2_image_preview_fn
    elif key == constants.DEM_KEY:
        func = greylevel_monoband_preview_fn
    else:
        pass
    return func


class PreviewsCallback(keras.callbacks.Callback):
    """
    This callback creates previews at each end of epoch
    """
    def __init__(self, sample, logdir, input_keys=None, target_keys=None, every_nth_epochs=1):
        """
        :param sample: the sample to evaluate, tuple (inputs, target)
        :param logdir:
        :param input_keys: list of inputs to consider
        :param target_keys: list of outputs to consider
        :param every_nth_epochs: optional.
        """
        super().__init__()
        self.logdir = logdir
        self.every_nth_epochs = every_nth_epochs
        self.test_data, self.target = sample
        # if the user didn't specify the keys, all of the dataset are considered
        self.input_keys = input_keys if input_keys is not None else self.test_data.keys()
        self.target_keys = target_keys if target_keys is not None else self.target.keys()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_nth_epochs == 0:
            # Use the model to predict the values from the validation dataset.
            predicted = self.model.predict(self.test_data)

            # Log the images summary.
            file_writer = tf.summary.create_file_writer(system.pathify(self.logdir) + 'previews')
            with file_writer.as_default():
                for key in self.target_keys:
                    tf.summary.image("predicted: " + key, get_preview_fn(key)(predicted[key]), step=epoch)
                    tf.summary.image("target: " + key, get_preview_fn(key)(self.target[key]), step=epoch)

                for key in self.input_keys:
                    tf.summary.image("input: " + key, get_preview_fn(key)(self.test_data[key]), step=epoch)


def only_first_4_patches(tensor, n_imgs=4):
    """
    Batch truncation function
    :param tensor: batch
    :param n_imgs: number of patches to keep
    :return: truncated batch
    """
    return tensor[0:n_imgs]


def mean_std_stretch(tensor, std_mult=2.0):
    """
    Mean/Standard deviation based stretching.
    new_elem bust have a shape of dimension 4 (like [None, szy, szx, nb_channels])
    Output values are stretched between mean-std_mult*std and mean+std_mult*std for all channels separately.
    :param tensor: input tensor
    :param std_mult: scalar
    :return: a uint8 tensor, returning only the first 4 patches of the batch
    """
    if len(tensor.shape) != 4:
        raise Exception("Please provide a tensor with shape of dimension 4 (like [None, szy, szx, nb_channels])")

    tensor = tf.cast(tensor, tf.float32)
    _mean = tf.reduce_mean(tensor, axis=[1, 2], keepdims=True)  # [-1, 1, 1, N]
    _std = tf.sqrt(tf.reduce_mean(tf.square(tensor - _mean), axis=[1, 2], keepdims=True) + 1e-8)
    range_lo = _mean - std_mult * _std
    range_hi = _mean + std_mult * _std
    tensor = (tensor - range_lo) / (range_hi - range_lo)
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    tensor = only_first_4_patches(tensor)
    return tf.cast(255 * tensor, tf.uint8)


def s1_image_preview_fn(tensor):
    """
    Function that converts SAR images into a nice RGB image using (R=VV, G=VH, B=VH)
    :param tensor: SAR images tensor
    :return: linearly stretched, colorized SAR image for summary purpose
    """
    tensor = mean_std_stretch(tensor)
    return tf.stack([tensor[:, :, :, band] for band in [0, 1, 1]], axis=3)  # (R=VV, G=VH, B=VH)


def s2_image_preview_fn(tensor):
    """
    Linear stretch and reorder B4, B3 and B2 bands of a Sentinel-2 image, for image summary purpose
    :param tensor: S2 images tensor
    :return: nice RBG images
    """
    tensor = mean_std_stretch(tensor)
    return tf.stack([tensor[:, :, :, band] for band in [2, 1, 0]], axis=3)  # (R=B4, G=B3, B=B2)


def greylevel_monoband_preview_fn(tensor):
    """
    Linear stretch between min and max value of the patches
    :param tensor: a tensor of shape [-1, pszy, pszx, 1]
    :return: grey level image
    """
    mins = tf.math.reduce_min(tensor)
    maxs = tf.math.reduce_max(tensor)
    tensor = tf.divide(tensor - mins, maxs - mins)
    return tf.cast(255 * tensor, tf.uint8)
