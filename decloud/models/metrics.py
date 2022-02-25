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
Functions for metrics (PSNR, MSE, SSIM, SAM)
A valid metric is simply a function that takes `y_true` and  `y_pred` as inputs and returns a scalar as output

"""
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.metrics import MeanMetricWrapper
from decloud.preprocessing.normalization import denormalize_s2


def to_float32(tensor):
    """
    Cast a tensor to float32
    :param tensor: input tensor
    :return: float tensor
    """
    return tf.cast(tensor, tf.float32)


def psnr_from_mse(mse):
    """
    Compute PSNR from MSE
    :param mse: MSE tensor (scalar)
    :return: PSNR tensor (scalar)
    """
    squared_max = 10000 ** 2
    imse = tf.divide(squared_max, mse)
    psnr = 10.0 * tf.divide(tf.math.log(imse), math.log(10))
    return psnr


@tf.keras.utils.register_keras_serializable()
class MeanSquaredError(keras.metrics.MeanSquaredError):
    """
    Mean squared error metric on denormalized images. This class only handles the denormalization,
    the parent class handles the main work.
    """
    def __init__(self, name='MSE', **kwargs):
        # Variables
        super().__init__(name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = denormalize_s2(to_float32(y_pred))
        y_true = denormalize_s2(to_float32(y_true))
        super().update_state(y_true, y_pred, sample_weight)


@tf.keras.utils.register_keras_serializable()
class PSNR(MeanSquaredError):
    """
    Peak Signal to Noise Ratio metric. Inherits from MeanSquaredError class, which handles the main work.
    Here we just transform to PSNR the result computed by MeanSquaredError
    """
    def __init__(self, name='PSNR', **kwargs):
        # Variables
        super().__init__(name, **kwargs)

    def result(self):
        # self.total and self.count have been computed in `update_state` by MeanSquaredError class
        mse = tf.math.divide_no_nan(self.total, self.count)
        return psnr_from_mse(mse)


@tf.keras.utils.register_keras_serializable()
class SpectralAngle(keras.metrics.CosineSimilarity):
    """
    Spectral Angle Mapper. Inherits from CosineSimilarity

    Fred A Kruse, AB Lefkoff, JW Boardman, KB Heidebrecht, AT Shapiro, PJ Barloon, and AFH Goetz.
    The spectral image processing system (sips) - interactive visualization and analysis of imaging spectrometer data.
    In AIP Conference Proceedings, volume 283, pages 192–201. American Institute of Physics, 1993.
    """
    def __init__(self, name='SAM', **kwargs):
        # Variables
        super().__init__(name, **kwargs)

    def result(self):
        # self.total and self.count have been computed in `update_state` by CosineSimilarity class
        cosine_similarity = tf.math.divide_no_nan(self.total, self.count)
        return tf.math.acos(cosine_similarity)


@tf.keras.utils.register_keras_serializable()
class StructuralSimilarity(MeanMetricWrapper):
    """
    Structural similarity metric. Inherits from MeanMetricWrapper, which handles computing the mean across all batches

    Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: From error visibility to
    structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, avril 2004.
    """
    def __init__(self, name='SSIM', **kwargs):
        # We pass the ssim function and its kwarg `max_val`
        super().__init__(tf.image.ssim, name=name, max_val=1.0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred, sample_weight)
