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
"""Base class for gapfilling models"""
import tensorflow as tf
from decloud.models.model import Model


class gapfilling_base(Model):
    """
    Base class for gapfilling models
    """

    @staticmethod
    def interpolate(prec_data, next_data, prec_date, next_date, target_date):
        """
        Linear interpolation of a pixel between two dates
        """
        diff_data = tf.math.subtract(next_data, prec_data)
        diff_time = tf.math.subtract(next_date, prec_date)

        delta_time = tf.math.subtract(target_date, prec_date)
        divide_time = tf.math.divide(delta_time, diff_time)
        divide_time = tf.reshape(divide_time, shape=[tf.shape(diff_data)[0], 1, 1, 1])

        diff_data = tf.cast(diff_data, tf.float32)
        divide_time = tf.cast(divide_time, tf.float32)
        inter = tf.math.multiply(diff_data, divide_time)
        return tf.math.add(prec_data, inter)
