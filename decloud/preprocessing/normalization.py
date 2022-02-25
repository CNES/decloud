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
"""Sentinel images and DEM normalization """
import tensorflow as tf
import decloud.preprocessing.constants as constants


def normalize_s1(input_image, dtype=tf.float32):
    """ Normalize Sentinel-1 image """
    return constants.S1_SCALE_COEF * tf.cast(input_image, dtype)


def normalize_s2(input_image, dtype=tf.float32):
    """ Normalize Sentinel-2 image """
    return constants.S2_SCALE_COEF * tf.cast(input_image, dtype)


def normalize_dem(input_image, dtype=tf.float32):
    """ Normalize DEM """
    return constants.DEM_SCALE_COEF * tf.cast(input_image, dtype)


def normalize(key, placeholder):
    """
    Normalize an input placeholder, knowing its key
    :param key: placeholder key
    :param placeholder: placeholder
    :return: normalized placeholder
    """
    func = None
    if key.startswith("s1"):
        func = normalize_s1
    elif key.startswith("s2"):
        func = normalize_s2
    elif key == constants.DEM_KEY:
        func = normalize_dem
    else:
        # Do not normalize
        pass
    if func is not None:
        return func(placeholder)
    return placeholder


def denormalize_s2(input_image, dtype=tf.float32):
    """ De-normalize Sentinel-2 image """
    return constants.S2_UNSCALE_COEF * tf.cast(input_image, dtype)
