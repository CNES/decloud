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
"""Constants"""
import re


MINDB = -10
MAXDB = 3
UINT16 = 65535
EPSILON = 10**-6
SUFFIX_S1 = "from{}to{}dB".format(MINDB, MAXDB)
SUFFIX_STATS_S1 = "edge.tif"
SUFFIX_STATS_S2 = "stats.tif"
SENTINEL_TILE = re.compile("T[0-9]+[A-Z]*")
PATCHSIZE_REF = 64
S1_UNSCALE_COEF = 65535.0
S1_SCALE_COEF = 1.0 / S1_UNSCALE_COEF
S2_UNSCALE_COEF = 10000.0
S2_SCALE_COEF = 1.0 / S2_UNSCALE_COEF
DEM_SCALE_COEF = 0.0001
DEM_KEY = "dem"
PADS = [32, 64, 128, 256]
IS_TRAINING = "is_training"
DROPOUT_RATE = "drop_rate"
LEARNING_RATE = "lr"

# BandMath expression for Sentinel-1 channel normalization
# - 0 if pixel == 0,
# - 1 if pixel < exp(mindb),
# - UINT16 if pixel > exp(maxdb),
# - 1+({UINT16}-1)/({maxdb}-{mindb})*(ln(abs(im1b1)+{eps})-{mindb} else
S1_NORMALIZATION_BM_EXPR = "im1b1==0?0:im1b1<exp({mindb})?1:im1b1>exp({maxdb})?{UINT16}:" \
                           "1+({UINT16}-1)/({maxdb}-{mindb})*(ln(abs(im1b1)+{eps})-{mindb}" \
                           ")".format(mindb=MINDB, maxdb=MAXDB, UINT16=UINT16, eps=EPSILON)


def padded_tensor_name(tensor_name, pad):
    """
    A name for the padded tensor
    :param tensor_name: tensor name
    :param pad: pad value
    :return: name
    """
    return "{}_pad{}".format(tensor_name, pad)
