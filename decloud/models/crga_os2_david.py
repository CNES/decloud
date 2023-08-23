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
"""CRGA OS2 David model"""
from tensorflow.keras import layers
from decloud.models.crga_os2_base import crga_os2_base
import decloud.preprocessing.constants as constants
from tensorflow import concat


class crga_os2_david(crga_os2_base):
    """
    CRGA OS2 David model
    David (like in "David versus Goliath") is a tiny model that is studied to see
    how does a small model perform against a lot of data, compared to a bigger one
    (Goliath aka CRGA OS2 UNet model).
    """

    def get_outputs(self, normalized_inputs):

        input_dict = {"ante": [normalized_inputs["s1_tm1"], normalized_inputs["s2_tm1"]],
                      "current": [normalized_inputs["s1_t"], normalized_inputs["s2_t"]],
                      "post": [normalized_inputs["s1_tp1"], normalized_inputs["s2_tp1"]]}

        # The network
        features = []
        conv1 = layers.Conv2D(64, 5, 1, activation='relu', name="conv1_relu", padding="same")
        conv1_dem = layers.Conv2D(64, 3, 1, activation='relu', name="conv1_dem_relu", padding="same")
        conv2 = layers.Conv2D(128, 3, 2, activation='relu', name="conv2_bn_relu", padding="same")
        conv3 = layers.Conv2D(256, 3, 2, activation='relu', name="conv3_bn_relu", padding="same")
        deconv1 = layers.Conv2DTranspose(128, 3, 2, activation='relu', name="deconv1_bn_relu", padding="same")
        deconv2 = layers.Conv2DTranspose(64, 3, 2, activation='relu', name="deconv2_bn_relu", padding="same")
        conv4 = layers.Conv2D(4, 5, 1, activation='relu', name="s2_estim", padding="same")
        for input_image in input_dict:
            net = concat(input_dict[input_image], axis=-1)
            net = conv1(net)  # 256
            net = conv2(net)  # 128
            if self.has_dem():
                net_dem = conv1_dem(normalized_inputs[constants.DEM_KEY])
                net = concat([net, net_dem], axis=-1)
            net = conv3(net)  # 64
            features.append(net)

        net = concat(features, axis=-1)
        net = deconv1(net)    # 128
        net = deconv2(net)    # 256
        s2_out = conv4(net)   # 256

        return {"s2_target": s2_out}  # key must correspond to the key from the dataset
