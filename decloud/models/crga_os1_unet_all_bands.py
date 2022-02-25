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
"""CRGA OS1 UNet model (all bands)"""
from tensorflow.keras import layers
import decloud.preprocessing.constants as constants
from decloud.models.crga_os1_base_all_bands import crga_os1_base_all_bands


class crga_os1_unet_all_bands(crga_os1_base_all_bands):
    """
    CRGA OS1 UNet model (all bands)
    """
    def get_outputs(self, normalized_inputs):

        input_dict = {"ante": [normalized_inputs["s1_tm1"],
                               normalized_inputs["s2_tm1"],
                               normalized_inputs["s2_20m_tm1"]],
                      "current": normalized_inputs["s1_t"],
                      "post": [normalized_inputs["s1_tp1"],
                               normalized_inputs["s2_tp1"],
                               normalized_inputs["s2_20m_tp1"]]}

        # The network
        features = {factor: [] for factor in [1, 2, 4, 8, 16, 32]}
        conv1_s1 = layers.Conv2D(64, 5, 1, activation='relu', name="conv1_s1_relu", padding="same")
        conv1_s1s2 = layers.Conv2D(64, 5, 1, activation='relu', name="conv1_s1s2_relu", padding="same")
        conv1_dem = layers.Conv2D(64, 3, 1, activation='relu', name="conv1_dem_relu", padding="same")
        conv1_20m = layers.Conv2D(64, 3, 1, activation='relu', name="conv1_20m_relu", padding="same")
        conv2 = layers.Conv2D(128, 3, 2, activation='relu', name="conv2_bn_relu", padding="same")
        conv2_20m = layers.Conv2D(128, 3, 1, activation='relu', name="conv2_20m_bn_relu", padding="same")
        conv3 = layers.Conv2D(256, 3, 2, activation='relu', name="conv3_bn_relu", padding="same")
        conv4 = layers.Conv2D(512, 3, 2, activation='relu', name="conv4_bn_relu", padding="same")
        conv5 = layers.Conv2D(512, 3, 2, activation='relu', name="conv5_bn_relu", padding="same")
        conv6 = layers.Conv2D(512, 3, 2, activation='relu', name="conv6_bn_relu", padding="same")
        deconv1 = layers.Conv2DTranspose(512, 3, 2, activation='relu', name="deconv1_bn_relu", padding="same")
        deconv2 = layers.Conv2DTranspose(512, 3, 2, activation='relu', name="deconv2_bn_relu", padding="same")
        deconv3 = layers.Conv2DTranspose(256, 3, 2, activation='relu', name="deconv3_bn_relu", padding="same")
        deconv4 = layers.Conv2DTranspose(128, 3, 2, activation='relu', name="deconv4_bn_relu", padding="same")
        deconv5 = layers.Conv2DTranspose(64, 3, 2, activation='relu', name="deconv5_bn_relu", padding="same")
        deconv5_20m = layers.Conv2DTranspose(64, 3, 1, activation='relu', name="deconv5_20m_bn_relu", padding="same")
        conv_final = layers.Conv2D(4, 5, 1, name="s2_estim", padding="same")
        conv_20m_final = layers.Conv2D(6, 3, 1, name="s2_20m_estim", padding="same")

        # The network
        features = {factor: [] for factor in [1, 2, 4, 8, 16, 32]}
        for input_image in input_dict:
            if input_image == "current":  # there is only s1
                net_10m = conv1_s1(input_dict[input_image])  # 256
                features[1].append(net_10m)
                net = conv2(net_10m)  # 128
            else:  # for post & ante, the is s1, s2 and s2_20m
                net_10m = layers.concatenate(input_dict[input_image][:2], axis=-1)
                net_10m = conv1_s1s2(net_10m)  # 256
                features[1].append(net_10m)
                net_10m = conv2(net_10m)  # 128
                net_20m = conv1_20m(input_dict[input_image][2])  # 128
                features_20m = [net_10m, net_20m]
                if self.has_dem():
                    features_20m.append(conv1_dem(normalized_inputs[constants.DEM_KEY]))
                net = layers.concatenate(features_20m, axis=-1)
                net = conv2_20m(net)  # 128

            features[2].append(net)
            net = conv3(net)  # 64
            features[4].append(net)
            net = conv4(net)  # 32
            features[8].append(net)
            net = conv5(net)  # 16
            features[16].append(net)
            net = conv6(net)  # 8
            features[32].append(net)

        # Decoder
        def _combine(factor, x=None):
            if x is not None:
                features[factor].append(x)
            return layers.concatenate(features[factor], axis=-1)

        net = _combine(factor=32)
        net = deconv1(net)  # 16
        net = _combine(factor=16, x=net)
        net = deconv2(net)  # 32
        net = _combine(factor=8, x=net)
        net = deconv3(net)  # 64
        net = _combine(factor=4, x=net)
        net = deconv4(net)  # 128
        net = _combine(factor=2, x=net)
        net_20m = deconv5_20m(net)  # 128
        net = deconv5(net)  # 256
        net_10m = _combine(factor=1, x=net)

        s2_out = conv_final(net_10m)
        s2_20m_out = conv_20m_final(net_20m)

        # 10m-resampled stack that will be the output for inference (not used for training)
        s2_20m_resampled = layers.UpSampling2D(size=(2, 2))(s2_20m_out)
        s2_all_bands = layers.concatenate([s2_out, s2_20m_resampled], axis=-1)

        return {"s2_t": s2_out, "s2_20m_t": s2_20m_out, 's2_all_bands_estim': s2_all_bands}
