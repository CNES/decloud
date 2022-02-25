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
"""UNet model implementation (monthly synthesis of 6 optical & SAR couple of images)"""
from tensorflow.keras import layers
from decloud.models.model import Model
import decloud.preprocessing.constants as constants


class monthly_synthesis_6_s2s1_images(Model):
    def __init__(self, dataset_shapes,
                 dataset_input_keys=["s2_t0", "s2_t1", "s2_t2", "s2_t3", "s2_t4", "s2_t5",
                                     "s1_t0", "s1_t1", "s1_t2", "s1_t3", "s1_t4", "s1_t5", constants.DEM_KEY],
                 model_output_keys=["s2_target"]):
        super().__init__(dataset_input_keys=dataset_input_keys, model_output_keys=model_output_keys,
                         dataset_shapes=dataset_shapes)

    def get_outputs(self, normalized_inputs):

        # The network
        features = {factor: [] for factor in [1, 2, 4, 8, 16, 32]}
        conv1_s2 = layers.Conv2D(64, 5, 1, activation='relu', name="conv1_s2_relu", padding="same")
        conv1_s1 = layers.Conv2D(64, 5, 1, activation='relu', name="conv1_s1_relu", padding="same")
        conv2_s2 = layers.Conv2D(128, 3, 2, activation='relu', name="conv2_s2_relu", padding="same")
        conv2_s1 = layers.Conv2D(128, 3, 2, activation='relu', name="conv2_s1_relu", padding="same")
        conv3 = layers.Conv2D(256, 3, 2, activation='relu', name="conv3_bn_relu", padding="same")
        conv4 = layers.Conv2D(512, 3, 2, activation='relu', name="conv4_bn_relu", padding="same")
        conv5 = layers.Conv2D(512, 3, 2, activation='relu', name="conv5_bn_relu", padding="same")
        conv6 = layers.Conv2D(512, 3, 2, activation='relu', name="conv6_bn_relu", padding="same")
        deconv1 = layers.Conv2DTranspose(512, 3, 2, activation='relu', name="deconv1_bn_relu", padding="same")
        deconv2 = layers.Conv2DTranspose(512, 3, 2, activation='relu', name="deconv2_bn_relu", padding="same")
        deconv3 = layers.Conv2DTranspose(256, 3, 2, activation='relu', name="deconv3_bn_relu", padding="same")
        deconv4 = layers.Conv2DTranspose(128, 3, 2, activation='relu', name="deconv4_bn_relu", padding="same")
        deconv5 = layers.Conv2DTranspose(64, 3, 2, activation='relu', name="deconv5_bn_relu", padding="same")
        conv_final = layers.Conv2D(4, 5, 1, name="s2_estim", padding="same")

        for key, input_image in normalized_inputs.items():
            if key != constants.DEM_KEY:
                if key.startswith('s1'):
                    net = conv1_s1(input_image)
                    features[1].append(net)
                    net = conv2_s1(net)
                elif key.startswith('s2'):
                    net = conv1_s2(input_image)
                    features[1].append(net)
                    net = conv2_s2(net)
                if self.has_dem():
                    net = layers.concatenate([net, normalized_inputs[constants.DEM_KEY]], axis=-1)
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
        net = deconv5(net)  # 256
        net = _combine(factor=1, x=net)

        s2_out = conv_final(net)

        return {"s2_target": s2_out}  # key must correspond to the key from the dataset
