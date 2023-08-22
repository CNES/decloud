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
"""David model implementation (monthly synthesis of 6 optical images)"""
from tensorflow.keras import layers
from decloud.models.model import Model
from tensorflow import concat


class monthly_synthesis_6_s2_images_david(Model):
    def __init__(self, dataset_shapes,
                 dataset_input_keys=["s2_t0", "s2_t1", "s2_t2", "s2_t3", "s2_t4", "s2_t5"],
                 model_output_keys=["s2_target"]):
        super().__init__(dataset_input_keys=dataset_input_keys, model_output_keys=model_output_keys,
                         dataset_shapes=dataset_shapes)

    def get_outputs(self, normalized_inputs):
        # The network
        features = []
        conv1 = layers.Conv2D(64, 5, 1, activation='relu', name="conv1_s2_relu", padding="same")
        conv2 = layers.Conv2D(128, 3, 2, activation='relu', name="conv2_bn_relu", padding="same")
        conv3 = layers.Conv2D(256, 3, 2, activation='relu', name="conv3_bn_relu", padding="same")
        deconv1 = layers.Conv2DTranspose(128, 3, 2, activation='relu', name="deconv1_bn_relu", padding="same")
        deconv2 = layers.Conv2DTranspose(64, 3, 2, activation='relu', name="deconv2_bn_relu", padding="same")
        conv4 = layers.Conv2D(4, 5, 1, activation='relu', name="s2_estim", padding="same")

        for key, input_image in normalized_inputs.items():
            net = conv1(input_image)  # 256
            net = conv2(net)  # 128
            net = conv3(net)  # 64
            features.append(net)

        net = concat(features, axis=-1)
        net = deconv1(net)  # 128
        net = deconv2(net)  # 256
        s2_out = conv4(net)  # 256

        return {"s2_target": s2_out}  # key must correspond to the key from the dataset
