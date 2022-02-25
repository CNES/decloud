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
"""Implementation of the Meraner et al. original network"""
from tensorflow.keras import layers
from decloud.models.model import Model


class meraner_original(Model):
    """
    Implementation of the Meraner et al. original network
    see https://doi.org/10.1016/j.isprsjprs.2020.05.013
    """

    def __init__(self, dataset_shapes, dataset_input_keys=["s1_t", "s2_t"], model_output_keys=["s2_target"]):
        super().__init__(dataset_input_keys=dataset_input_keys, model_output_keys=model_output_keys,
                         dataset_shapes=dataset_shapes)

    def get_outputs(self, normalized_inputs):
        resblocks_dim = 256
        n_resblocks = 16

        # ResNet block helper
        def _resblock(x, i):
            resconv1 = layers.Conv2D(resblocks_dim, 3, 1, name="block{}_conv1_relu".format(i), padding="same",
                                     activation='relu')
            resconv2 = layers.Conv2D(resblocks_dim, 3, 1, name="block{}_conv2_relu".format(i), padding="same")
            out = resconv1(x)
            out = resconv2(out)
            return layers.ReLU()(0.1 * out + x)  # Residual scaling

        # The network
        conv1 = layers.Conv2D(resblocks_dim, 3, 1, activation='relu', name="conv1_relu", padding="same")
        net = layers.concatenate([normalized_inputs["s1_t"], normalized_inputs["s2_t"]], axis=-1)
        net = conv1(net)
        for i in range(n_resblocks):
            net = _resblock(net, i)
        conv2 = layers.Conv2D(4, 3, 1, name="conv2", padding="same")
        net = conv2(net)

        net = layers.Add()([net, normalized_inputs["s2_t"]])

        return {"s2_target": net}
