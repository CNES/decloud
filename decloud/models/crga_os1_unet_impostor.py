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
"""CRGA OS1 UNet impostor model"""
from decloud.models.crga_os1_unet import crga_os1_unet
import decloud.preprocessing.constants as constants


class crga_os1_unet_impostor(crga_os1_unet):
    """
    CRGA OS1 UNet impostor model
    This model can be used on the CRGA_OS2-3 acquisitions layout to compute metrics and compare with the crga_os2_unet
    model. This model only uses "s2_target" as reference instead of "s2_t".
    """

    def __init__(self, dataset_shapes,
                 dataset_input_keys=["s1_tm1", "s1_target", "s1_tp1", "s2_tm1", "s2_tp1", constants.DEM_KEY],
                 model_output_keys=["s2_target"]):
        super().__init__(dataset_input_keys=dataset_input_keys, model_output_keys=model_output_keys,
                         dataset_shapes=dataset_shapes)

    def get_outputs(self, normalized_inputs):
        # The dataset has been loaded with the right data ('s1_target'). But now we need to replace its key by 's1_t'
        # so that it can be used in crga_os1_unet
        normalized_inputs['s1_t'] = normalized_inputs.pop('s1_target')
        outputs = super().get_outputs(normalized_inputs=normalized_inputs)
        return {"s2_target": outputs["s2_t"]}
