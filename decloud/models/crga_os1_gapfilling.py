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
"""Gapfilling model implementation (OS1)"""
from decloud.models.gapfilling_base import gapfilling_base


class crga_os1_gapfilling(gapfilling_base):
    """
    Gapfilling model implementation (OS1)
    """

    def __init__(self, dataset_shapes,
                 dataset_input_keys=["s2_tm1", "s2_tp1",
                                     "s2_timestamp_tm1", "s2_timestamp_t", "s2_timestamp_tp1"],
                 model_output_keys=["s2_t"]):
        super().__init__(dataset_input_keys=dataset_input_keys, model_output_keys=model_output_keys,
                         dataset_shapes=dataset_shapes)

    def get_outputs(self, normalized_inputs):
        interp = self.interpolate(prec_data=normalized_inputs["s2_tm1"],
                                  next_data=normalized_inputs["s2_tp1"],
                                  prec_date=normalized_inputs["s2_timestamp_tm1"],
                                  next_date=normalized_inputs["s2_timestamp_tp1"],
                                  target_date=normalized_inputs["s2_timestamp_t"])
        return {"s2_t": interp}
