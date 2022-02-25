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
"""Base class for CRGA OS2 models (all bands)"""
from decloud.models.model import Model
import decloud.preprocessing.constants as constants


class crga_os2_base_all_bands(Model):
    """
    Base class for CRGA OS2 models (all bands)
    """

    def __init__(self, dataset_shapes,
                 dataset_input_keys=["s1_tm1", "s1_t", "s1_tp1", "s2_tm1", "s2_t", "s2_tp1",
                                     "s2_20m_tm1", "s2_20m_t", "s2_20m_tp1", constants.DEM_KEY],
                 model_output_keys=["s2_target", "s2_20m_target", "s2_all_bands_estim"]):
        super().__init__(dataset_input_keys=dataset_input_keys, model_output_keys=model_output_keys,
                         dataset_shapes=dataset_shapes)
