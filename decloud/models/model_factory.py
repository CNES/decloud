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
"""Model factory"""
import importlib


class ModelFactory:
    """
    Factory to load a model.
    Model's file must be have the same name as model and in decloud/models repository
    Ex :
        Model crga_os2_unet's file must be named crga_os2_unet.py
    """
    def __init__(self):
        pass

    @staticmethod
    def get_model(name, **kwargs):
        """
        :param name: Model's name to load
        :param kwargs: Input data's dictionary
        :return: Model instance
        """
        try:
            cls = getattr(importlib.import_module('decloud.models.{}'.format(name)), name)

            return cls(**kwargs)

        except ImportError as e:
            raise e
