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
"""Model class and helpers"""
import abc
import tensorflow as tf
from tensorflow import keras
import decloud.preprocessing.constants as constants
from decloud.preprocessing.normalization import normalize
from decloud.models.utils import _is_chief


# ------------------------------------------------- Model class --------------------------------------------------------


class Model(abc.ABC):
    """
    Base class for all models
    """

    @abc.abstractmethod
    def __init__(self, dataset_input_keys, model_output_keys, dataset_shapes):
        """
        Model base class

        :param dataset_input_keys: list of dataset keys used for the training
        :param model_output_keys: list of the model outputs keys
        :param dataset_shapes: a dict() of shapes
        """
        self.dataset_input_keys = dataset_input_keys
        self.model_output_keys = model_output_keys
        self.dataset_shapes = dataset_shapes
        self.model = None

    def __getattr__(self, name):
        """This method is called when the default attribute access fails. We choose to try to access the attribute of
        self.model. Thus, any method of keras.Model() can be used transparently,
        e.g. model.summary() or model.fit()"""
        if not self.model:
            raise Exception("model is None. Call create_network() before using it!")
        return getattr(self.model, name)

    def get_inputs(self):
        """
        This method returns the dict of inputs
        """
        # Create model inputs with S1, S2 and DEM images
        model_inputs = {}
        for key in self.dataset_input_keys:
            shape = self.dataset_shapes[key]
            # Remove the potential batch dimension, because keras.Input() doesn't want the batch dimension
            if len(shape) > 3:
                shape = shape[1:]
            # Here we modify the x and y dims of >2D tensors to enable any image size at input
            if len(shape) > 2:
                shape[0] = None
                shape[1] = None
            placeholder = keras.Input(shape=shape, name=key)
            model_inputs.update({key: placeholder})
        return model_inputs

    def create_network(self):
        """
        This method returns the Keras model. This needs to be called **inside** the strategy.scope()
        :return: the keras model
        """

        # Get the model inputs
        model_inputs = self.get_inputs()

        # Normalize the inputs
        normalized_inputs = {key: normalize(key, input) for key, input in model_inputs.items()}

        # Build the model
        outputs = self.get_outputs(normalized_inputs)

        # Add extra outputs
        extra_outputs = {}
        for out_key, out_tensor in outputs.items():
            for pad in constants.PADS:
                extra_output_key = constants.padded_tensor_name(out_key, pad)
                extra_output_name = constants.padded_tensor_name(out_tensor._keras_history.layer.name, pad)
                scale = constants.S2_UNSCALE_COEF
                extra_output = tf.keras.layers.Cropping2D(cropping=pad, name=extra_output_name)(scale * out_tensor)
                extra_outputs[extra_output_key] = extra_output
        outputs.update(extra_outputs)

        # Return the keras model
        self.model = keras.Model(inputs=model_inputs, outputs=outputs, name=self.__class__.__name__)

    def has_dem(self):
        """
        :return: True is the model has a DEM_KEY in its inputs
        """
        return constants.DEM_KEY in self.dataset_input_keys

    def get_loss(self):
        """
        Loss type.
        Can be re-implemented in child classes.
        """
        return 'mean_absolute_error'

    @abc.abstractmethod
    def get_outputs(self, normalized_inputs):
        """
        Implementation of the model
        :param normalized_inputs: normalized inputs
        :return: a dict of outputs tensors of the model
        """
        pass

    def summary(self, strategy=None):
        """
        Wraps the summary printing of the model. When multiworker strategy, only prints if the worker is chief
        """
        if not strategy or _is_chief(strategy):
            self.model.summary(line_length=150)

    def plot(self, output_path, strategy=None):
        """
        Enables to save a figure representing the architecture of the network.
        //!\\ only works if create_network() has been called beforehand
        """
        # When multiworker strategy, only plot if the worker is chief
        if not strategy or _is_chief(strategy):
            # Build a simplified model, without normalization nor extra outputs.
            # This model is only used for plotting the architecture thanks to `keras.utils.plot_model`
            inputs = self.get_inputs()  # inputs without normalization
            outputs = self.get_outputs(inputs)  # raw model outputs
            model_simplified = keras.Model(inputs=inputs, outputs=outputs, name=self.__class__.__name__ + '_simplified')
            keras.utils.plot_model(model_simplified, output_path)
