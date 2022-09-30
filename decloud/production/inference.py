#!/usr/bin/env python3
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
"""
Base component of inference for models
"""

from decloud.core import system
import otbApplication
import logging
from decloud.preprocessing.constants import padded_tensor_name
import pyotb


def inference(sources, sources_scales, pad, ts, savedmodel_dir, out_tensor, out_nodatavalue, out_pixeltype,
              nodatavalues=None):
    """
    Uses OTBTF TensorflowModelServe for the inference, perform some post-processing to keep only valid pixels.

    :param sources: a dict of sources, with keys=placeholder name, and value=str or pyotb object
    :param sources_scales: a dict of sources scales (1=unit)
    :param pad: Margin size for blocking artefacts removal
    :param ts: Tile size. Tune this to process larger output image chunks, and speed up the process.
    :param savedmodel_dir: SavedModel directory
    :param out_tensor: output tensor name
    :param out_nodatavalue: NoData value for the output reconstructed S2t image
    :param out_pixeltype: PixelType for the output reconstructed S2t image
    :param nodatavalues: Optional, dictionary of NoData with keys=placeholder name
    """

    logging.info("Setup inference pipeline")
    logging.info("Input sources: {}".format(sources))
    logging.info("Input pad: {}".format(pad))
    logging.info("Input tile size: {}".format(ts))
    logging.info("SavedModel directory: {}".format(savedmodel_dir))
    logging.info("Output tensor name: {}".format(out_tensor))

    # Receptive/expression fields
    gen_fcn = pad
    efield = ts  # Expression field
    if efield % 64 != 0:
        logging.fatal("Please chose a tile size that is a multiple of 64")
        quit()
    rfield = int(efield + 2 * gen_fcn)  # Receptive field
    logging.info("Receptive field: {}, Expression field: {}".format(rfield, efield))

    # Setup TensorFlowModelServe
    system.set_env_var("OTB_TF_NSOURCES", str(len(sources)))
    parameters = {}

    # Inputs
    for i, (placeholder, source) in enumerate(sources.items()):
        logging.info("Preparing source {} for placeholder {}".format(i + 1, placeholder))

        def get_key(key):
            """ Return the parameter key for the current source """
            return "source{}.{}".format(i + 1, key)

        src_rfield = rfield
        if placeholder in sources_scales:
            src_rfield = int(rfield / sources_scales[placeholder])

        parameters.update({get_key("il"): [source],
                           get_key("rfieldx"): src_rfield,
                           get_key("rfieldy"): src_rfield,
                           get_key("placeholder"): placeholder})

    # Model
    parameters.update({"model.dir": savedmodel_dir, "model.fullyconv": True,
                       "output.names": [padded_tensor_name(out_tensor, pad)],
                       "output.efieldx": efield, "output.efieldy": efield,
                       "optim.tilesizex": efield, "optim.tilesizey": efield,
                       "optim.disabletiling": True})
    infer = pyotb.TensorflowModelServe(parameters)

    # Post Processing
    # For ESA Sentinel-2, remove potential zeros the network may have introduced in the valid parts of the image
    if out_pixeltype == otbApplication.ImagePixelType_uint16:
        infer = pyotb.where(infer <= 1, 1, infer)

    # Applying the NoDatas from input sources to the output result
    if nodatavalues:
        # Potentially transform filepath to pyotb object
        sources = {k: pyotb.Input(v) if isinstance(v, str) else v for k, v in sources.items()}
        # Defining valid data masks for each source (1=valid data, 0=NoData); i.e. when at least one band is not NoData
        source_masks = [pyotb.any((sources[placeholder] != nodata)) for placeholder, nodata in nodatavalues.items()]
        # Merging all the masks: all sources must be valid for the merged mask to be considered valid
        merged_mask = pyotb.all(*source_masks)

        # Closing post processing mask to remove small groups of NoData pixels
        closing = pyotb.BinaryMorphologicalOperation(merged_mask, filter="closing", foreval=1, structype="box",
                                                     xradius=5, yradius=5)

        # Erode post processing mask
        erode = pyotb.BinaryMorphologicalOperation(closing, filter="erode", foreval=1, structype="box",
                                                   xradius=pad, yradius=pad)

        # Superimpose the eroded post processing mask
        resample = pyotb.Superimpose(inm=erode, interpolator="nn", lms=192, inr=infer)

        # Apply nodata where the post processing mask is "0"
        mnodata = pyotb.ManageNoData({"in": infer, "mode": "apply", "mode.apply.mask": resample,
                                      "mode.apply.ndval": out_nodatavalue})
        mnodata.SetParameterOutputImagePixelType("out", out_pixeltype)
        return mnodata

    infer.SetParameterOutputImagePixelType("out", out_pixeltype)
    return infer
