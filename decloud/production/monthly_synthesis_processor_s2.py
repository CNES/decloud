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
"""Process time series with Meraner-like models"""
import argparse
import datetime
import os
import sys
import logging

import otbApplication

from decloud.core import system
from decloud.preprocessing.constants import padded_tensor_name
from decloud.production.products import Factory as ProductsFactory
import pyotb


def monthly_synthesis_inference(sources, sources_scales, pad, ts, savedmodel_dir, out_tensor, out_nodatavalue,
                                out_pixeltype, nodatavalues=None):
    """
    Uses OTBTF TensorflowModelServe for the inference, perform some post-processing to keep only valid pixels.

    :param sources: a dict of sources, with keys=placeholder name, and value=str/otbimage
    :param sources_scales: a dict of sources scales (1=unit)
    :param pad: Margin size for blocking artefacts removal
    :param ts: Tile size. Tune this to process larger output image chunks, and speed up the process.
    :param savedmodel_dir: SavedModel directory
    :param out_tensor: output tensor name
    :param out_nodatavalue: NoData value for the output reconstructed S2t image
    :param out_pixeltype: PixelType for the output reconstructed S2t image
    :param nodatavalues: Optional, dictionary of NoData with keys=placeholder name
    :param with_20m_bands: Whether to compute the 20m bands. Default False
    """
    if nodatavalues is None:
        nodatavalues = {"s2_t0": -10000, "s2_t1": -10000, "s2_t2": -10000, "s2_t3": -10000, "s2_t4": -10000,
                        "s2_t5": -10000}

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
    infer = pyotb.App("TensorflowModelServe", frozen=True)

    # Setup BandMath for post processing
    bm = pyotb.App("BandMath", frozen=True)
    mask_expr = "0"

    # Inputs
    k = 0  # counter used for the im# of postprocessing mask
    for i, (placeholder, source) in enumerate(sources.items()):
        logging.info("Preparing source {} for placeholder {}".format(i + 1, placeholder))

        def get_key(key):
            """ Return the parameter key for the current source """
            return "source{}.{}".format(i + 1, key)

        src_rfield = rfield
        if placeholder in sources_scales:
            src_rfield = int(rfield / sources_scales[placeholder])

        infer.set_parameters({get_key("il"): [source]})

        # Update post processing BandMath expression
        if placeholder != 'dem' and '20m' not in placeholder:
            nodatavalue = nodatavalues[placeholder]
            n_channels = pyotb.get_nbchannels(source)
            mask_expr += "||"
            mask_expr += "&&".join(["im{}b{}=={}".format(k + 1, b, nodatavalue) for b in range(1, 1 + n_channels)])
            bm.set_parameters(il=[source])
            k += 1

        infer.set_parameters({get_key("rfieldx"): src_rfield,
                              get_key("rfieldy"): src_rfield,
                              get_key("placeholder"): placeholder})

    # Model
    infer.set_parameters({"model.dir": savedmodel_dir, "model.fullyconv": "on",
                          "output.names": [padded_tensor_name(out_tensor, pad)],
                          "output.efieldx": efield, "output.efieldy": efield,
                          "optim.tilesizex": efield, "optim.tilesizey": efield,
                          "optim.disabletiling": 1})
    infer.Execute()

    # For ESA Sentinel-2, remove potential zeros the network may have introduced in the valid parts of the image
    if out_pixeltype == otbApplication.ImagePixelType_uint16:
        n_channels = pyotb.get_nbchannels(infer.out)
        exp = ';'.join([f'(im1b{b}<=1 ? 1 : im1b{b})' for b in range(1, 1 + n_channels)])
        rmzeros = pyotb.App("BandMathX", il=[infer.out], exp=exp)
        rmzeros.SetParameterOutputImagePixelType("out", out_pixeltype)
    else:
        rmzeros = infer

    # Mask for post processing
    mask_expr += "?0:255"
    bm.set_parameters(exp=mask_expr)

    # Closing post processing mask to remove small groups of NoData pixels
    closing = pyotb.App("BinaryMorphologicalOperation", bm, filter="closing", foreval=255, structype="box",
                        xradius=5, yradius=5)

    # Erode post processing mask
    erode = pyotb.App("BinaryMorphologicalOperation", closing, filter="erode", foreval=255, structype="box",
                      xradius=pad, yradius=pad)

    # Superimpose the eroded post processing mask
    resample = pyotb.App("Superimpose", inm=erode, interpolator="nn", lms=192, inr=infer)

    # Apply nodata where the post processing mask is "0"
    mnodata = pyotb.App("ManageNoData", {"in": rmzeros, "mode": "apply", "mode.apply.mask": resample,
                                         "mode.apply.ndval": out_nodatavalue})

    mnodata.SetParameterOutputImagePixelType("out", out_pixeltype)
    return mnodata


if __name__ == "__main__":
    # Logger
    system.basic_logging_init()

    # Parser
    parser = argparse.ArgumentParser(
        description="Remove clouds in a time series of Sentinel-2 image, using joint optical and SAR images.")

    # Input images
    parser.add_argument("--il_s2", nargs='+', help="List of Sentinel-2 images, can be a list of paths or "
                                                   "a .txt file containing paths")
    parser.add_argument("--s2_dir", help="Directory of Sentinel-2 images. Enables to treat all the images of "
                                         "a directory. Used only if il_s2 is not specified")
    parser.add_argument("--dem", help="DEM path")
    parser.add_argument("--out_dir", required=True, help="Output directory for the monthly synthesis")
    parser.add_argument("--model", required=True, help="Path to the saved model directory, containing saved_model.pb")
    parser.add_argument("--ulx", help="Upper Left X of the ROI, in geographic coordinates. Optional", type=float)
    parser.add_argument("--uly", help="Upper Left Y of the ROI, in geographic coordinates. Optional", type=float)
    parser.add_argument("--lrx", help="Lower Right X of the ROI, in geographic coordinates. Optional", type=float)
    parser.add_argument("--lry", help="Lower Right Y of the ROI, in geographic coordinates. Optional", type=float)
    parser.add_argument("--year", help="Starting date, format YYYY-MM-DD. Optional")
    parser.add_argument("--month", help="End date, format YYYY-MM-DD. Optional")
    parser.add_argument('--ts', default=256, type=int,
                        help="Tile size. Tune this to process larger output image chunks, and speed up the process.")
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help="Whether to overwrite results if already exist")
    parser.set_defaults(overwrite=False)
    parser.add_argument('--write_intermediate', dest='write_intermediate', action='store_true',
                        help="Whether to write S1t & S2t input rasters used by the model.")
    parser.set_defaults(write_intermediate=False)

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    params = parser.parse_args()

    if not (params.il_s2 or params.s2_dir):
        raise Exception('Missing --il_s2 or --s2_dir argument')

    if params.il_s2 and params.s2_dir:
        logging.warning('Both --il_s2 and --s2_dir were specified. Discarding --s2_dir')
        params.s2_dir = None

    # Getting all the S2 filepaths
    if params.s2_dir:
        s2_image_paths = [os.path.join(params.s2_dir, name) for name in os.listdir(params.s2_dir)]
    elif params.il_s2[0].endswith('.txt'):
        with open(params.il_s2[0], 'r') as f:
            s2_image_paths = [x.strip() for x in f.readlines()]
    else:
        s2_image_paths = params.il_s2

    # Converting filepaths to S2 products
    input_s2_products = {}
    product_count, invalid_count = 0, 0
    for product_path in s2_image_paths:
        product = ProductsFactory.create(product_path, 's2', verbose=False)
        if product:
            input_s2_products[product_path] = product
            product_count += 1
        else:
            invalid_count += 1
    logging.info('Retrieved {} S2 products from disk. '
                 'Discarded {} paths that were not S2 products'.format(product_count, invalid_count))

    if not system.is_dir(params.out_dir):
        system.mkdir(params.out_dir)
    output_path = os.path.join(params.out_dir, 'monthly_synthesis_s2_{}{}.tif'.format(params.year, params.month))

    # ==================
    # Input parameters
    # ==================
    central_date = datetime.datetime(int(params.year), int(params.month), 15)
    delta_days = 22
    model_nb_images = 6

    # looping through the files
    candidates = []
    for s2_filepath, s2_product in input_s2_products.items():
        # We consider only images with no NoData
        if (abs(s2_product.get_date() - central_date) < datetime.timedelta(days=delta_days) and
                s2_product.get_nodata_percentage() < 0.05):
            candidates.append(s2_product)

    # If too many images, keeping only the 6 best images for synthesis
    if len(candidates) > model_nb_images:
        candidates.sort(key=lambda x: x.get_cloud_percentage())
        candidates = candidates[:model_nb_images]

    # Duplicating if not enough images
    if len(candidates) < model_nb_images:
        candidates.sort(key=lambda x: x.get_cloud_percentage())
        candidates = [candidates[0]] * (model_nb_images - len(candidates)) + candidates

    # Gathering as a dictionary
    sources = {}
    for i, candidate in enumerate(candidates):
        sources.update({'s2_t{}'.format(i): candidate.get_raster_10m()})

    # Sources scales
    sources_scales = {}

    if params.dem is not None:
        sources.update({"dem": params.dem})
        sources_scales.update({"dem": 2})

    # Inference
    out_tensor = "s2_estim"
    processor = monthly_synthesis_inference(sources=sources, sources_scales=sources_scales, pad=64,
                                            ts=params.ts, savedmodel_dir=params.model, out_tensor=out_tensor,
                                            out_nodatavalue=-10000, out_pixeltype=otbApplication.ImagePixelType_int16)

    # If needed, extracting ROI of the reconstructed image
    if params.lrx and params.lry and params.ulx and params.uly:
        processor = pyotb.App('ExtractROI',
                              {'in': processor, 'mode': 'extent', 'mode.extent.unit': 'phy',
                               'mode.extent.ulx': params.ulx, 'mode.extent.uly': params.uly,
                               'mode.extent.lrx': params.lrx, 'mode.extent.lry': params.lry})

    # OTB extended filename that will be used for all writing
    filename_extension = ("&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue={}&"
                          "gdal:co:COMPRESS=DEFLATE&gdal:co:TILED=YES".format(params.ts))

    processor.write(out=output_path, filename_extension=filename_extension)

    # Writing the inputs sources of the model
    if params.write_intermediate:
        for name, source in sources.items():
            if name != 'dem':
                # If needed, extracting ROI of every rasters
                if params.lrx and params.lry and params.ulx and params.uly:
                    source = pyotb.App('ExtractROI',
                                       {'in': source, 'mode': 'extent', 'mode.extent.unit': 'phy',
                                        'mode.extent.ulx': params.ulx, 'mode.extent.uly': params.uly,
                                        'mode.extent.lrx': params.lrx, 'mode.extent.lry': params.lry})
                pyotb.Input(source).write(os.path.join(os.path.dirname(output_path),
                                                       os.path.basename(output_path).replace('monthly_synthesis',
                                                                                             name)),
                                          pixel_type='int32', filename_extension=filename_extension)
