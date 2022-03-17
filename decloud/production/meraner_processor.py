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
"""Processor for Meraner-like models"""
import argparse
import os
import sys

from decloud.production.products import Factory as ProductsFactory
from decloud.production.inference import inference
import decloud.preprocessing.constants as constants
from decloud.core import system
import pyotb


def meraner_processor(il_s1, in_s2, savedmodel, dem=None, output=None, output_20m=None,
                      s1_Nimages=12, ts=256, pad=64, with_20m_bands=False, with_intermediate=False):
    """
    Apply Meraner model to input sources.

    :param il_s1: list of paths to Sentinel 1 products
    :param in_s2: path of Sentinel 2 product to be reconstructed
    :param savedmodel: path of the savedmodel (folder containing the .pb)
    :param dem: path of the DEM
    :param s1_Nimages: max number of Sentinel-1 images to consider for the mosaic
    :param output: output path for the reconstructed image. If not specified, the result is returned as a pyotb object
    :param output_20m: output path for the reconstructed 20m bands image
    :param pad: Margin size for blocking artefacts removal
    :param ts: Tile size. Tune this to process larger output image chunks, and speed up the process.
    :param with_20m_bands: whether to compute the 20m bands. If True, the saved model must have been trained accordingly
    :param with_intermediate: whether to write/return input sources (s1_t and s2_t)

    :return output, (sources): if output path is not specified, returns reconstructed in-memory pyotb object
                               optionally, if with_indermediate, also returns the input sources
    """

    s2t_product = ProductsFactory.create(in_s2, 's2')
    input_s1_products = [ProductsFactory.create(pth, 's1') for pth in il_s1]

    def _closest_date(x):
        """
        Helper to sort S1 products
        :param x: input product
        :return: rank
        """
        return abs(s2t_product.get_timestamp() - x.get_timestamp())

    input_s1_products.sort(key=_closest_date, reverse=True)
    input_s1_products = input_s1_products[:s1_Nimages]

    # Getting the 10m rasters
    input_s1_images_10m = [product.get_raster_10m() for product in input_s1_products]
    s2t = s2t_product.get_raster_10m()

    sources = {"s1_t": pyotb.Mosaic(il=input_s1_images_10m, nodata=0),
               "s2_t": s2t}

    if with_20m_bands:
        s2t_20m = s2t_product.get_raster_20m()
        sources.update({"s2_20m_t": s2t_20m})
    if dem:
        sources.update({"dem": dem})

    # Resolution factor
    sources_scales = {"dem": 2, "s2_20m_t": 2}

    # OTB extended filename that will be used for all writing
    filename_extension = ("&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue={}&"
                          "gdal:co:COMPRESS=DEFLATE&gdal:co:TILED=YES".format(ts))

    # Inference
    if with_20m_bands:
        out_tensor = "s2_all_bands_estim"  # stack of the 10m reconstruction and 20m reconstruction (resampled to 10m)
        resampled_all_bands = inference(sources, sources_scales, pad=pad, ts=ts,
                                        savedmodel_dir=savedmodel, out_tensor=out_tensor,
                                        out_nodatavalue=s2t_product.get_nodatavalue(),
                                        out_pixeltype=s2t_product.get_raster_10m_encoding(),
                                        nodatavalues={"s1_t": 0, "s2_t": -10000})

        # If the user didn't specify output paths, return in-memory object
        if not (output and output_20m):
            if with_intermediate:
                return resampled_all_bands, sources
            return resampled_all_bands

        # temporary path for the stack of 10m+20m bands
        temp_outpath = system.join(system.dirname(output_20m), 'temp_all_bands_' + system.basename(output_20m))
        resampled_all_bands.write(out=temp_outpath, filename_extension=filename_extension)

        stack = pyotb.Input(temp_outpath)
        # Writing 10m bands
        stack[:, :, :4].write(out=output, filename_extension=filename_extension)
        # Resampling and writing 20m bands
        final_res_20m = pyotb.RigidTransformResample({'in': stack[:, :, 4:], 'interpolator': 'nn',
                                                      'transform.type.id.scaley': 0.5, 'transform.type.id.scalex': 0.5})
        final_res_20m.write(out=output_20m, filename_extension=filename_extension)
        os.remove(temp_outpath)

    else:
        out_tensor = "s2_estim"
        processor = inference(sources, sources_scales, pad=pad, ts=ts,
                              savedmodel_dir=savedmodel, out_tensor=out_tensor,
                              out_nodatavalue=s2t_product.get_nodatavalue(),
                              out_pixeltype=s2t_product.get_raster_10m_encoding(),
                              nodatavalues={"s1_t": 0, "s2_t": -10000})

        # If the user didn't specify output path, return in-memory object
        if not output:
            if with_intermediate:
                return processor, sources
            return processor

        processor.write(out="{}".format(output), filename_extension=filename_extension)

        if with_intermediate and output:
            # Writing the input sources, in the same directory as output with a suffix
            for name, source in sources.items():
                if name != 'dem':
                    source.write(os.path.splitext(output)[0] + '_{}.tif'.format(name),
                                 pixel_type='int32',
                                 filename_extension=filename_extension)


# ------------------------------------------------------- Main ---------------------------------------------------------
def main():
    # Logger
    system.basic_logging_init()

    # Parser
    parser = argparse.ArgumentParser(description="Remove clouds in a Sentinel-2 image using SAR images.")

    # Input images
    parser.add_argument("--il_s1", required=True, help="Sentinel-1 products to use at date t", nargs='+', default=[])
    parser.add_argument("--in_s2", required=True, help="Sentinel-2 product at date t to reconstruct")
    parser.add_argument("--in_dem", help="DEM")
    parser.add_argument('--with_20m_bands', dest='with_20m_bands', action='store_true',
                        help="Remove clouds on all 10m & 20m bands")
    parser.set_defaults(with_20m_bands=False)

    # Output image
    parser.add_argument("--output", required=True, help="Output file for the Sentinel-2 image de-clouded at date t")
    parser.add_argument("--output_20m", help="Output file for the Sentinel-2 20m image de-clouded at date t")

    # Model parameters
    parser.add_argument("--savedmodel", required=True, help="Path to the SavedModel directory")
    parser.add_argument('--pad', type=int, default=64, const=64, nargs="?", choices=constants.PADS,
                        help="Margin size for blocking artefacts removal")
    parser.add_argument('--ts', default=256, type=int,
                        help="Tile size. Tune this to process larger output image chunks, and speed up the process.")

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    params = parser.parse_args()

    meraner_processor(params.il_s1, params.in_s2, params.savedmodel, params.in_dem,
                      params.output, params.output_20m, ts=params.ts, pad=params.pad,
                      with_20m_bands=params.with_20m_bands)


if __name__ == "__main__":
    sys.exit(main())
