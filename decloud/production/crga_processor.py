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
"""Processor for CRGA models"""
import argparse
import os
import sys

import decloud.preprocessing.constants as constants
from decloud.core import system
import pyotb
from decloud.production.products import Factory as ProductsFactory
from decloud.production.inference import inference


def crga_processor(il_s1after, il_s1before, il_s1, il_s2after, il_s2before, in_s2, dem, savedmodel,
                   output=None, output_20m=None, ts=256, pad=64,
                   with_20m_bands=False, maxgap=144, with_intermediate=False):
    """
    Apply CRGA model to input sources.

    :param il_s1after: list of paths to Sentinel 1 products
    :param il_s1before: list of paths to Sentinel 1 products
    :param il_s1: list of paths to Sentinel 1 products
    :param il_s2after: list of paths to Sentinel 2 products
    :param il_s2before: list of paths to Sentinel 2 products
    :param in_s2: path of Sentinel 2 product to be reconstructed
    :param dem: path of the DEM
    :param savedmodel: path of the savedmodel (folder containing the .pb)
    :param output: output path for the reconstructed image. If not specified, the result is returned as a pyotb object
    :param output_20m: output path for the reconstructed 20m bands image
    :param pad: Margin size for blocking artefacts removal
    :param ts: Tile size. Tune this to process larger output image chunks, and speed up the process.
    :param with_20m_bands: whether to compute the 20m bands. If True, the saved model must have been trained accordingly
    :param maxgap: max gap (in hours) between S1 and S2 images
    :param with_intermediate: whether to write/return intermediate results (T-1, T+1 images output of the pre-processor)

    :return output, (sources): if output path is not specified, returns reconstructed in-memory pyotb object
                               optionally, if with_indermediate, also returns the input sources
    """

    # Create input products
    s1tp1_products = [ProductsFactory.create(pth, 's1') for pth in il_s1after]
    s1tm1_products = [ProductsFactory.create(pth, 's1') for pth in il_s1before]
    s1t_products = [ProductsFactory.create(pth, 's1') for pth in il_s1]
    s2tp1_products = [ProductsFactory.create(pth, 's2') for pth in il_s2after]
    s2tm1_products = [ProductsFactory.create(pth, 's2') for pth in il_s2before]
    s2t_product = ProductsFactory.create(in_s2, 's2')
    # gathering as a dictionary
    products_dic = {'s2_t': [s2t_product], 's2_tp1': s2tp1_products, 's2_tm1': s2tm1_products,
                    's1_t': s1t_products, 's1_tp1': s1tp1_products, 's1_tm1': s1tm1_products}

    # Splitting every product lists into 2 lists: list of images and list of dates
    images = {k: [product.get_raster_10m() for product in products] for k, products in products_dic.items()}
    dates = {k: [str(product.get_timestamp()) for product in products] for k, products in products_dic.items()}

    # Handling potential 20m bands
    if with_20m_bands:
        s2_20m = {k + '_20m': [product.get_raster_20m() for product in products] for k, products in products_dic.items()
                  if k.startswith('s2')}
        # we create a downsampled radar image that has the same resolution as 20m images
        s1_20m = {k + '_20m': [pyotb.RigidTransformResample({'in': image_10m, 'transform.type.id.scalex': 0.5,
                                                             'transform.type.id.scaley': 0.5, 'interpolator': 'nn'})
                               for image_10m in images_10m] for k, images_10m in images.items() if k.startswith('s1')}
        images.update(**s1_20m, **s2_20m)

    # Pre-Processing: "merging" all available images by creating S1/S2 pairs that satisfy a S2/S1 maxgap parameter
    preprocessor_tm1 = pyotb.DecloudTimeSeriesPreProcessor(maxgap=maxgap * 3600, ilsar=images['s1_tm1'],
                                                           ilopt=images['s2_tm1'], timestampssar=dates['s1_tm1'],
                                                           timestampsopt=dates['s2_tm1'], sorting="asc")
    preprocessor_tp1 = pyotb.DecloudTimeSeriesPreProcessor(maxgap=maxgap * 3600, ilsar=images['s1_tp1'],
                                                           ilopt=images['s2_tp1'], timestampssar=dates['s1_tp1'],
                                                           timestampsopt=dates['s2_tp1'], sorting="des")

    # For T date, there is only one S2 image, thus a simple mosaic of S1 images with the closest ones on top
    def _closest_date(x):
        """Helper to sort S1 products"""
        return abs(s2t_product.get_timestamp() - x.get_timestamp())

    s1t_products.sort(key=_closest_date, reverse=True)
    # Getting the 10m rasters
    input_s1_images_10m = [product.get_raster_10m() for product in s1t_products]
    s2t = s2t_product.get_raster_10m()

    sources = {'s1_tm1': preprocessor_tm1.outsar1,
               's2_tm1': preprocessor_tm1.outopt1,
               's1_tp1': preprocessor_tp1.outsar1,
               's2_tp1': preprocessor_tp1.outopt1,
               's1_t': pyotb.Mosaic(il=input_s1_images_10m, nodata=0),
               's2_t': s2t,
               "dem": dem}

    # Pre-Processing 20m bands
    if with_20m_bands:
        preproc_20m_tm1 = pyotb.DecloudTimeSeriesPreProcessor(maxgap=maxgap * 3600, ilsar=images['s1_tm1_20m'],
                                                              ilopt=images['s2_tm1_20m'], timestampssar=dates['s1_tm1'],
                                                              timestampsopt=dates['s2_tm1'], sorting="asc")
        preproc_20m_tp1 = pyotb.DecloudTimeSeriesPreProcessor(maxgap=maxgap * 3600, ilsar=images['s1_tp1_20m'],
                                                              ilopt=images['s2_tp1_20m'], timestampssar=dates['s1_tp1'],
                                                              timestampsopt=dates['s2_tp1'], sorting="des")
        sources.update({"s2_20m_tm1": preproc_20m_tm1.outopt1,
                        "s2_20m_tp1": preproc_20m_tp1.outopt1,
                        "s2_20m_t": images['s2_t_20m'][0]})

    # Resolution factor
    sources_scales = {"dem": 2, 's2_20m_tm1': 2, 's2_20m_tp1': 2, 's2_20m_t': 2}

    # OTB extended filename that will be used for all writing
    filename_extension = ("&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue={}&"
                          "gdal:co:COMPRESS=DEFLATE&gdal:co:TILED=YES".format(ts))

    # Inference
    if with_20m_bands:
        out_tensor = "s2_all_bands_estim"  # stack of the 10m reconstruction and 20m reconstruction (resampled to 10m)
        resampled_all_bands = inference(sources=sources, sources_scales=sources_scales, pad=pad,
                                        ts=ts, savedmodel_dir=savedmodel,
                                        out_tensor=out_tensor,
                                        out_nodatavalue=s2t_product.get_nodatavalue(),
                                        out_pixeltype=s2t_product.get_raster_10m_encoding(),
                                        nodatavalues={"s1_tm1": 0, "s2_tm1": -10000, "s1_tp1": 0,
                                                      "s2_tp1": -10000, "s1_t": 0, "s2_t": -10000})

        # If the user didn't specify output paths, return in-memory object
        if not (output and output_20m):
            if with_intermediate:
                return resampled_all_bands, sources
            return resampled_all_bands

        # temporary path for the stack of 10m+20m bands
        temp_outpath = system.join(system.dirname(output_20m),
                                   'temp_all_bands_' + system.basename(output_20m))
        resampled_all_bands.write(out=temp_outpath,
                                  filename_extension=filename_extension)

        stack = pyotb.Input(temp_outpath)
        # Writing 10m bands
        stack[:, :, :4].write(out=output, filename_extension=filename_extension)
        # Resampling and writing 20m bands
        final_res_20m = pyotb.RigidTransformResample({'in': stack[:, :, 4:], 'interpolator': 'nn',
                                                      'transform.type.id.scaley': 0.5,
                                                      'transform.type.id.scalex': 0.5})
        final_res_20m.write(out=output_20m, filename_extension=filename_extension)
        os.remove(temp_outpath)

    else:
        out_tensor = "s2_estim"
        processed_10m = inference(sources=sources, sources_scales=sources_scales, pad=pad,
                                  ts=ts, savedmodel_dir=savedmodel,
                                  out_tensor=out_tensor,
                                  out_nodatavalue=s2t_product.get_nodatavalue(),
                                  out_pixeltype=s2t_product.get_raster_10m_encoding(),
                                  nodatavalues={"s1_tm1": 0, "s2_tm1": -10000, "s1_tp1": 0,
                                                "s2_tp1": -10000, "s1_t": 0, "s2_t": -10000})

        # If the user didn't specify output path, return in-memory object
        if not output:
            if with_intermediate:
                return processed_10m, sources
            return processed_10m

        processed_10m.write(out=output, filename_extension=filename_extension)

    if with_intermediate and output:
        # Writing the outputs of Preprocessor, in the same directory as output with a suffix
        for name, source in sources.items():
            if name != 'dem':
                source.write(os.path.splitext(output)[0] + '_{}.tif'.format(name),
                             pixel_type=products_dic[name][0].get_raster_10m_encoding(),
                             filename_extension=filename_extension)


# ------------------------------------------------------- Main ---------------------------------------------------------
def main():
    # Logger
    system.basic_logging_init()

    # Parser
    parser = argparse.ArgumentParser(description="Remove clouds in a Sentinel-2 image from joint optical and SAR time"
                                                 " series.")

    # Input images
    parser.add_argument("--il_s1before", required=True, nargs='+', help="List of Sentinel-1 products near date t-1")
    parser.add_argument("--il_s2before", required=True, nargs='+', help="List of Sentinel-2 products near date t-1")
    parser.add_argument("--il_s1", required=True, nargs='+', help="List of Sentinel-1 products near date t")
    parser.add_argument("--in_s2", required=True, help="Sentinel-2 product at date t to reconstruct")
    parser.add_argument("--il_s1after", required=True, nargs='+', help="List of Sentinel-1 products near date t+1")
    parser.add_argument("--il_s2after", required=True, nargs='+', help="List of Sentinel-2 products near date t+1")
    parser.add_argument("--dem", required=True, help="DEM")
    parser.add_argument('--with_20m_bands', dest='with_20m_bands', action='store_true',
                        help="Remove clouds on all 10m & 20m bands")
    parser.set_defaults(with_20m_bands=False)
    parser.add_argument('--write_intermediate', dest='write_intermediate', action='store_true',
                        help="Whether to write intermediary T-1 & T+1 input rasters used by the model.")
    parser.set_defaults(write_intermediate=False)

    # Output image
    parser.add_argument("--output", required=True, help="Output file for the Sentinel-2 10m image de-clouded at date t")
    parser.add_argument("--output_20m", help="Output file for the Sentinel-2 20m image de-clouded at date t")

    # Model parameters
    parser.add_argument("--savedmodel", required=True, help="Path to the SavedModel directory")
    parser.add_argument('--pad', type=int, default=64, const=64, nargs="?", choices=constants.PADS,
                        help="Margin size for blocking artefacts removal")
    parser.add_argument('--ts', default=256, type=int,
                        help="Tile size. Tune this to process larger output image chunks, and speed up the process.")
    parser.add_argument('--maxgap', default=72, type=int,
                        help="Max gap (in hours) between S1 and S2 images for the selection of before and after pairs.")

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    params = parser.parse_args()

    crga_processor(params.il_s1after, params.il_s1before, params.il_s1, params.il_s2after, params.il_s2before,
                   params.in_s2, params.dem, params.savedmodel,
                   params.output, params.output_20m, params.ts, params.pad,
                   params.with_20m_bands, params.maxgap, params.write_intermediate)


if __name__ == "__main__":
    sys.exit(main())
