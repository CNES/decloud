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
"""Process time series with CRGA models"""
import argparse
import datetime
import os
import sys
import logging
import heapq
import numpy as np
from decloud.core import system
from decloud.production.products import Factory as ProductsFactory
from decloud.production.crga_processor import crga_processor
import pyotb


def get_nclosest(n, s2t, product_dic, period=None):
    """
    Finds the n temporally closest images of a given S2 product.

    :param n: number of images to select
    :param s2t: S2ProductBase image
    :param product_dic: ProductBase images
    :param period: Optional. Period of interest, can be 'before' or 'after' or any
    :return res: list of filepaths
    """

    # Searching for candidates and gathering as dictionary {product:timedelta}
    if period == 'before':
        candidates = dict([(file, s2t.get_date() - product.get_date())
                           for file, product in product_dic.items() if product.get_date() < s2t.get_date()])
    elif period == 'after':
        candidates = dict([(file, product.get_date() - s2t.get_date())
                           for file, product in product_dic.items() if product.get_date() > s2t.get_date()])
    else:
        candidates = dict([(file, abs(product.get_date() - s2t.get_date())) for file, product in product_dic.items()])

    # selecting the N best among the candidates
    res = heapq.nsmallest(n, candidates, key=candidates.get)

    return res


if __name__ == "__main__":
    # Logger
    system.basic_logging_init()

    # Parser
    parser = argparse.ArgumentParser(
        description="Remove clouds in a time series of Sentinel-2 image, using joint optical and SAR images.")

    # Input images
    parser.add_argument("--il_s2", nargs='+', help="List of Sentinel-2 images, can be a list of paths or "
                                                   "a .txt file containing paths")
    parser.add_argument("--il_s1", nargs='+', help="List of Sentinel-1 images, can be a list of paths or "
                                                   "a .txt file containing paths")
    parser.add_argument("--s2_dir", help="Directory of Sentinel-2 images. Enables to treat all the images of "
                                         "a directory. Used only if il_s2 is not specified")
    parser.add_argument("--s1_dir", help="Directory of Sentinel-1 images. Enables to treat all the images of "
                                         "a directory. Used only if il_s1 is not specified")
    parser.add_argument("--dem", required=True, help="DEM path")
    parser.add_argument("--out_dir", required=True, help="Output directory for the reconstructed optical time series")
    parser.add_argument("--model", required=True, help="Path to the saved model directory, containing saved_model.pb")
    parser.add_argument("--ulx", help="Upper Left X of the ROI, in geographic coordinates. Optional", type=float)
    parser.add_argument("--uly", help="Upper Left Y of the ROI, in geographic coordinates. Optional", type=float)
    parser.add_argument("--lrx", help="Lower Right X of the ROI, in geographic coordinates. Optional", type=float)
    parser.add_argument("--lry", help="Lower Right Y of the ROI, in geographic coordinates. Optional", type=float)
    parser.add_argument("--start", help="Starting date, format YYYY-MM-DD. Optional")
    parser.add_argument("--end", help="End date, format YYYY-MM-DD. Optional")
    parser.add_argument('--ts', default=256, type=int,
                        help="Tile size. Tune this to process larger output image chunks, and speed up the process.")
    parser.add_argument('--write_intermediate', dest='write_intermediate', action='store_true',
                        help="Whether to write intermediary T-1 & T+1 input rasters used by the model.")
    parser.set_defaults(write_intermediate=False)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help="Whether to overwrite results if already exist")
    parser.set_defaults(overwrite=False)
    parser.add_argument('--skip_nodata_images', dest='skip_nodata_images', action='store_true',
                        help="Whether to skip the reconstruction of the optical image if it is all NoData")
    parser.set_defaults(skip_nodata_images=False)

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    params = parser.parse_args()

    if not (params.il_s2 or params.s2_dir):
        raise Exception('Missing --il_s2 or --s2_dir argument')
    if not (params.il_s1 or params.s1_dir):
        raise Exception('Missing --il_s1 or --s1_dir argument')

    if params.il_s2 and params.s2_dir:
        logging.warning('Both --il_s2 and --s2_dir were specified. Discarding --s2_dir')
        params.s2_dir = None
    if params.il_s1 and params.s1_dir:
        logging.warning('Both --il_s1 and --s1_dir were specified. Discarding --s1_dir')
        params.s1_dir = None

    # Getting all the S2 filepaths
    if params.s2_dir:
        s2_image_paths = [os.path.join(params.s2_dir, name) for name in os.listdir(params.s2_dir)]
    elif params.il_s2[0].endswith('.txt'):
        with open(params.il_s2[0], 'r') as f:
            s2_image_paths = [x.strip() for x in f.readlines()]
    else:
        s2_image_paths = params.il_s2

    # Getting all the S1 filepaths
    if params.s1_dir:
        s1_image_paths = [os.path.join(params.s1_dir, name) for name in os.listdir(params.s1_dir)]
    elif params.il_s1[0].endswith('.txt'):
        with open(params.il_s1[0], 'r') as f:
            s1_image_paths = [x.strip() for x in f.readlines()]
    else:
        s1_image_paths = params.il_s1

    # Converting filepaths to products
    input_s2_products, input_s1_products = {}, {}
    for imtype, image_paths, input_products in zip(['s2', 's1'], [s2_image_paths, s1_image_paths],
                                                   [input_s2_products, input_s1_products]):
        product_count, invalid_count = 0, 0
        for product_path in image_paths:
            product = ProductsFactory.create(product_path, imtype, verbose=False)
            if product:
                input_products[product_path] = product
                product_count += 1
            else:
                invalid_count += 1
        logging.info(f'Retrieved {product_count} {imtype.upper()} products from disk. '
                     f'Discarded {invalid_count} paths that were not {imtype.upper()} products')

    if params.start or params.end:
        logging.info('Filtering timerange of inputs products, to match user timerange : '
                     'From {} to {}'.format(params.start, params.end))
        start = datetime.datetime.strptime(params.start, '%Y-%m-%d') if params.start else None
        end = datetime.datetime.strptime(params.end, '%Y-%m-%d') if params.end else None

    if not system.is_dir(params.out_dir):
        system.mkdir(params.out_dir)

    s1_Nimages, s2_Nimages = 12, 12  # number of images to consider for s1t, s1tm1, s1tp1 & s2tm1, s2tp1

    # OTB extended filename that will be used for all writing
    filename_extension = ("&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue={}&"
                          "gdal:co:COMPRESS=DEFLATE&gdal:co:TILED=YES".format(params.ts))

    # looping through the dates
    for s2_filepath, s2t_product in input_s2_products.items():
        if (params.start and s2t_product.get_date() < start) or (params.end and s2t_product.get_date() > end):
            # skipping invalid timerange product
            continue

        output_filename = os.path.splitext(os.path.basename(s2_filepath))[0]+'_reconstructed.tif'
        output_path = os.path.join(params.out_dir, output_filename)
        if params.overwrite or (not os.path.exists(output_path)):
            s2tp1_paths = get_nclosest(s2_Nimages, s2t_product, input_s2_products, 'after')
            s2tm1_paths = get_nclosest(s2_Nimages, s2t_product, input_s2_products, 'before')
            s1t_paths = get_nclosest(s1_Nimages, s2t_product, input_s1_products)
            s1tp1_paths = get_nclosest(s1_Nimages, s2t_product, input_s1_products, 'after')
            s1tm1_paths = get_nclosest(s1_Nimages, s2t_product, input_s1_products, 'before')

            if any([len(paths) == 0 for paths in [s1tp1_paths, s1tm1_paths, s1t_paths, s2tp1_paths, s2tm1_paths]]):
                logging.warning('Could not find some T-1 or T+1 or S1T products. '
                                'Skipping inference: {}'.format(os.path.basename(s2_filepath)))
                continue

            # Potentially skip the inference if the s2_t image is all NoData
            if params.skip_nodata_images:
                # we consider the 20m image (because it is smaller than 10m image)
                s2t_20m = s2t_product.get_raster_20m()
                # If needed, extracting ROI of all rasters
                if params.lrx and params.lry and params.ulx and params.uly:
                    s2t_20m = pyotb.ExtractROI({'in': s2t_20m, 'mode': 'extent', 'mode.extent.unit': 'phy',
                                                'mode.extent.ulx': params.ulx, 'mode.extent.uly': params.uly,
                                                'mode.extent.lrx': params.lrx, 'mode.extent.lry': params.lry})
                s2t_20m = pyotb.Input(s2t_20m) if isinstance(s2t_20m, str) else s2t_20m
                if np.max(np.asarray(s2t_20m)) <= 0:
                    logging.warning(f'SKIPPING all NoData image: {s2_filepath}')
                    continue

            if params.write_intermediate:
                processor, sources = crga_processor(s1tp1_paths, s1tm1_paths, s1t_paths, s2tp1_paths, s2tm1_paths,
                                                    s2_filepath, params.dem, params.model, ts=params.ts,
                                                    with_intermediate=True)
            else:
                processor = crga_processor(s1tp1_paths, s1tm1_paths, s1t_paths, s2tp1_paths, s2tm1_paths,
                                           s2_filepath, params.dem, params.model, ts=params.ts)

            # If needed, extracting ROI of the reconstructed image
            if params.lrx and params.lry and params.ulx and params.uly:
                processor = pyotb.ExtractROI({'in': processor, 'mode': 'extent', 'mode.extent.unit': 'phy',
                                              'mode.extent.ulx': params.ulx, 'mode.extent.uly': params.uly,
                                              'mode.extent.lrx': params.lrx, 'mode.extent.lry': params.lry},
                                             propagate_pixel_type=True)

            # Writing result
            processor.write(out=output_path, filename_extension=filename_extension)

            # Writing intermediate results: s2t and the outputs of preprocessor (s1tm1, s1tp1, s1t, s2tm1, s2t)
            if params.write_intermediate:
                for name, image in sources.items():
                    if name != 'dem':
                        # If needed, extracting ROI
                        if params.lrx and params.lry and params.ulx and params.uly:
                            image = pyotb.ExtractROI({'in': image, 'mode': 'extent', 'mode.extent.unit': 'phy',
                                                      'mode.extent.ulx': params.ulx, 'mode.extent.uly': params.uly,
                                                      'mode.extent.lrx': params.lrx, 'mode.extent.lry': params.lry},
                                                     propagate_pixel_type=True)

                        if isinstance(image, str):  # if needed transform the filepath to pyotb in-memory object
                            image = pyotb.Input(image)
                        image.write(os.path.join(params.out_dir, output_filename.replace('reconstructed', name)),
                                    pixel_type='int32', filename_extension=filename_extension)
