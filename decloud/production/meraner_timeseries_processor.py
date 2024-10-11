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
from decloud.core import system
import numpy as np

from decloud.production.products import Factory as ProductsFactory
from decloud.production.meraner_processor import meraner_processor
import pyotb

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
    parser.add_argument("--dem", help="DEM path")
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
                        help="Whether to write S1t & S2t input rasters used by the model.")
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

    # Converting filepaths to S1 products
    input_s1_products = {}
    product_count, invalid_count = 0, 0
    for product_path in s1_image_paths:
        product = ProductsFactory.create(product_path, 's1', verbose=False)
        if product:
            input_s1_products[product_path] = product
            product_count += 1
        else:
            invalid_count += 1
    logging.info('Retrieved {} S1 products from disk. '
                 'Discarded {} paths that were not S1 products'.format(product_count, invalid_count))
    il_s1 = list(input_s1_products.keys())

    if params.start or params.end:
        logging.info('Filtering timerange of inputs products, to match user timerange : '
                     'From {} to {}'.format(params.start, params.end))
        start = datetime.datetime.strptime(params.start, '%Y-%m-%d') if params.start else None
        end = datetime.datetime.strptime(params.end, '%Y-%m-%d') if params.end else None

    if not system.is_dir(params.out_dir):
        system.mkdir(params.out_dir)

    # OTB extended filename that will be used for all writing
    ext_fname = (
        "&streaming:type=tiled"
        "&streaming:sizemode=height"
        f"&streaming:sizevalue={params.ts}"
        "&gdal:co:COMPRESS=DEFLATE"
        "&gdal:co:TILED=YES"
    )

    # looping through the input Sentinel-2 images
    for s2_filepath, s2t_product in input_s2_products.items():
        if (params.start and s2t_product.get_date() < start) or (params.end and s2t_product.get_date() > end):
            # skipping invalid timerange product
            continue

        output_filename = os.path.splitext(os.path.basename(s2_filepath))[0] + '_reconstructed.tif'
        output_path = os.path.join(params.out_dir, output_filename)
        if params.overwrite or (not os.path.exists(output_path)):
            # Potentially skip the inference if the s2_t image is all NoData
            if params.skip_nodata_images:
                # we consider the 20m image (because it is smaller than 10m image)
                s2t_20m = s2t_product.get_raster_20m()
                # If needed, extracting ROI of all rasters
                if params.lrx and params.lry and params.ulx and params.uly:
                    s2t_20m = pyotb.ExtractROI({
                        'in': s2t_20m, 
                        'mode': 'extent', 
                        'mode.extent.unit': 'phy',
                        'mode.extent.ulx': params.ulx, 
                        'mode.extent.uly': params.uly,
                        'mode.extent.lrx': params.lrx, 
                        'mode.extent.lry': params.lry
                    })
                s2t_20m = pyotb.Input(s2t_20m) if isinstance(s2t_20m, str) else s2t_20m
                if np.max(np.asarray(s2t_20m)) <= 0:
                    logging.warning(f'SKIPPING all NoData image: {s2_filepath}')
                    continue

            if params.write_intermediate:
                processor, sources = meraner_processor(
                    il_s1, 
                    s2_filepath, 
                    params.model, 
                    params.dem, 
                    s1_Nimages=12,
                    ts=params.ts, 
                    with_intermediate=True
                )
            else:
                processor = meraner_processor(
                    il_s1, 
                    s2_filepath, 
                    params.model, 
                    params.dem, 
                    s1_Nimages=12,
                    ts=params.ts
                )

            # If needed, extracting ROI of the reconstructed image
            if params.lrx and params.lry and params.ulx and params.uly:
                processor = pyotb.ExtractROI({
                    'in': processor, 
                    'mode': 'extent', 
                    'mode.extent.unit': 'phy',
                    'mode.extent.ulx': params.ulx, 
                    'mode.extent.uly': params.uly,
                    'mode.extent.lrx': params.lrx, 
                    'mode.extent.lry': params.lry},
                    propagate_pixel_type=True
                )

            processor.write(out=output_path, filename_extension=ext_fname)

            # Writing the inputs sources of the model
            if params.write_intermediate:
                for name, source in sources.items():
                    if name != 'dem':
                        # If needed, extracting ROI of every rasters
                        if params.lrx and params.lry and params.ulx and params.uly:
                            source = pyotb.ExtractROI({
                                'in': source, 
                                'mode': 'extent', 
                                'mode.extent.unit': 'phy',
                                'mode.extent.ulx': params.ulx, 
                                'mode.extent.uly': params.uly,
                                'mode.extent.lrx': params.lrx, 
                                'mode.extent.lry': params.lry},
                                propagate_pixel_type=True
                            )

                        if isinstance(source, str):  # if needed transform the filepath to pyotb in-memory object
                            source = pyotb.Input(source)
                        source.write(
                            os.path.join(
                                params.out_dir, 
                                output_filename.replace('reconstructed', name)
                            ),
                            pixel_type='int16', 
                            filename_extension=ext_fname
                        )
