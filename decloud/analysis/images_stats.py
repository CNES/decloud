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
Compute statistics over S1 or S2 images
"""
import argparse
import glob
import os
import sys
import json
import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal

parser = argparse.ArgumentParser(description="dataset test")
parser.add_argument("--input_dir", help="A directory containing S1 or S2 files", required=True)
parser.add_argument("--tmpdir", required=True)
parser.add_argument("--nodata", type=int, default=-10000)
parser.add_argument("--prefix", default="")
params = parser.parse_args()


def compute_stats(dir_path, prefix, nodata):
    """
        Compute for all images containing a prefix in a folder the statistics for each band of them

        :param dir_path: Folder containing images
        :param prefix: Prefix to select images
        :param nodata: Nodata value
        :return: Return a dictionary containing for each band of images : min_value, max_value, mean_value, std
    """

    images_list = [x for x in glob.glob("{}/*".format(dir_path)) if x.endswith(".tif") and prefix in x]
    checkpoint = len(images_list) // 10
    stats = dict()

    for idx, image in enumerate(images_list):

        if idx % checkpoint == 0:
            print(idx)

        img = gdal.Open(image)
        stats[os.path.basename(image)] = dict()

        for band in range(img.RasterCount):

            stats[os.path.basename(image)][band] = dict()

            mask_no_data = np.ma.masked_values(img.GetRasterBand(band+1).ReadAsArray(), nodata).astype(float)

            stats[os.path.basename(image)][band] = dict()
            stats[os.path.basename(image)][band]["min"] = int(np.min(mask_no_data))
            stats[os.path.basename(image)][band]["max"] = int(np.max(mask_no_data))
            stats[os.path.basename(image)][band]["mean"] = int(np.mean(mask_no_data))
            stats[os.path.basename(image)][band]["std"] = int(np.std(mask_no_data))

    return stats


def get_min_max(path, band):
    """
        Return min and max value of statistics file for a band

        :param path: Path of json statistic file
        :param band: Band number
        :return: [min_value, max_value]
    """

    with open(path, "r") as in_file:
        data = json.load(in_file)

    min_val = sys.float_info.max
    max_val = sys.float_info.min

    for i in data.keys():
        if float(data[i][str(band)]["min"]) < min_val and float(data[i][str(band)]["min"]) != 0:
            min_val = float(data[i][str(band)]["min"])

        if float(data[i][str(band)]["max"]) > max_val:
            max_val = float(data[i][str(band)]["max"])

    return min_val, max_val


def get_bin_edges(min_val, max_val, nbins):
    """
        Return a edges list between min and max values with nbins number

        :param min_val: Min edge value
        :param max_val: Max edge value
        :param nbins: Number of bins
        :return: A list containing bins edges
    """
    return np.linspace(min_val, max_val, nbins + 1)


def compute_histogram(dir_path, in_prefix, output_dir, prefix="", nb_band=4):
    """
        Compute for all images containing a prefix in a folder the histogram for each band of them

        :param dir_path: Folder containing images
        :param in_prefix: Prefix to select images
        :param output_dir: Results folder
        :param prefix: Output prefix for histogram file
        :param nb_band: Image band number
    """
    nbins = 1000
    images_list = [x for x in glob.glob("{}/*".format(dir_path)) if x.endswith(".tif") and in_prefix in x]
    checkpoint = len(images_list) // 100

    for i in range(nb_band):
        min_value, max_value = get_min_max('{}/stats_{}.json'.format(output_dir, prefix), i)

        bin_edges = get_bin_edges(min_value, max_value, nbins=nbins)

        hist = np.zeros(nbins)

        for idx, image in enumerate(images_list):

            if idx % checkpoint == 0:
                print(idx)

            img = gdal.Open(image)

            data = img.GetRasterBand(i+1).ReadAsArray().astype(float)
            tmp, _ = np.histogram(data, bins=bin_edges)
            hist += tmp

        with open("{}/data_hist_{}_{}.txt".format(output_dir, prefix, i), "w") as histfile:
            np.savetxt(histfile, hist, delimiter=",")


def plot_histogram(prefix, output_dir, nb_band=4, nbins=1000):
    """
        Save histogram in png format

        :param prefix: Prefix to add png file
        :param output_dir: Folder containing images
        :param nb_band: Image band number
        :param nbins: Number of bins
    """
    color = ["blue", "green", 'red', "purple"]

    for i in range(nb_band):
        min_value, max_value = get_min_max('{}/stats_{}.json'.format(output_dir, prefix), i)

        bin_edges = get_bin_edges(min_value, max_value, nbins=nbins)

        with open("{}/data_hist_{}_{}.txt".format(output_dir, prefix, i), "r") as histfile:
            hist = np.loadtxt(histfile)

        plt.figure()
        plt.bar(bin_edges[:-1], np.array(hist), edgecolor=color[i])
        plt.savefig("{}/hist_{}_{}.png".format(output_dir, prefix, i))


def compute_dataset_stats(in_dir, output_dir, in_prefix="", prefix="", nodata=-10000):
    """
        Call stats compute function and save them in json format file

        :param in_dir: Folder containing images
        :param output_dir: Output folder
        :param in_prefix: Prefix to select images
        :param prefix: Prefix to add output json file
        :param nodata: Nodata value
    """
    with open('{}/stats_{}.json'.format(output_dir, prefix), 'w') as json_file:
        stats = compute_stats(dir_path=in_dir, prefix=in_prefix, nodata=nodata)
        json.dump(stats, json_file, sort_keys=True, indent=4)


if __name__ == "__main__":

    compute_dataset_stats(in_dir=params.input_dir, output_dir=params.tmpdir, in_prefix="", prefix=params.prefix,
                          nodata=params.nodata)
    compute_histogram(dir_path=params.input_dir, in_prefix="", output_dir=params.tmpdir, prefix=params.prefix,
                      nb_band=2)
    plot_histogram(prefix=params.prefix, output_dir=params.tmpdir, nb_band=2)
