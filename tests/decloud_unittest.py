#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import unittest
import filecmp
from osgeo import gdal
import otbApplication as otb
from abc import ABC
from decloud.core.system import get_env_var, basename


class DecloudTest(ABC, unittest.TestCase):

    DECLOUD_DATA_DIR = get_env_var("DECLOUD_DATA_DIR")

    def get_path(self, path):
        pth = os.path.join(self.DECLOUD_DATA_DIR, path)
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Directory {pth} not found!")
        return pth

    def compare_images(self, image, reference, mae_threshold=0.01):

        nbchannels_reconstruct = gdal.Open(image).RasterCount
        nbchannels_baseline = gdal.Open(reference).RasterCount

        self.assertTrue(nbchannels_reconstruct == nbchannels_baseline)

        for i in range(1, 1+nbchannels_baseline):
            comp = otb.Registry.CreateApplication('CompareImages')
            comp.SetParameterString('ref.in', reference)
            comp.SetParameterInt('ref.channel', i)
            comp.SetParameterString('meas.in', image)
            comp.SetParameterInt('meas.channel', i)
            comp.Execute()
            mae = comp.GetParameterFloat('mae')

            self.assertTrue(mae < mae_threshold)

    def compare_file(self, file, reference):
        self.assertTrue(filecmp.cmp(file, reference))

    def compare_raster_metadata(self, image, reference):
        baseline_gdalinfo_path = '/tmp/baseline_{}_gdalinfo'.format(basename(reference))
        subprocess.call('gdalinfo {} | grep --invert-match -e "Files:" -e "METADATATYPE" -e "OTB_VERSION" '
                      '-e "NoData Value" > {}'.format(reference, baseline_gdalinfo_path), shell=True)

        image_gdalinfo_path = '/tmp/image_{}_gdalinfo'.format(basename(image))
        subprocess.call('gdalinfo {} | grep --invert-match -e "Files:" -e "METADATATYPE" -e "OTB_VERSION" '
                        '-e "NoData Value" > {}'.format(image, image_gdalinfo_path), shell=True)

        with open(baseline_gdalinfo_path) as f:
            baseline_gdalinfo = f.read()
        with open(image_gdalinfo_path) as f:
            image_gdalinfo_path = f.read()
        self.assertEqual(baseline_gdalinfo, image_gdalinfo_path)
