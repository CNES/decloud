#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from decloud.preprocessing import sentinel1_prepare
from .decloud_unittest import DecloudTest


class Sentinel1PrepareTest(DecloudTest):

    def test_sentinel1_prepare(self):
        sentinel1_prepare.main(["--input_s1_vh", self.get_path("inputs/THEIA/s1b_31TEJ_vh_DES_110_20201011t060008.tif"),
                                "--out_s1_dir", "/tmp/s1_prepare"])

        self.compare_raster_metadata("/tmp/s1_prepare/s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif",
                                     self.get_path("baseline/PREPARE/S1_PREPARE/T31TEJ/"
                                     "s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif"))
        self.compare_images("/tmp/s1_prepare/s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif",
                            self.get_path("baseline/PREPARE/S1_PREPARE/T31TEJ/"
                            "s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif"))


if __name__ == '__main__':
    unittest.main()
