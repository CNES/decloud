#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from decloud.preprocessing import dem_prepare
from .decloud_unittest import DecloudTest


class DEMTest(DecloudTest):

    def test_dem(self):
        dem_prepare.main(["--reference",
                          self.get_path("baseline/PREPARE/S2_PREPARE/T31TEJ/"
                          "SENTINEL2B_20201026-103901-924_L2A_T31TEJ_C_V2-2/"
                          "SENTINEL2B_20201026-103901-924_L2A_T31TEJ_C_V2-2_FRE_20m.tif"),
                          "--output", "/tmp/dem_prepare.tif", "--tmp", "/tmp/DEM"])

        self.compare_images("/tmp/dem_prepare.tif", self.get_path("baseline/PREPARE/DEM_PREPARE/T31TEJ.tif"))


if __name__ == '__main__':
    unittest.main()
