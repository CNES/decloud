#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from decloud.preprocessing import sentinel2_prepare
from .decloud_unittest import DecloudTest


class Sentinel2PrepareTest(DecloudTest):

    def test_sentinel2_prepare(self):
        sentinel2_prepare.main(["--in_image", self.get_path("inputs/THEIA/SENTINEL2B_20201012-105848"
                                              "-497_L2A_T31TEJ_C_V2-2"),
                                "--out_s2_dir", "/tmp/s2_prepare"])

        self.compare_images("/tmp/s2_prepare/T31TEJ/SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2/"
                            "SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2_FRE_10m.tif",
                            self.get_path("baseline/PREPARE/S2_PREPARE/T31TEJ/"
                            "SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2/"
                            "SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2_FRE_10m.tif"))


if __name__ == '__main__':
    unittest.main()
