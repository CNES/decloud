#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
from decloud.models import train_from_tfrecords
from .decloud_unittest import DecloudTest

SAVEDMODEL_FILENAME = "saved_model.pb"


def is_savedmodel_written(args_list):
    out_savedmodel = "/tmp/savedmodel"
    base_args = ["--logdir", "/tmp/logdir",
                 "--out_savedmodel", out_savedmodel,
                 "--epochs", "1",
                 "-bt", "1",
                 "-bv", "1",
                 "--strategy", "singlecpu"]
    train_from_tfrecords.main(args_list + base_args)

    for dir, sub_dirs, files in os.walk(out_savedmodel):
        if SAVEDMODEL_FILENAME in files:
            return True
    return False


OS2_TFREC_PTH = "baseline/TFRecord/CRGA"
OS2_ALL_BANDS_TFREC_PTH = "baseline/TFRecord/CRGA_all_bands"
MERANER_ALL_BANDS_TFREC_PTH = "baseline/TFRecord/CRGA_all_bands"
ERRMSG = f"File {SAVEDMODEL_FILENAME} not found !"


class TrainFromTFRecordsTest(DecloudTest):

    def test_trainFromTFRecords_os1_unet(self):
        self.assertTrue(is_savedmodel_written(["--training_record", self.get_path(OS2_TFREC_PTH),
                                               "--model", "crga_os1_unet"]),
                        ERRMSG)

    def test_trainFromTFRecords_os2_david(self):
        self.assertTrue(is_savedmodel_written(["--training_record", self.get_path(OS2_TFREC_PTH),
                                               "--model", "crga_os2_david"]),
                        ERRMSG)

    def test_trainFromTFRecords_os2_unet(self):
        self.assertTrue(is_savedmodel_written(["--training_record", self.get_path(OS2_TFREC_PTH),
                                               "--model", "crga_os2_unet"]),
                        ERRMSG)

    def test_trainFromTFRecords_os1_unet_all_bands(self):
        self.assertTrue(is_savedmodel_written(["--training_record", self.get_path(OS2_ALL_BANDS_TFREC_PTH),
                                               "--model", "crga_os1_unet_all_bands"]),
                        ERRMSG)

    def test_trainFromTFRecords_os2_david_all_bands(self):
        self.assertTrue(is_savedmodel_written(["--training_record", self.get_path(OS2_ALL_BANDS_TFREC_PTH),
                                               "--model", "crga_os2_david_all_bands"]),
                        ERRMSG)

    def test_trainFromTFRecords_os2_unet_all_bands(self):
        self.assertTrue(is_savedmodel_written(["--training_record", self.get_path(OS2_ALL_BANDS_TFREC_PTH),
                                               "--model", "crga_os2_unet_all_bands"]),
                        ERRMSG)

    def test_trainFromTFRecords_meraner_unet(self):
        self.assertTrue(is_savedmodel_written(["--training_record", self.get_path(MERANER_ALL_BANDS_TFREC_PTH),
                                               "--model", "meraner_unet"]),
                        ERRMSG)

    def test_trainFromTFRecords_meraner_unet_all_bands(self):
        self.assertTrue(is_savedmodel_written(["--training_record", self.get_path(MERANER_ALL_BANDS_TFREC_PTH),
                                               "--model", "meraner_unet_all_bands"]),
                        ERRMSG)


if __name__ == '__main__':
    unittest.main()
