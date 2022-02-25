#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from decloud.models import create_tfrecords
from .decloud_unittest import DecloudTest


class CreateTFRecordsTest(DecloudTest):

    def test_createTFRecords_CRGA(self):
        create_tfrecords.main(["--acquisition_train", "decloud/acquisitions/AL-TEST.json",
                               "--output_train", "/tmp/train", "--tiles", self.get_path("inputs/dataset_test.json"),
                               "--rois", self.get_path("inputs/roi_test.json")])

        self.compare_file("/tmp/train/output_shape.json", self.get_path("baseline/TFRecord/CRGA/output_shape.json"))
        self.compare_file("/tmp/train/output_types.json", self.get_path("baseline/TFRecord/CRGA/output_types.json"))

    def test_createTFRecords_CRGA_all_bands(self):
        create_tfrecords.main(["--acquisition_train", "decloud/acquisitions/AL-TEST.json",
                               "--output_train", "/tmp/train_all_bands", "--tiles",
                               self.get_path("inputs/dataset_test.json"),
                               "--rois", self.get_path("inputs/roi_test.json"), "--with_20m_bands"])

        self.compare_file("/tmp/train_all_bands/output_shape.json",
                          self.get_path("baseline/TFRecord/CRGA_all_bands/output_shape.json"))
        self.compare_file("/tmp/train_all_bands/output_types.json",
                          self.get_path("baseline/TFRecord/CRGA_all_bands/output_types.json"))

    def test_createTFRecords_Meraner_all_bands(self):
        create_tfrecords.main(["--acquisition_train", "decloud/acquisitions/AL-TEST-Meraner.json",
                               "--output_train", "/tmp/train_all_bands", "--tiles",
                               self.get_path("inputs/dataset_test.json"),
                               "--rois", self.get_path("inputs/roi_test.json"), "--with_20m_bands"])

        self.compare_file("/tmp/train_all_bands/output_shape.json",
                          self.get_path("baseline/TFRecord/Meraner/output_shape.json"))
        self.compare_file("/tmp/train_all_bands/output_types.json",
                          self.get_path("baseline/TFRecord/Meraner/output_types.json"))

if __name__ == '__main__':
    unittest.main()
