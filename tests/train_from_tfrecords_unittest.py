#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
from decloud.models import train_from_tfrecords
from decloud.core import system
from decloud.production import crga_processor
from decloud.production import meraner_processor
from .decloud_unittest import DecloudTest
import tempfile
import os
from pathlib import Path

SAVEDMODEL_FILENAME = "saved_model.pb"
out_savedmodel = "/tmp/savedmodel"

def get_last_savedmodel():
    paths = sorted(Path(out_savedmodel).iterdir(), key=os.path.getmtime)
    last_pth = os.path.join(out_savedmodel, paths[-1])
    print(f"last savedmodel: {last_pth}")
    return last_pth


def is_savedmodel_written(args_list):
    base_args = [
        "--logdir", "/tmp/logdir",
        "--out_savedmodel", out_savedmodel,
        "--epochs", "1",
        "-bt", "1",
        "-bv", "1",
        "--strategy", "singlecpu"
    ]
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

    def is_inference_working_meraner(self):
        s1_t = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201013txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20201012txxxxxx_from-10to3dB.tif')
        ]
        s2_t = self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2')

        with tempfile.TemporaryDirectory() as tmpdirname:
            output = os.path.join(tmpdirname, 'reconstructed_w_preprocessor.tif')

            meraner_processor.meraner_processor(
                il_s1=s1_t, 
                in_s2=s2_t, 
                savedmodel=get_last_savedmodel(), 
                dem=self.get_path('baseline/PREPARE/DEM_PREPARE/T31TEJ.tif'), 
                output=output, 
                ts=256, 
                pad=64
            )

            self.assertTrue(system.file_exists(output))


    def is_inference_working_crga(self):
        # Input sources
        s1_tm1 = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20200929t060008_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20200930txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201001txxxxxx_from-10to3dB.tif')
        ]
        s2_tm1 = [
            self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20200926-103901-393_L2A_T31TEJ_C_V2-2/'),
            self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20200929-104857-489_L2A_T31TEJ_C_V2-2/')
        ]
        s1_t = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201013txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20201012txxxxxx_from-10to3dB.tif')
        ]
        s2_t = self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2')
        s1_tp1 = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201025txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20201024txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20201023t060008_from-10to3dB.tif')
        ]
        s2_tp1 = [
            self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20201026-103901-924_L2A_T31TEJ_C_V2-2'),
            self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2A_20201024-104859-766_L2A_T31TEJ_C_V2-2/')
        ]

        with tempfile.TemporaryDirectory() as tmpdirname:
            output = os.path.join(tmpdirname, 'reconstructed_w_preprocessor.tif')
            crga_processor.crga_processor(
                il_s1before=s1_tm1, 
                il_s2before=s2_tm1,
                il_s1=s1_t, 
                in_s2=s2_t,
                il_s1after=s1_tp1, 
                il_s2after=s2_tp1,
                dem=self.get_path('baseline/PREPARE/DEM_PREPARE/T31TEJ.tif'),
                output=output,
                maxgap=48, 
                savedmodel=get_last_savedmodel()
            )

            self.assertTrue(system.file_exists(output))


    def test_trainFromTFRecords_os1_unet(self):

        self.assertTrue(
            is_savedmodel_written([
                "--training_record", self.get_path(OS2_TFREC_PTH),
                "--model", "crga_os1_unet"
            ]),
            ERRMSG
        )


    def test_trainFromTFRecords_os2_david(self):
        self.assertTrue(
            is_savedmodel_written([
                "--training_record", self.get_path(OS2_TFREC_PTH),
                "--model", "crga_os2_david"
            ]),
            ERRMSG
        )
        self.is_inference_working_crga()

    def test_trainFromTFRecords_os2_unet(self):
        self.assertTrue(
            is_savedmodel_written([
                "--training_record", self.get_path(OS2_TFREC_PTH),
                "--model", "crga_os2_unet"
            ]),
            ERRMSG
        )
        self.is_inference_working_crga()

    def test_trainFromTFRecords_os1_unet_all_bands(self):
        self.assertTrue(
            is_savedmodel_written([
                "--training_record", self.get_path(OS2_ALL_BANDS_TFREC_PTH),
                "--model", "crga_os1_unet_all_bands"
            ]),
            ERRMSG
        )

    def test_trainFromTFRecords_os2_david_all_bands(self):
        self.assertTrue(
            is_savedmodel_written([
                "--training_record", self.get_path(OS2_ALL_BANDS_TFREC_PTH),
                "--model", "crga_os2_david_all_bands"
            ]),
            ERRMSG
        )

    def test_trainFromTFRecords_os2_unet_all_bands(self):
        self.assertTrue(
            is_savedmodel_written([
                "--training_record", self.get_path(OS2_ALL_BANDS_TFREC_PTH),
                "--model", "crga_os2_unet_all_bands"
            ]),
            ERRMSG
        )

    def test_trainFromTFRecords_meraner_unet(self):
        self.assertTrue(
            is_savedmodel_written([
                "--training_record", self.get_path(MERANER_ALL_BANDS_TFREC_PTH),
                "--model", "meraner_unet"
            ]),
            ERRMSG
        )
        self.is_inference_working_meraner()

    def test_trainFromTFRecords_meraner_unet_all_bands(self):
        self.assertTrue(
            is_savedmodel_written([
                "--training_record", self.get_path(MERANER_ALL_BANDS_TFREC_PTH),
                "--model", "meraner_unet_all_bands"
            ]),
            ERRMSG
        )


if __name__ == '__main__':
    unittest.main()
