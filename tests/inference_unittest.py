#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Processor for CRGA models"""
import otbApplication
from decloud.core import system
import pyotb
import unittest
from decloud.production import crga_processor
from decloud.production.inference import inference
from .decloud_unittest import DecloudTest
import datetime


def get_timestamp(yyyymmdd):
    dt = datetime.datetime.strptime(yyyymmdd, '%Y%m%d')
    ts = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return str(ts)


class InferenceTest(DecloudTest):

    def test_inference_with_mosaic(self):
        # Logger
        system.basic_logging_init()

        # Baseline
        baseline_path = self.get_path("baseline/reconstructed_baseline_w_mosaic.tif")

        # Model
        model_path = self.get_path("models/crga_os2david_occitanie_pretrained")

        # Input sources
        s1_tm1 = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20200929t060008_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20200930txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201001txxxxxx_from-10to3dB.tif')]
        s2_tm1 = [
            self.get_path(
                'baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20200926-103901-393_L2A_T31TEJ_C_V2-2/SENTINEL2B_20200926-103901-393_L2A_T31TEJ_C_V2-2_FRE_10m.tif'),
            self.get_path(
                'baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20200929-104857-489_L2A_T31TEJ_C_V2-2/SENTINEL2B_20200929-104857-489_L2A_T31TEJ_C_V2-2_FRE_10m.tif')]
        s1_t = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201013txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20201012txxxxxx_from-10to3dB.tif')]
        s2_t = [
            self.get_path(
                'baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2/SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2_FRE_10m.tif')]
        s1_tp1 = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201025txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20201024txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20201023t060008_from-10to3dB.tif')]
        s2_tp1 = [
            self.get_path(
                'baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20201026-103901-924_L2A_T31TEJ_C_V2-2/SENTINEL2B_20201026-103901-924_L2A_T31TEJ_C_V2-2_FRE_10m.tif'),
            self.get_path(
                'baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2A_20201024-104859-766_L2A_T31TEJ_C_V2-2/SENTINEL2A_20201024-104859-766_L2A_T31TEJ_C_V2-2_FRE_10m.tif')]

        # Input sources
        sources = {'s1_tm1': pyotb.Mosaic(il=s1_tm1, nodata=0),
                   's2_tm1': pyotb.Mosaic(il=s2_tm1, nodata=-10000),
                   's1_tp1': pyotb.Mosaic(il=s1_tp1, nodata=0),
                   's2_tp1': pyotb.Mosaic(il=s2_tp1, nodata=-10000),
                   's1_t': pyotb.Mosaic(il=s1_t, nodata=0),
                   's2_t': pyotb.Mosaic(il=s2_t, nodata=-10000),
                   'dem': self.get_path('baseline/PREPARE/DEM_PREPARE/T31TEJ.tif')}

        # Sources scales
        sources_scales = {"dem": 2}

        # Inference
        out_tensor = "s2_estim"
        outpath = '/tmp/reconstructed_w_mosaic.tif'
        processor = inference(sources=sources, sources_scales=sources_scales, pad=64,
                              ts=256, savedmodel_dir=model_path, out_tensor=out_tensor, out_nodatavalue=-10000,
                              out_pixeltype=otbApplication.ImagePixelType_int16,
                              nodatavalues={"s1_tm1": 0, "s2_tm1": -10000, "s1_tp1": 0,
                                            "s2_tp1": -10000, "s1_t": 0, "s2_t": -10000})
        processor.write(out=outpath, filename_extension="&streaming:type=tiled&streaming:sizemode=height&"
                                                        "streaming:sizevalue=256&"
                                                        "gdal:co:COMPRESS=DEFLATE&gdal:co:TILED=YES")

        # Just a dummy test
        self.assertTrue(system.file_exists(outpath))

        self.compare_images(outpath, baseline_path)
        self.compare_raster_metadata(outpath, baseline_path)

    def test_inference_with_generic_preprocessor(self):
        # Logger
        system.basic_logging_init()

        # Baseline
        baseline_path = self.get_path("baseline/reconstructed_baseline_w_preprocessor.tif")

        # Model
        model_path = self.get_path("models/crga_os2david_occitanie_pretrained")

        # Input sources
        s1_tm1 = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20200929t060008_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20200930txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201001txxxxxx_from-10to3dB.tif')]
        s2_tm1 = [
            self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20200926-103901-393_L2A_T31TEJ_C_V2-2/'),
            self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20200929-104857-489_L2A_T31TEJ_C_V2-2/')]
        s1_t = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201013txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20201012txxxxxx_from-10to3dB.tif')]
        s2_t = self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2')
        s1_tp1 = [
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201025txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20201024txxxxxx_from-10to3dB.tif'),
            self.get_path('baseline/PREPARE/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20201023t060008_from-10to3dB.tif')]
        s2_tp1 = [
            self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2B_20201026-103901-924_L2A_T31TEJ_C_V2-2'),
            self.get_path('baseline/PREPARE/S2_PREPARE/T31TEJ/SENTINEL2A_20201024-104859-766_L2A_T31TEJ_C_V2-2/')]

        outpath = '/tmp/reconstructed_w_preprocessor.tif'
        crga_processor.crga_processor(il_s1before=s1_tm1, il_s2before=s2_tm1,
                                      il_s1=s1_t, in_s2=s2_t,
                                      il_s1after=s1_tp1, il_s2after=s2_tp1,
                                      dem=self.get_path('baseline/PREPARE/DEM_PREPARE/T31TEJ.tif'),
                                      output=outpath, maxgap=48, savedmodel=model_path)

        # Just a dummy test
        self.assertTrue(system.file_exists(outpath))

        self.compare_images(outpath, baseline_path)
        self.compare_raster_metadata(outpath, baseline_path)


if __name__ == '__main__':
    unittest.main()
