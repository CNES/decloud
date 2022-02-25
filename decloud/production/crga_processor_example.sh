#!/bin/bash
python production/crga_processor.py \
--il_s1before \
  /path/to/s1b_31TEJ_vvvh_DES_139_20201001txxxxxx_from-10to3dB.tif \
  /path/to/s1a_31TEJ_vvvh_DES_037_20200930txxxxxx_from-10to3dB.tif \
  /path/to/s1b_31TEJ_vvvh_DES_110_20200929t060008_from-10to3dB.tif \
--il_s1 \
  /path/to/s1b_31TEJ_vvvh_DES_139_20201013txxxxxx_from-10to3dB.tif \
  /path/to/s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif \
  /path/to/s1a_31TEJ_vvvh_DES_037_20201012txxxxxx_from-10to3dB.tif \
--il_s1after \
  /path/to/s1b_31TEJ_vvvh_DES_139_20201025txxxxxx_from-10to3dB.tif \
  /path/to/s1a_31TEJ_vvvh_DES_037_20201024txxxxxx_from-10to3dB.tif \
  /path/to/s1b_31TEJ_vvvh_DES_110_20201023t060008_from-10to3dB.tif \
--il_s2before \
  /path/to/SENTINEL2B_20200929-104857-489_L2A_T31TEJ_C_V2-2 \
  /path/to/SENTINEL2B_20200926-103901-393_L2A_T31TEJ_C_V2-2 \
--il_s2after \
  /path/to/SENTINEL2B_20201026-103901-924_L2A_T31TEJ_C_V2-2 \
  /path/to/SENTINEL2A_20201024-104859-766_L2A_T31TEJ_C_V2-2 \
--in_s2 SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2 \
--dem T31TEJ.tif \
--savedmodel /path/to/crga_os2_occitanie_pretrained \
--output reconstructed_20201012.tif
