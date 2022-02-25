## URL
Models are located here: https://nextcloud.inrae.fr/s/DEy4PgR2igSQKKH

## Models available

### CRGA OS2
TODO: illustration inputs / output

This section covers how to run these pre-trained models:
- crga_os2_occitanie_pretrained
- crga_os2david_occitanie_pretrained
- crga_os2_burkina_pretrained

### CRGA OS1
TODO: illustration inputs / output

This section covers how to run these pre-trained models:
- crga_os1_occitanie_pretrained
- crga_os1_burkina_pretrained

### Meraner
TODO: illustration inputs / output

This section covers how to run these pre-trained models:
- meraner_occitanie_pretrained
- meraner_burkina_pretrained

### Monthly synthesis S2/S1
TODO: illustration inputs / output

This section covers how to run these pre-trained models:
- monthly_synthesis_s2s1_occitanie_pretrained
- monthly_synthesis_s2s1_david_occitanie_pretrained

### Monthly synthesis S2
TODO: illustration inputs / output

This section covers how to run these pre-trained models:
- monthly_synthesis_s2_david_occitanie_pretrained

## How to run a model

For instance, we use `crga_processor.py` to perform the inference of the *crga* models.
This program not only performs the inference, but also takes care of preparing the right input images to feed the model, and also the post-processing steps (like removing inferred no-data pixels).
It is built exclusively using OTB application pipelines, and is fully streamable (not limitation or images size).

Below is an example of use : 

```yaml
python production/crga_processor.py \
--il_s1before \
  /data/decloud/bucket/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201001txxxxxx_from-10to3dB.tif \
  /data/decloud/bucket/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20200930txxxxxx_from-10to3dB.tif \
  /data/decloud/bucket/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20200929t060008_from-10to3dB.tif \
--il_s1 \
  /data/decloud/bucket/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201013txxxxxx_from-10to3dB.tif \
  /data/decloud/bucket/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20201011t060008_from-10to3dB.tif \
  /data/decloud/bucket/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20201012txxxxxx_from-10to3dB.tif \
--il_s1after \
  /data/decloud/bucket/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_139_20201025txxxxxx_from-10to3dB.tif \
  /data/decloud/bucket/S1_PREPARE/T31TEJ/s1a_31TEJ_vvvh_DES_037_20201024txxxxxx_from-10to3dB.tif \
  /data/decloud/bucket/S1_PREPARE/T31TEJ/s1b_31TEJ_vvvh_DES_110_20201023t060008_from-10to3dB.tif \
--il_s2before \
  /data/decloud/bucket/S2_PREPARE/T31TEJ/SENTINEL2B_20200929-104857-489_L2A_T31TEJ_C_V2-2 \
  /data/decloud/bucket/S2_PREPARE/T31TEJ/SENTINEL2B_20200926-103901-393_L2A_T31TEJ_C_V2-2 \
--il_s2after \
  /data/decloud/bucket/S2_PREPARE/T31TEJ/SENTINEL2B_20201026-103901-924_L2A_T31TEJ_C_V2-2 \
  /data/decloud/bucket/S2_PREPARE/T31TEJ/SENTINEL2A_20201024-104859-766_L2A_T31TEJ_C_V2-2 \
--in_s2 /data/decloud/bucket/S2_PREPARE/T31TEJ/SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2 \
--dem /data/decloud/bucket/DEM_PREPARE/T31TEJ.tif \
--savedmodel /data/decloud/todel/savedmodel_david2/09-04-21_224907_various_enhancements_and_todos_93228_crga_os2_david_bt48_bv48 \
--output /data/decloud/results/theia_data/SENTINEL2B_20201012-105848-497_L2A_T31TEJ_C_V2-2_FRE_10m_reconst_reference.tif
```
