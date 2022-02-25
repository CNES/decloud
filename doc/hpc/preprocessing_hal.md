# Pre-processings on HAL

This document explains how to perform the pre-processing of images on HAL (CNES cluster).
It shows how to use the scripts that are in located in `shell`

## Prepare Sentinel-2 images

The `sentinel2_prepare.pbs` script runs the `preprocessing/sentinel2_prepare.py` program, that process a single Sentinel-2 product.
It can be submitted on multiple tiles with the following script:
```
#!/bin/bash

ROOT_IN=/.../.../SENTINEL2
ROOT_OUT=/.../.../S2_PREPARE

for tile_dir in $ROOT_IN/*; do
  for s2_image in $tile_dir/*; do
    qsub -v in_image="$s2_image",out_s2_dir="$ROOT_OUT" ~/decloud/shell/sentinel2_prepare.pbs
  done
done
```
With:
- ROOT_IN: the path containing tiles of Sentinel-2 images (T31TEJ, T31TEK, ...) each one containing symlinks to datalake
- ROOT_OUT: the output directory, for the pre-processed Sentinel-2 images
The program is able to identify the name of the Sentinel-2 tile, and create the output directory for the tile in the ROOT_IN output directory. 
  No need to create each tile directory, just let the program do it.

## Prepare Sentinel-1 images
The `sentinel1_prepare.sh` script submit one job that runs the `preprocessing/sentinel1_prepare.py` program, processing **a single SAR image**.

Let denote:
- `$IN_S1_DIR`: the directory containing all Sentinel-1 images produced by **s1tiling**,
- `$OUT_S1_DIR`: the output directory, where the Sentinel-1 images are generated.

To process an entire tile, the following command can be used:
```
find $IN_S1_DIR -type f -iname "*_vh_*" \
-exec ~/decloud/shell/sentinel1_prepare.sh {} "$OUT_S1_DIR" \;
```

## Compute clouds coverage and pixel validity statistics

The `tile_coverage.sh` script submit one job that computes the cloud coverage, and the percent of valid pixels in an entire tile.
The script takes as input (1) the tile file (.json) and (2) the output directory for the generated files.

Let denote:
- `$OUT_STATS_DIR`: the output directory, where the files are generated.

```
~/decloud/shell/tile_coverage.sh T31TFK.json "$OUT_STATS_DIR"
```
