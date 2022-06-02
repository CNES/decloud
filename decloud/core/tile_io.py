# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2022 INRAE

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
"""Classes for Sentinel images access"""
import os
import datetime
import itertools
import json
import logging
import multiprocessing
from abc import ABC, abstractmethod
import numpy as np
import rtree
from scipy import spatial
import otbApplication
from decloud.core import system
from decloud.preprocessing import constants
from decloud.core import raster

# --------------------------------------------------- Constants --------------------------------------------------------


KEY_S1_BANDS_10M = "s1_10m"
KEY_S2_BANDS_10M = "s2_10m"
KEY_S2_BANDS_20M = "s2_20m"
KEY_S2_CLOUDMASK_10M = "s2_cld_10m"
KEY_DEM_20M = "dem_20m"

DTYPE = {KEY_S1_BANDS_10M: np.uint16,
         KEY_S2_BANDS_10M: np.int16,
         KEY_S2_BANDS_20M: np.int16,
         KEY_S2_CLOUDMASK_10M: np.uint8,
         KEY_DEM_20M: np.int16}

NB_CHANNELS = {KEY_S1_BANDS_10M: 2,
               KEY_S2_BANDS_10M: 4,
               KEY_S2_BANDS_20M: 6,
               KEY_S2_CLOUDMASK_10M: 1,
               KEY_DEM_20M: 1}


# ---------------------------------------------------- Helpers ---------------------------------------------------------


def s1_filename_to_md(filename):
    """
    This function converts the S1 filename into a small dict of metadata
    :param filename: S1 raster filename
    :return: dict, a small dict with useful metadata
    """
    basename = filename[filename.rfind("/"):]
    metadata = dict()
    splits = basename.split("_")
    if len(splits) != 7:
        raise Exception(f"{filename} not a S1 image (wrong number of splits between \"_\" in filename)")
    if len(splits[5]) < 15:
        raise Exception(f"{filename} not a S1 archive (wrong date format)")
    date_str = splits[5][:15]
    metadata["tile"] = splits[1]
    if date_str[9:15] == "xxxxxx":
        date_str = date_str[0:8] + "t054500"  # We should find a way to use the real acquisition time here
    metadata["date"] = datetime.datetime.strptime(date_str, '%Y%m%dt%H%M%S')
    metadata["orbit"] = splits[3]
    metadata["pol"] = splits[2]
    metadata["filename"] = filename
    return metadata


def get_s1images_in_directory(pth, ref_patchsize, patchsize_10m):
    """
    Returns a list of S1Images instantiated from the designed path
    :param pth: the path containing the S1 products
    :param ref_patchsize: reference patch size
    :param patchsize_10m: size of the 10m-patch
    :return: a list of S1Image instances
    """
    files = system.get_files(pth, "dB.tif")
    return [create_s1_image(vvvh_gtiff=fn, ref_patchsize=ref_patchsize, patchsize_10m=patchsize_10m) for fn in files]


def create_s1_image(vvvh_gtiff, ref_patchsize, patchsize_10m):
    """
    Instantiate a S1Image from the given GeoTiff. The input image must be 2 channel (vv/vh) processed using
    sentinel1_prepare.py

    :param vvvh_gtiff: input geotiff image
    :param ref_patchsize: reference patch size
    :param patchsize_10m: size of 10m-patch
    :return: an S1Image instance
    """
    metadata = s1_filename_to_md(vvvh_gtiff)
    pth = system.dirname(vvvh_gtiff)

    # Compute stats
    edge_stats_fn = os.path.join(pth, system.new_bname(metadata["filename"], constants.SUFFIX_STATS_S1))
    compute_patches_stats(image=metadata["filename"], output_stats=edge_stats_fn, expr="im1b1==0&&im1b2==0",
                          patchsize=ref_patchsize)

    return S1Image(acq_date=metadata["date"], edge_stats_fn=edge_stats_fn, vvvh_fn=metadata["filename"],
                   ascending=metadata["orbit"].lower() == "asc", patchsize_10m=patchsize_10m)


def get_s2images_in_directory(pth, ref_patchsize, patchsize_10m, with_cld_mask, with_20m_bands):
    """
    Returns a list of S2Images instantiated from the designed path
    :param pth: the path containing the prepared Sentinel-2 products
    :param ref_patchsize: reference patch size
    :param patchsize_10m: size of the 10m-patch
    :param with_cld_mask: True/False. If True, the S2Image are instantiated with cloud mask support
    :param with_20m_bands: True/False. If True, the S2Image are instantiated with 20m spacing bands support
    :return: a list of S2Image instances
    """
    return [create_s2_image_from_dir(s2_product,
                                     ref_patchsize=ref_patchsize,
                                     patchsize_10m=patchsize_10m,
                                     with_cld_mask=with_cld_mask,
                                     with_20m_bands=with_20m_bands) for s2_product in system.get_directories(pth)]


def s2_filename_to_md(filename):
    """
    This function converts the S2 filename into a small dict of metadata
    :param filename:
    :return: dict
    """
    basename = system.basename(filename)
    metadata = dict()
    splits = basename.split("_")
    if len(splits) < 4:
        raise Exception(f"{filename} might not be a S2 product")
    metadata["tile"] = splits[3]
    datestr = splits[1]
    metadata["date"] = datetime.datetime.strptime(datestr[:-1], '%Y%m%d-%H%M%S-%f')
    return metadata


def compute_patches_stats(image, output_stats, patchsize, expr=""):
    """
    Run the "SquarePatchesSelection" OTB application over the input image, only if the output file does not exist
    :param image: input image(s). Either a string, or a string list. For string list, the
                  "expr" parameter must be set
    :param output_stats: output image
    :param expr: BandMath expression (optional for 1 input image, mandatory for multiple input image)
    :param patchsize: the patch size
    """
    logging.debug("Computing stats for %s. Result will be stored in %s.", image, output_stats)
    if system.is_complete(output_stats):
        logging.debug("File %s already exists. Skipping.", output_stats)
    else:
        app = otbApplication.Registry.CreateApplication("SquarePatchesSelection")
        if expr:
            thresh = otbApplication.Registry.CreateApplication("BandMath")
            if isinstance(image, str):
                thresh.SetParameterStringList("il", [image])
            else:
                thresh.SetParameterStringList("il", image)
            thresh.SetParameterString("exp", expr)
            thresh.Execute()
            app.SetParameterInputImage("in", thresh.GetParameterOutputImage("out"))
        else:
            if not isinstance(image, str):
                raise Exception("\"image\" must be of type str, if no expr is provided!")
            app.SetParameterString("in", image)
        app.SetParameterInt("patchsize", patchsize)
        app.SetParameterString("out", f"{output_stats}?&gdal:co:COMPRESS=DEFLATE")
        app.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_uint16)
        app.ExecuteAndWriteOutput()
        system.declare_complete(output_stats)


def create_s2_image_from_dir(s2_product_dir, ref_patchsize, patchsize_10m, with_cld_mask, with_20m_bands):
    """
    Create a S2Image instance from one S2 product
    :param s2_product_dir: directory containing:
        SENTINEL2A_20170209-103304-620_L2A_T31TEJ_D_V1-4_FRE_10m.tif
        SENTINEL2A_20170209-103304-620_L2A_T31TEJ_D_V1-4_FRE_20m.tif
        SENTINEL2A_20170209-103304-620_L2A_T31TEJ_D_V1-4_CLM_R1.tif
        SENTINEL2A_20170209-103304-620_L2A_T31TEJ_D_V1-4_EDG_R1.tif
    :param ref_patchsize: patch size to compute statistics
    :param patchsize_10m: patch size used at the 10m spacing resolution
    :param with_cld_mask: True/False. If True, the S2Image are instantiated with cloud mask support
    :param with_20m_bands: True/False. If True, the S2Image are instantiated with 20m spacing bands support
    :return: an S2Image instance
    """
    logging.debug("Processing %s", s2_product_dir)
    files = system.get_files(s2_product_dir, ext=".tif")
    edg_mask, cld_mask, b10m_imgs, b20m_imgs = None, None, None, None
    for file in files:
        if "EDG_R1.tif" in file:
            edg_mask = file
        if "CLM_R1.tif" in file:
            cld_mask = file
        if "FRE_10m.tif" in file:
            b10m_imgs = file
        if "FRE_20m.tif" in file:
            b20m_imgs = file

    # Check that files exists
    def _check(title, filename):
        if filename is None:
            raise Exception(f"File for {title} does not exist in product {s2_product_dir}")

    _check("edge mask", edg_mask)
    _check("cloud mask", cld_mask)
    _check("10m bands stack", b10m_imgs)
    _check("20m bands stack", b20m_imgs)

    # Print infos
    logging.debug("Cloud mask:\t%s", cld_mask)
    logging.debug("Edge mask:\t%s", edg_mask)
    logging.debug("Channels:")
    logging.debug("\t10m bands: %s", b10m_imgs)
    logging.debug("\t20m bands: %s", b20m_imgs)

    # Compute stats
    clouds_stats_fn = os.path.join(s2_product_dir, system.new_bname(cld_mask, constants.SUFFIX_STATS_S2))
    edge_stats_fn = os.path.join(s2_product_dir, system.new_bname(edg_mask, constants.SUFFIX_STATS_S2))
    compute_patches_stats(image=cld_mask, output_stats=clouds_stats_fn, expr="im1b1>0", patchsize=ref_patchsize)
    compute_patches_stats(image=edg_mask, output_stats=edge_stats_fn, patchsize=ref_patchsize)

    # Return a s2 image class
    metadata = s2_filename_to_md(s2_product_dir)
    return S2Image(acq_date=metadata["date"],
                   edge_stats_fn=edge_stats_fn,
                   bands_10m_fn=b10m_imgs,
                   bands_20m_fn=b20m_imgs if with_20m_bands is True else None,
                   cld_mask_fn=cld_mask if with_cld_mask is True else None,
                   clouds_stats_fn=clouds_stats_fn,
                   patchsize_10m=patchsize_10m)


# --------------------------------------------- Patch reader class -----------------------------------------------------


class PatchReader:
    """ A patch reader """

    def __init__(self, filename, psz, dtype, gdal_cachemax="32"):
        """
        Initializer
        :param filename: The image filename
        :param psz: The patch size
        """

        raster.set_gdal_cachemax(gdal_cachemax)

        # Set GDAL DS
        self.gdal_ds = raster.gdal_open(filename)

        # Set GDAL GeoTransform
        self.ulx, self.resolution_x, _, self.uly, _, self.resolution_y = self.gdal_ds.GetGeoTransform()

        # Set patches sizes
        self.patch_size = psz

        # dtype
        self.dtype = dtype

    def get(self, patch_location):
        """
        Read a patch as numpy array
        :return A numpy array
        """
        # Read array
        myarray = self.gdal_ds.ReadAsArray(patch_location[0] * self.patch_size, patch_location[1] * self.patch_size,
                                           self.patch_size, self.patch_size)

        # Re-order bands (when there is > 1 band)
        if len(myarray.shape) == 3:
            axes = (1, 2, 0)
            myarray = np.transpose(myarray, axes=axes)
        else:
            myarray = np.expand_dims(myarray, axis=2)

        return myarray.astype(self.dtype)

    def get_geographic_info(self, patch_location):
        """
        Get the geographic info of a patch
        :param patch_location: tuple
        :return the coordinates of the bounding box (Upper left and Lower Right), in lat/lon 4326 coordinate system
        """
        # Getting the Upper Left info of the patch
        patch_ulx = self.ulx + patch_location[0] * self.resolution_x * self.patch_size
        patch_uly = self.uly + patch_location[1] * self.resolution_y * self.patch_size

        # Deduce the Lower Right
        patch_lrx = patch_ulx + self.patch_size * self.resolution_x
        patch_lry = patch_uly + self.patch_size * self.resolution_y

        # Convert to 4326
        patch_ul_lon, patch_ul_lat = raster.convert_to_4326((patch_ulx, patch_uly), self.gdal_ds)
        patch_lr_lon, patch_lr_lat = raster.convert_to_4326((patch_lrx, patch_lry), self.gdal_ds)

        return patch_ul_lon, patch_ul_lat, patch_lr_lon, patch_lr_lat  # (lon, lat) is the standard for GeoJSON


# ---------------------------------------------- Image base classes ----------------------------------------------------


class AbstractImage(ABC):
    """
    Abstract class for images
    """

    @abstractmethod
    def __init__(self):
        self.patch_sources = dict()

    def get_patch(self, key, patch_location):
        """
        Returns one patch from the selected patch_source at the given location
        :param key: patch source key
        :param patch_location: the patch location
        :return: a numpy array
        """
        if key not in self.patch_sources:
            raise Exception(f"Key {key} not in patches sources. Available sources keys: {self.patch_sources}")
        return self.patch_sources[key].get(patch_location=patch_location)

    @abstractmethod
    def get(self, patch_location):
        """
        Returns all existing data
        :param patch_location: the patch location
        :return: a numpy array
        """


class SentinelImage(AbstractImage):
    """
    Abstract class for Sentinel images
    """

    @abstractmethod
    def __init__(self, acq_date, edge_stats_fn, patchsize_10m):
        """

        :param acq_date: date of the Sentinel image (datetime.datetime)
        :param edge_stats_fn: filename of the edges stats image
        :param patchsize_10m: size of the 10m-patches
        """
        super().__init__()
        self.acq_date = acq_date
        self.edge_stats_fn = edge_stats_fn
        self.edge_stats = raster.read_as_np(edge_stats_fn)
        self.patchsize_10m = patchsize_10m
        self.timestamp = self.acq_date.replace(tzinfo=datetime.timezone.utc).timestamp()

    def get_timestamp(self):
        """
        Returns the timestamps (in seconds)
        :return: int
        """
        return self.timestamp

    def get(self, patch_location):
        return {"timestamp": np.asarray(self.get_timestamp())}


# ------------------------------------------------- DEM image class ----------------------------------------------------


class SRTMDEMImage(AbstractImage):
    """
    DEM image class.
    Handles a single raster access.
    """

    def __init__(self, raster_20m_filename, patchsize_20m):
        """

        :param raster_20m_filename: filename of the 20m-spacing raster
        :param patchsize_20m: patch size (the actual patch is 20m resolution)
        """
        super().__init__()
        self.raster_20m_filename = raster_20m_filename
        self.patchsize_20m = patchsize_20m

        # Patches sources
        self.patch_sources[KEY_DEM_20M] = PatchReader(filename=self.raster_20m_filename, psz=self.patchsize_20m,
                                                      dtype=DTYPE[KEY_DEM_20M])

    def get(self, patch_location):
        return {constants.DEM_KEY: self.patch_sources[KEY_DEM_20M].get(patch_location=patch_location)}


# ------------------------------------------- Sentinel images classes --------------------------------------------------


class S2Image(SentinelImage):
    """
    Sentinel-2 image class.
    Keeps Sentinel-2 product metadata, provide an access to image patches.
    """

    def __init__(self, acq_date, edge_stats_fn, bands_10m_fn, clouds_stats_fn, patchsize_10m, bands_20m_fn=None,
                 cld_mask_fn=None):
        super().__init__(acq_date=acq_date, edge_stats_fn=edge_stats_fn, patchsize_10m=patchsize_10m)
        self.bands_10m_fn = bands_10m_fn
        self.bands_20m_fn = bands_20m_fn
        self.cld_mask_fn = cld_mask_fn
        self.clouds_stats_fn = clouds_stats_fn
        self.clouds_stats = raster.read_as_np(clouds_stats_fn)

        # Prepare patches sources
        self.patch_sources[KEY_S2_BANDS_10M] = PatchReader(filename=self.bands_10m_fn, psz=self.patchsize_10m,
                                                           dtype=DTYPE[KEY_S2_BANDS_10M])
        if self.bands_20m_fn is not None:
            self.patch_sources[KEY_S2_BANDS_20M] = PatchReader(filename=self.bands_20m_fn,
                                                               psz=int(self.patchsize_10m / 2),
                                                               dtype=DTYPE[KEY_S2_BANDS_20M])
        if self.cld_mask_fn is not None:
            self.patch_sources[KEY_S2_CLOUDMASK_10M] = PatchReader(filename=self.cld_mask_fn, psz=self.patchsize_10m,
                                                                   dtype=DTYPE[KEY_S2_CLOUDMASK_10M])

    def get(self, patch_location):
        ret = {"s2_timestamp": np.asarray(self.get_timestamp())}
        ret.update({"s2": self.patch_sources[KEY_S2_BANDS_10M].get(patch_location=patch_location)})
        if self.bands_20m_fn is not None:
            ret.update({"s2_20m": self.patch_sources[KEY_S2_BANDS_20M].get(patch_location=patch_location)})
        if self.cld_mask_fn is not None:
            ret.update({"s2_cld10m": self.patch_sources[KEY_S2_CLOUDMASK_10M].get(patch_location=patch_location)})
        return ret


class S1Image(SentinelImage):
    """
    Sentinel-1 image class.
    Keeps Sentinel-1 product metadata, provide an access to image patches.
    """

    def __init__(self, acq_date, edge_stats_fn, vvvh_fn, ascending, patchsize_10m):
        super().__init__(acq_date=acq_date, edge_stats_fn=edge_stats_fn, patchsize_10m=patchsize_10m)
        self.vvvh_fn = vvvh_fn
        self.ascending = ascending

        # Prepare patches sources
        self.patch_sources[KEY_S1_BANDS_10M] = PatchReader(filename=self.vvvh_fn, psz=self.patchsize_10m,
                                                           dtype=DTYPE[KEY_S1_BANDS_10M])

    def get(self, patch_location):
        ret = {"s1_timestamp": np.asarray(self.get_timestamp())}
        ret.update({"s1_ascending": np.asarray(self.ascending)})
        ret.update({"s1": self.patch_sources[KEY_S1_BANDS_10M].get(patch_location=patch_location)})
        return ret


# ---------------------------------------------- Tile Handler class ----------------------------------------------------


class TileHandler:
    """
    TilesHandler performs every I/O operations, build indexation structures in one S2 tile
    """

    @staticmethod
    def new_bbox(timeframe_low, timeframe_hi, cld_cov_min, cld_cov_max, validity, closest_s1_gap_min,
                 closest_s1_gap_max):
        """
        Return a bounding box in the domain (Time, Cloud coverage, Validity, Closest s1 temporal gap)
        """
        return (timeframe_low, cld_cov_min, validity, closest_s1_gap_min,
                timeframe_hi, cld_cov_max, validity, closest_s1_gap_max)

    def for_each_pos(self, apply_fn):
        """
        Iterate over every (pos_x, pos_y) positions and runs "apply_fn(pos)" with pos = (pos_x, pos_y)
        :param apply_fn: The function to call for each pos. Must have a single argument, "pos" a tuple (pos_x, pos_y)
        :return: nothing
        """
        for pos_x in range(self.grid_size_x):
            for pos_y in range(self.grid_size_y):
                pos = (pos_x, pos_y)
                apply_fn(pos)

    def find_s2(self, pos, timeframe_low, timeframe_hi, cld_cov_min, cld_cov_max, validity, closest_s1_gap_max):
        """
        Return all candidates that intersect the bounding box in the domain (Time, Cloud coverage, Validity,
            Closest s1 temporal gap)
        """
        if closest_s1_gap_max is None:
            closest_s1_gap_max = self.max_distance
        bbox_search = self.new_bbox(timeframe_low, timeframe_hi, cld_cov_min, cld_cov_max, validity,
                                    closest_s1_gap_min=0, closest_s1_gap_max=closest_s1_gap_max)
        return self.s2_trees[pos].intersection(bbox_search)

    def __init__(self, s1_dir, s2_dir, patchsize_10m, tile, dem_20m=None, with_s2_cldmsk=False,
                 with_20m_bands=False):
        """
        TileHandler delivers patches for one given tile.
        Patches are delivered through the read_tuple() function, in the form of a dict with the following structure:
            {"s2_someKey1": np.array([..]),
            "s1_someKey1": np.array([..]),
            ...,
            "dem": np.array([..])}

        :param s1_dir: The directory where the tiled/uint16 S1 images are stored
            (these images have been processed with `preprocessing/sentinel1_prepare.py`)
        :param s2_dir: The directory where the tiled/int16 S2 images are stored
            (these images have been processed with `preprocessing/sentinel2_prepare.py`)
        :param patchsize_10m: The patch size for the 10m resolution. Must be a multiple of 64.
        :param tile: the name of the tile, e.g. 'T31TCJ'
        :param dem_20m: optional raster for the 20m spacing DEM. If value is None, it is not delivered.
        :param with_s2_cldmsk: True or False. True: the cloud mask patches are delivered
        :param with_20m_bands: True or False. True: the 20m-spacing bands patches are delivered
        """

        self.tile = tile
        # This is the size of a patch of the 10m bands of Sentinel-2.
        self.patchsize_10m = patchsize_10m

        # List S1 images
        if s1_dir is not None:
            self.s1_images = get_s1images_in_directory(pth=s1_dir, ref_patchsize=constants.PATCHSIZE_REF,
                                                       patchsize_10m=self.patchsize_10m)
            self.s1_images.sort(key=lambda x: x.acq_date)
            logging.info("Found %i S1 images in %s", len(self.s1_images), s1_dir)

        # List S2 images
        self.s2_images = get_s2images_in_directory(pth=s2_dir, ref_patchsize=constants.PATCHSIZE_REF,
                                                   patchsize_10m=self.patchsize_10m, with_cld_mask=with_s2_cldmsk,
                                                   with_20m_bands=with_20m_bands)
        self.s2_images.sort(key=lambda x: x.acq_date)
        logging.info("Found %i S2 images in %s", len(self.s2_images), s2_dir)

        # Get grid size
        gdal_ds = raster.gdal_open(self.s2_images[0].edge_stats_fn)
        self.grid_size_x = int(gdal_ds.RasterXSize * constants.PATCHSIZE_REF / self.patchsize_10m)
        self.grid_size_y = int(gdal_ds.RasterYSize * constants.PATCHSIZE_REF / self.patchsize_10m)

        # Index images
        # Create one grid of cloud coverage, and one grid of nodatas.
        # The grid has cells of size (patch_size)

        def _index(sx_images, read_fn, process_fn):
            """
            This function is used to build a numpy array of shape (N, grid_size_x, grid_size_y) that store for each
            patch some useful stuff.

            :param sx_images: a list of SentinelImage objects (either S1Image objects or S2Image objects)
            :param read_fn: the function used to retrieve the raster file that is read as a numpy array
            :param process_fn: the function used to process the value retrieved from the sub numpy array
            :return: A numpy array of shape (N, grid_size_x, grid_size_y)
            """
            output = np.zeros((len(sx_images), self.grid_size_x, self.grid_size_y))
            for sx_image_idx, sx_image in enumerate(sx_images):
                gdal_ds = raster.gdal_open(read_fn(sx_image))
                image_as_np = gdal_ds.ReadAsArray()

                def compute_value(pos, full_arr=image_as_np, image_idx=sx_image_idx):
                    sub_arr = raster.get_sub_arr(full_arr,
                                                 patch_location=pos,
                                                 patch_size=self.patchsize_10m,
                                                 ref_patch_size=constants.PATCHSIZE_REF)
                    value = process_fn(sub_arr)
                    pos_x, pos_y = pos
                    output[image_idx, pos_x, pos_y] = value

                self.for_each_pos(compute_value)

            return output

        def _reject_no_data(np_arr):
            """ Returns the map of patches validity """
            return np.amax(np_arr) == 0

        def _get_edge_stats_fn(sx_image):
            """ Returns the edge statistics raster file name """
            return sx_image.edge_stats_fn

        def _average_cloud_coverage_values(np_arr):
            """ Returns the map of average cloud percentage inside patches """
            return np.mean(100.0 * np_arr / (constants.PATCHSIZE_REF * constants.PATCHSIZE_REF))

        def _get_clouds_stats_fn(s2_image):
            """ Returns the clouds statistics raster file name """
            return s2_image.clouds_stats_fn

        def _print_np_stats(np_arr, title="some"):
            """ Print some statistics of the input numpy array """
            msg = "{} stats: Shape={}, Min={:.2f}, Max={:.2f}, Mean={:.2f}, Standard deviation={:.2f}".format(
                title, np_arr.shape, np.amin(np_arr), np.amax(np_arr), np.mean(np_arr), np.std(np_arr))
            logging.info(msg)

        if s1_dir is not None:
            logging.info("Computing S1 patches statistics")
            self.s1_images_validity = _index(self.s1_images, read_fn=_get_edge_stats_fn, process_fn=_reject_no_data)
            _print_np_stats(self.s1_images_validity, "Validity")

        logging.info("Computing S2 patches statistics")
        self.s2_images_validity = _index(self.s2_images, read_fn=_get_edge_stats_fn, process_fn=_reject_no_data)
        _print_np_stats(self.s2_images_validity, "Validity")
        self.s2_images_cloud_coverage = _index(self.s2_images, read_fn=_get_clouds_stats_fn,
                                               process_fn=_average_cloud_coverage_values)
        _print_np_stats(self.s2_images_cloud_coverage, "Cloud coverage")

        # Build a dict() of the closest s1_image for each pos, and for each s2_image
        # The dict structure is like: self.closest_s1[pos][s2_idx]
        class Closest:
            """ Simple class to store index/distance to compute the closest image """

            def __init__(self, index, distance):
                self.index = index
                self.distance = distance

            def update(self, other):
                """ Update the closest one """
                if other.distance < self.distance:
                    self.distance = other.distance
                    self.index = other.index

        self.closest_s1 = dict()
        self.max_distance = 10 * 12 * 31 * 24 * 3600  # Maximum distance to search
        if s1_dir is not None:
            # Build KDTree to index s1 images timestamps
            logging.info("Build KDTrees")
            s1_timestamps_kdtrees = dict()
            s1_timestamps_indices = dict()

            def build_kdtree(pos):
                """
                Build a KDTree at the specified location (pos_x, pos_y)
                This function modifies:
                    s1_timestamps_kdtrees
                    s1_timestamps_indices
                """
                timestamps = []
                timestamps_index = []
                pos_x, pos_y = pos
                for s1_index, s1_image in enumerate(self.s1_images):
                    validity = self.s1_images_validity[s1_index, pos_x, pos_y]
                    if validity:
                        timestamps.append(s1_image.get_timestamp())
                        timestamps_index.append(s1_index)
                s1_timestamps_kdtrees[pos] = spatial.KDTree(list(zip(np.asarray(timestamps).ravel())))
                s1_timestamps_indices[pos] = timestamps_index

            self.for_each_pos(build_kdtree)

            def find_closest_s1_image(pos):
                """
                Find the closest s1 image for each s2 images, at the specified location (pos_x, pos_y)
                This function modifies:
                    self.closest_s1
                """
                closest_s1 = {}
                for s2_idx, s2_image in enumerate(self.s2_images):
                    timestamp_query = np.array([s2_image.get_timestamp()])
                    _value, timestamp_idx = s1_timestamps_kdtrees[pos].query(timestamp_query)
                    s1_idx = s1_timestamps_indices[pos][timestamp_idx]
                    if _value < self.max_distance:
                        closest = Closest(index=s1_idx, distance=_value)
                        if s2_idx not in closest_s1:
                            closest_s1[s2_idx] = closest
                        else:
                            closest_s1[s2_idx].update(closest)
                self.closest_s1[pos] = closest_s1

            self.for_each_pos(find_closest_s1_image)
        else:
            # When S1 images aren't used
            # Closest S1 dict is composed of virtual S1 images
            def set_virtual_closest_s1_image(pos):
                """ Set a virtual S1 image very close to each S2 image """
                self.closest_s1[pos] = {s2_idx: Closest(index=-1,
                                                        distance=0) for s2_idx, _ in enumerate(self.s2_images)}

            self.for_each_pos(set_virtual_closest_s1_image)

        # Build RTrees (Cloud_coverage, Date, Validity, Closest S1 (timestamp))
        logging.info("Build RTrees (Cloud_coverage, Date, Validity, Closest S1)")
        properties = rtree.index.Property()
        properties.dimension = 4
        self.s2_trees = dict()

        def build_rtree(pos):
            """
            Build a RTree for the specified location pos=(pos_x, pos_y)
            This function modifies:
                self.s2_trees
            """
            closest_s1 = self.closest_s1[pos]
            self.s2_trees[pos] = rtree.index.Index(properties=properties)
            pos_x, pos_y = pos
            for s2_image_idx, s2_image in enumerate(self.s2_images):
                timestamp = s2_image.get_timestamp()
                cld_cov_value = self.s2_images_cloud_coverage[s2_image_idx, pos_x, pos_y]
                validity_value = self.s2_images_validity[s2_image_idx, pos_x, pos_y]
                closest_s1_gap = closest_s1[s2_image_idx].distance if s2_image_idx in closest_s1 else self.max_distance
                bbox = self.new_bbox(timeframe_low=timestamp, timeframe_hi=timestamp,
                                     cld_cov_min=cld_cov_value, cld_cov_max=cld_cov_value,
                                     validity=validity_value,
                                     closest_s1_gap_min=closest_s1_gap,
                                     closest_s1_gap_max=self.max_distance)
                self.s2_trees[pos].insert(s2_image_idx, bbox)

        self.for_each_pos(build_rtree)

        # Reading lock
        self.read_lock = multiprocessing.Lock()

        # Setup DEM
        self.dem_image = None if dem_20m is None else SRTMDEMImage(raster_20m_filename=dem_20m,
                                                                   patchsize_20m=int(self.patchsize_10m / 2))

        logging.info("Done")

    def tuple_search(self, acquisitions_layout, roi=None):
        """
        The function performs a search of every tuples of patches that fulfil the acquisition layout.

        :param acquisitions_layout: the acquisition layout that specify how the scenes are acquired
        :param roi: the roi (geotiff file name) describes for each reference patch is the patch has to be used or not.
                    Basically the roi.tif results in the rasterization of one vector layer over the raster grid formed
                    by the reference patch size over the Sentinel-2 image (e.g. 640m spacing is the PATCHSIZE_REF
                    is 64 pixels)
        :return: the tuples, stored in a dict()
        """

        roi_np = None if roi is None else raster.read_as_np(roi)

        # A function that returns True is the (new_elem, y) pos is inside the ROI
        def is_inside_roi(patch_location):
            if roi_np is not None:
                sub_np_arr = raster.get_sub_arr(roi_np, patch_size=self.patchsize_10m, patch_location=patch_location,
                                                ref_patch_size=constants.PATCHSIZE_REF)
                if np.amin(sub_np_arr) == 0:  # if there is at least one cell with "0": it is not entirely inside
                    return False
            return True

        # Summarize acquisition layout
        logging.info("Tile %s, seeking the following acquisition layout:", self.tile)
        acquisitions_layout.summarize()

        # To fetch timestamp origin in acquisitions layout:
        # acquisition_ref_key = acquisitions_layout.get_ref_name()

        # Begin filtering
        acquisition_candidates_grid = dict()

        def collect(pos):
            """
            Collect the samples that match with the search criterion at the given pos=(pos_x, pos_y)
            This function modifies:
                acquisition_candidates_grid
            """
            if is_inside_roi(pos):
                acquisition_candidates_grid[pos] = []

                def _filter(acquisition_name, ref_timestamp):
                    """
                    Function that filter from the available s2 images, given the acquisition and the timestamp
                    """
                    s2_acquisition = acquisitions_layout.get_s2_acquisition(acquisition_name)

                    # Cloud coverage
                    cld_cov_max = s2_acquisition.max_cloud_percent
                    if (isinstance(s2_acquisition.min_cloud_percent, str)
                            and s2_acquisition.min_cloud_percent.startswith('random')):
                        cld_cov_min = min(eval(s2_acquisition.min_cloud_percent), cld_cov_max)
                    elif isinstance(s2_acquisition.min_cloud_percent, (int, float)):
                        cld_cov_min = s2_acquisition.min_cloud_percent
                    else:
                        raise Exception('Wrong format for min cloud percent, must be a number or random.[whatever]')

                    # Timestamp window
                    timeframe_begin, timeframe_end = acquisitions_layout.get_timestamp_range(acquisition_name)

                    # If the S1S2 gap isn't defined, it is because we don't need one S1 image.
                    # So we just put None, so that the find_s2() knows that the timestamp delta is not a filtering
                    # criterion.
                    closest_s1_gap_max = acquisitions_layout.get_s1s2_max_timestamp_delta(acquisition_name)

                    # Call to RTree query
                    result = self.find_s2(pos=pos,
                                          validity=1,
                                          cld_cov_min=cld_cov_min,
                                          cld_cov_max=cld_cov_max,
                                          timeframe_low=ref_timestamp + timeframe_begin,
                                          timeframe_hi=ref_timestamp + timeframe_end,
                                          closest_s1_gap_max=closest_s1_gap_max)
                    return list(result)

                for idx, s2_image in enumerate(self.s2_images):
                    acquisition_candidates = dict()
                    ref_timestamp = s2_image.get_timestamp()

                    for acquisition_name in acquisitions_layout:
                        # Here we check that the images fulfill the constraints
                        ret = _filter(acquisition_name=acquisition_name, ref_timestamp=ref_timestamp)

                        if idx in ret and \
                                not acquisitions_layout.is_siblings(acquisition_candidates.keys(), acquisition_name) \
                                and idx in acquisition_candidates.values():
                            ret.remove(idx)

                        # if acquisition_name == acquisition_ref_key:
                        #     assert (len(ret) <= 1)
                        if len(ret) == 0:
                            break

                        acquisition_candidates[acquisition_name] = ret

                    # We check that we have candidates for each key. If not, we skip.
                    if len(acquisition_candidates) == len(acquisitions_layout):
                        acquisition_candidates_grid[pos].append(acquisition_candidates)

        self.for_each_pos(collect)

        # Add s1/s2 key
        candidates_grid = dict()
        for pos, candidates in acquisition_candidates_grid.items():
            for candidate in candidates:
                new_candidate = dict()
                for key, values in candidate.items():
                    # key: "s_t-1", "s_t", "s_t+1", ...
                    # values : 45, 48, 49, ...
                    new_val = []
                    for value in values:
                        new_entry = {"s2": value}
                        if acquisitions_layout.has_s1_acquisition(key):
                            if value in self.closest_s1[pos]:
                                closest_s1_idx = self.closest_s1[pos][value].index
                                new_entry.update({"s1": closest_s1_idx})
                        new_val.append(new_entry)
                    new_candidate[key] = new_val
                if pos not in candidates_grid:
                    candidates_grid[pos] = [new_candidate]
                else:
                    candidates_grid[pos].append(new_candidate)

        # Should be:
        # candidates_grid[(0, 0)] = [{"t-1": [{"s2": 45}],
        #                             "t": [{"s1:" 12, "s2": 47}],
        #                             "t+1": [{"s2": 48}, {"s2": 49}]},
        #                             ...]

        # Generate every possible combinations from candidates
        tuples_grid = dict()

        # convert acquisition_candidates_grid --> tuples_grid
        # The structure of tuples_grid should be:
        #
        #  tuples_grid[(0,0)] = [{"t-1": {"s2": 5},
        #                         "t":   {"s1": 3, "s2": 7},
        #                         "t+1": {"s2": 11}},
        #                          ...
        #                         {"t-1": {"s2": 345},
        #                          "t":   {"s1": 453, "s2": 344},
        #                          "t+1": {"s2": 346}}]
        #  tuples_grid[(0,1)] = [...]
        #  ...
        #  tuples_grid[(n,n)] = [...]
        index = acquisitions_layout.keys()
        for pos, candidates in candidates_grid.items():
            tuples_grid[pos] = [dict(zip(index, list(x))) for candidate in candidates
                                for x in list(itertools.product(*list(candidate.values())))]

        nb_samples = sum(len(lst) for lst in tuples_grid.values())
        logging.info("Tile %s, found %s samples satisfying the acquisition layout", self.tile, nb_samples)

        return tuples_grid

    def read_tuple(self, tuple_pos, tuple_indices):
        """
        Read a tuple of Sentinel images patches.

        :param tuple_pos: the tuple position in the tile_handler grid. The position is a tuple (pos_x, pos_y)

            e.g. tuple_pos = (23, 41)
        :param tuple_indices: the tuple indices.

            e.g. tuple_indices = {"t-1": {"s2": 345},
                                  "t":   {"s1": 453, "s2": 344},
                                  "t+1": {"s2": 346}}

        :return: new_sample, a dict of the acquisition layout keys prefixed with "s1" or "s2" depending on the sensor,
            with the numpy arrays. If the DEM is used, the {"dem": <numpy_array@0x..>} item updates the returned dict.

            e.g. new_sample = {"s2_t-1": <numpy_array@0x..>,
                               "s1_t": <numpy_array@0x..>,
                               "s2_t": <numpy_array@0x..>,
                               "s2_t+1": <numpy_array@0x..>}

        """

        with self.read_lock:
            new_sample = dict()

            # fill the sample with s1/s2 keys
            for key, values in tuple_indices.items():
                for sx_key, sx_idx in values.items():
                    if sx_key == "s1":
                        src = self.s1_images[sx_idx]
                    elif sx_key == "s2":
                        src = self.s2_images[sx_idx]
                        # Add the geographic info
                        new_sample['geoinfo'] = \
                            src.patch_sources[KEY_S2_BANDS_10M].get_geographic_info(patch_location=tuple_pos)
                    else:
                        raise Exception(f"Unknown key {sx_key}!")
                    src_dict = src.get(patch_location=tuple_pos)
                    for src_key, src_np_arr in src_dict.items():
                        # the final key is composed in concatenating key, "_", src_key
                        new_sample[src_key + "_" + key] = src_np_arr

            # update the sample with the DEM
            if self.dem_image is not None:
                new_sample.update(self.dem_image.get(patch_location=tuple_pos))

            return new_sample


# ---------------------------------------------- Tiles loader class ----------------------------------------------------


class TilesLoader(dict):
    """
    A class that instantiate some TileHandler objects from a json file
    Keys:
     - "S1_ROOT_DIR": str (Optional)
     - "S2_ROOT_DIR": str
     - "DEM_ROOT_DIR": str (Optional)
     - "TILES": list

    Example of a .json file:
    {
      "S1_ROOT_DIR": "/data/decloud/S1_PREPARE",
      "S2_ROOT_DIR": "/data/decloud/S2_PREPARE",
      "DEM_ROOT_DIR": "/data/decloud/DEM_PREPARE",
      "TILES": ["T31TCK", "T31TDJ", "T31TEJ"]
    }
    """

    def __init__(self, the_json, patchsize_10m, with_20m_bands=False):
        """
        :param the_json: The .json file
        :param patchsize_10m: Patches size (64, 128, 256, ...) must be a multiple of 64
        :param with_20m_bands: True or False. True: the 20m-spacing bands patches are delivered
        """
        super().__init__()
        logging.info("Loading tiles from %s", the_json)
        with open(the_json) as json_file:
            data = json.load(json_file)

        def get_pth(key):
            """
            Retrieve the path
            :param key: path key
            :return: the path value
            """
            if key in data:
                value = data[key]
                assert isinstance(value, str)
                return value
            return None

        # Paths
        self.s1_tiles_root_dir = get_pth("S1_ROOT_DIR")
        self.s2_tiles_root_dir = get_pth("S2_ROOT_DIR")
        self.dem_tiles_root_dir = get_pth("DEM_ROOT_DIR")

        if self.s2_tiles_root_dir is None:
            raise Exception(f"S2_ROOT_DIR key not found in {the_json}")

        # Tiles list
        self.tiles_list = data["TILES"]
        if self.tiles_list is None:
            raise Exception(f"TILES key not found in {the_json}")
        if not isinstance(self.tiles_list, list):
            raise Exception("TILES value must be a list of strings!")

        # Instantiate tile handlers
        for tile in self.tiles_list:

            def _get_tile_pth(root_dir, current_tile=tile):
                """ Returns the directory for the current tile """
                if root_dir is not None:
                    return os.path.join(root_dir, current_tile)
                return None

            s1_dir = _get_tile_pth(self.s1_tiles_root_dir)
            s2_dir = _get_tile_pth(self.s2_tiles_root_dir)
            dem_tif = _get_tile_pth(self.dem_tiles_root_dir)
            if dem_tif is not None:
                dem_tif += ".tif"
            logging.info("Creating TileHandler for \"%s\"", tile)

            tile_handler = TileHandler(s1_dir=s1_dir, s2_dir=s2_dir, dem_20m=dem_tif, patchsize_10m=patchsize_10m,
                                       with_20m_bands=with_20m_bands, tile=tile)
            self.update({tile: tile_handler})
