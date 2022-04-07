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
"""Classes for Sentinel products handling"""
from decloud.core import system
from decloud.core.system import logging_info
import decloud.preprocessing.constants as constants
import abc
import datetime
import os
import numpy as np
import otbApplication
import pyotb


# -------------------------------------------------------- Base --------------------------------------------------------


class ProductBase(abc.ABC):
    """
    Base class for Sentinel products
    """

    @abc.abstractmethod
    def __init__(self, product_path):
        assert isinstance(product_path, str)
        self.product_path = product_path
        pass

    @abc.abstractmethod
    def get_date(self):
        """
        :return: a datetime.datetime object
        """
        pass

    @abc.abstractmethod
    def get_nodatavalue(self):
        """
        :return: the no-data value
        """
        pass

    def get_timestamp(self):
        return self.get_date().timestamp()


# ----------------------------------------------------- Sentinel-1 -----------------------------------------------------


class S1ProductBase(ProductBase):
    """
    Base class for Sentinel-1 products
    """
    INTERNAL_NODATA_VALUE = 0.0

    @abc.abstractmethod
    def __init__(self, product_path):
        super().__init__(product_path)

    @abc.abstractmethod
    def get_raster_10m(self):
        """
        - Bands order must be vv, vh
        - Each channel must be in the "processed dynamic": see preprocessing.constants.S1_NORMALIZATION_BM_EXPR
          Meaning:
          1. Value "0" is for no-data
          2. Values are passed to log scale, then linearly rescaled between [1, UINT16 - 1] using:

                x --> 1 + (UINT16 - 1) / (MAXDB - MINDB) * (ln(abs(x) + EPS) - MINDB)
                The values are clipped in the [1, UINT16 - 1] range.

        :return: a str or an OTB Application OutputParameterImage
        """
        pass


class S1_TILED(S1ProductBase):
    def __init__(self, product_path, verbose=True):
        logging_info("Init. S1_TILED product", verbose)
        super().__init__(product_path)

        # check product type
        if not system.file_exists(product_path):
            raise Exception("Not a S1_TILED product. File {} not found".format(product_path))
        suffix = "_{}.tif".format(constants.SUFFIX_S1)
        if not product_path.endswith(suffix):
            raise Exception("Not a S1_TILED product. File {} not ending with {}".format(product_path, suffix))

        # 10m bands
        self.bands_10m_file = product_path
        logging_info("10m spacing bands: {}".format(self.bands_10m_file), verbose)

        # Edge mask
        self.edge_raster = os.path.splitext(self.bands_10m_file)[0] + '_edge.tif'

        # Date
        onefile = system.basename(self.bands_10m_file)
        datestr = onefile.split("_")[5]
        datestr = datestr.replace("x", "1")
        # now format should be %Y%m%dt%H%M%S
        self.date = datetime.datetime.strptime(datestr, '%Y%m%dt%H%M%S')
        logging_info("Date: {}".format(self.date), verbose)

    def get_raster_10m(self):
        return self.bands_10m_file

    def get_date(self):
        return self.date

    def get_nodatavalue(self):
        return 0.0

    def get_nodata_percentage(self):
        nodatas = (pyotb.Input(self.edge_raster) != 0)
        return np.mean(nodatas)

    def get_raster_10m_encoding(self):
        return otbApplication.ImagePixelType_uint16


class S1_THEIA(S1ProductBase):
    def __init__(self, product_path):
        """product_path can either be the VV or the VH path"""
        super().__init__(product_path)

        # From one polarization, deducing the other
        product_name = system.basename(product_path)
        product_dir = system.dirname(product_path)
        if '_vh_' in product_name:
            self.vh_file = product_path
            self.vv_file = system.join(product_dir, product_name.replace('_vh_', '_vv_'))
        elif '_vv_' in product_name:
            self.vv_file = product_path
            self.vh_file = system.join(product_dir, product_name.replace('_vv_', '_vh_'))
        else:
            raise Exception("Not a S1_THEIA product : {} not a VV nor a VH image".format(product_path))

        if not system.file_exists(self.vv_file):
            raise Exception("Not a S1_THEIA product : {} does not exist".format(self.vv_file))
        if not system.file_exists(self.vh_file):
            raise Exception("Not a S1_THEIA product : {} does not exist".format(self.vh_file))

        datestr = product_name.split('_')[5].split('t')[0]
        self.date = datetime.datetime.strptime(datestr, '%Y%m%d')

    def get_raster_10m(self):
        def _bm(input_fn):
            """ BandMath for SAR normalization """
            bm = otbApplication.Registry.CreateApplication("BandMath")
            bm.SetParameterStringList("il", [input_fn])
            bm.SetParameterString("exp", constants.S1_NORMALIZATION_BM_EXPR)
            bm.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_uint16)
            bm.Execute()
            return bm

        bm_vv = _bm(self.vv_file)
        bm_vh = _bm(self.vh_file)

        conc = otbApplication.Registry.CreateApplication("ConcatenateImages")
        conc.ConnectImage("il", bm_vv, 'out')
        conc.ConnectImage("il", bm_vh, 'out')
        # TODO: vérifier si PixelType est utile ?
        conc.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_uint16)

        return conc

    def get_date(self):
        return self.date

    def get_nodatavalue(self):
        return 0.0

    def get_raster_10m_encoding(self):
        return otbApplication.ImagePixelType_uint16


# ----------------------------------------------------- Sentinel-2 -----------------------------------------------------


class S2ProductBase(ProductBase):
    """
    Base class for Sentinel-2 products
    """
    INTERNAL_NODATA_VALUE = -10000

    @abc.abstractmethod
    def __init__(self, product_path):
        super().__init__(product_path)

    @abc.abstractmethod
    def get_raster_10m(self):
        """
        Bands order must be 2, 3, 4, 8
        :return: a str or an OTB Application OutputParameterImage
        """
        pass

    @abc.abstractmethod
    def get_raster_20m(self):
        """
        Bands order must be 5, 6, 7, 8a, 11, 12
        :return: a str or an OTB Application OutputParameterImage
        """
        pass

    def get_raster_all_bands(self):
        """
        Stack the 10m + 20m bands in a raster with 10m resolution
        :return: an App object
        """
        raster_20m_resampled = pyotb.Superimpose(inr=self.get_raster_10m(), inm=self.get_raster_20m(),
                                                 interpolator='nn')
        stack = pyotb.ConcatenateImages([self.get_raster_10m(), raster_20m_resampled])
        return stack

    @abc.abstractmethod
    def get_raster_10m_encoding(self):
        """
        Encoding of the 10m bands
        :return: a otbApplication.ImagePixelType
        """
        pass


class S2_ESA(S2ProductBase):
    def __init__(self, product_path, verbose=True):
        logging_info("Init. S2_ESA product", verbose)
        super().__init__(product_path)

        def _get_files():
            is_zip = product_path.lower().endswith(".zip")
            if is_zip:
                logging_info("Input type is a .zip archive", verbose)
                files = system.list_files_in_zip(product_path)
                files = [system.to_vsizip(product_path, f) for f in files]
            else:
                logging_info("Input type is a directory", verbose)
                files = system.get_files(product_path, '.jp2')
            return files

        def _filter_files(files, endswith):
            filelist = [f for f in files if f.endswith(endswith)]
            if len(filelist) == 0:
                raise Exception("{} not a S2_ESA product : {} is missing".format(product_path, endswith))
            return filelist

        # Rasters
        files = _get_files()
        self.band2_file = _filter_files(files, "_B02_10m.jp2")[0]
        self.band3_file = _filter_files(files, "_B03_10m.jp2")[0]
        self.band4_file = _filter_files(files, "_B04_10m.jp2")[0]
        self.band8_file = _filter_files(files, "_B08_10m.jp2")[0]
        self.band5_file = _filter_files(files, "_B05_20m.jp2")[0]
        self.band6_file = _filter_files(files, "_B06_20m.jp2")[0]
        self.band7_file = _filter_files(files, "_B07_20m.jp2")[0]
        self.band8a_file = _filter_files(files, "_B8A_20m.jp2")[0]
        self.band11_file = _filter_files(files, "_B11_20m.jp2")[0]
        self.band12_file = _filter_files(files, "_B12_20m.jp2")[0]

        # Date
        onefile = self.band2_file
        onefile = system.basename(onefile)
        datestr = onefile.split("_")[1]
        self.date = datetime.datetime.strptime(datestr, '%Y%m%dT%H%M%S')

        def conc_and_set_nodata(il):
            """
            Concatenate and set the NoData to a value common to all products type (THEIA & ESA)

            :param il: input filenames
            :return: concatenate pipeline
            """
            conc = otbApplication.Registry.CreateApplication("ConcatenateImages")
            conc.SetParameterStringList("il", il)
            conc.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_int32)

            mnnodata = otbApplication.Registry.CreateApplication('ManageNoData')
            mnnodata.ConnectImage('in', conc, 'out')
            mnnodata.SetParameterString('mode', 'changevalue')
            mnnodata.SetParameterFloat('mode.changevalue.newv', self.INTERNAL_NODATA_VALUE)
            mnnodata.Execute()

            return mnnodata

        self.conc_10m = conc_and_set_nodata([self.band2_file, self.band3_file, self.band4_file, self.band8_file])
        self.conc_20m = conc_and_set_nodata([self.band5_file, self.band6_file, self.band7_file, self.band8a_file,
                                             self.band11_file, self.band12_file])

    def get_raster_10m(self):
        return self.conc_10m

    def get_raster_20m(self):
        return self.conc_20m

    def get_date(self):
        return self.date

    def get_raster_10m_encoding(self):
        return otbApplication.ImagePixelType_uint16

    def get_nodatavalue(self):
        return 0


class S2_THEIA(S2ProductBase):
    def __init__(self, product_path, verbose=True):
        logging_info("Init. S2_THEIA product", verbose)
        super().__init__(product_path)

        def _get_files():
            is_zip = product_path.lower().endswith(".zip")
            if is_zip:
                logging_info("Input type is a .zip archive", verbose)
                files = system.list_files_in_zip(product_path)
                files = [system.to_vsizip(product_path, f) for f in files]
            else:
                logging_info("Input type is a directory", verbose)
                files = system.get_files(product_path, '.tif')
            return files

        def _filter_files(files, endswith):
            filelist = [f for f in files if f.endswith(endswith)]
            if len(filelist) == 0:
                raise Exception("{} not a S2_THEIA product : {} is missing".format(product_path, endswith))
            return filelist

        # Rasters
        files = _get_files()
        self.band2_file = _filter_files(files, "_FRE_B2.tif")[0]
        self.band3_file = _filter_files(files, "_FRE_B3.tif")[0]
        self.band4_file = _filter_files(files, "_FRE_B4.tif")[0]
        self.band8_file = _filter_files(files, "_FRE_B8.tif")[0]
        self.band5_file = _filter_files(files, "_FRE_B5.tif")[0]
        self.band6_file = _filter_files(files, "_FRE_B6.tif")[0]
        self.band7_file = _filter_files(files, "_FRE_B7.tif")[0]
        self.band8a_file = _filter_files(files, "_FRE_B8A.tif")[0]
        self.band11_file = _filter_files(files, "_FRE_B11.tif")[0]
        self.band12_file = _filter_files(files, "_FRE_B12.tif")[0]
        self.cld_msk_file = _filter_files(files, "_CLM_R1.tif")[0]
        self.edg_msk_file = _filter_files(files, "_EDG_R1.tif")[0]

        # Date
        onefile = self.band2_file
        onefile = system.basename(onefile)
        datestr = onefile.split("_")[1]
        self.date = datetime.datetime.strptime(datestr, '%Y%m%d-%H%M%S-%f')

        def conc(il):
            """
            Concatenate.

            :param il: input filenames
            :return: concatenate pipeline
            """
            conc = otbApplication.Registry.CreateApplication("ConcatenateImages")
            conc.SetParameterStringList("il", il)
            conc.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_int16)
            conc.Execute()

            return conc

        self.conc_10m = conc([self.band2_file, self.band3_file, self.band4_file, self.band8_file])
        self.conc_20m = conc([self.band5_file, self.band6_file, self.band7_file, self.band8a_file, self.band11_file,
                              self.band12_file])

    def get_raster_10m(self):
        return self.conc_10m

    def get_raster_20m(self):
        return self.conc_20m

    def get_date(self):
        return self.date

    def get_raster_10m_encoding(self):
        return otbApplication.ImagePixelType_int16

    def get_nodatavalue(self):
        return -10000


class S2_TILED(S2ProductBase):
    def __init__(self, product_path, verbose=True):
        logging_info("Init. S2_TILED product", verbose)
        super().__init__(product_path)

        def _get_files(endswith, number_expected):
            files = system.get_files(product_path, endswith)
            if len(files) != number_expected:
                raise Exception("Not a S2_TILED product (expected {} files ending with {})".format(number_expected,
                                                                                                   endswith))
            return files

        # Rasters
        self.bands_10m_file = _get_files("_FRE_10m.tif", 1)[0]
        self.bands_20m_file = _get_files("_FRE_20m.tif", 1)[0]
        self.cld_msk_file = _get_files("_CLM_R1.tif", 1)[0]
        self.edg_msk_file = _get_files("_EDG_R1.tif", 1)[0]

        # Date
        onefile = self.bands_10m_file
        onefile = system.basename(onefile)
        datestr = onefile.split("_")[1]
        self.date = datetime.datetime.strptime(datestr, '%Y%m%d-%H%M%S-%f')

    def get_raster_10m(self):
        # TODO: Use the statistics + ExtractROI to crop the output image
        return self.bands_10m_file

    def get_raster_20m(self):
        # TODO: Use the statistics + ExtractROI to crop the output image
        return self.bands_20m_file

    def get_date(self):
        return self.date

    def get_raster_10m_encoding(self):
        return otbApplication.ImagePixelType_int16

    def get_nodata_percentage(self):
        nodatas = (pyotb.Input(self.edg_msk_file) != 0)
        return np.mean(nodatas)

    def get_cloud_percentage(self):
        clouds = (pyotb.Input(self.cld_msk_file) != 0)
        return np.mean(clouds)

    def get_nodatavalue(self):
        return -10000


# ----------------------------------------------------- Factory --------------------------------------------------------

class Factory:
    @staticmethod
    def create(product_path, product_type, verbose=True):
        """
        :param product_path: product path (.zip or directory)
        :param product_type: "s1" or "s2"
        :param verbose: default True, prints info about what succeeded or failed
        :return: either a ESA product or a THEIA product
        """
        assert product_type in ["s1", "s2"]
        ret = None

        def _try_load(product_class):
            nonlocal ret
            if ret is None:
                try:
                    product = product_class(product_path=product_path, verbose=verbose)
                    ret = product
                except Exception as exp:
                    logging_info("Product is not of type {} (Exception: {})".format(product_type, exp), verbose)
                    pass

        if product_type == "s1":
            _try_load(S1_TILED)
            _try_load(S1_THEIA)
        elif product_type == "s2":
            _try_load(S2_TILED)
            _try_load(S2_THEIA)
            _try_load(S2_ESA)
        else:
            raise Exception("product_type must be either \"s1\" or \"s2\" (value: {})".format(product_type))

        return ret
