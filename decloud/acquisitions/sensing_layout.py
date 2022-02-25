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
"""Classes for acquisition layouts"""
from dataclasses import dataclass


# --------------------------------------------- Acquisition classes ----------------------------------------------------


@dataclass
class GenericAcquisition:
    """ Sentinel Acquisition """
    timestamp: int


@dataclass
class S1Acquisition:
    """ Sentinel-1 Acquisition """
    ascending: bool = False


@dataclass
class S2Acquisition:
    """ Sentinel-2 Acquisition """
    min_cloud_percent: 'typing.Any'
    max_cloud_percent: float


# ------------------------------------------- Acquisition layout class -------------------------------------------------


class AcquisitionsLayout(dict):
    """
    Class storing the acquisition layout
    """
    SENSOR_S1_KEY = "S1"
    SENSOR_S2_KEY = "S2"
    MAX_S1S2_TIMESTAMP_DELTA_KEY = "MAX_S1S2_TIMESTAMP_DELTA_KEY"
    TIMESTAMP_BEGIN_KEY = "TIMESTAMP_BEGIN_KEY"
    TIMESTAMP_END_KEY = "TIMESTAMP_END_KEY"
    OPTIONS = {}

    def new_acquisition(self, name, s1_acquisition=None, s2_acquisition=None, max_s1s2_gap_hours=None,
                        timeframe_origin=False, timeframe_start_hours=0, timeframe_end_hours=0):
        """
        Add a new acquisition in the layout.

        Parameters:
            name: acquisition name (e.g. "Img_t-1")
            s1_acquisition: a S1Acquisition instance or None
            s2_acquisition: a S2Acquisition instance or None
            max_s1s2_gap_hours: if both s1_acquisition and s2_acquisition are not None, this must be provided (maximum
                number of hours between the s1_acquisition date and the s2_acquisition date)
            timeframe_origin: True or False. If True, the acquisition will be considered as the temporal origin for the
                other acquisitions. Meaning that the timeframe_start_hours and timeframe_end_hours or the other
                acquisitions will be relative to the date of the s2_acquisition created with timeframe_origin=True. Only
                one acquisition can be set as origin.
            timeframe_start_hours, timeframe_end_hours: range of the timeframe. Relative to the acquisition which is the
                reference (i.e. the only acquisition that has timeframe_origin=True)
        """

        # Add a new key in the dict
        assert name not in self
        self[name] = dict()
        if s1_acquisition is not None:
            assert isinstance(s1_acquisition, S1Acquisition)
            self[name][self.SENSOR_S1_KEY] = s1_acquisition
        if s2_acquisition is not None:
            assert isinstance(s2_acquisition, S2Acquisition)
            self[name][self.SENSOR_S2_KEY] = s2_acquisition

        # At least s1_acquisition or s2_acquisition must be different than None
        assert s1_acquisition is not None or s2_acquisition is not None

        # If both s1_acquisition and s2_acquisition are different than None, we must have the max_s1s2_gap_hours
        # its default value is None
        self[name][self.MAX_S1S2_TIMESTAMP_DELTA_KEY] = None
        if s1_acquisition is not None and s2_acquisition is not None:
            assert max_s1s2_gap_hours is not None
            assert max_s1s2_gap_hours > 0
            self[name][self.MAX_S1S2_TIMESTAMP_DELTA_KEY] = max_s1s2_gap_hours * 3600

        # No need to set a [start, end] range if the acquisition is set as the reference (it will be the origin)
        if timeframe_origin is True:
            # Can be only one acquisition set as reference
            for acquisition_name in self:
                if acquisition_name != name:
                    assert not self.is_ref(acquisition_name)
            assert timeframe_start_hours == 0
            assert timeframe_end_hours == 0
        self[name][self.TIMESTAMP_BEGIN_KEY] = min(timeframe_start_hours, timeframe_end_hours) * 3600
        self[name][self.TIMESTAMP_END_KEY] = max(timeframe_start_hours, timeframe_end_hours) * 3600

    def options(self, options):
        self.OPTIONS = options

    def is_siblings(self, acq_list, acq_name):
        """
        Return True if the two acquisitions is set as sibling.
        """
        if "siblings" in self.OPTIONS.keys():
            for sibling in self.OPTIONS["siblings"]:
                for acq in acq_list:
                    if acq in sibling and acq_name in sibling:
                        return True

        return False

    def is_ref(self, name):
        """
        Return True if the acquisition is set as the temporal origin.
        """
        return self[name][self.TIMESTAMP_BEGIN_KEY] == 0 and self[name][self.TIMESTAMP_END_KEY] == 0

    def get_ref_name(self):
        """
        Return the acquisition name which is the temporal origin.
        """
        for acquisition_name in self:
            if self.is_ref(acquisition_name):
                return acquisition_name
        raise Exception("No temporal origin found! You must set one reference using "
                        "new_acquisition(..., timeframe_origin=True, ...)")

    def get_timestamp_range(self, name):
        """
        Return the timestamp range (timestamp_begin, timestamp_end)
        """
        return self[name][self.TIMESTAMP_BEGIN_KEY], self[name][self.TIMESTAMP_END_KEY]

    def get_s1s2_max_timestamp_delta(self, name):
        """
        Return the get_s1s2_max_timestamp_delta of the specified acquisition.
        None can be returned (if only a single s1_acquisition or a single s2_acquisition)
        """
        return self[name][self.MAX_S1S2_TIMESTAMP_DELTA_KEY]

    def _get_sx_acquisition(self, name, sensor_key):
        if sensor_key not in self[name]:
            return None
        return self[name][sensor_key]

    def has_s1_acquisition(self, name):
        """
        Return True if the acquisition has a S1Acquisition
        """
        return self.SENSOR_S1_KEY in self[name]

    def has_s2_acquisition(self, name):
        """
        Return True if the acquisition has a S2Acquisition
        """
        return self.SENSOR_S2_KEY in self[name]

    def get_s1_acquisition(self, name):
        """
        Return the s1_acquisition of the specified acquisition
        name: acquisition name
        """
        return self._get_sx_acquisition(name=name, sensor_key=self.SENSOR_S1_KEY)

    def get_s2_acquisition(self, name):
        """
        Return the s2_acquisition of the specified acquisition
        name: acquisition name
        """
        return self._get_sx_acquisition(name=name, sensor_key=self.SENSOR_S2_KEY)

    def summarize(self):
        """
        This function summarizes the acquisition layout, displaying a nice table
        Example:
            Cresson et al. layout:

                            |      t-1       |       t        |      t+1
            ----------------+----------------+----------------+----------------
                   S1       |       /        |     +/-24h     |       /
            ----------------+----------------+----------------+----------------
                   S2       | [-360h, -120h] |    [0h, 0h]    |  [120h, 360h]
                            |   0-0% cld.    |   0-0% cld.    |   0-0% cld.

        """

        max_head_len = max([len(key) for key in self] + [16])

        def _cell(msg):
            return msg.center(max_head_len, " ")

        def _round_hours(n_seconds):
            return "{}h".format(int(n_seconds / 3600))

        def _summarize_timerange(n_seconds_1, n_seconds_2):
            ts_min = min(n_seconds_1, n_seconds_2)
            ts_max = max(n_seconds_1, n_seconds_2)
            return "[{}, {}]".format(_round_hours(ts_min), _round_hours(ts_max))

        def _cell_s1(key):
            content = "/"
            if self.has_s1_acquisition(key):
                delta = self.get_s1s2_max_timestamp_delta(key)
                content = "+/-{}".format(_round_hours(delta))
            return _cell(content)

        def _cell_s2(key):
            content = "/"
            if self.has_s2_acquisition(key):
                ts_begin, ts_end = self.get_timestamp_range(key)
                content = _summarize_timerange(ts_begin, ts_end)
            return _cell(content)

        def _cell_s2b(key):
            content = "/"
            if self.has_s2_acquisition(key):
                s2_acquisition = self.get_s2_acquisition(key)
                content = "{}-{}% cld.".format(s2_acquisition.min_cloud_percent, s2_acquisition.max_cloud_percent)
            return _cell(content)

        keys = self.keys()
        headers = [_cell("")] + [_cell(key) for key in keys]
        line1 = [_cell("S1")] + [_cell_s1(key) for key in keys]
        line2 = [_cell("S2")] + [_cell_s2(key) for key in keys]
        line3 = [_cell("")] + [_cell_s2b(key) for key in keys]

        horizontal_line = "+".join(["-" * max_head_len for k in range(len(keys) + 1)])
        msg = "|".join(headers)
        msg += "\n" + horizontal_line
        msg += "\n" + "|".join(line1)
        msg += "\n" + horizontal_line
        msg += "\n" + "|".join(line2)
        msg += "\n" + "|".join(line3)
        print(msg)
