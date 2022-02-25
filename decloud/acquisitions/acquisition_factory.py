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
"""The AcquisitionsLayout factory enables to create AcquisitionsLayout from .json files"""
import json
from decloud.acquisitions.sensing_layout import S1Acquisition, S2Acquisition, AcquisitionsLayout


class AcquisitionFactory:

    def __init__(self):
        pass

    @staticmethod
    def get_acquisition(name):

        acquisition = AcquisitionsLayout()

        with open(name, "r") as file:
            structure = json.load(file)

        for step in structure:
            s1_acquisition = None
            s2_acquisition = None

            if step == "options":
                acquisition.options(structure[step])
            else:
                for acq in structure[step]["acquisition"]:
                    if structure[step]["acquisition"][acq]["type"] == "S1":
                        s1_acquisition = S1Acquisition(ascending=acq["orbit"] if "orbit" in acq else None)
                    elif structure[step]["acquisition"][acq]["type"] == "S2":
                        s2_acquisition = S2Acquisition(
                            min_cloud_percent=structure[step]["acquisition"][acq]["min_cloud_percent"]
                            if "min_cloud_percent" in structure[step]["acquisition"][acq] else 0,
                            max_cloud_percent=structure[step]["acquisition"][acq]["max_cloud_percent"]
                            if "max_cloud_percent" in structure[step]["acquisition"][acq] else 0)

                if "timeframe_origin" not in structure[step]:
                    assert "timeframe_start_hours" in structure[step] and "timeframe_end_hours" in structure[step], \
                        "{} not origin frame, timeframe_start_hours and timeframe_end_hours must be filled in"
                else:
                    assert "timeframe_start_hours" not in structure[step] and \
                           "timeframe_end_hours" not in structure[step], \
                        "{} origin frame, timeframe_start_hours and timeframe_end_hours must not be filled in"

                acquisition.new_acquisition(step,
                                            s1_acquisition=s1_acquisition,
                                            s2_acquisition=s2_acquisition,

                                            max_s1s2_gap_hours=structure[step]["max_s1s2_gap_hours"]
                                            if "max_s1s2_gap_hours" in structure[step] else None,

                                            timeframe_origin=structure[step]["timeframe_origin"]
                                            if "timeframe_origin" in structure[step] else False,

                                            timeframe_start_hours=structure[step]["timeframe_start_hours"]
                                            if "timeframe_start_hours" in structure[step] else 0,

                                            timeframe_end_hours=structure[step]["timeframe_end_hours"]
                                            if "timeframe_end_hours" in structure[step] else 0
                                            )

        return acquisition
