#!/usr/bin/env python3
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
"""
This scripts summarizes the number of samples that we can get from an AcquisitionsLayout
suited for single optical image reconstruction from date SAR/optical pair, for different
parameters of the AcquisitionsLayout
"""
import argparse
from decloud.acquisitions.sensing_layout import AcquisitionsLayout, S1Acquisition, S2Acquisition
from decloud.core.tile_io import TilesLoader
from decloud.core.dataset import RandomIterator
from decloud.core import system


def create_meraner_al(max_s1s2_gap_hours, max_s2s2_days):
    """
    Create an AcquisitionsLayout suited for Meraner-like models
    """
    new_al = AcquisitionsLayout()
    new_al.new_acquisition("t",
                           s1_acquisition=S1Acquisition(),
                           s2_acquisition=S2Acquisition(min_cloud_percent=1, max_cloud_percent=100),
                           max_s1s2_gap_hours=max_s1s2_gap_hours,
                           timeframe_origin=True)
    new_al.new_acquisition("target",
                           s2_acquisition=S2Acquisition(min_cloud_percent=0, max_cloud_percent=0),
                           timeframe_start_hours=-24 * max_s2s2_days,
                           timeframe_end_hours=24 * max_s2s2_days)
    return new_al


system.basic_logging_init()

parser = argparse.ArgumentParser(description="Acquisition layout analysis")
parser.add_argument("--tiles", required=True, help="Path to tile handler file (.json)")
parser.add_argument("--patch_size", type=int, default=256)
params = parser.parse_args()

# Tiles handlers
th = TilesLoader(params.tiles, patchsize_10m=params.patch_size)

# Compute
res = []
for h in [12, 18, 24, 36]:
    for d in [3, 4, 5, 6, 7, 8, 9, 10]:
        al = create_meraner_al(h, d)
        it = RandomIterator(tile_handlers=th, acquisitions_layout=al, tile_rois={})
        res.append([h, d, it.nb_of_tuples])

# Display
for s in ["h", "d", "nb_of_tuples"]:
    print("| {}".format(s).ljust(25), end="")
print("|")
for r in res:
    for s in r:
        print("| {}".format(s).ljust(25), end="")
    print("|")

# Occitanie
# | h                      | d                      | nb_of_tuples           |
# | 12                     | 3                      | 98024                  |
# | 12                     | 4                      | 206352                 |
# | 12                     | 5                      | 344919                 |
# | 12                     | 6                      | 483895                 |
# | 12                     | 7                      | 601102                 |
# | 12                     | 8                      | 601102                 |
# | 12                     | 9                      | 699354                 |
# | 12                     | 10                     | 858913                 |
# | 18                     | 3                      | 98024                  |
# | 18                     | 4                      | 206352                 |
# | 18                     | 5                      | 344919                 |
# | 18                     | 6                      | 483895                 |
# | 18                     | 7                      | 601102                 |
# | 18                     | 8                      | 601102                 |
# | 18                     | 9                      | 699354                 |
# | 18                     | 10                     | 858913                 |
# | 24                     | 3                      | 170993                 |
# | 24                     | 4                      | 364111                 |
# | 24                     | 5                      | 597569                 |
# | 24                     | 6                      | 851263                 |
# | 24                     | 7                      | 1046634                |
# | 24                     | 8                      | 1046634                |
# | 24                     | 9                      | 1221418                |
# | 24                     | 10                     | 1506905                |
# | 36                     | 3                      | 239242                 |
# | 36                     | 4                      | 497910                 |
# | 36                     | 5                      | 836453                 |
# | 36                     | 6                      | 1191613                |
# | 36                     | 7                      | 1461675                |
# | 36                     | 8                      | 1461675                |
# | 36                     | 9                      | 1704018                |
# | 36                     | 10                     | 2090504                |

# Afrique
# | h                      | d                      | nb_of_tuples           |
# | 12                     | 3                      | 2079                   |
# | 12                     | 4                      | 4498                   |
# | 12                     | 5                      | 37455                  |
# | 12                     | 6                      | 67499                  |
# | 12                     | 7                      | 70265                  |
# | 12                     | 8                      | 70265                  |
# | 12                     | 9                      | 72629                  |
# | 12                     | 10                     | 99690                  |
# | 18                     | 3                      | 5069                   |
# | 18                     | 4                      | 9924                   |
# | 18                     | 5                      | 54411                  |
# | 18                     | 6                      | 93778                  |
# | 18                     | 7                      | 100131                 |
# | 18                     | 8                      | 100131                 |
# | 18                     | 9                      | 104949                 |
# | 18                     | 10                     | 146446                 |
# | 24                     | 3                      | 5239                   |
# | 24                     | 4                      | 10231                  |
# | 24                     | 5                      | 66886                  |
# | 24                     | 6                      | 122652                 |
# | 24                     | 7                      | 129059                 |
# | 24                     | 8                      | 129059                 |
# | 24                     | 9                      | 134174                 |
# | 24                     | 10                     | 189333                 |
# | 36                     | 3                      | 6522                   |
# | 36                     | 4                      | 13081                  |
# | 36                     | 5                      | 94408                  |
# | 36                     | 6                      | 165248                 |
# | 36                     | 7                      | 173282                 |
# | 36                     | 8                      | 173492                 |
# | 36                     | 9                      | 180435                 |
# | 36                     | 10                     | 261768                 |
