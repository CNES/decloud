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
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="decloud",
    version="1.3",
    author="Remi Cresson, Nicolas Narçon, Benjamin Commandre",
    author_email="remi.cresson@inrae.fr",
    description="Deep learning based reconstruction of optical time series using SAR imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/TBD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    keywords="remote sensing, deep learning, gapfilling, remove clouds, satellite imagery, otb, orfeotoolbox",
    entry_points={
        'console_scripts': [
            'crga_processor=decloud.production.crga_processor:main',
            'meraner_processor=decloud.production.meraner_processor:main'
        ],
    }
)
