#!/usr/bin/python

# common.py
# Copyright 2016
#   Guilherme Folego (gfolego@gmail.com)
#   Otavio Gomes (otaviolmiro@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""
============================================================
Common utilities for feature extraction
============================================================
"""


import os
import argparse
import numpy as np


# PyTorch specific constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32


# Classification constants
VG_CLASS = 1
NVG_CLASS = 0
VG_PREFIX = 'vg'
NVG_PREFIX = 'nvg'
LABEL_SEPARATOR = '_'


# Model training constants
K_FOLD = 3
N_ITER = 10
WINDOW_SIZE = 224


# Score model constants
SCORE_MAX_ITER = 1000
SCORE_K_FOLD = 5


# Global values
_VERBOSE_LEVEL = 0
n_cores = 1


def dir_type(path):
    """Directory type for argparse."""
    if not os.path.isdir(path):
        msg = "{0} is not a directory".format(path)
        raise argparse.ArgumentTypeError(msg)
    return path


def file_type(path):
    """File type for argparse."""
    if not os.path.isfile(path):
        msg = "{0} is not a regular file".format(path)
        raise argparse.ArgumentTypeError(msg)
    return path


def iter_type(x):
    """Iteration type for argparse."""
    x = int(x)
    if x <= 0:
        raise argparse.ArgumentTypeError("Minimum number of iterations is 1")
    return x


def get_verbose_level():
    """Get verbose level."""
    return _VERBOSE_LEVEL


def set_verbose_level(level):
    """Set verbose level."""
    global _VERBOSE_LEVEL
    _VERBOSE_LEVEL = level


def print_verbose(msg, level=1):
    """Print message according to verbose level."""
    if _VERBOSE_LEVEL >= level:
        print(msg)


def get_n_cores():
    """Get number of cores."""
    global n_cores
    return n_cores


def set_n_cores(n):
    """Set number of cores."""
    global n_cores
    n_cores = n
