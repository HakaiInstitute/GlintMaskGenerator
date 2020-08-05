"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-08-05
Description: 
"""
from pathlib import Path
from typing import Generator

import numpy as np


def case_insensitive_glob(dir_path: str, pattern: str) -> Generator:
    """ Find files with a glob pattern while ignore upper/lower case."""
    return Path(dir_path).glob(
        ''.join('[%s%s]' % (char.lower(), char.upper()) if char.isalpha() else char for char in pattern))


def normalize_img(img: np.ndarray, /, *, bit_depth) -> np.ndarray:
    """ Normalize the values of an image with arbitrary bit_depth to range [0, 1]."""
    return img / ((1 << bit_depth) - 1)
