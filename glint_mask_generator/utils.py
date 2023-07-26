"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""
from pathlib import Path
from typing import Iterable

import numpy as np


def normalize_img(img: np.ndarray, bit_depth: int) -> np.ndarray:
    """Normalize the values of an image with arbitrary bit_depth to range [0, 1].

    Parameters
    ----------
    img
        The image data
    bit_depth
        The bit depth of the image.
        e.g. 8 for 8-bit, 16 for 16-bit, 32 for 32-bit.

    Returns
    -------
    np.ndarray
        The image normalized to range [0,1].
    """
    return img / ((1 << bit_depth) - 1)


def list_images(img_dir) -> Iterable[str]:
    """List all image files in img_dir.

    Returns an iterator that lists the files to process. Subclasses may want to override
    this to return specific image types or filter the results. By default, will list all
    images in self.img_dir if the file extension is in the extensions list.

    Returns
    -------
    Iterable[str]
        The list of files to be used for generating the masks.
    """
    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    paths = Path(img_dir).glob("**/*")
    paths = filter(lambda p: p.is_file() and p.suffix.lower() in extensions, paths)
    return (str(p) for p in paths)


def make_circular_kernel(size: int) -> np.ndarray:
    """Create circular kernel"""
    y, x = np.ogrid[-size : size + 1, -size : size + 1]
    dist_m: np.ndarray = x**2 + y**2
    return dist_m <= size**2
