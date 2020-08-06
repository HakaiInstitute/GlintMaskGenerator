"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-08-05
Description: Utility functions used in the glint masking tools.
"""

import numpy as np


def normalize_img(img: np.ndarray, bit_depth: int) -> np.ndarray:
    """ Normalize the values of an image with arbitrary bit_depth to range [0, 1].

    Args:
        img: np.ndarray
            The image data
        bit_depth: int
            The bit depth of the image, e.g. 8-bit, 16-bit, 32-bit...
            Must be specified using a kwarg.

    Returns:
        np.ndarray: The image normalized to range [0,1].
    """
    return img / ((1 << bit_depth) - 1)
