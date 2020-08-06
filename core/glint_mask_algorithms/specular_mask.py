"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-05-22
Description: Generate masks for glint regions in in RGB images using a thresholding on the estimated specular
    component of reflection.
"""

import math

import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_closing, binary_opening

Image.MAX_IMAGE_PIXELS = None

# For numerical stability in division ops
EPSILON = 1e-8


def estimate_specular_reflection_component(img: np.ndarray, percent_diffuse: float) -> np.ndarray:
    # noinspection SpellCheckingInspection
    """Estimate the specular reflection component of pixels in an image.

        Based on method from:
            Wang, S., Yu, C., Sun, Y. et al. Specular reflection removal
            of ocean surface remote sensing images from UAVs. Multimedia Tools
            Appl 77, 11363–11379 (2018). https://doi.org/10.1007/s11042-017-5551-7

        Parameters
        ----------
        img: numpy.ndarray, shape=(H,W,C)
            A numpy ndarray of an RGB image.
        percent_diffuse
            An estimate of the % of pixels that show purely diffuse reflection.

        Returns
        -------
        numpy.ndarray, shape=(H,W)
            An 1D image where values are an estimate of the component of specular reflectance.
        """
    # Calculate the pixel-wise max intensity and intensity range over RGB channels
    i_max = np.amax(img, axis=2)
    i_min = np.amin(img, axis=2)
    i_range = i_max - i_min

    # Calculate intensity ratio
    q = np.divide(i_max, i_range + EPSILON)

    # Select diffuse only pixels using the PERCENTILE_THRESH
    # i.e. A percentage of PERCENTILE_THRESH pixels are supposed to have no
    #     specular reflection
    num_thresh = math.ceil(percent_diffuse * q.size)

    # Get intensity ratio for a pixel dividing the image into diffuse and specular sections
    q_x_hat = np.partition(q.ravel(), num_thresh)[num_thresh]

    # Estimate the spectral component of each pixel
    spec_ref_est = np.clip(i_max - (q_x_hat * i_range), 0, None)

    return spec_ref_est


def make_single_mask(img: np.ndarray, percent_diffuse: float = 0.95, mask_thresh: float = 0.99,
                     opening: int = 15, closing: int = 15) -> np.ndarray:
    """Create and return a glint mask for RGB imagery.

    Parameters
    ----------
    img: np.ndarray shape=(H,W,3)
        Path to a 3-channel RGB numpy image normalized to values in [0,1].
    percent_diffuse
        An estimate of the percentage of pixels in an image that show pure diffuse reflectance, and
        thus no specular reflectance (glint).
    mask_thresh
        The threshold on the specular reflectance estimate image to convert into a mask.
        e.g. if more than 50% specular reflectance is unacceptable, use 0.5.
    opening
        The number of morphological opening iterations on the produced mask.
        Useful for closing small holes in the mask.
    closing
        The number of morphological closing iterations on the produced mask.
        Useful for removing small bits of mask.

    Returns
    -------
    numpy.ndarray, shape=(H,W)
        Numpy array of glint mask for img at input_path.
    """
    spec_ref = estimate_specular_reflection_component(img, percent_diffuse)

    # Generate the mask
    mask = (spec_ref >= mask_thresh).astype(np.uint8)

    # Fill in small holes in the mask
    if opening > 0:
        mask = binary_opening(mask, iterations=opening).astype(np.uint8)

    # Remove small bits of mask
    if closing > 0:
        mask = binary_closing(mask, iterations=closing).astype(np.uint8)

    # Save the mask
    mask = mask * 255

    return mask
