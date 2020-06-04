# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-22
#
# Description: Generate masks for glint regions in in RGB images using a thresholding on the estimated specular
# component of reflection.

import math
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from scipy.ndimage.morphology import binary_opening, binary_closing

from core.common import save_mask

Image.MAX_IMAGE_PIXELS = None

# For numerical stability in division ops
EPSILON = 1e-8


def estimate_specular_reflection_component(img: np.ndarray, percent_diffuse: float) -> np.ndarray:
    """Estimate the specular reflection component of pixels in an image.

    Based on method from:
        Wang, S., Yu, C., Sun, Y. et al. Specular reflection removal
        of ocean surface remote sensing images from UAVs. Multimed Tools
        Appl 77, 11363–11379 (2018). https://doi.org/10.1007/s11042-017-5551-7

    Parameters
    ----------
    img: numpy.ndarray, shape=(H,W,C)
        A numpy ndarray of an RGB image.
    percent_diffuse: float
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


def make_single_mask(img_path: str, percent_diffuse: float = 0.1, mask_thresh: float = 0.8,
                     opening: int = 5, closing: int = 5) -> np.ndarray:
    """Create and return a glint mask for RGB imagery.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images.
        Supported formats for single image processing are here:
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.

    percent_diffuse: Optional[float]
        An estimate of the percentage of pixels in an image that show pure diffuse reflectance, and
        thus no specular reflectance (glint). Defaults to 0.1. Try playing with values, low ones typically work well.

    mask_thresh: Optional[float]
        The threshold on the specular reflectance estimate image to convert into a mask.
        E.g. if more than 50% specular reflectance is unacceptable, use 0.5. Default is 0.8.

    opening: Optional[int]
        The number of morphological opening iterations on the produced mask.
        Useful for closing small holes in the mask. 5 by default.

    closing: Optional[int]
        The number of morphological closing iterations on the produced mask.
        Useful for removing small bits of mask. 5 by default.

    Returns
    -------
    numpy.ndarray, shape=(H,W)
        Numpy array of glint mask for img at input_path.
    """
    # Open the image
    img = Image.open(img_path)

    # Use RGB bands only
    img = np.array(img)[:, :, :3] / 255.

    spec_ref = estimate_specular_reflection_component(img, percent_diffuse)

    # Generate the mask
    mask = (spec_ref <= mask_thresh)

    # Fill in small holes in the mask
    if opening > 0:
        mask = binary_opening(mask.astype(np.uint8), iterations=opening).astype(np.bool)

    # Remove small bits of mask
    if closing > 0:
        mask = binary_closing(mask.astype(np.uint8), iterations=closing).astype(np.bool)

    # Save the mask
    mask = mask.astype(np.uint8) * 255

    return mask


def make_and_save_single_mask(img_path: str, mask_out_path: str, percent_diffuse: float = 0.1, mask_thresh: float = 0.8,
                              opening: int = 5, closing: int = 5) -> List[str]:
    """Create and return a glint mask for RGB imagery.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images.
        Supported formats for single image processing are here:
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.

    mask_out_path: str
        The directory where the image mask should be saved.

    percent_diffuse: Optional[float]
        An estimate of the percentage of pixels in an image that show pure diffuse reflectance, and
        thus no specular reflectance (glint). Defaults to 0.1. Try playing with values, low ones typically work well.

    mask_thresh: Optional[float]
        The threshold on the specular reflectance estimate image to convert into a mask.
        E.g. if more than 50% specular reflectance is unacceptable, use 0.5. Default is 0.8.

    opening: Optional[int]
        The number of morphological opening iterations on the produced mask.
        Useful for closing small holes in the mask. 5 by default.

    closing: Optional[int]
        The number of morphological closing iterations on the produced mask.
        Useful for removing small bits of mask. 5 by default.

    Returns
    -------
    List[str]
        The name of the saved masks.
    """
    out_path = Path(mask_out_path).joinpath(f"{Path(img_path).stem}_mask.png")
    mask = make_single_mask(img_path=img_path, percent_diffuse=percent_diffuse, mask_thresh=mask_thresh,
                            opening=opening, closing=closing)
    return [save_mask(out_path, mask)]
