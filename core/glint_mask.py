# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-22
# Based on Matlab script by Tom Bell written 6/28/2019
#
# Description: Generate masks for glint regions in in RGB images using Tom Bell's blue-channel binning algorithm.

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from .common import save_mask

Image.MAX_IMAGE_PIXELS = None
EPSILON = 1e-8


def make_single_mask(img_path: str, img_type: Optional[str] = 'rgb', glint_threshold: float = 0.9,
                     mask_buffer_sigma: int = 20, num_bins: int = 8) -> np.ndarray:
    """Create and return a glint mask for RGB imagery.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images.
        Supported formats for single image processing are here:
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.

    img_type: Optional[str]
        Str describing what kind of files to look for in the specified img_path.
        Should be one of 'rgb', 'micasense_ms', 'dji_ms'. Default is 'rgb'.

    glint_threshold: Optional[float]
        The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
        Play with this value. Default is 0.9.

    mask_buffer_sigma: Optional[int]
        The sigma for the Gaussian kernel used to buffer the mask. Defaults to 20.

    num_bins: Optional[int]
        The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.

    Returns
    -------
    numpy.ndarray, shape=(H,W)
        Numpy array of mask for the image at img_path.
    """
    # Open the image
    img = Image.open(img_path)
    img = np.array(img)

    if img_type == 'rgb':
        # Get the Blue Channel
        img = img[:, :, 2]

        # Quantize blue channel into num_bins bins
        bins = np.linspace(0, 255, num_bins, endpoint=False)
    elif img_type in ['micasense_ms', 'dji_ms']:
        # Quantize 16-bit red edge values
        max_value = (1 << 16) - 1
        bins = np.linspace(0, max_value, num_bins, endpoint=False)
    else:
        raise ValueError(f"Invalid value for img_type {img_type}")

    si = np.digitize(img, bins)
    # Rescale to (0, 1) range
    sis = (si.astype(np.float) - si.min(initial=0)) / ((si.max(initial=0) - si.min(initial=0)) + EPSILON)

    # Find Glint Threshold and Set those Pixels to 1
    mask = (sis <= glint_threshold)

    # Create Buffered Mask for RGB
    if img_type == 'rgb':
        mask_buffered = gaussian_filter(mask.astype(np.float), mask_buffer_sigma)
        mask = mask_buffered >= 0.99

    # Save the mask
    mask = mask.astype(np.uint8) * 255

    return mask


def make_and_save_single_mask(img_path: str, mask_out_path: str, img_type: Optional[str] = 'rgb',
                              glint_threshold: float = 0.9, mask_buffer_sigma: int = 20,
                              num_bins: int = 8) -> List[str]:
    """Create and save a glint mask for RGB imagery.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images.
        Supported formats for single image processing are here:
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.

    mask_out_path: str
        The directory where the image mask should be saved.

    img_type: Optional[str]
        Str describing what kind of files to look for in the specified img_path.
            Should be one of 'rgb', 'micasense_ms', 'dji_ms'. Default is 'rgb'.

    glint_threshold: Optional[float]
        The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
        Play with this value. Default is 0.9.

    mask_buffer_sigma: Optional[int]
        The sigma for the Gaussian kernel used to buffer the mask. Defaults to 20.

    num_bins: Optional[int]
        The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.

    Returns
    -------
    List[str]
        The name of the saved masks.
    """
    mask = make_single_mask(img_path=img_path, img_type=img_type, glint_threshold=glint_threshold,
                            mask_buffer_sigma=mask_buffer_sigma,
                            num_bins=num_bins)

    if img_type == 'micasense_ms':
        out_paths = [Path(mask_out_path).joinpath(f"{Path(img_path).stem[:-1]}{i}_mask.png") for i in range(1, 6)]
        return [save_mask(out_path, mask) for out_path in out_paths]

    elif img_type == 'dji_ms':
        out_paths = [Path(mask_out_path).joinpath(f"{Path(img_path).stem[:-1]}{i}_mask.png") for i in range(6)]
        return [save_mask(out_path, mask) for out_path in out_paths]

    elif img_type == 'rgb':
        out_path = Path(mask_out_path).joinpath(f"{Path(img_path).stem}_mask.png")
        return [save_mask(out_path, mask)]

    else:
        raise ValueError(f"Invalid value for img_type {img_type}")
