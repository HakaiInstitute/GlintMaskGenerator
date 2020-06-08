# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-22
# Based on Matlab script by Tom Bell written 6/28/2019
#
# Description: Generate masks for glint regions in in RGB images using Tom Bell's blue-channel binning algorithm.

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from .common import save_mask, is_dji_red_edge, is_micasense_red_edge

Image.MAX_IMAGE_PIXELS = None
EPSILON = 1e-8


def make_single_mask(img_path: str, red_edge: bool = False, glint_threshold: float = 0.9, mask_buffer_sigma: int = 20,
                     num_bins: int = 8) -> np.ndarray:
    """Create and return a glint mask for RGB imagery.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images.
        Supported formats for single image processing are here:
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.

    red_edge: Optional[bool]
        Flag indicating if image is a red edge image or not. If false, assumes first three image channels are RGB.

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

    if not red_edge:
        # Get the Blue Channel
        img = img[:, :, 2]

        # Quantize blue channel into num_bins bins
        bins = np.linspace(0, 255, num_bins, endpoint=False)
    else:
        # Quantize 16-bit red edge values
        bins = np.linspace(0, (1 << 16) - 1, num_bins, endpoint=False)

    si = np.digitize(img, bins)
    # Rescale to (0, 1) range
    sis = (si.astype(np.float) - si.min(initial=0)) / ((si.max(initial=0) - si.min(initial=0)) + EPSILON)

    # Find Glint Threshold and Set those Pixels to 1
    mask = (sis <= glint_threshold)

    # Create Buffered Mask
    if not red_edge:
        mask_buffered = gaussian_filter(mask.astype(np.float), mask_buffer_sigma)
        mask = mask_buffered >= 0.99

    # Save the mask
    mask = mask.astype(np.uint8) * 255

    return mask


def make_and_save_single_mask(img_path: str, mask_out_path: str, red_edge: bool = False, glint_threshold: float = 0.9,
                              mask_buffer_sigma: int = 20, num_bins: int = 8) -> List[str]:
    """Create and save a glint mask for RGB imagery.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images.
        Supported formats for single image processing are here:
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.

    mask_out_path: str
        The directory where the image mask should be saved.

    red_edge: Optional[bool]
        Flag indicating if image is a red edge image or not. If false, assumes first three image channels are RGB.

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
    mask = make_single_mask(img_path=img_path, red_edge=red_edge, glint_threshold=glint_threshold,
                            mask_buffer_sigma=mask_buffer_sigma,
                            num_bins=num_bins)

    if red_edge:
        if is_dji_red_edge(img_path):
            out_paths = [Path(mask_out_path).joinpath(f"{Path(img_path).stem[:-1]}{i}_mask.png") for i in range(6)]
        elif is_micasense_red_edge(img_path):
            out_paths = [Path(mask_out_path).joinpath(f"{Path(img_path).stem[:-1]}{i}_mask.png") for i in range(1, 6)]
        else:
            raise RuntimeError(f"Could not identify what kind of red edge file was processed for file {str(img_path)}")

        return [save_mask(out_path, mask) for out_path in out_paths]
    else:
        out_path = Path(mask_out_path).joinpath(f"{Path(img_path).stem}_mask.png")
        return [save_mask(out_path, mask)]
