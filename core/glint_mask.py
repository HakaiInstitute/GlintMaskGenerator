# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-22
# Based on Matlab script by Tom Bell written 6/28/2019
#
# Description: Generate masks for glint regions in in RGB images using Tom Bell's blue-channel binning algorithm.

from functools import partial
from typing import Optional, Callable

import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from .common import get_img_paths, process_imgs

Image.MAX_IMAGE_PIXELS = None
EPSILON = 1e-8


def red_edge_make_and_save_mask(img_path: str, mask_out_path: str, glint_threshold: float = 0.9,
                                mask_buffer_sigma: int = 20, num_bins: int = 8,
                                processes: Optional[int] = None, callback: Optional[Callable] = None) -> None:
    """Generate masks for glint regions in Red Edge imagery using Tom Bell's binning algorithm.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images. If img_path is a directory, all IMG_xxxx_6.tif
        and DJI_***4.TIF files will be processed.

    mask_out_path: str
        The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
        mask_out_path must be a directory if img_path is specified as a directory.

    glint_threshold: Optional[float]
        The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
        Play with this value. Default is 0.9.

    mask_buffer_sigma: Optional[int]
        The sigma for the Gaussian kernel used to buffer the mask. Defaults to 20.

    num_bins: Optional[int]
        The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.

    processes: Optional[int]
        The number of processes to use for parallel processing. Defaults to number of CPUs.

    callback: Optional[Callable]
        Optional callback function passed the name of each input and output mask files after processing it. Ignore
        in command line interface.

    Returns
    -------
    None
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    img_paths = get_img_paths(img_path, mask_out_path, red_edge=True)
    f = partial(make_single_mask, red_edge=True, glint_threshold=glint_threshold,
                mask_buffer_sigma=mask_buffer_sigma, num_bins=num_bins)
    return process_imgs(f, img_paths, mask_out_path, processes=processes, callback=callback)


def rgb_make_and_save_mask(img_path: str, mask_out_path: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 20,
                           num_bins: int = 8, processes: Optional[int] = None,
                           callback: Optional[Callable] = None) -> None:
    """Generate masks for glint regions in RGB imagery using Tom Bell's binning algorithm.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images. If img_path is a directory, all tif, jpg, jpeg,
        and png images in that directory will be processed.

    mask_out_path: str
        The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
        mask_out_path must be a directory if img_path is specified as a directory.

    glint_threshold: Optional[float]
        The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
        Play with this value. Default is 0.9.

    mask_buffer_sigma: Optional[int]
        The sigma for the Gaussian kernel used to buffer the mask. Defaults to 20.

    num_bins: Optional[int]
        The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.

    processes: Optional[int]
        The number of processes to use for parallel processing. Defaults to number of CPUs.

    callback: Optional[Callable]
            Optional callback function passed the name of each input and output mask files after processing it. Ignore
            in command line interface.

    Returns
    -------
    None
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    img_paths = get_img_paths(img_path, mask_out_path, red_edge=False)
    f = partial(make_single_mask, red_edge=False, glint_threshold=glint_threshold, mask_buffer_sigma=mask_buffer_sigma,
                num_bins=num_bins)
    return process_imgs(f, img_paths, mask_out_path, processes=processes, callback=callback)


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

    callback: Optional[Callable]
            Optional callback function passed the name of each input and output mask files after processing it.

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
    mask_buffered = gaussian_filter(mask.astype(np.float), mask_buffer_sigma)
    mask = mask_buffered >= 0.99

    # Save the mask
    mask = mask.astype(np.uint8) * 255

    return mask
