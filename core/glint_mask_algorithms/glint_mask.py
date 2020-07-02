"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-05-22
Description: Generate masks for glint regions in in RGB images using Tom Bell's blue-channel binning algorithm.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

EPSILON = 1e-8


def make_single_mask(img: np.ndarray, glint_threshold: float = 0.9, mask_buffer_sigma: int = 20,
                     num_bins: int = 8) -> np.ndarray:
    """Create and return a glint mask for RGB imagery.

    Parameters
    ----------
    img: np.ndarray shape=(H,W)
        Path to a single channel numpy image normalized to values in [0,1].

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
    bins = np.linspace(0., 1., num_bins, endpoint=False)
    si = np.digitize(img, bins)
    # Rescale to (0, 1) range
    sis = (si.astype(np.float) - si.min(initial=0)) / ((si.max(initial=0) - si.min(initial=0)) + EPSILON)

    # Find Glint Threshold and Set those Pixels to 1
    mask = (sis <= glint_threshold)

    # Buffer the mask
    mask_buffered = gaussian_filter(mask.astype(np.float), mask_buffer_sigma)
    mask = mask_buffered >= 0.99

    # Save the mask
    mask = mask.astype(np.uint8) * 255

    return mask
