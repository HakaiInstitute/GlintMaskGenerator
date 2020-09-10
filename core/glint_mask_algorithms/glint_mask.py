"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-05-22
Description: Generate masks for glint regions in in RGB images using Tom Bell's blue-channel binning algorithm.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

EPSILON = 1e-8


def make_single_mask(img: np.ndarray, glint_threshold: float = 0.875, mask_buffer_sigma: int = 0) -> np.ndarray:
    """Create and return a glint mask for RGB imagery.

    Parameters
    ----------
    img: np.ndarray shape=(H,W)
        Path to a single channel numpy image normalized to values in [0,1].
    glint_threshold
        The pixel band threshold indicating glint.
        Domain for values is (0.0, 1.0).
    mask_buffer_sigma
        The sigma for the Gaussian kernel used to buffer the mask.

    Returns
    -------
    numpy.ndarray, shape=(H,W)
        Numpy array of mask for the image at img_path.
    """
    # TOM's method:
    # bins = np.linspace(0., 1., num_bins, endpoint=False)
    # si = np.digitize(img, bins)
    # sis = (si.astype(np.float) - si.min(initial=0)) / ((si.max(initial=0) - si.min(initial=0)) + EPSILON)
    #
    # # Find Glint Threshold and Set those Pixels to 1
    # mask = (sis <= glint_threshold)

    # Much more efficient method.
    # With glint_threshold=0.875, it is equivalent to the above with num_bins=8 and glint_threshold=0.9
    mask = img < glint_threshold

    # Buffer the mask
    if mask_buffer_sigma > 0:
        mask_buffered = gaussian_filter(mask.astype(np.float), mask_buffer_sigma)
        mask = mask_buffered >= 0.99

    # Convert to format required by Metashape
    mask = mask.astype(np.uint8) * 255

    return mask
