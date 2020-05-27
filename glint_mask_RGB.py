# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-22
# Based on Matlab script by Tom Bell written 6/28/2019
#
# Description: Generate masks for glint regions in in RGB Images.

from pathlib import Path

import numpy as np
from PIL import Image
from fire import Fire
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


def make_mask(img_path, mask_out_path, glint_threshold=0.9, mask_buffer_sigma=20, num_bins=8):
    """
    Create and return a glint mask for RGB imagery.

    :param img_path: The path to a named input image or directory containing images.
        Supported formats for single image processing are here:
        https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html.
        If img_path is a directory, all jpg, jpeg, and png images in that directory will be processed.
    :param mask_out_path: The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
        mask_out_path must be a directory if img_path is specified as a directory.
    :param glint_threshold: The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
        Play with this value. Default is 0.9.
    :param mask_buffer_sigma: The sigma for the Gaussian kernel used to buffer the mask. Defaults to 20
    :param num_bins: The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.
    :return: Numpy array of mask for results on single files, None for directory processing.
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    if Path(img_path).is_dir():
        if not Path(mask_out_path).is_dir():
            raise ValueError(
                "img_path and mask_out_path must both be a path to a directory, or both be a path to a named file.")

        # Get all images in the specified directory
        for ext in tqdm(("png", "PNG", "jpg", "JPG", "jpeg", "JPEG")):
            for path in tqdm(list(Path(img_path).glob(f"*.{ext}"))):
                out_path = Path(mask_out_path).joinpath(f"{path.stem}_mask.png")
                _ = make_mask(str(path), str(out_path),
                              glint_threshold=glint_threshold, mask_buffer_sigma=mask_buffer_sigma, num_bins=num_bins)

    else:
        # Open the image
        img = Image.open(img_path)

        # Use RGB bands only
        img = np.array(img)[:, :, :3]

        # Get the Blue Channel
        blue = img[:, :, 2]

        # Quantize blue channel into num_bins bins
        bins = np.linspace(0, 255, num_bins, endpoint=False)
        si = np.digitize(blue, bins)
        # Rescale to (0, 1) range
        sis = (si.astype(np.float) - si.min()) / (si.max() - si.min())

        # Find Glint Threshold and Set those Pixels to 1
        mask = (sis <= glint_threshold)

        # Create Buffered Mask
        mask_buffered = gaussian_filter(mask.astype(np.float), mask_buffer_sigma)
        mask = mask_buffered >= 0.99

        # Save the mask
        mask = mask.astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(mask_out_path)

        return mask


if __name__ == '__main__':
    Fire(make_mask)
