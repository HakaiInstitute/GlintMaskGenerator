# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-30
# Description: Command line interface to the glint-mask-tools.

# Based on Matlab script by Tom Bell written 6/28/2019
#
# Description: Generate masks for glint regions in in RGB images using Tom Bell's blue-channel binning algorithm.
from functools import partial

import fire
from PIL import Image
from tqdm import tqdm

from core.common import get_img_paths, process_imgs
from core.glint_mask import make_and_save_single_mask as tom_make_and_save_single_mask
from core.specular_mask import make_and_save_single_mask as specular_make_and_save_single_mask

Image.MAX_IMAGE_PIXELS = None
EPSILON = 1e-8


def _tom(img_path: str, red_edge_files, mask_out_path: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 20,
         num_bins: int = 8) -> None:
    img_paths = get_img_paths(img_path, mask_out_path, red_edge=red_edge_files)
    pbar = tqdm(total=len(img_paths))
    f = partial(tom_make_and_save_single_mask, mask_out_path=mask_out_path, red_edge=red_edge_files,
                glint_threshold=glint_threshold,
                mask_buffer_sigma=mask_buffer_sigma, num_bins=num_bins)
    process_imgs(f, img_paths, callback=lambda _: pbar.update())
    pbar.close()


def red_edge(img_path: str, mask_out_path: str, glint_threshold: float = 0.9,
             mask_buffer_sigma: int = 20, num_bins: int = 8) -> None:
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

    Returns
    -------
    None
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    _tom(img_path, red_edge_files=True, mask_out_path=mask_out_path, glint_threshold=glint_threshold,
         mask_buffer_sigma=mask_buffer_sigma, num_bins=num_bins)


def rgb(img_path: str, mask_out_path: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 20,
        num_bins: int = 8) -> None:
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

    Returns
    -------
    None
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    _tom(img_path, red_edge_files=False, mask_out_path=mask_out_path, glint_threshold=glint_threshold,
         mask_buffer_sigma=mask_buffer_sigma, num_bins=num_bins)


def specular(img_path: str, mask_out_path: str, percent_diffuse: float = 0.1, mask_thresh: float = 0.8,
             opening: int = 5, closing: int = 5) -> None:
    """Generate masks for glint regions in RGB imagery using Tom Bell's binning algorithm.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images. If img_path is a directory, all tif, jpg, jpeg,
        and png images in that directory will be processed.

    mask_out_path: str
        The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
        mask_out_path must be a directory if img_path is specified as a directory.

    percent_diffuse: Optional[float]
        An estimate of the percentage of pixels in an image that show pure diffuse reflectance, and
        thus no specular reflectance (glint). Defaults to 0.1. Try playing with values, low ones typically work well.

    mask_thresh: Optional[float]
        The threshold on the specular reflectance estimate image to convert into a mask.
        E.g. if more than 50% specular reflectance is unacceptable, use 0.5. Default is 0.4.

    opening: Optional[int]
        The number of morphological opening iterations on the produced mask.
        Useful for closing small holes in the mask. Set to 0 by default (i.e. it's shut off).

    closing: Optional[int]
        The number of morphological closing iterations on the produced mask.
        Useful for removing small bits of mask. Set to 0 by default (i.e. it's shut off).

    Returns
    -------
    None
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    img_paths = get_img_paths(img_path, mask_out_path, red_edge=False)
    progress = tqdm(total=len(img_paths))
    f = partial(specular_make_and_save_single_mask, mask_out_path=mask_out_path, percent_diffuse=percent_diffuse,
                mask_thresh=mask_thresh, opening=opening, closing=closing)
    process_imgs(f, img_paths, callback=lambda _: progress.update())
    progress.close()


if __name__ == '__main__':
    fire.Fire({
        'rgb': rgb,
        'specular': specular,
        'red_edge': red_edge
    })
