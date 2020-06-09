# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-30
# Description: Command line interface to the glint-mask-tools.

# Based on Matlab script by Tom Bell written 6/28/2019
#
# Description: Generate masks for glint regions in in RGB images using Tom Bell's blue-channel binning algorithm.
import sys
from functools import partial
from typing import Callable, Iterable, Optional

import fire
from PIL import Image
from tqdm import tqdm

from core.common import get_img_paths, process_imgs
from core.glint_mask import make_and_save_single_mask as tom_make_and_save_single_mask
from core.specular_mask import make_and_save_single_mask as specular_make_and_save_single_mask

Image.MAX_IMAGE_PIXELS = None
EPSILON = 1e-8


def _err_callback(path, exception):
    tqdm.write(f"{path} failed with err:\n{exception}", file=sys.stderr)


def _process(img_paths: Iterable[str], mask_func: Callable, max_workers: Optional[int] = None) -> None:
    with tqdm(total=len(list(img_paths))) as progress:
        return process_imgs(mask_func, img_paths, max_workers=max_workers,
                            callback=lambda _: progress.update(), err_callback=partial(_err_callback))


def _helper_rgb_ms(img_path: str, mask_out_path: str, img_type: str, glint_threshold: float,
                   mask_buffer_sigma: int, num_bins: int, max_workers: int) -> None:
    img_paths = get_img_paths(img_path, mask_out_path, img_type=img_type)
    mask_func = partial(tom_make_and_save_single_mask, mask_out_path=mask_out_path, img_type=img_type,
                        glint_threshold=glint_threshold, mask_buffer_sigma=mask_buffer_sigma, num_bins=num_bins)
    return _process(img_paths, mask_func, max_workers=max_workers)


def dji_ms(img_path: str, mask_out_path: str, glint_threshold: float = 0.9,
           mask_buffer_sigma: int = 20, num_bins: int = 8, max_workers: Optional[int] = None) -> None:
    """Generate masks for glint regions in multispectral imagery from the DJI camera using Tom Bell's algorithm.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images. If img_path is a directory, all DJI_***4.TIF
        files will be processed.

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

    max_workers: Optional[int]
        The maximum number of image processing workers. Useful for limiting memory usage.
        Defaults to the number of CPUs * 5.

    Returns
    -------
    None
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    return _helper_rgb_ms(img_path, mask_out_path, 'dji_ms',
                          glint_threshold, mask_buffer_sigma, num_bins, max_workers)


def micasense_ms(img_path: str, mask_out_path: str, glint_threshold: float = 0.9,
                 mask_buffer_sigma: int = 20, num_bins: int = 8, max_workers: Optional[int] = None) -> None:
    """Generate masks for glint regions in multispectral imagery from the Micasense camera using Tom Bell's algorithm.

    Parameters
    ----------
    img_path: str
        The path to a named input image or directory containing images. If img_path is a directory, all IMG_xxxx_6.tif
        files will be processed.

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

    max_workers: Optional[int]
        The maximum number of image processing workers. Useful for limiting memory usage.
        Defaults to the number of CPUs * 5.

    Returns
    -------
    None
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    return _helper_rgb_ms(img_path, mask_out_path, 'micasense_ms',
                          glint_threshold, mask_buffer_sigma, num_bins, max_workers)


def rgb(img_path: str, mask_out_path: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 20,
        num_bins: int = 8, max_workers: Optional[int] = None) -> None:
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

    max_workers: Optional[int]
        The maximum number of image processing workers. Useful for limiting memory usage.
        Defaults to the number of CPUs * 5.

    Returns
    -------
    None
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    return _helper_rgb_ms(img_path, mask_out_path, 'rgb',
                          glint_threshold, mask_buffer_sigma, num_bins, max_workers)


def specular(img_path: str, mask_out_path: str, percent_diffuse: float = 0.1, mask_thresh: float = 0.8,
             opening: int = 5, closing: int = 5, max_workers: Optional[int] = None) -> None:
    """Generate masks for glint regions in RGB imagery by setting a threshold on estimated specular reflectance.

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

    max_workers: Optional[int]
        The maximum number of image processing workers. Useful for limiting memory usage.
        Defaults to the number of CPUs * 5.

    Returns
    -------
    None
        Side effects are that the mask is saved to the specified mask_out_path location.
    """
    img_paths = get_img_paths(img_path, mask_out_path, img_type='rgb')
    mask_func = partial(specular_make_and_save_single_mask, mask_out_path=mask_out_path,
                        percent_diffuse=percent_diffuse, mask_thresh=mask_thresh, opening=opening, closing=closing)
    return _process(img_paths, mask_func, max_workers=max_workers)


if __name__ == '__main__':
    fire.Fire({
        'rgb': rgb,
        'specular': specular,
        'micasense_ms': micasense_ms,
        'dji_ms': dji_ms
    })
