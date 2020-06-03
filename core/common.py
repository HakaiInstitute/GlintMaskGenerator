# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-01
# Description: Common functions used by all glint mask algorithms

import concurrent.futures
import itertools
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image


def get_img_paths(img_path: str, mask_out_path: str, red_edge: Optional[bool] = False):
    """Get the paths of images to process from location img_path.

    Args:
        img_path: str
            The filename or directory of images to process.

        mask_out_path: str
            The directory where the output masks should be saved.

        red_edge: Optional[bool]
            Flag indicating if the images at the specified img_path are Micasense or DJI images containing red edge band
            files.

    Returns:
        None
    """
    if not Path(mask_out_path).is_dir():
        raise ValueError("Specify a valid directory for mask_out_path.")

    if Path(img_path).is_file():
        img_paths = [Path(img_path)]

    elif Path(img_path).is_dir():
        # Get all images in the specified directory
        if red_edge:
            micasense_paths = Path(img_path).glob("IMG_*[0-9]_5.tif")
            dji_paths = Path(img_path).glob("DJI_*[0-9]4.TIF")
            # Add more here if we get new cameras or anything changes
            img_paths = itertools.chain(micasense_paths, dji_paths)
        else:
            extensions = ("png", "jpg", "jpeg", "tif", "tiff")
            if not os.name == 'nt':
                extensions = extensions + tuple(e.upper() for e in extensions)
            img_paths = itertools.chain.from_iterable((Path(img_path).glob(f"*.{ext}") for ext in extensions))
    else:
        raise ValueError("Check that img_path is a valid file or directory location.")

    return [str(p) for p in list(img_paths)]


def process_imgs(process_func: Callable, img_paths: Iterable[str],
                 callback: Optional[Callable] = None, err_callback: Optional[Callable] = None):
    """Compute the glint masks for all images in img_paths using the process_func and save to the mask_out_path.

    Args:
        process_func: Callable
            The function to process individual imgs at of the img_paths.

        img_paths: Iterable(str)
            An iterable of str image paths to process.

        processes: Optional[int]
            The number of processes to use to process images in parallel. Default to 1.

        callback: Optional[Callable]
            Optional callback function passed the name of each input and output mask files after processing it.

        err_callback: Optional[Callable]
            Optional callback function passed exception object on processing failure.

    Returns:

    """
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(process_func, path): path for path in img_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                data = future.result()
                if callback is not None:
                    callback(data)

            except Exception as exc:
                if err_callback is not None:
                    err_callback(path, exc)

                for f in future_to_path:
                    f.cancel()
                return


def save_mask(out_path, mask):
    """Save the image mask to location out_path.

    Args:
        out_path : str
            The path where the file should be saved, including img extension.

        mask : np.ndarray shape=(H,W)
            2D image mask to save into the image format specified in the out_path.

    Returns:
        None
    """
    mask_img = Image.fromarray(mask, mode='L')
    mask_img.save(str(out_path))
    return out_path
