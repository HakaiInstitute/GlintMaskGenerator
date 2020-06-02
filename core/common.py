# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-01
# Description: Common functions used by all glint mask algorithms

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image
from tqdm import tqdm


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
            micasense_paths = Path(img_path).glob(f"IMG_*[0-9]_4.tif")
            dji_paths = Path(img_path).glob(f"DJI_*[0-9]4.TIF")
            # Add more here if we get new cameras or anything changes
            img_paths = itertools.chain(micasense_paths, dji_paths)
        else:
            extensions = ("png", "PNG", "jpg", "JPG", "jpeg", "JPEG", "tif", "TIF")
            img_paths = itertools.chain.from_iterable((Path(img_path).glob(f"*.{ext}") for ext in extensions))
    else:
        raise ValueError("Check that img_path is a valid file or directory location.")

    return [str(p) for p in list(img_paths)]


def process_imgs(process_func: Callable, img_paths: Iterable[str], mask_out_path: str, processes: Optional[int] = 1,
                 callback: Optional[Callable] = None):
    """Compute the glint masks for all images in img_paths using the process_func and save to the mask_out_path.

    Args:
        process_func: Callable
            The function to process individual imgs at of the img_paths.

        img_paths: Iterable(str)
            An iterable of str image paths to process.

        mask_out_path: str
            The directory where the produced glint masks should be saved.

        processes: Optional[int]
            The number of processes to use to process images in parallel. Default to 1.

        callback: Optional[Callable]
            Optional callback function passed the name of each input and output mask files after processing it.

    Returns:

    """
    # progress = tqdm(total=len(img_paths))

    with ProcessPoolExecutor(max_workers=processes) as executor:
        for path in img_paths:
            out_path = Path(mask_out_path).joinpath(f"{Path(path).stem}_mask.png")

            f = executor.submit(process_func, path)
            # f.add_done_callback(save_mask)
            
            # f.add_done_callback(lambda f: progress.update())
            if callback is not None:
                f.add_done_callback(callback)

            # if callback is not None:
            #     callback(path, out_path)

    # progress.close()


def save_mask(out_path, mask):
    mask_img = Image.fromarray(mask, mode='L')
    mask_img.save(str(out_path))