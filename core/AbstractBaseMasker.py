"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-06-12
Description: Abstract Base Class for GlintMaskGenerators. Implements a number of functions that allows for
  multi-threaded processing of images and a simple interface for which the GUI and CLIs can be easily hook into.
"""

import concurrent.futures
import os
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Iterator, Optional

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class AbstractBaseMasker(ABC):
    """ Abstract class for image mask generators."""

    def __init__(self, img_dir, out_dir):
        super().__init__()

        self.img_dir = img_dir
        self.out_dir = out_dir

    def __call__(self, max_workers, callback, err_callback) -> None:
        """ Allows calling self.process by calling the object as a function."""
        return self.process(max_workers, callback, err_callback)

    def __len__(self) -> int:
        """ Get and return the number of files to process."""
        return sum(1 for _ in self.img_paths)

    @property
    def img_paths(self) -> Iterator[str]:
        """ Property that lists the file paths to process in self.img_dir. Returns an iterator that lists the files to
            process. Subclasses may want to override this to return specific image types or filter the results.
            By default, will list all images in self.img_dir if the file extension is in the extensions list.

        Returns:
            List[str]
                The list of files to be used for generating the masks.
        """
        extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

        paths = Path(self.img_dir).glob("**/*")
        paths = filter(lambda p: p.is_file() and p.suffix.lower() in extensions, paths)

        return map(lambda p: str(p), paths)

    # noinspection PyMethodMayBeStatic
    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """ Optional pre-processor hook. Subclasses may implement this method to do preprocessing step on image before
            passing it to the actual masking function.
            Common use-cases are selecting appropriate bands and value normalization.

        Args:
            img: np.ndarray
                The image data as a numpy array with dtype=float32.

        Returns:
            np.ndarray
                This function must return the converted image which is next passed to the the masking function.
        """
        return img

    # noinspection PyMethodMayBeStatic
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """ Optional post-processor hook for masks. Subclasses may implement this method to do preprocessing step on
            the produced mask before it saved to the mask out paths.

        Args:
            mask: np.ndarray
                The mask created by the glint masking algorithm.

        Returns:
            np.ndarray:
                This function must return a mask which is then saved to the mask out paths.
        """
        return mask

    @abstractmethod
    def mask_img(self, img: np.ndarray) -> np.ndarray:
        """ Abstract mask img method. All subclasses must implement this method. Takes a numpy ndarray that has been
            normalized and generates the glint mask.

        Args:
            img: Any
                The preprocessed image in numpy array format.

        Returns:
            np.ndarray
                The calculated glint mask.
        """
        raise NotImplemented

    @abstractmethod
    def get_mask_save_paths(self, in_path: str) -> List[str]:
        """ Get the out path for where to save the mask corresponding to image at in_path.

        Args:
            in_path: str
                The image path for which a mask is generated. Used to generate an appropriate out path for the mask.

        Returns:
            List[str]
                A list of paths where the mask for the image at location in_path should be saved.
        """
        raise NotImplemented

    def process(self, max_workers: int = os.cpu_count() * 5,
                callback: Optional[Callable[[str], None]] = None,
                err_callback: Optional[Callable[[str, Exception], None]] = None
                ) -> None:
        """ Compute the glint masks for all images in img_paths using the process_func and save to the mask_out_paths.

        Args:
            max_workers: int
                The maximum number of image processing workers. Useful for limiting memory usage.
                Defaults to the number of CPUs * 5.

            callback: Optional[Callable[[str], None]]
                Optional callback function passed the name of each input and output mask files after processing it.
                Will receive img_path: str as arg.

            err_callback: Optional[Callable[[str, Exception], None]]
                Optional callback function passed exception object on processing failure.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self.process_one_file, path): path for path in self.img_paths}
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    _ = future.result()
                    if callback is not None:
                        callback(path)

                except Exception as exc:
                    if err_callback is not None:
                        err_callback(path, exc)
                    executor.shutdown(wait=False)
                    return

    def process_one_file(self, img_path: str) -> None:
        """ Generates and saves a glint mask for the image located at img_path and saves it to all paths returned by
            self.get_mask_save_paths(img_path).

        Args:
            img_path: str
                The path to the image to generate a glint mask for.
        """
        img = self.read_img(img_path)
        img = self.preprocess_img(img)

        mask = self.mask_img(img)
        mask = self.postprocess_mask(mask)

        for path in self.get_mask_save_paths(img_path):
            self.save_mask(mask, path)

    @staticmethod
    def save_mask(mask: np.ndarray, out_path: str) -> None:
        """ Utility to save a mask to the location out_path.

        Args:
            mask : np.ndarray shape=(H,W)
                2D image mask to save into the image format specified in the out_path.

            out_path : str
                The path where the file should be saved, including img extension.
        """
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(str(out_path))

    @staticmethod
    def read_img(img_path: str) -> np.ndarray:
        """ Convenience method to turn img paths into numpy arrays of type float.

        Args:
            img_path: str
                The path to the image to open.

        Returns:
            np.ndarray: The image data in a np.ndarray with dtype=float32.
        """
        img = Image.open(img_path)
        return np.array(img).astype(np.float)
