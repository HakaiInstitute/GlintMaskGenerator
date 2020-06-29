# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-12
# Description: Abstract Base Class for GlintMaskGenerators. Implements a number of functions that allows for
#   multi-threaded processing of images and a simple interface for which the GUI and CLIs can be easily extended.

import concurrent.futures
import itertools
import os
from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Callable, List, Any, Generator

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def case_insensitive_glob(dir_path: str, pattern: str) -> Generator:
    """Find files with a glob pattern while ignore upper/lower case."""
    return Path(dir_path).glob(
        ''.join('[%s%s]' % (char.lower(), char.upper()) if char.isalpha() else char for char in pattern))


class AbstractBaseMasker(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._files = self.get_files()

    def __call__(self, max_workers, callback, err_callback) -> Any:
        """Allows calling self.process by calling the object as a function."""
        return self.process(max_workers, callback, err_callback)

    def __len__(self) -> int:
        """Get and return the number of files to process."""
        return len(self.file_paths)

    @property
    def file_paths(self):
        """Property that returns the list of detected files for the masker."""
        return self._files

    def process(self, max_workers: Optional[int] = os.cpu_count() * 5, callback: Callable = None,
                err_callback: Callable = None) -> Any:

        """Compute the glint masks for all images in img_paths using the process_func and save to the mask_out_path.

        Args:
            max_workers: Optional[int]
                The maximum number of image processing workers. Useful for limiting memory usage.
                Defaults to the number of CPUs * 5.

            callback: Optional[Callable]
                Optional callback function passed the name of each input and output mask files after processing it.

            err_callback: Optional[Callable]
                Optional callback function passed exception object on processing failure.

        Returns:
            None
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self.process_one_file, path): path for path in self._files}
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    data = future.result()
                    if callback is not None:
                        callback(data)

                except Exception as exc:
                    if err_callback is not None:
                        err_callback(path, exc)
                    executor.shutdown(wait=False)
                    return

    @abstractmethod
    def get_files(self) -> List[str]:
        """Abstract file path getter method all subclasses must implement this method. Must return a list of files to
            process. This method is called on class instantiation.

        Returns:
            List[str]
                The list of files to be used for generating the masks.
        """
        raise NotImplementedError

    @abstractmethod
    def process_one_file(self, img_path: str) -> Any:
        """Abstract processor method. All subclasses must implement this method. Takes an image path and does the mask
            generation or saving logic. This method is called by the process function for each img_path in self._files.

        Returns:
            Any
                This function may return any type which will then be passed to the callback function in process.
        """
        raise NotImplementedError

    @staticmethod
    def list_img_files(img_dir: str) -> List[str]:
        """Utility function to get the paths of all images in directory img_dir.

        Args:
            img_dir: str
                The directory to search for images.

        Returns:
            List[str]
                List of all image files in the directory
        """
        extensions = ("png", "jpg", "jpeg", "tif", "tiff")
        img_dirs = itertools.chain.from_iterable(case_insensitive_glob(img_dir, f"*.{ext}") for ext in extensions)
        return [str(p) for p in list(img_dirs)]

    @staticmethod
    def save_mask(mask: np.ndarray, out_path: str) -> None:
        """Utility to save a mask to the location out_path.

        Args:
            mask : np.ndarray shape=(H,W)
                2D image mask to save into the image format specified in the out_path.

            out_path : str
                The path where the file should be saved, including img extension.
        """
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(str(out_path))

    @staticmethod
    def read_img(img_path: str):
        """Convenience method to turn img paths into numpy arrays of type float."""
        img = Image.open(img_path)
        return np.array(img).astype(np.float)

    @abstractmethod
    def get_out_paths(self, in_path: str) -> List[str]:
        """Get the out path for where to save the mask corresponding to image at in_path.

        Args:
            in_path: str
                The image path for which a mask is generated. Used to generate an appropriate out path for the mask.

        Returns:
            List[str]
                A list of paths where the mask for the image at location in_path should be saved.
        """
        raise NotImplementedError