"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
Description: 
"""
import concurrent.futures
import os
from functools import cached_property
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from .glint_algorithms import GlintAlgorithm, IntensityRatioAlgorithm, ThresholdAlgorithm
from .image_loaders import ImageLoader, MicasenseRedEdgeLoader, P4MSLoader, RGB8BitLoader


class Masker(object):
    def __init__(self, algorithm: GlintAlgorithm, image_loader: ImageLoader):
        self.algorithm = algorithm
        self.image_loader = image_loader

    @staticmethod
    def save_mask(mask: np.ndarray, out_path: str):
        """Utility function to save a mask to the location out_path.

                Parameters
                ----------
                mask
                    2D image mask to save into the image format specified in the out_path.
                out_path
                    The path where the file should be saved, including img extension.
                """
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(str(out_path))

    # noinspection PyMethodMayBeStatic
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Method which can be overridden to do any postprocessing on the generated boolean numpy mask."""
        return mask

    @staticmethod
    def to_metashape_mask(mask: np.ndarray):
        return np.logical_not(mask).astype(np.uint8) * 255

    def __len__(self) -> int:
        """Get and return the number of files to process.

        Returns
        -------
        int
            The number of files that need to be processed.
        """
        return len(self.image_loader)

    def __call__(self, max_workers: int,
                 callback: Optional[Callable[[str], None]] = None,
                 err_callback: Optional[Callable[[str, Exception], None]] = None
                 ) -> None:
        """Allows calling self.process by calling the object as a function.

        Parameters
        ----------
        max_workers
            The number of threads to use during processing. Useful for limiting memory consumption.
        callback
            Optional callback that receives the img_path as an arg after it is successfully processed.
        err_callback
            Optional callback that receives the img_path and an Exception as args after a processing failure.
        """
        if max_workers == 0:
            return self.process_unthreaded(callback, err_callback)
        else:
            return self.process(max_workers, callback, err_callback)

    # noinspection SpellCheckingInspection
    def process_unthreaded(self, callback: Optional[Callable[[List[str]], None]] = None,
                           err_callback: Optional[Callable[[List[str], Exception], None]] = None) -> None:
        for img, paths in zip(self.image_loader.images, self.image_loader.paths):
            try:
                self._process_one(img, paths)
                if callback is not None:
                    callback(paths)

            except Exception as exc:
                if err_callback is not None:
                    err_callback(paths, exc)
                return

    def process(self, max_workers: int = os.cpu_count() * 5, callback: Optional[Callable[[List[str]], None]] = None,
                err_callback: Optional[Callable[[List[str], Exception], None]] = None) -> None:
        """Compute all glint masks.

        Computes masks for all images in self.img_paths using the process_func and save to the mask_out_paths.

        Parameters
        ----------
        max_workers
            The maximum number of image processing workers.
            Useful for limiting memory usage.
        callback
            Optional callback function passed the name of each input and output mask files after processing it.
            Will receive img_path: str as arg.
        err_callback
            Optional callback function passed exception object on processing failure.
            Will receive img_path: str, and the Exception as args.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paths = {
                executor.submit(self._process_one, img, paths): paths
                for img, paths in zip(self.image_loader.images, self.image_loader.paths)
            }
            for future in concurrent.futures.as_completed(future_to_paths):
                paths = future_to_paths[future]
                try:
                    future.result()
                    if callback is not None:
                        callback(paths)

                except Exception as exc:
                    if err_callback is not None:
                        err_callback(paths, exc)
                    executor.shutdown(wait=False)
                    return

    def _process_one(self, img: np.ndarray, paths: Union[List[str], str]) -> None:
        """Generates and saves a glint mask for the image located at img_path.

        Saves the generated mask to all path locations returned by self.get_mask_save_paths(img_path).

        Parameters
        ----------
        img
            Numpy array representing the image
        paths
            The file paths used to create the image. Can be single file path or list of path to multiple files
        """
        mask = self.algorithm(img)
        mask = self.postprocess_mask(mask)
        mask = self.to_metashape_mask(mask)

        for path in self.image_loader.get_mask_save_paths(paths):
            self.save_mask(mask, path)


class PixelBufferMixin:
    def __init__(self, *args, pixel_buffer: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixel_buffer = pixel_buffer

    @cached_property
    def kernel(self) -> np.ndarray:
        """Create circular kernel"""
        y, x = np.ogrid[-self.pixel_buffer:self.pixel_buffer + 1, -self.pixel_buffer:self.pixel_buffer + 1]
        dist_m: np.ndarray = x ** 2 + y ** 2
        return dist_m <= self.pixel_buffer ** 2

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.pixel_buffer <= 0:
            return mask
        return (convolve(mask, self.kernel, mode='constant', cval=0) > 0).astype(np.int)


class RGBThresholdMasker(PixelBufferMixin, Masker):
    def __init__(self, img_dir: str, mask_dir: str,
                 thresholds: Sequence[float] = (1, 1, 0.875), pixel_buffer: int = 0):
        super().__init__(algorithm=ThresholdAlgorithm(thresholds),
                         image_loader=RGB8BitLoader(img_dir, mask_dir),
                         pixel_buffer=pixel_buffer)


class P4MSThresholdMasker(PixelBufferMixin, Masker):
    def __init__(self, img_dir: str, mask_dir: str,
                 thresholds: Sequence[float] = (0.875, 1, 1, 1, 1), pixel_buffer: int = 0):
        super().__init__(algorithm=ThresholdAlgorithm(thresholds),
                         image_loader=P4MSLoader(img_dir, mask_dir),
                         pixel_buffer=pixel_buffer)


class MicasenseRedEdgeThresholdMasker(PixelBufferMixin, Masker):
    def __init__(self, img_dir: str, mask_dir: str,
                 thresholds: Sequence[float] = (0.875, 1, 1, 1, 1), pixel_buffer: int = 0):
        super().__init__(algorithm=ThresholdAlgorithm(thresholds),
                         image_loader=MicasenseRedEdgeLoader(img_dir, mask_dir),
                         pixel_buffer=pixel_buffer)


class RGBIntensityRatioMasker(PixelBufferMixin, Masker):
    def __init__(self, img_dir: str, mask_dir: str,
                 percent_diffuse: float = 0.95, threshold: float = 0.99, pixel_buffer: int = 0):
        super().__init__(algorithm=IntensityRatioAlgorithm(percent_diffuse, threshold),
                         image_loader=RGB8BitLoader(img_dir, mask_dir),
                         pixel_buffer=pixel_buffer)


class P4MSIntensityRatioMasker(PixelBufferMixin, Masker):
    def __init__(self, img_dir: str, mask_dir: str,
                 percent_diffuse: float = 0.95, threshold: float = 0.99, pixel_buffer: int = 0):
        super().__init__(algorithm=IntensityRatioAlgorithm(percent_diffuse, threshold),
                         image_loader=P4MSLoader(img_dir, mask_dir),
                         pixel_buffer=pixel_buffer)


class MicasenseRedEdgeIntensityRatioMasker(PixelBufferMixin, Masker):
    def __init__(self, img_dir: str, mask_dir: str,
                 percent_diffuse: float = 0.95, threshold: float = 0.99, pixel_buffer: int = 0):
        super().__init__(algorithm=IntensityRatioAlgorithm(percent_diffuse, threshold),
                         image_loader=MicasenseRedEdgeLoader(img_dir, mask_dir),
                         pixel_buffer=pixel_buffer)
