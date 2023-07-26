"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""
import concurrent.futures
import os
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from scipy.ndimage import convolve

from .glint_algorithms import GlintAlgorithm, ThresholdAlgorithm
from .image_loaders import (
    CIRLoader,
    ImageLoader,
    MicasenseRedEdgeLoader,
    P4MSLoader,
    RGBLoader,
)
from .utils import make_circular_kernel


class Masker(object):
    def __init__(
        self,
        algorithm: GlintAlgorithm,
        image_loader: ImageLoader,
        pixel_buffer: int = 0,
    ):
        self.algorithm = algorithm
        self.image_loader = image_loader
        self.pixel_buffer = pixel_buffer
        self.buffer_kernel = make_circular_kernel(self.pixel_buffer)

    # noinspection PyMethodMayBeStatic
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Hook to do any postprocessing on the generated boolean numpy mask."""
        if self.pixel_buffer <= 0:
            return mask
        return (convolve(mask, self.buffer_kernel, mode="constant", cval=0) > 0).astype(
            int
        )

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

    def __call__(
        self,
        max_workers: int,
        callback: Optional[Callable[[str], None]] = None,
        err_callback: Optional[Callable[[str, Exception], None]] = None,
    ) -> None:
        """Calling Masker.process by calling the instance as a function.

        Parameters
        ----------
        max_workers
            The number of threads to use during processing. Useful for limiting memory
            consumption.
        callback
            Optional callback that receives the img_path as an arg after it is
            successfully processed.
        err_callback
            Optional callback that receives the img_path and an Exception as args after
            a processing failure.
        """
        if max_workers == 0:
            return self.process_unthreaded(callback, err_callback)
        else:
            return self.process(max_workers, callback, err_callback)

    # noinspection SpellCheckingInspection
    def process_unthreaded(
        self,
        callback: Optional[Callable[[List[str]], None]] = None,
        err_callback: Optional[Callable[[List[str], Exception], None]] = None,
    ) -> None:
        for paths in self.image_loader.paths:
            try:
                self._process_one(paths)
                if callback is not None:
                    callback(paths)

            except Exception as exc:
                if err_callback is not None:
                    err_callback(paths, exc)
                return

    def process(
        self,
        max_workers: int = os.cpu_count(),
        callback: Optional[Callable[[List[str]], None]] = None,
        err_callback: Optional[Callable[[List[str], Exception], None]] = None,
    ) -> None:
        """Compute all glint masks.

        Computes masks for all images in self.img_paths using the process_func and save
        to the mask_out_paths.

        Parameters
        ----------
        max_workers
            The maximum number of image processing workers.
            Useful for limiting memory usage.
        callback
            Optional callback function passed the name of each input and output mask
            files after processing it. Will receive img_path: str as arg.
        err_callback
            Optional callback function passed exception object on processing failure.
            Will receive img_path: str, and the Exception as args.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paths = {
                executor.submit(self._process_one, paths): paths
                for paths in self.image_loader.paths
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

    def _process_one(self, paths: Union[List[str], str]) -> None:
        """Generates and saves a glint mask for the image located at img_path.

        Saves the generated mask to all path locations returned by
        self.get_mask_save_paths(img_path).

        Parameters
        ----------
        paths
            The file paths used to create the image. Can be single file path or list of
            path to multiple files
        """
        self.image_loader.apply_masker(paths, self)


class RGBThresholdMasker(Masker):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: Sequence[float] = (1, 1, 0.875),
        pixel_buffer: int = 0,
    ):
        super().__init__(
            algorithm=ThresholdAlgorithm(thresholds),
            image_loader=RGBLoader(img_dir, mask_dir),
            pixel_buffer=pixel_buffer,
        )


class CIRThresholdMasker(Masker):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: Sequence[float] = (1, 1, 0.875, 1),
        pixel_buffer: int = 0,
    ):
        super().__init__(
            algorithm=ThresholdAlgorithm(thresholds),
            image_loader=CIRLoader(img_dir, mask_dir),
            pixel_buffer=pixel_buffer,
        )


class P4MSThresholdMasker(Masker):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: Sequence[float] = (0.875, 1, 1, 1, 1),
        pixel_buffer: int = 0,
    ):
        super().__init__(
            algorithm=ThresholdAlgorithm(thresholds),
            image_loader=P4MSLoader(img_dir, mask_dir),
            pixel_buffer=pixel_buffer,
        )


class MicasenseRedEdgeThresholdMasker(Masker):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: Sequence[float] = (0.875, 1, 1, 1, 1),
        pixel_buffer: int = 0,
    ):
        super().__init__(
            algorithm=ThresholdAlgorithm(thresholds),
            image_loader=MicasenseRedEdgeLoader(img_dir, mask_dir),
            pixel_buffer=pixel_buffer,
        )


if __name__ == "__main__":
    CIRThresholdMasker("ignore/in", "ignore/out", pixel_buffer=10).process_unthreaded()
