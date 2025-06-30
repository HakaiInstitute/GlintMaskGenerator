"""Module containing the main Masker classes for different comninations of glint masking algorithms and sensors.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import TYPE_CHECKING, Callable

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

if TYPE_CHECKING:
    from collections.abc import Sequence


class Masker:
    """The main class for masking glint in imagery. It is composed of an image loader and glint masking algorithm."""

    def __init__(
        self,
        algorithm: GlintAlgorithm,
        image_loader: ImageLoader,
        pixel_buffer: int = 0,
    ) -> None:
        """Create the Masker object."""
        self.algorithm = algorithm
        self.image_loader = image_loader
        self.pixel_buffer = pixel_buffer
        self.buffer_kernel = make_circular_kernel(self.pixel_buffer)

    # noinspection PyMethodMayBeStatic
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Postprocess the generated boolean numpy mask. Can be overridden to customize behavior."""
        if self.pixel_buffer <= 0:
            return mask
        return (convolve(mask, self.buffer_kernel, mode="constant", cval=0) > 0).astype(
            int,
        )

    @staticmethod
    def to_metashape_mask(mask: np.ndarray) -> np.ndarray:
        """Scale the mask values to work with Agisoft Metashape expectations."""
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
        callback: Callable[[list[str]], None] = lambda _: None,
        err_callback: Callable[[list[str], Exception], None] = lambda _s, _e: None,
    ) -> None:
        """Run the masker processing.

        Parameters
        ----------
        max_workers
            The number of threads to use during processing. Useful for limiting memory
            consumption.
        callback
            Callback that receives the img_path as an arg after it is
            successfully processed.
        err_callback
            Callback that receives the img_path and an Exception as args after
            a processing failure.

        """
        if max_workers == 0:
            return self.process_unthreaded(callback, err_callback)
        return self.process(max_workers, callback, err_callback)

    # noinspection SpellCheckingInspection
    def process_unthreaded(
        self,
        callback: Callable[[list[str]], None] = lambda _: None,
        err_callback: Callable[[list[str], Exception], None] = lambda _s, _e: None,
    ) -> None:
        """Process all the images within the main process."""
        cur = None
        try:
            for paths in self.image_loader.paths:
                cur = paths
                self._process_one(paths)
                callback(paths)

        except Exception as exc:
            err_callback(cur, exc)
            return

    def process(
        self,
        max_workers: int = os.cpu_count(),
        callback: Callable[[list[str]], None] = lambda _: None,
        err_callback: Callable[[list[str], Exception], None] = lambda _s, _e: None,
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
            Callback function passed the name of each input and output mask
            files after processing it. Will receive img_path: str as arg.
        err_callback
            Callback function passed exception object on processing failure.
            Will receive img_path: str, and the Exception as args.

        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paths = {executor.submit(self._process_one, paths): paths for paths in self.image_loader.paths}
            for future in concurrent.futures.as_completed(future_to_paths):
                paths = future_to_paths[future]
                try:
                    future.result()
                    callback(paths)

                except Exception as exc:
                    err_callback(paths, exc)
                    executor.shutdown(wait=False)
                    return

    def _process_one(self, paths: list[str] | str) -> None:
        """Generate and saves a glint mask for the image located at img_path.

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
    """The main Masker class for threshold masking 3-band RGB imagery."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: Sequence[float] = (1, 1, 0.875),
        pixel_buffer: int = 0,
    ) -> None:
        """Create a new RGBThresholdMasker."""
        super().__init__(
            algorithm=ThresholdAlgorithm(thresholds),
            image_loader=RGBLoader(img_dir, mask_dir),
            pixel_buffer=pixel_buffer,
        )


class CIRThresholdMasker(Masker):
    """The main Masker class for threshold masking 4-band CIR imagery."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: Sequence[float] = (1, 1, 0.875, 1),
        pixel_buffer: int = 0,
    ) -> None:
        """Create a new CIRThresholdMasker."""
        super().__init__(
            algorithm=ThresholdAlgorithm(thresholds),
            image_loader=CIRLoader(img_dir, mask_dir),
            pixel_buffer=pixel_buffer,
        )


class P4MSThresholdMasker(Masker):
    """The main Masker class for threshold masking Phantom 4 MS imagery."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: Sequence[float] = (0.875, 1, 1, 1, 1),
        pixel_buffer: int = 0,
    ) -> None:
        """Create a new P4MSThresholdMasker."""
        super().__init__(
            algorithm=ThresholdAlgorithm(thresholds),
            image_loader=P4MSLoader(img_dir, mask_dir),
            pixel_buffer=pixel_buffer,
        )


class MicasenseRedEdgeThresholdMasker(Masker):
    """The main Masker class for threshold masking Micasense Red Edge imagery."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: Sequence[float] = (0.875, 1, 1, 1, 1),
        pixel_buffer: int = 0,
    ) -> None:
        """Create a new MicasenseRedEdgeThresholdMasker."""
        super().__init__(
            algorithm=ThresholdAlgorithm(thresholds),
            image_loader=MicasenseRedEdgeLoader(img_dir, mask_dir),
            pixel_buffer=pixel_buffer,
        )


if __name__ == "__main__":
    CIRThresholdMasker("ignore/in", "ignore/out", pixel_buffer=10).process_unthreaded()
