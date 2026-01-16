"""Module containing the main Masker classes for different comninations of glint masking algorithms and _known_sensors.

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

from .utils import make_circular_kernel

if TYPE_CHECKING:
    from .band_alignment import BandAligner
    from .glint_algorithms import GlintAlgorithm
    from .image_loaders import ImageLoader


class Masker:
    """The main class for masking glint in imagery. It is composed of an image loader and glint masking algorithm."""

    def __init__(  # noqa: PLR0913
        self,
        algorithm: GlintAlgorithm,
        image_loader: ImageLoader,
        image_preprocessor: Callable[[np.ndarray], np.ndarray],
        pixel_buffer: int = 0,
        *,
        per_band: bool = False,
        band_aligner: BandAligner | None = None,
    ) -> None:
        """Create the Masker object."""
        self.algorithm = algorithm
        self.image_loader = image_loader
        self.image_preprocessor = image_preprocessor
        self.pixel_buffer = pixel_buffer
        self.per_band = per_band
        self.band_aligner = band_aligner
        self.buffer_kernel = make_circular_kernel(self.pixel_buffer)

    # noinspection PyMethodMayBeStatic
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Postprocess the generated boolean numpy mask. Can be overridden to customize behavior."""
        if self.pixel_buffer <= 0:
            return mask
        if mask.ndim == 2:  # noqa: PLR2004
            # Single combined mask
            return (convolve(mask, self.buffer_kernel, mode="constant", cval=0) > 0).astype(int)
        # Per-band masks: apply buffer to each channel
        result = np.empty_like(mask, dtype=int)
        for i in range(mask.shape[2]):
            result[:, :, i] = (convolve(mask[:, :, i], self.buffer_kernel, mode="constant", cval=0) > 0).astype(int)
        return result

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
        img = self.image_loader.load_image(paths)

        offsets = None
        if self.band_aligner is not None:
            img, offsets = self.band_aligner.align(img)

        img = self.image_preprocessor(img)
        mask = self.algorithm(img)
        mask = self.postprocess_mask(mask)

        # Shift masks back to original unaligned coordinates for each band
        # This works for both union masks (2D) and per-band masks (3D)
        if self.band_aligner is not None:
            mask = self.band_aligner.unalign_mask(mask, offsets)

        mask = self.to_metashape_mask(mask)

        # Save per-band if we have a 3D mask (from alignment) or per_band mode
        save_per_band = mask.ndim == 3 or self.per_band  # noqa: PLR2004
        self.image_loader.save_masks(mask, paths, per_band=save_per_band)
