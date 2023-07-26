"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-05-30
Description: Command line interface to the glint-mask-tools.
"""
import os
import sys
from typing import List

import fire
from tqdm import tqdm

from .maskers import (
    Masker,
    MicasenseRedEdgeThresholdMasker,
    P4MSThresholdMasker,
    RGBThresholdMasker,
    CIRThresholdMasker,
)


class CLI(object):
    def __init__(self, max_workers: int = min(4, os.cpu_count())):
        """Command Line Interface Class for glint mask generators.

        Parameters
        ----------
        max_workers
            The maximum number of threads to use for processing.
        """
        self.max_workers = max_workers

    @staticmethod
    def _err_callback(path, exception):
        tqdm.write(f"{path} failed with err:\n{exception}", file=sys.stderr)

    def _process(self, masker: Masker):
        with tqdm(total=len(masker)) as progress:
            masker(
                max_workers=self.max_workers,
                callback=lambda _: progress.update(1),
                err_callback=self._err_callback,
            )

    def rgb_threshold(
        self,
        img_dir: str,
        out_dir: str,
        thresholds: List[float] = (1, 1, 0.875),
        pixel_buffer: int = 0,
    ) -> None:
        """Generate masks for glint regions in RGB imagery using Tom Bell's binning
            algorithm.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images. If img_dir
            is a directory, all tif, jpg, jpeg, and png images in that directory will
            be processed.
        out_dir
            The path to send your out image including the file name and type.
            e.g. "/path/to/mask.png". out_dir must be a directory if img_dir is
            specified as a directory.
        thresholds
            The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).
            Default is [1, 1, 0.875].
        pixel_buffer
            The pixel distance to buffer out the mask. Defaults to 0 (off).

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self._process(RGBThresholdMasker(img_dir, out_dir, thresholds, pixel_buffer))

    def cir_threshold(
        self,
        img_dir: str,
        out_dir: str,
        thresholds: List[float] = (1, 1, 0.875, 1),
        pixel_buffer: int = 0,
    ) -> None:
        """Generate masks for glint regions in 4 Band CIR imagery using Tom Bell's
            binning algorithm.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images. If img_dir
            is a directory, all tif, jpg, jpeg, and png images in that directory will
            be processed.
        out_dir
            The path to send your out image including the file name and type.
            e.g. "/path/to/mask.png". out_dir must be a directory if img_dir is
            specified as a directory.
        thresholds
            The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).
            Default is [1, 1, 0.875].
        pixel_buffer
            The pixel distance to buffer out the mask. Defaults to 0 (off).

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self._process(CIRThresholdMasker(img_dir, out_dir, thresholds, pixel_buffer))

    def p4ms_threshold(
        self,
        img_dir: str,
        out_dir: str,
        thresholds: List[float] = (0.875, 1, 1, 1, 1),
        pixel_buffer: int = 0,
    ) -> None:
        """Generate masks for glint regions in multispectral imagery from the DJI camera
            using Tom Bell's algorithm on the Blue image band.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images. If img_dir
            is a directory, all tif, jpg, jpeg, and png images in that directory will
            be processed.
        out_dir
            The path to send your out image including the file name and type.
            e.g. "/path/to/mask.png". out_dir must be a directory if img_dir is
            specified as a directory.
        thresholds
            The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).
            Default is [0.875, 1, 1, 1, 1].
        pixel_buffer
            The pixel distance to buffer out the mask. Defaults to 0 (off).

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self._process(P4MSThresholdMasker(img_dir, out_dir, thresholds, pixel_buffer))

    def micasense_threshold(
        self,
        img_dir: str,
        out_dir: str,
        thresholds: List[float] = (0.875, 1, 1, 1, 1),
        pixel_buffer: int = 0,
    ) -> None:
        """Generate masks for glint regions in multispectral imagery from the
            Micasense camera using Tom Bell's algorithm on the blue image band.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images.
            If img_dir is a directory, all tif, jpg, jpeg, and png images in that
            directory will be processed.
        out_dir
            The path to send your out image including the file name and type.
            e.g. "/path/to/mask.png". out_dir must be a directory if img_dir is
            specified as a directory.
        thresholds
            The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).
            Default is [0.875, 1, 1, 1, 1].
        pixel_buffer
            The pixel distance to buffer out the mask. Defaults to 0 (off).

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self._process(
            MicasenseRedEdgeThresholdMasker(img_dir, out_dir, thresholds, pixel_buffer)
        )


def main():
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
