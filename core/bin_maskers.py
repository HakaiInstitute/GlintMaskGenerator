"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-06-12
Description: Classes for processing images using Tom's bin-based glint masking technique for various types of image
    files
"""

import re
from abc import ABCMeta
from pathlib import Path
from typing import List, Union, Iterator

import numpy as np

from core.AbstractBaseMasker import AbstractBaseMasker
from core.glint_mask_algorithms.glint_mask import make_single_mask
from core.utils import normalize_img


class AbstractBinMasker(AbstractBaseMasker, metaclass=ABCMeta):
    """ Abstract class for all maskers that use Tom Bell's binning algorithm.
        Defines behaviour common to all Bin maskers."""

    def __init__(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 0,
                 num_bins: int = 8) -> None:
        """ Create and return a glint mask for RGB imagery.

        Args:
            img_dir: str
                The path to a directory containing images to process.

            out_dir: str
                Path to the directory where the image masks should be saved.

            glint_threshold: float
                The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
                Play with this value. Default is 0.9.

            mask_buffer_sigma: int
                The sigma for the Gaussian kernel used to buffer the mask. Defaults to 0.

            num_bins: int
                The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.
        """
        super().__init__(img_dir, out_dir)

        self.glint_threshold = glint_threshold
        self.mask_buffer_sigma = mask_buffer_sigma
        self.num_bins = num_bins

    def mask_img(self, img: np.ndarray) -> np.ndarray:
        """ Generates and saves a glint mask for the image at path img_path using Tom's method."""
        return make_single_mask(img, self.glint_threshold, self.mask_buffer_sigma, self.num_bins)


class BlueBinMasker(AbstractBinMasker):
    """ Tom Bell's method masker for RGB imagery."""

    def __init__(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 20,
                 num_bins: int = 8) -> None:
        """ Create and return a glint mask for RGB imagery.

        Args:
            img_dir: str
                The path to a directory containing images to process.

            out_dir: str
                Path to the directory where the image masks should be saved.

            glint_threshold: float
                The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
                Play with this value. Default is 0.9.

            mask_buffer_sigma: int
                The sigma for the Gaussian kernel used to buffer the mask. Defaults to 20.

            num_bins: int
                The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.
        """
        # This method is only different from the base class in terms of mask_buffer_sigma default value
        super().__init__(img_dir, out_dir, glint_threshold, mask_buffer_sigma, num_bins)

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """ Select the blue channel and normalize to [0, 1]."""
        return normalize_img(img[:, :, 2], bit_depth=8)

    def get_mask_save_paths(self, in_path: str) -> List[str]:
        """ Implement get_out_paths as required by AbstractBinMasker."""
        return [str(Path(self.out_dir).joinpath(f"{Path(in_path).stem}_mask.png"))]


class DJIMultispectralMasker(AbstractBinMasker):
    """ Tom Bell method masker for DJI multi-spectral imagery."""

    @property
    def img_paths(self) -> Iterator[str]:
        """ Generates a list of files which should be used as input to generate glint masks. For DJI Multispectral
            masking, this is a list of paths that correspond to the RedEdge band of the files output from a DJI
            multispectral camera."""
        return filter(self.is_dji_red_edge, super().img_paths)

    @staticmethod
    def is_dji_red_edge(filename: str) -> bool:
        """ Determine if the filename belongs to a DJI multispectral red edge image."""
        matcher = re.compile("(.*[\\\\/])?DJI_[0-9]{2}[1-9]4.TIF", flags=re.IGNORECASE)
        return matcher.match(str(filename)) is not None

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """ Given a 16 bit image, normalize the values to the range [0, 1]"""
        return normalize_img(img, bit_depth=16)

    def get_mask_save_paths(self, in_path: str) -> List[str]:
        """ Generates a list of output mask paths for each input image file path. For DJI multispectral, we save a mask
            file for each of the bands even though only the red edge band was used to generate the masks."""
        return [str(Path(self.out_dir).joinpath(f"{Path(in_path).stem[:-1]}{i}_mask.png")) for i in range(6)]


class MicasenseRedEdgeMasker(AbstractBinMasker):
    """ Tom Bell method masker for Micasense RedEdge Camera imagery."""

    @property
    def img_paths(self) -> Iterator[str]:
        """ Generates a list of files which should be used as input to generate glint masks. For Micasense Red Edge
            masking, this should be a list of paths that correspond to the RedEdge band of the files output from a
            Micasense camera."""
        # Filter out all but the red edge band files
        return filter(self.is_micasense_red_edge, super().img_paths)

    @staticmethod
    def is_micasense_red_edge(filename: Union[Path, str]) -> bool:
        """ Determine if the filename belongs to a Micasense red edge image."""
        matcher = re.compile("(.*[\\\\/])?IMG_[0-9]{4}_5.tif", flags=re.IGNORECASE)
        return matcher.match(str(filename)) is not None

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """ Given a 16 bit image, normalize the values to the range [0, 1]."""
        return normalize_img(img, bit_depth=16)

    def get_mask_save_paths(self, in_path: str) -> List[str]:
        """ Generates a list of output mask paths for each input image file path. For Micasense, we wants a mask file
            for each of the bands even though only the red edge band was used to generate the masks."""
        return [str(Path(self.out_dir).joinpath(f"{Path(in_path).stem[:-1]}{i}_mask.png")) for i in range(1, 6)]
