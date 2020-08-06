"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-06-12
Description: Classes for processing images using Tom's bin-based glint masking technique for various types of image
    files.
"""

import re
from abc import ABCMeta
from pathlib import Path
from typing import Iterator, List, Union

import numpy as np

from core.AbstractBaseMasker import AbstractBaseMasker
from core.glint_mask_algorithms.glint_mask import make_single_mask
from core.utils import normalize_img


class AbstractBinMasker(AbstractBaseMasker, metaclass=ABCMeta):
    """ Abstract class for all maskers that use Tom Bell's binning algorithm.
        Defines behaviour common to all Bin maskers."""

    def __init__(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 0,
                 num_bins: int = 8) -> None:
        """Create and return a glint mask for RGB imagery.

        Parameters
        ----------
        img_dir: str
            The path to a directory containing images to process.
        out_dir: str
            Path to the directory where the image masks should be saved.
        glint_threshold: float
            The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
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
        """Generates and saves a glint mask for the image at path img_path using Tom's method.

        Parameters
        ----------
        img: np.ndarray
            The image data normalized to range [0,1].

        Returns
        -------
        np.ndarray
            The calculated mask.
        """
        return make_single_mask(img, self.glint_threshold, self.mask_buffer_sigma, self.num_bins)


class BlueBinMasker(AbstractBinMasker):
    """Tom Bell's method masker for RGB imagery."""

    def __init__(self, *args, mask_buffer_sigma=20, **kwargs) -> None:
        # This method is only different from the base class in terms of mask_buffer_sigma default value
        super().__init__(*args, mask_buffer_sigma=mask_buffer_sigma, **kwargs)

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """Selects the blue channel and normalize to [0, 1]."""
        return normalize_img(img[:, :, 2], bit_depth=8)

    def get_mask_save_paths(self, in_path: str) -> List[str]:
        """Implement get_out_paths as required by AbstractBinMasker."""
        return [str(Path(self.out_dir).joinpath(f"{Path(in_path).stem}_mask.png"))]


class DJIMultispectralMasker(AbstractBinMasker):
    """ Tom Bell method masker for DJI multi-spectral imagery."""

    _filename_matcher = re.compile("(.*[\\\\/])?DJI_[0-9]{2}[1-9]4.TIF", flags=re.IGNORECASE)

    @property
    def img_paths(self) -> Iterator[str]:
        """Generates an iterator with the files to process.

        For DJI Multispectral masking, this is a list of paths that correspond to the RedEdge band of the files output
        from a DJI multispectral camera.
        """
        return filter(self.is_dji_red_edge, super().img_paths)

    def is_dji_red_edge(self, filename: str) -> bool:
        """Determine if the filename belongs to a DJI multispectral red edge image."""
        return self._filename_matcher.match(str(filename)) is not None

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """Normalize 16-bit images to range [0,1]."""
        return normalize_img(img, bit_depth=16)

    def get_mask_save_paths(self, in_path: str) -> List[str]:
        """Generates a list of output mask paths for each input image file path.

        For DJI multispectral, we save a mask file for each of the bands even though only the red edge band was used to
        generate the masks.
        """
        in_path_root = Path(in_path).stem[:-1]
        out_dir = Path(self.out_dir)
        return [str(out_dir.joinpath(f"{in_path_root}{i}_mask.png")) for i in range(6)]


class MicasenseRedEdgeMasker(AbstractBinMasker):
    """Tom Bell method masker for Micasense RedEdge Camera imagery."""

    _filename_matcher = re.compile("(.*[\\\\/])?IMG_[0-9]{4}_5.tif", flags=re.IGNORECASE)

    @property
    def img_paths(self) -> Iterator[str]:
        """Generates the list of files to process.

        For Micasense Red Edge masking, this should be a list of paths that correspond to the RedEdge band of the files
        output from a Micasense camera.
        """
        # Filter out all but the red edge band files
        return filter(self.is_micasense_red_edge, super().img_paths)

    def is_micasense_red_edge(self, filename: Union[Path, str]) -> bool:
        """Determine if the filename belongs to a Micasense red edge image."""
        return self._filename_matcher.match(str(filename)) is not None

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """Normalize 16-bit imagery to have values in the range [0,1]."""
        return normalize_img(img, bit_depth=16)

    def get_mask_save_paths(self, in_path: str) -> List[str]:
        """Generates a list of output mask paths for each input image file path.

        For Micasense, we wants a mask file for each of the bands even though only the red edge band was used to
        generate the masks."""
        in_path_root = Path(in_path).stem[:-1]
        out_dir = Path(self.out_dir)
        return [str(out_dir.joinpath(f"{in_path_root}{i}_mask.png")) for i in range(1, 6)]
