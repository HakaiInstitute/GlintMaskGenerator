"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-06-12
Description: Classes for processing images using Tom's bin-based glint masking technique for various types of image
    files.
"""

import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from .abstract_masker import Masker

EPSILON = 1e-8


class BinMasker(Masker, metaclass=ABCMeta):
    """ Abstract class for all maskers that use Tom Bell's binning algorithm.
        Defines behaviour common to all Bin maskers."""

    def __init__(self, img_dir: str, out_dir: str, glint_threshold: float = 0.875, mask_buffer_sigma: int = 0) -> None:
        """Create and return a glint mask for RGB imagery.

        Parameters
        ----------
        img_dir
            The path to a directory containing images to process.
        out_dir
            Path to the directory where the image masks should be saved.
        glint_threshold
            The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
        mask_buffer_sigma
            The sigma for the Gaussian kernel used to buffer the mask.
        """
        super().__init__(img_dir, out_dir)

        self.glint_threshold = glint_threshold
        self.mask_buffer_sigma = mask_buffer_sigma

    def mask_img(self, img: np.ndarray) -> np.ndarray:
        """Generates and saves a glint mask for the image at path img_path using Tom's method.

        Parameters
        ----------
        img
            The image data normalized to range [0,1].

        Returns
        -------
        np.ndarray
            The calculated mask.
        """
        # TOM's method:
        # bins = np.linspace(0., 1., num_bins, endpoint=False)
        # si = np.digitize(img, bins)
        # sis = (si.astype(np.float) - si.min(initial=0)) / ((si.max(initial=0) - si.min(initial=0)) + EPSILON)
        #
        # # Find Glint Threshold and Set those Pixels to 1
        # mask = (sis <= glint_threshold)

        # Much more efficient method.
        # With glint_threshold=0.875, it is equivalent to the above with num_bins=8 and glint_threshold=0.9
        return img < self.glint_threshold

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        # Buffer the mask
        if self.mask_buffer_sigma > 0:
            mask_buffered = gaussian_filter(mask.astype(np.float), self.mask_buffer_sigma)
            mask = mask_buffered >= 0.99

        # Convert to format required by Metashape
        return mask.astype(np.uint8) * 255


class RGBBinMasker(BinMasker):
    """Tom Bell's method masker for RGB imagery."""

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """Selects the blue channel and normalize to [0, 1]."""
        return self.normalize_img(img[:, :, 2], bit_depth=8)

    def get_mask_save_paths(self, in_path: str) -> Iterable[str]:
        """Implement get_out_paths as required by AbstractBinMasker."""
        return [str(Path(self.out_dir).joinpath(f"{Path(in_path).stem}_mask.png"))]


class _P4MSBinMasker(BinMasker, metaclass=ABCMeta):
    """ Tom Bell method masker for DJI multi-spectral imagery."""

    @property
    @abstractmethod
    def filename_matcher(self):
        raise NotImplemented

    def is_dji_red_edge(self, filename: str) -> bool:
        """Determine if the filename belongs to a DJI multispectral red edge image."""
        return self.filename_matcher.match(str(filename)) is not None

    @property
    def img_paths(self) -> Iterable[str]:
        """Generates an iterator with the files to process.

        For DJI Multispectral masking, this is a list of paths that correspond to the RedEdge band of the files output
        from a DJI multispectral camera.
        """
        return filter(self.is_dji_red_edge, super().img_paths)

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """Normalize 16-bit images to range [0,1]."""
        return self.normalize_img(img, bit_depth=16)

    def get_mask_save_paths(self, in_path: str) -> Iterable[str]:
        """Generates a list of output mask paths for each input image file path.

        For DJI multispectral, we save a mask file for each of the bands even though only the red edge band was used to
        generate the masks.
        """
        in_path_root = Path(in_path).stem[:-1]
        out_dir = Path(self.out_dir)
        return [str(out_dir.joinpath(f"{in_path_root}{i}_mask.png")) for i in range(1, 6)]


class P4MSRedEdgeBinMasker(_P4MSBinMasker):
    """ Tom Bell method masker for DJI multi-spectral imagery."""

    @property
    def filename_matcher(self):
        return re.compile("(.*[\\\\/])?DJI_[0-9]{3}4.TIF", flags=re.IGNORECASE)


class P4MSBlueBinMasker(_P4MSBinMasker):
    """ Tom Bell method masker for DJI multi-spectral imagery."""

    @property
    def filename_matcher(self):
        return re.compile("(.*[\\\\/])?DJI_[0-9]{3}1.TIF", flags=re.IGNORECASE)


class _MicasenseRedEdgeBinMasker(BinMasker, metaclass=ABCMeta):
    """Tom Bell method masker for Micasense RedEdge Camera imagery."""

    @property
    @abstractmethod
    def filename_matcher(self):
        raise NotImplemented

    def is_micasense_red_edge(self, filename: Union[Path, str]) -> bool:
        """Determine if the filename belongs to a Micasense red edge image."""
        return self.filename_matcher.match(str(filename)) is not None

    @property
    def img_paths(self) -> Iterable[str]:
        """Generates the list of files to process.

        For Micasense Red Edge masking, this should be a list of paths that correspond to the RedEdge band of the files
        output from a Micasense camera.
        """
        # Filter out all but the red edge band files
        return filter(self.is_micasense_red_edge, super().img_paths)

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """Normalize 16-bit imagery to have values in the range [0,1]."""
        return self.normalize_img(img, bit_depth=16)

    def get_mask_save_paths(self, in_path: str) -> Iterable[str]:
        """Generates a list of output mask paths for each input image file path.

        For Micasense, we wants a mask file for each of the bands even though only the red edge band was used to
        generate the masks."""
        in_path_root = Path(in_path).stem[:-1]
        out_dir = Path(self.out_dir)
        return [str(out_dir.joinpath(f"{in_path_root}{i}_mask.png")) for i in range(1, 6)]


class MicasenseRedEdgeBinMasker(_MicasenseRedEdgeBinMasker):
    """Tom Bell method masker for Micasense RedEdge Camera imagery."""

    @property
    def filename_matcher(self):
        return re.compile("(.*[\\\\/])?IMG_[0-9]{4}_5.tif", flags=re.IGNORECASE)


class MicasenseBlueBinMasker(_MicasenseRedEdgeBinMasker):
    """Tom Bell method masker for Micasense RedEdge Camera imagery."""

    @property
    def filename_matcher(self):
        return re.compile("(.*[\\\\/])?IMG_[0-9]{4}_1.tif", flags=re.IGNORECASE)
