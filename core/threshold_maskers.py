"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-16
Description: 
"""

import re
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterable, Pattern, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from core.abstract_masker import Masker

EPSILON = 1e-8


class ThresholdMasker(Masker, metaclass=ABCMeta):
    """ Abstract class for all maskers that use threshold masking algorithm.
        Defines behaviour common to all Bin maskers."""

    def __init__(self, img_dir: str, out_dir: str,
                 glint_thresholds: Iterable[float], mask_buffer_sigma: int = 0) -> None:
        """Create and return a glint mask for RGB imagery.

        Parameters
        ----------
        img_dir
            The path to a directory containing images to process.
        out_dir
            Path to the directory where the image masks should be saved.
        glint_thresholds
            Threshold for each band where values with pixels above this are considered glint. Masks are generated for
            each band and then combined using a union operation. Values must be in domain [0., 1.].
        mask_buffer_sigma
            The sigma for the Gaussian kernel used to buffer the mask.
        """
        super().__init__(img_dir, out_dir)

        self.glint_thresholds = glint_thresholds
        self.mask_buffer_sigma = mask_buffer_sigma

    def mask_img(self, img: np.ndarray) -> np.ndarray:
        """Generates and saves a glint mask for the image at path img_path using Tom's method.

        Parameters
        ----------
        img
            The image data normalized to range [0,1]. Channel order is H, W, C

        Returns
        -------
        np.ndarray
            The calculated mask.
        """
        return np.any(img < self.glint_thresholds, axis=2)

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Buffer the mask and convert to format required by Agisoft."""
        # Buffer the mask
        if self.mask_buffer_sigma > 0:
            mask_buffered = gaussian_filter(mask.astype(np.float), self.mask_buffer_sigma)
            mask = mask_buffered >= 0.99

        # Convert to format required by Metashape
        return mask.astype(np.uint8) * 255


class GenericThresholdMasker(ThresholdMasker):
    """Threshold masker for RGB and ACO imagery."""

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] and switch band order to BGR."""
        # e.g. [0,255] -> [0,1]
        return self.normalize_img(img, bit_depth=8)

    def get_mask_save_paths(self, in_path: str) -> Iterable[str]:
        """Implement get_out_paths as required by AbstractBinMasker."""
        return [str(Path(self.out_dir).joinpath(f"{Path(in_path).stem}_mask.png"))]


class _MultiFileThresholdMasker(ThresholdMasker, metaclass=ABCMeta):
    """Threshold masker for imagery split over multiple files."""

    @property
    @abstractmethod
    def _file_matcher(self) -> Pattern:
        raise NotImplemented

    def is_first_band_file(self, filename: str) -> bool:
        """Determine if the filename belongs to a P4MS blue band image."""
        return self._file_matcher.match(str(filename)) is not None

    @property
    def img_paths(self) -> Iterable[str]:
        return filter(self.is_first_band_file, super().img_paths)

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """Normalize 16-bit images to range [0,1]."""
        return self.normalize_img(img, bit_depth=16)

    def read_img(self, img_path: str) -> np.ndarray:
        """Overrides baseclass image reader to read multiple files into a single numpy array.

        Parameters
        ----------
        img_path
            The path to the first image of 5.

        Returns
        -------
        np.ndarray
            The image data in a np.ndarray with dtype=float32.
        """
        img_paths = self.get_grouped_img_paths(img_path)
        imgs = [np.asarray(Image.open(p)) for p in img_paths]
        result = np.stack(imgs, axis=2).astype(np.float)
        return result

    @staticmethod
    @abstractmethod
    def get_grouped_img_paths(first_band_path: str) -> Sequence[str]:
        raise NotImplemented


class P4MSThresholdMasker(_MultiFileThresholdMasker):
    """Threshold masker for DJI P4MS imagery."""

    @property
    def _file_matcher(self) -> Pattern:
        return re.compile("(.*[\\\\/])?DJI_[0-9]{3}1.TIF", flags=re.IGNORECASE)

    @staticmethod
    def get_grouped_img_paths(first_band_path: str) -> Sequence[str]:
        """Returns a tuple with the files to process.

        e.g. DJI P4MS, this is a list of paths that correspond to the all files output
        from a DJI multispectral camera for a single capture event.
        """
        in_path_root = Path(first_band_path).stem[:-1]
        return [str(Path(first_band_path).with_name(f"{in_path_root}{i}.TIF")) for i in range(1, 6)]

    def get_mask_save_paths(self, in_path: str) -> Iterable[str]:
        """Generates a list of output mask paths for each input image file path.

        For DJI multispectral, we save a mask file for each of the bands even though only the red edge band was used to
        generate the masks.
        """
        in_path_root = Path(in_path).stem[:-1]
        out_dir = Path(self.out_dir)
        return (str(out_dir.joinpath(f"{in_path_root}{i}_mask.png")) for i in range(1, 6))


class MicasenseRedEdgeThresholdMasker(_MultiFileThresholdMasker):
    """Threshold masker for MicasenseRedEdge imagery."""

    @property
    def _file_matcher(self) -> Pattern:
        return re.compile("(.*[\\\\/])?IMG_[0-9]{4}_1.tif", flags=re.IGNORECASE)

    @staticmethod
    def get_grouped_img_paths(first_band_path: str) -> Sequence[str]:
        """Returns a tuple with the files to process.

        e.g. DJI P4MS, this is a list of paths that correspond to the all files output
        from a DJI multispectral camera for a single capture event.
        """
        in_path_root = Path(first_band_path).stem[:-1]
        return [str(Path(first_band_path).with_name(f"{in_path_root}{i}.tif")) for i in range(1, 6)]

    def get_mask_save_paths(self, in_path: str) -> Iterable[str]:
        """Generates a list of output mask paths for each input image file path.

        For Micasense, we wants a mask file for each of the bands even though only the red edge band was used to
        generate the masks."""
        in_path_root = Path(in_path).stem[:-1]
        out_dir = Path(self.out_dir)
        return (str(out_dir.joinpath(f"{in_path_root}{i}_mask.png")) for i in range(1, 6))


if __name__ == '__main__':
    masker = MicasenseRedEdgeThresholdMasker(
        '/media/taylor/Samsung_T5/Datasets/ExampleImages/MicasenseRededge',
        '/tmp',
        glint_thresholds=[0.875, 1., 1., 1., 1.]
    )
    masker(max_workers=1, callback=print, err_callback=print)

    # masker = P4MSThresholdMasker(
    #     '/media/taylor/Samsung_T5/Datasets/ExampleImages/P4MS',
    #     '/tmp',
    #     glint_thresholds=[0.875, 1., 1., 1., 1.]
    # )
    # masker(max_workers=1, callback=print, err_callback=print)
