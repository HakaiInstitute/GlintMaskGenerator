"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-06-12
Description: Classes for generating glint masks using the specular reflection estimation technique for various
    types of image files.
"""

from pathlib import Path
from typing import List

import numpy as np

from core.AbstractBaseMasker import AbstractBaseMasker
from core.glint_mask_algorithms.specular_mask import make_single_mask


class RGBSpecularMasker(AbstractBaseMasker):
    """Specular masker method for RGB imagery."""

    def __init__(self, img_dir: str, out_dir: str, percent_diffuse: float = 0.95, mask_thresh: float = 0.99,
                 opening: int = 15, closing: int = 15) -> None:
        """Create and return a glint mask for RGB imagery.

        Args:
            img_dir: str
                The path to a directory containing images to process.

            out_dir: str
                Path to the directory where the image masks should be saved.

            percent_diffuse: Optional[float]
                An estimate of the percentage of pixels in an image that show pure diffuse reflectance, and
                thus no specular reflectance (glint). Defaults to 0.95.

            mask_thresh: Optional[float]
                The threshold on the specular reflectance estimate image to convert into a mask.
                E.g. if more than 50% specular reflectance is unacceptable, use 0.5. Default is 0.99.

            opening: Optional[int]
                The number of morphological opening iterations on the produced mask.
                Useful for closing small holes in the mask. 15 by default.

            closing: Optional[int]
                The number of morphological closing iterations on the produced mask.
                Useful for removing small bits of mask. 15 by default.
        """
        super().__init__()

        self._img_dir = img_dir
        self._out_dir = out_dir
        self._percent_diffuse = percent_diffuse
        self._mask_thresh = mask_thresh
        self._opening = opening
        self._closing = closing

    def get_img_paths(self) -> List[str]:
        """Implements abstract method required by AbstractBaseMasker."""
        return self.list_img_files(self._img_dir)

    def normalize_img(self, img: np.ndarray) -> np.ndarray:
        """Normalizes 8-bit pixel values and select only the RGB channels."""
        return img[:, :, :3] / 255

    def mask_img(self, img: np.ndarray) -> np.ndarray:
        """Generates and saves a glint mask for the image at path img_path using the specular method."""
        return make_single_mask(img, self._percent_diffuse, self._mask_thresh, self._opening, self._closing)

    def get_mask_save_paths(self, in_path: str) -> List[str]:
        """Get the out path for where to save the mask corresponding to image at in_path."""
        return [str(Path(self._out_dir).joinpath(f"{Path(in_path).stem}_mask.png"))]
