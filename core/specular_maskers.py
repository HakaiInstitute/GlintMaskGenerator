# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-12
# Description: Classes for generating glint masks using the specular reflection estimation technique for various
#     types of image files.

from pathlib import Path
from typing import List, Any

import numpy as np

from core.AbstractBaseMasker import AbstractBaseMasker
from core.glint_mask_algorithms.specular_mask import make_single_mask


class RGBSpecularMasker(AbstractBaseMasker):
    def __init__(self, img_dir: str, out_dir: str, percent_diffuse: float = 0.1, mask_thresh: float = 0.8,
                 opening: int = 5, closing: int = 5) -> None:
        """Create and return a glint mask for RGB imagery.

        Args:
            img_dir: str
                The path to a directory containing images to process.

            out_dir: str
                Path to the directory where the image masks should be saved.

            percent_diffuse: Optional[float]
                An estimate of the percentage of pixels in an image that show pure diffuse reflectance, and
                thus no specular reflectance (glint). Defaults to 0.1. Try playing with values, low ones typically work well.

            mask_thresh: Optional[float]
                The threshold on the specular reflectance estimate image to convert into a mask.
                E.g. if more than 50% specular reflectance is unacceptable, use 0.5. Default is 0.8.

            opening: Optional[int]
                The number of morphological opening iterations on the produced mask.
                Useful for closing small holes in the mask. 5 by default.

            closing: Optional[int]
                The number of morphological closing iterations on the produced mask.
                Useful for removing small bits of mask. 5 by default.
        """
        self._img_dir = img_dir
        self._out_dir = out_dir
        self._percent_diffuse = percent_diffuse
        self._mask_thresh = mask_thresh
        self._opening = opening
        self._closing = closing

        super().__init__()

    def process_one_file(self, img_path: str) -> Any:
        """Generates and saves a glint mask for the image at path img_path.

        Args:
            img_path: str
                The path to the image to generate a glint mask for.

        Returns:
            Tuple(str, np.ndarray)
                The path to the generated glint mask and an ndarray containing the 8-bit mask.
        """
        img = self.read_img(img_path)
        img = self.normalize_img(img)

        mask = make_single_mask(img, self._percent_diffuse, self._mask_thresh, self._opening, self._closing)
        out_path = self.get_out_paths(img_path)
        self.save_mask(mask, out_path)

        return out_path, mask

    def get_files(self) -> List[str]:
        """Implements abstract method required by AbstractBaseMasker."""
        return self.list_img_files(self._img_dir)

    def get_out_paths(self, in_path: str) -> str:
        """Get the out path for where to save the mask corresponding to image at in_path.

        Args:
            in_path: str
                The image path for which a mask is generated. Used to generate an appropriate out path for the mask.

        Returns:
            str
                The path where the mask for the image at location in_path should be saved.
        """
        return Path(self._out_dir).joinpath(f"{Path(in_path).stem}_mask.png")

    @staticmethod
    def normalize_img(img: np.ndarray) -> np.ndarray:
        """Normalizes 8-bit pixel values and select only the RGB channels."""
        return img[:, :, :3] / 255
