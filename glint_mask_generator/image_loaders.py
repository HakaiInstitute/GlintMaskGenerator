"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""
import re
from abc import ABC, ABCMeta, abstractmethod
from functools import singledispatchmethod
from pathlib import Path
from typing import Iterable, List, Union, TYPE_CHECKING

import numpy as np
from PIL import Image

from .utils import list_images, normalize_img

if TYPE_CHECKING:
    from .maskers import Masker

Image.MAX_IMAGE_PIXELS = None


class ImageLoader(ABC):
    def __init__(
        self, image_directory: Union[str, Path], mask_directory: Union[str, Path]
    ):
        super().__init__()
        self.image_directory = Path(image_directory)
        self.mask_directory = Path(mask_directory)

    def __len__(self):
        return sum(1 for _ in self.paths)

    @staticmethod
    @abstractmethod
    def load_image(path: str) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def paths(self) -> Iterable[Union[List[str], str]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def _bit_depth(self) -> int:
        raise NotImplementedError

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        return normalize_img(img, bit_depth=self._bit_depth)

    @singledispatchmethod
    def get_mask_save_paths(self, img_paths: List[str]) -> Iterable[str]:
        img_names = (Path(p).stem for p in img_paths)
        return (str(self.mask_directory.joinpath(f"{p}_mask.png")) for p in img_names)

    @get_mask_save_paths.register
    def _(self, img_path: str) -> Iterable[str]:
        return self.get_mask_save_paths([img_path])

    @staticmethod
    def save_mask(mask: np.ndarray, out_path: str):
        """Utility function to save a mask to the location out_path.

        Parameters
        ----------
        mask
            2D image mask to save into the image format specified in the out_path.
        out_path
            The path where the file should be saved, including img extension.
        """
        mask_img = Image.fromarray(mask, mode="L")
        mask_img.save(str(out_path))

    def apply_masker(
        self, img_paths: Union[List[str], str], masker: "Masker"
    ):  # noqa: F821
        img = self.load_image(img_paths)
        img = self.preprocess_image(img)
        mask = masker.algorithm(img)
        mask = masker.postprocess_mask(mask)
        mask = masker.to_metashape_mask(mask)

        for path in self.get_mask_save_paths(img_paths):
            self.save_mask(mask, path)


class SingleFileImageLoader(ImageLoader, metaclass=ABCMeta):
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        img = Image.open(path)
        # noinspection PyTypeChecker
        return np.array(img).astype(float)

    @property
    def paths(self) -> Iterable[str]:
        return list_images(self.image_directory)


class RGBLoader(SingleFileImageLoader):
    _bit_depth = 8


class CIRLoader(SingleFileImageLoader):
    _bit_depth = 8
    _crop_size = 256

    def apply_masker(
        self, img_paths: Union[List[str], str], masker: "Masker"
    ):  # noqa: F821
        """Compute the image mask by moving a window over the input.
        These CIR images are too large to be read into memory simultaneously so this
        image masking routine is special.

        This is the setup in the following logic:
        The inner square is the region to be written to the output.
        The outer square is a padding section used to ensure the pixel buffers are
        computed properly.
        x1,y1
           ,-----------------,
           |  x,y            |
           |   ,--------,    |
           |   |        |    |
           |   |        |    |
           |   |________|    |
           |          xc,yc  |
           |_________________|
                           x2,y2
        """
        if not isinstance(img_paths, str):
            raise RuntimeError("CIRLoader can only operate on images at a single path")
        else:
            img_path = img_paths
            mask_path = next(self.get_mask_save_paths(img_paths))

        with Image.open(img_path, "r") as src:
            height = src.height
            width = src.width

            pad = masker.pixel_buffer

            # Make an all black image to store the data
            with Image.new("L", (width, height)) as dest:
                for y in range(0, height, self._crop_size):
                    for x in range(0, width, self._crop_size):
                        xc = min(x + self._crop_size, width)
                        yc = min(y + self._crop_size, height)

                        x0 = max(x - pad, 0)
                        x1 = min(xc + pad, width)
                        y0 = max(y - pad, 0)
                        y1 = min(yc + pad, height)

                        # noinspection PyTypeChecker
                        img = np.array(src.crop((x0, y0, x1, y1)))
                        img = self.preprocess_image(img)
                        mask = masker.algorithm(img)
                        mask = masker.postprocess_mask(mask)
                        mask = masker.to_metashape_mask(mask)

                        # Remove padding
                        mask = mask[(y - y0) :, (x - x0) :]
                        mask = mask[
                            : self._crop_size + pad - (y1 - yc),
                            : self._crop_size + pad - (x1 - xc),
                        ]

                        # Write the mask section
                        dest.paste(Image.fromarray(mask), (x, y))
                dest.save(mask_path)


class MultiFileImageLoader(ImageLoader, metaclass=ABCMeta):
    @staticmethod
    def load_image(paths: List[Union[str, Path]]) -> np.ndarray:
        # noinspection PyTypeChecker
        imgs = [np.asarray(Image.open(p)) for p in paths]
        return np.stack(imgs, axis=2).astype(float)

    @property
    @abstractmethod
    def paths(self) -> Iterable[List[str]]:
        raise NotImplementedError


class MicasenseRedEdgeLoader(MultiFileImageLoader):
    _blue_band_pattern = re.compile(
        "(.*[\\\\/])?IMG_[0-9]{4}_1.tif", flags=re.IGNORECASE
    )
    _bit_depth = 16

    def _is_blue_band_path(self, path: Union[str, Path]) -> bool:
        return self._blue_band_pattern.match(str(path)) is not None

    @property
    def _blue_band_paths(self) -> Iterable[str]:
        return filter(self._is_blue_band_path, list_images(self.image_directory))

    @staticmethod
    def _blue_band_path_to_band_paths(path: str) -> List[str]:
        in_path_root = Path(path).stem[:-1]
        return [
            str(Path(path).with_name(f"{in_path_root}{i}.tif")) for i in range(1, 6)
        ]

    @property
    def paths(self) -> Iterable[List[str]]:
        return map(self._blue_band_path_to_band_paths, self._blue_band_paths)


class P4MSLoader(MultiFileImageLoader):
    _blue_band_pattern = re.compile(
        "(.*[\\\\/])?DJI_[0-9]{3}1.TIF", flags=re.IGNORECASE
    )
    _bit_depth = 16

    def _is_blue_band_path(self, path: Union[str, Path]) -> bool:
        return self._blue_band_pattern.match(str(path)) is not None

    @property
    def _blue_band_paths(self) -> Iterable[str]:
        return filter(self._is_blue_band_path, list_images(self.image_directory))

    @staticmethod
    def _blue_band_path_to_band_paths(path: Union[str, Path]) -> List[str]:
        in_path_root = Path(path).stem[:-1]
        return [
            str(Path(path).with_name(f"{in_path_root}{i}.TIF")) for i in range(1, 6)
        ]

    @property
    def paths(self) -> Iterable[List[str]]:
        return map(self._blue_band_path_to_band_paths, self._blue_band_paths)
