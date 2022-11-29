"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
Description: 
"""
import os
import re
from abc import ABC, ABCMeta, abstractmethod
from functools import singledispatchmethod
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import rasterio
from PIL import Image

from .utils import list_images, normalize_img

Image.MAX_IMAGE_PIXELS = None


class ImageLoader(ABC):
    def __init__(self, image_directory: str, mask_directory: str):
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
    def _bit_depth(self):
        return NotImplementedError

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
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(str(out_path))

    def mask_images(self, img_paths: Union[List[str], str], masker: "Masker"):
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
        return np.array(img).astype(np.float)

    @property
    def paths(self) -> Iterable[str]:
        # Filter anything smaller than 1kb, since it's probably corrupt
        return filter(lambda p: os.stat(p).st_size > (1 << 20), list_images(self.image_directory))


class RGBLoader(SingleFileImageLoader):
    _bit_depth = 8


class CIRLoader(SingleFileImageLoader):
    _bit_depth = 8

    def mask_images(self, img_paths: Union[List[str], str], masker: 'Masker'):
        if not isinstance(img_paths, str):
            raise RuntimeError("CIRLoader can only operate on images at a single path")
        else:
            img_path = img_paths[0]
            mask_path = self.get_mask_save_paths(img_paths)[0]
        with rasterio.open(img_path, 'r') as src:
            profile = src.profile

            profile.update(
                dtype=rasterio.uint8,
                count=1,
                compress='lzw')

            with rasterio.open(mask_path, 'w', **profile) as dest:
                for ji, window in src.block_windows(1):
                    img = src.read(1, window=window)
                    img = self.preprocess_image(img)
                    mask = masker.algorithm(img)
                    mask = masker.postprocess_mask(mask)
                    mask = masker.to_metashape_mask(mask)
                    dest.write(mask, window=window, indexes=1)


class MultiFileImageLoader(ImageLoader, metaclass=ABCMeta):
    @staticmethod
    def load_image(paths: List[str]) -> np.ndarray:
        imgs = [np.asarray(Image.open(p)) for p in paths]
        return np.stack(imgs, axis=2).astype(np.float)

    @property
    @abstractmethod
    def paths(self) -> Iterable[List[str]]:
        raise NotImplementedError


class MicasenseRedEdgeLoader(MultiFileImageLoader):
    _blue_band_pattern = re.compile("(.*[\\\\/])?IMG_[0-9]{4}_1.tif", flags=re.IGNORECASE)
    _bit_depth = 16

    def _is_blue_band_path(self, path: str) -> bool:
        return self._blue_band_pattern.match(str(path)) is not None

    @property
    def _blue_band_paths(self) -> Iterable[str]:
        return filter(self._is_blue_band_path, list_images(self.image_directory))

    @staticmethod
    def _blue_band_path_to_band_paths(path: str) -> List[str]:
        in_path_root = Path(path).stem[:-1]
        return [str(Path(path).with_name(f"{in_path_root}{i}.tif")) for i in range(1, 6)]

    @property
    def paths(self) -> Iterable[List[str]]:
        return map(self._blue_band_path_to_band_paths, self._blue_band_paths)


class P4MSLoader(MultiFileImageLoader):
    _blue_band_pattern = re.compile("(.*[\\\\/])?DJI_[0-9]{3}1.TIF", flags=re.IGNORECASE)
    _bit_depth = 16

    def _is_blue_band_path(self, path: str) -> bool:
        return self._blue_band_pattern.match(str(path)) is not None

    @property
    def _blue_band_paths(self) -> Iterable[str]:
        return filter(self._is_blue_band_path, list_images(self.image_directory))

    @staticmethod
    def _blue_band_path_to_band_paths(path: str) -> List[str]:
        in_path_root = Path(path).stem[:-1]
        return [str(Path(path).with_name(f"{in_path_root}{i}.TIF")) for i in range(1, 6)]

    @property
    def paths(self) -> Iterable[List[str]]:
        return map(self._blue_band_path_to_band_paths, self._blue_band_paths)
