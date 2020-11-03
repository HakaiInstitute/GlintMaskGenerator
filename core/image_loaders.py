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

    @property
    @abstractmethod
    def images(self) -> Iterable[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def paths(self) -> Iterable[Union[List[str], str]]:
        raise NotImplementedError

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        return img

    @singledispatchmethod
    def get_mask_save_paths(self, img_paths: List[str]) -> Iterable[str]:
        img_names = (Path(p).stem for p in img_paths)
        return (str(self.mask_directory.joinpath(f"{p}_mask.png")) for p in img_names)

    @get_mask_save_paths.register
    def _(self, img_path: str) -> Iterable[str]:
        return self.get_mask_save_paths([img_path])


class SingleFileImageLoader(ImageLoader, metaclass=ABCMeta):
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        img = Image.open(path)
        return np.array(img).astype(np.float)

    @property
    def images(self) -> Iterable[np.ndarray]:
        return (self.preprocess_image(self.load_image(p)) for p in self.paths)


class RGB8BitLoader(SingleFileImageLoader):
    @property
    def paths(self) -> Iterable[str]:
        # Filter anything smaller than 1kb, since it's probably corrupt
        return filter(lambda p: os.stat(p).st_size > (1 << 20), list_images(self.image_directory))

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        return normalize_img(img, bit_depth=8)


class MultiFileImageLoader(ImageLoader, metaclass=ABCMeta):
    @property
    @abstractmethod
    def paths(self) -> Iterable[List[str]]:
        raise NotImplementedError

    @staticmethod
    def load_image(paths: List[str]) -> np.ndarray:
        imgs = [np.asarray(Image.open(p)) for p in paths]
        result = np.stack(imgs, axis=2).astype(np.float)
        return result

    @property
    def images(self) -> Iterable[np.ndarray]:
        return (self.preprocess_image(self.load_image(p)) for p in self.paths)


class MicasenseRedEdgeLoader(MultiFileImageLoader):
    _blue_band_pattern = re.compile("(.*[\\\\/])?IMG_[0-9]{4}_1.tif", flags=re.IGNORECASE)

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

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        return normalize_img(img, bit_depth=16)


class P4MSLoader(MultiFileImageLoader):
    _blue_band_pattern = re.compile("(.*[\\\\/])?DJI_[0-9]{3}1.TIF", flags=re.IGNORECASE)

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

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        return normalize_img(img, bit_depth=16)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    loader = P4MSLoader("/media/taylor/Samsung_T5/Datasets/ExampleImages/P4MS", "/tmp")
    for im in loader.images:
        plt.figure()
        plt.imshow(im[:, :, [2, 1, 0]])
        plt.show()

    loader = MicasenseRedEdgeLoader("/media/taylor/Samsung_T5/Datasets/ExampleImages/MicasenseRedEdge", "/tmp")
    for im in loader.images:
        plt.figure()
        plt.imshow(im[:, :, [2, 1, 0]])
        plt.show()
