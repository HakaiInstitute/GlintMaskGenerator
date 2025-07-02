"""Module with classes that handle imagery I/O.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18.
"""

from __future__ import annotations

import re
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterable
from functools import singledispatchmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from PIL import Image

from .utils import list_images

if TYPE_CHECKING:
    from .maskers import Masker

Image.MAX_IMAGE_PIXELS = None


class ImageLoader(ABC):
    """Abstract base class for all image loader classes which handle loading the data for each sensor capture."""

    def __init__(
        self,
        image_directory: str | Path,
        mask_directory: str | Path,
    ) -> None:
        """Create a new ImageLoader."""
        super().__init__()
        self.image_directory = Path(image_directory)
        self.mask_directory = Path(mask_directory)

    def __len__(self) -> int:
        """Get the number of captures."""
        return sum(1 for _ in self.paths)

    @staticmethod
    @abstractmethod
    def load_image(path: str) -> np.ndarray:
        """Load the image at path into a numpy array."""
        raise NotImplementedError

    @property
    @abstractmethod
    def paths(self) -> Iterable[list[str] | str]:
        """Get path or paths of imagery to load per capture."""
        raise NotImplementedError

    @singledispatchmethod
    def get_mask_save_paths(self, img_paths: list[str]) -> Iterable[str]:
        """Get a list of paths where the output mask images should be saved."""
        img_names = (Path(p).stem for p in img_paths)
        return (str(self.mask_directory.joinpath(f"{p}_mask.png")) for p in img_names)

    @get_mask_save_paths.register
    def _(self, img_path: str) -> Iterable[str]:
        return self.get_mask_save_paths([img_path])

    def save_masks(self, mask: np.ndarray, img_paths: list[str] | str) -> None:
        """Save the mask to appropriate locations based on the img_paths."""
        mask_img = Image.fromarray(mask)

        for out_path in self.get_mask_save_paths(img_paths):
            mask_img.save(str(out_path))


class SingleFileImageLoader(ImageLoader):
    """Abstract class for handling the loading of imagery contained within a single file."""

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load the image into a numpy array."""
        img = Image.open(path)
        # noinspection PyTypeChecker
        return np.array(img).astype(float)

    @property
    def paths(self) -> Iterable[str]:
        """Get paths of images to load."""
        return list_images(self.image_directory)


class BigTiffLoader(SingleFileImageLoader):
    """Class responsible for loading large single, tiff imagery, such as that output by IX Capture software."""

    _crop_size = 256

    def apply_masker(self, img_paths: list[str] | str, masker: Masker) -> None:
        """Compute the image mask by moving a window over the input.

        These CIR images are too large to be read into memory simultaneously, so this
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
            msg = "BigTiffLoader can only operate on images at a single path"
            raise TypeError(msg)
        img_path = img_paths
        mask_path = next(self.get_mask_save_paths(img_paths))

        src = tifffile.imread(img_path)
        height, width, bands = src.shape

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
                    img = src[y0:y1, x0:x1]
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
    """Abstract class for loading imagery where band data is stored in a multiple files."""

    @staticmethod
    def load_image(paths: list[str | Path]) -> np.ndarray:
        """Load all images into a numpy array."""
        # noinspection PyTypeChecker
        imgs = [np.asarray(Image.open(p)) for p in paths]
        return np.stack(imgs, axis=2).astype(float)

    @property
    @abstractmethod
    def paths(self) -> Iterable[list[str]]:
        """Get grouped paths of imagery to load."""
        raise NotImplementedError


class MicasenseRedEdgeLoader(MultiFileImageLoader):
    """Class responsible for loading imagery from Micasense Red Edge sensors."""

    _base_file_pattern = re.compile("(.*[\\\\/])?IMG_[0-9]{4}_1.tif", flags=re.IGNORECASE)

    @property
    def paths(self) -> Iterable[list[str]]:
        """Find and group related band files together."""
        base_files = filter(self._base_file_pattern.match, list_images(self.image_directory))
        base_files = [Path(f) for f in base_files]

        groups = []
        for base_file in sorted(base_files):
            # Find corresponding band files
            band_files = [str(base_file.with_name(f"{base_file.stem[:-1]}{i}.tif")) for i in range(1, 6)]

            # Verify all files exist
            if all(Path(f).exists() for f in band_files):
                groups.append(band_files)

        return groups


class P4MSLoader(MultiFileImageLoader):
    """Class responsible for loading imagery from Phantom 4 MS sensors."""

    _base_file_pattern = re.compile("(.*[\\\\/])?DJI_[0-9]{3}1.TIF", flags=re.IGNORECASE)

    @property
    def paths(self) -> Iterable[list[str]]:
        """Find and group related band files together."""
        base_files = filter(self._base_file_pattern.match, list_images(self.image_directory))
        base_files = [Path(f) for f in base_files]

        groups = []
        for base_file in sorted(base_files):
            # Find corresponding band files
            band_files = [str(base_file.with_name(f"{base_file.stem[:-1]}{i}.TIF")) for i in range(1, 6)]

            # Verify all files exist
            if all(Path(f).exists() for f in band_files):
                groups.append(band_files)

        return groups


class DJIM3MLoader(MultiFileImageLoader):
    """Class responsible for loading imagery from DJI Mavic 3 MS sensors."""

    _base_file_pattern = re.compile(r"(.*[\\/])?DJI_[0-9]+_[0-9]{4}_MS_G\.TIF", flags=re.IGNORECASE)

    @property
    def paths(self) -> Iterable[list[str]]:
        """Find and group related band files together."""
        base_files = filter(self._base_file_pattern.match, list_images(self.image_directory))

        groups = []
        for base_file in sorted(base_files):
            # Find corresponding band files
            base_name = base_file.replace("_G.TIF", "")
            band_files = [
                f"{base_name}_G.TIF",
                f"{base_name}_R.TIF",
                f"{base_name}_RE.TIF",
                f"{base_name}_NIR.TIF",
            ]

            # Verify all files exist
            if all(Path(f).exists() for f in band_files):
                groups.append(band_files)

        return groups
