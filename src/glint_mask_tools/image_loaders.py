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

import numpy as np
import tifffile
from PIL import Image

from .utils import list_images

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

    def save_masks(self, mask: np.ndarray, img_paths: list[str] | str, *, per_band: bool = False) -> None:
        """Save the mask to appropriate locations based on the img_paths.

        Parameters
        ----------
        mask
            The mask array to save. Can be 2D (H, W) or 3D (H, W, C) for per-band masks.
        img_paths
            The paths of the input images, used to determine output paths.
        per_band
            If True and mask is 3D, save each channel to its corresponding output file.

        """
        mask_paths = list(self.get_mask_save_paths(img_paths))

        if mask.ndim == 3 and per_band and len(mask_paths) == mask.shape[2]:  # noqa: PLR2004
            # Per-band mode: save each channel to its corresponding file
            for i, out_path in enumerate(mask_paths):
                mask_img = Image.fromarray(mask[:, :, i])
                mask_img.save(str(out_path))
        else:
            # Combined mode: save same mask to all outputs
            mask_2d = mask if mask.ndim == 2 else mask[:, :, 0]  # noqa: PLR2004
            mask_img = Image.fromarray(mask_2d)
            for out_path in mask_paths:
                mask_img.save(str(out_path))


class SingleFileImageLoader(ImageLoader):
    """Abstract class for handling the loading of imagery contained within a single file."""

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load the image into a numpy array."""
        img = np.array(Image.open(path).convert("RGB"))
        return img.astype(float)

    @property
    def paths(self) -> Iterable[str]:
        """Get paths of images to load."""
        return list_images(self.image_directory)


class BigTiffLoader(SingleFileImageLoader):
    """Class responsible for loading large single, tiff imagery, such as that output by IX Capture software."""

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load the image into a numpy array."""
        img = tifffile.imread(path)[:, :, :4]
        return np.array(img).astype(float)


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
    _num_bands = 5

    @property
    def paths(self) -> Iterable[list[str]]:
        """Find and group related band files together."""
        base_files = filter(self._base_file_pattern.match, list_images(self.image_directory))
        base_files = [Path(f) for f in base_files]

        groups = []
        for base_file in sorted(base_files):
            # Get the original extension (preserving case)
            original_ext = base_file.suffix
            # Find corresponding band files using original extension
            band_files = [
                str(base_file.with_name(f"{base_file.stem[:-1]}{i + 1}{original_ext}")) for i in range(self._num_bands)
            ]

            # Verify all files exist
            if all(Path(f).exists() for f in band_files):
                groups.append(band_files)

        return groups


class MicasenseRedEdgeDualLoader(MicasenseRedEdgeLoader):
    """Class responsible for loading imagery from Micasense Red Edge Dual sensors."""

    _num_bands = 10


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
            # Get the original extension (preserving case)
            original_ext = base_file.suffix
            # Find corresponding band files using original extension
            band_files = [str(base_file.with_name(f"{base_file.stem[:-1]}{i}{original_ext}")) for i in range(1, 6)]

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
