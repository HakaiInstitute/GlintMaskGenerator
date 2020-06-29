# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-06-12
# Description: Classes for processing images using Tom's bin-based glint masking technique for various types of image
#     files

import re
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Union

import numpy as np

from core.AbstractBaseMasker import AbstractBaseMasker
from core.glint_mask_algorithms.glint_mask import make_single_mask


class AbstractBinMasker(AbstractBaseMasker):
    def __init__(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 0,
                 num_bins: int = 8) -> None:
        """Create and return a glint mask for RGB imagery.

        Args:
            img_dir: str
                The path to a directory containing images to process.

            out_dir: str
                Path to the directory where the image masks should be saved.

            glint_threshold: Optional[float]
                The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
                Play with this value. Default is 0.9.

            mask_buffer_sigma: Optional[int]
                The sigma for the Gaussian kernel used to buffer the mask. Defaults to 0.

            num_bins: Optional[int]
                The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.
        """
        self._img_dir = img_dir
        self._out_dir = out_dir
        self._glint_threshold = glint_threshold
        self._mask_buffer_sigma = mask_buffer_sigma
        self._num_bins = num_bins

        super().__init__()

    def process_one_file(self, img_path: str) -> Any:
        """Reads the image at img_path and produces a mask using Toms binning method and saves the mask to the out dir.

        Args:
            img_path: str
                The path of the image to mask and save a mask for.

        Returns:
            Tuple[List[str], np.ndarray]
                The list of mask file paths created and the image mask as a numpy array.
        """
        img = self.read_img(img_path)
        img = self.normalize_img(img)
        mask = make_single_mask(img, self._glint_threshold, self._mask_buffer_sigma, self._num_bins)

        out_paths = self.get_out_paths(img_path)
        for path in out_paths:
            self.save_mask(mask, path)

        return out_paths, mask

    @staticmethod
    @abstractmethod
    def normalize_img(img: np.ndarray) -> np.ndarray:
        """Abstract method that converts a numpy array with unknown shape to a 2D array with values normalized to [0,1].

        Args:
            img: np.ndarray shape (H,W,C?,...)
                An array from an image read from the files to process list. Must be normalized by the max image pixel
                value such that the returned np.ndarray has values in the range [0,1]

        Returns:
            np.ndarray, shape=(H,W)
                The single channel, normalized array with values in range [0,1]
        """
        raise NotImplementedError

    @abstractmethod
    def get_out_paths(self, in_path: str) -> List[str]:
        """Generate the list of file paths where the generate glint mask should be saved.

        Args:
            in_path: The name of the image path for which a mask was generated.

        Returns:
            Union[str, List[str]]
                A path or list of paths where the glint mask corresponding to the img at in_path are to be saved.
        """
        raise NotImplementedError


class BlueBinMasker(AbstractBinMasker):
    def __init__(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 20,
                 num_bins: int = 8) -> None:
        """Create and return a glint mask for RGB imagery.

        Args:
            img_dir: str
                The path to a directory containing images to process.

            out_dir: str
                Path to the directory where the image masks should be saved.

            glint_threshold: Optional[float]
                The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
                Play with this value. Default is 0.9.

            mask_buffer_sigma: Optional[int]
                The sigma for the Gaussian kernel used to buffer the mask. Defaults to 0.

            num_bins: Optional[int]
                The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.
        """
        super().__init__(img_dir, out_dir, glint_threshold, mask_buffer_sigma, num_bins)

    def get_files(self) -> List[str]:
        """Implement get_files as required by AbstractBaseMasker."""
        return self.list_img_files(self._img_dir)

    def get_out_paths(self, in_path: str) -> List[str]:
        """Implement get_out_paths as required by AbstractBinMasker."""
        return [str(Path(self._out_dir).joinpath(f"{Path(in_path).stem}_mask.png"))]

    @staticmethod
    def normalize_img(img: np.ndarray) -> np.ndarray:
        """Implements normalize_img as required by AbstractBinMasker."""
        return img[:, :, 2] / 255


class DJIMultispectralMasker(AbstractBinMasker):
    def get_files(self) -> List[str]:
        """Generates a list of files which should be used as
            input to generate glint masks. For DJI Multispectral masking, this should be a list of paths that
            correspond to the RedEdge band of the files output from a DJI multispectral camera.

        Implements get_files as required by AbstractBaseMasker.

        Returns
            List[str]
                A list of paths for red edge images that should be processed using Tom's binning algorithm.
        """
        files = self.list_img_files(self._img_dir)
        return list(filter(self.is_dji_red_edge, files))

    def get_out_paths(self, in_path: str) -> List[str]:
        """Generates a list of output mask paths for each
            input image file path. For DJI multispectral, we wants a mask file for each of the bands even though only
            the red edge band was used to generate the masks.

        Implements get_out_paths as required by AbstractBinMasker.

        Args:
            in_path: str
                The path to the input image file that the output mask paths should be generated for.

        Returns:
            List[str]
                A list of paths to save the masks to for the corresponding image at in_path.
        """
        return [str(Path(self._out_dir).joinpath(f"{Path(in_path).stem[:-1]}{i}_mask.png")) for i in range(6)]

    @staticmethod
    def is_dji_red_edge(filename: str) -> bool:
        """Determine if the filename belongs to a DJI multispectral red edge image.

        Args:
            filename: Union[Path, str]
                A string or Path object pointing to an input image file path.

        Returns:
            bool
                A boolean indicating if the path filename corresponds to a micasense red edge band image.
        """
        matcher = re.compile("(.*[\\\\/])?DJI_[0-9]{2}[1-9]4.TIF", flags=re.IGNORECASE)
        return matcher.match(str(filename)) is not None

    @staticmethod
    def normalize_img(img: np.ndarray) -> np.ndarray:
        """Given a 16 bit image, normalize the values to the range [0, 1]

        Implements normalize_img as required by AbstractBinMasker.

        Args:
            img: np.ndarray, shape=(H,W)
                A numpy array of the 16-bit input image.

        Returns:
            np.ndarray, shape=(H,W)
                The normalized image as an ndarray.
        """
        return img / ((1 << 16) - 1)


class MicasenseRedEdgeMasker(AbstractBinMasker):
    def get_files(self) -> List[str]:
        """Generates a list of files which should be used as
            input to generate glint masks. For Micasense Red Edge masking, this should be a list of paths that
            correspond to the RedEdge band of the files output from a Micasense camera.

        Implements get_files as required by AbstractBaseMasker.

        Returns
            List[str]
                A list of paths for red edge images that should be processed using Tom's binning algorithm.
        """
        # Gets all image files
        files = self.list_img_files(self._img_dir)

        # Filter out all but the red edge band files
        return list(filter(self.is_micasense_red_edge, files))

    def get_out_paths(self, in_path: str) -> List[str]:
        """Generates a list of output mask paths for each
            input image file path. For Micasense, we wants a mask file for each of the bands even though only the
            red edge band was used to generate the masks.

        Implements get_out_paths as required by AbstractBinMasker.

        Args:
            in_path: str
                The path to the input image file that the output mask paths should be generated for.

        Returns:
            List[str]
                A list of paths to save the masks to for the corresponding image at in_path.
        """
        return [str(Path(self._out_dir).joinpath(f"{Path(in_path).stem[:-1]}{i}_mask.png")) for i in range(1, 6)]

    @staticmethod
    def is_micasense_red_edge(filename: Union[Path, str]) -> bool:
        """Determine if the filename belongs to a Micasense red edge image.

        Args:
            filename: Union[Path, str]
                A string or Path object pointing to an input image file path.

        Returns:
            bool
                A boolean indicating if the path filename corresponds to a micasense red edge band image.
        """
        matcher = re.compile("(.*[\\\\/])?IMG_[0-9]{4}_5.tif", flags=re.IGNORECASE)
        return matcher.match(str(filename)) is not None

    @staticmethod
    def normalize_img(img: np.ndarray) -> np.ndarray:
        """Given a 16 bit image, normalize the values to the range [0, 1]

        Implements normalize_img as required by AbstractBinMasker.

        Args:
            img: np.ndarray, shape=(H,W)
                A numpy array of the 16-bit input image.

        Returns:
            np.ndarray, shape=(H,W)
                The normalized image as an ndarray.
        """
        return img / ((1 << 16) - 1)
