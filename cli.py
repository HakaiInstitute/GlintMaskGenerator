"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-05-30
Description: Command line interface to the glint-mask-tools.
"""
import os
import sys

import fire
from tqdm import tqdm

from core.abstract_masker import Masker
from core.bin_maskers import DJIMultispectralBlueBinMasker, DJIMultispectralRedEdgeBinMasker, MicasenseBlueBinMasker, \
    MicasenseRedEdgeBinMasker, RGBBinMasker
from core.specular_maskers import RGBSpecularMasker


class CLI(object):
    def __init__(self, max_workers: int = os.cpu_count() * 5):
        """Command Line Interface Class for glint mask generators.

        Parameters
        ----------
        max_workers
            The maximum number of threads to use for processing.
        """
        self.max_workers = max_workers

    @staticmethod
    def _err_callback(path, exception):
        tqdm.write(f"{path} failed with err:\n{exception}", file=sys.stderr)

    def process(self, masker: Masker):
        with tqdm(total=len(masker)) as progress:
            return masker.process(max_workers=self.max_workers, callback=lambda _: progress.update(),
                                  err_callback=self._err_callback)

    def rgb(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 0,
            num_bins: int = 8) -> None:
        """Generate masks for glint regions in RGB imagery using Tom Bell's binning algorithm.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images. If img_dir is a directory, all tif, jpg, jpeg,
            and png images in that directory will be processed.
        out_dir
            The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
            out_dir must be a directory if img_dir is specified as a directory.
        glint_threshold
            The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
            Play with this value. Default is 0.9.
        mask_buffer_sigma
            The sigma for the Gaussian kernel used to buffer the mask. Defaults to 20.
        num_bins
            The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self.process(RGBBinMasker(img_dir, out_dir, glint_threshold, mask_buffer_sigma, num_bins))

    def dji_blue(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 0,
                 num_bins: int = 8) -> None:
        """Generate masks for glint regions in multispectral imagery from the DJI camera using Tom Bell's algorithm on
            the Blue image band.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images. If img_dir is a directory, all DJI_***4.TIF
            files will be processed.

        out_dir
            The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
            out_dir must be a directory if img_dir is specified as a directory.

        glint_threshold
            The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
            Play with this value. Default is 0.9.

        mask_buffer_sigma
            The sigma for the Gaussian kernel used to buffer the mask. Defaults to 0.

        num_bins
            The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self.process(DJIMultispectralBlueBinMasker(img_dir, out_dir, glint_threshold, mask_buffer_sigma, num_bins))

    def dji_rededge(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 0,
                    num_bins: int = 8) -> None:
        """Generate masks for glint regions in multispectral imagery from the DJI camera using Tom Bell's algorithm on
            the red edge image band.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images. If img_dir is a directory, all DJI_***4.TIF
            files will be processed.

        out_dir
            The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
            out_dir must be a directory if img_dir is specified as a directory.

        glint_threshold
            The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
            Play with this value. Default is 0.9.

        mask_buffer_sigma
            The sigma for the Gaussian kernel used to buffer the mask. Defaults to 0.

        num_bins
            The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self.process(DJIMultispectralRedEdgeBinMasker(img_dir, out_dir, glint_threshold, mask_buffer_sigma, num_bins))

    def micasense_blue(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 0,
                       num_bins: int = 8) -> None:
        """Generate masks for glint regions in multispectral imagery from the Micasense camera using Tom Bell's
            algorithm on the blue image band.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images.
            If img_dir is a directory, all IMG_dddd_6.tif files will be processed.
        out_dir
            The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
            out_dir must be a directory if img_dir is specified as a directory.
        glint_threshold
            The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
            Play with this value. Default is 0.9.
        mask_buffer_sigma
            The sigma for the Gaussian kernel used to buffer the mask. Defaults to 0.
        num_bins
            The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self.process(MicasenseBlueBinMasker(img_dir, out_dir, glint_threshold, mask_buffer_sigma, num_bins))

    def micasense_rededge(self, img_dir: str, out_dir: str, glint_threshold: float = 0.9, mask_buffer_sigma: int = 0,
                          num_bins: int = 8) -> None:
        """Generate masks for glint regions in multispectral imagery from the Micasense camera using Tom Bell's
            algorithm on the red edge image band.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images.
            If img_dir is a directory, all IMG_dddd_6.tif files will be processed.
        out_dir
            The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
            out_dir must be a directory if img_dir is specified as a directory.
        glint_threshold
            The amount of binned "blueness" that should be glint. Domain for values is (0.0, 1.0).
            Play with this value. Default is 0.9.
        mask_buffer_sigma
            The sigma for the Gaussian kernel used to buffer the mask. Defaults to 0.
        num_bins
            The number of bins the blue channel is slotted into. Defaults to 8 as in Tom's script.

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self.process(MicasenseRedEdgeBinMasker(img_dir, out_dir, glint_threshold, mask_buffer_sigma, num_bins))

    def specular(self, img_dir: str, out_dir: str, percent_diffuse: float = 0.95, mask_thresh: float = 0.99,
                 opening: int = 15, closing: int = 15) -> None:
        """Generate masks for glint regions in RGB imagery by setting a threshold on estimated specular reflectance.

        Parameters
        ----------
        img_dir
            The path to a named input image or directory containing images. If img_dir is a directory, all tif, jpg, jpeg,
            and png images in that directory will be processed.
        out_dir
            The path to send your out image including the file name and type. e.g. "/path/to/mask.png".
            out_dir must be a directory if img_dir is specified as a directory.
        percent_diffuse
            An estimate of the percentage of pixels in an image that show pure diffuse reflectance, and
            thus no specular reflectance (glint). Defaults to 0.95.
        mask_thresh
            The threshold on the specular reflectance estimate image to convert into a mask.
            E.g. if more than 50% specular reflectance is unacceptable, use 0.5. Default is 0.99.
        opening
            The number of morphological opening iterations on the produced mask.
            Useful for closing small holes in the mask. Set to 15 by default.
        closing
            The number of morphological closing iterations on the produced mask.
            Useful for removing small bits of mask. Set to 15 by default.

        Returns
        -------
        None
            Side effects are that the mask is saved to the specified out_dir location.
        """
        self.process(RGBSpecularMasker(img_dir, out_dir, percent_diffuse, mask_thresh, opening, closing))


if __name__ == '__main__':
    fire.Fire(CLI)
