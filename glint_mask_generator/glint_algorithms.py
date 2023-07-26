"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""
import math
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

EPSILON = 1e-8


class GlintAlgorithm(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Return a boolean glint mask for a given image.
        Output mask should have 1 for masked, 0 for unmasked.
        """
        raise NotImplementedError


class ThresholdAlgorithm(GlintAlgorithm):
    def __init__(self, thresholds: Sequence[float]):
        super().__init__()
        self.thresholds = thresholds

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.any(img > self.thresholds, axis=2)


class IntensityRatioAlgorithm(GlintAlgorithm):
    def __init__(self, percent_diffuse: float = 0.95, threshold: float = 0.99):
        """Create and return a glint mask for RGB imagery.

        Parameters
        ----------
        percent_diffuse
            An estimate of the percentage of pixels in an image that show pure diffuse
            reflectance, and thus no specular reflectance (glint).
        threshold
            Threshold on specular reflectance estimate to binarize into a mask.
            e.g. if more than 50% specular reflectance is unacceptable, use 0.5.
        """
        super().__init__()
        self.percent_diffuse = percent_diffuse
        self.threshold = threshold

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Create and return a glint mask for RGB imagery.

        Parameters
        ----------
        img: np.ndarray shape=(H,W,3)
            Path to a 3-channel RGB numpy image normalized to values in [0,1].

        Returns
        -------
        numpy.ndarray, shape=(H,W)
            Numpy array of glint mask for img at input_path.
        """
        return (
            self._estimate_specular_reflection_component(img, self.percent_diffuse)
            > self.threshold
        )

    @staticmethod
    def _estimate_specular_reflection_component(
        img: np.ndarray, percent_diffuse: float
    ) -> np.ndarray:
        """Estimate the specular reflection component of pixels in an image.

        Based on method from:
            Wang, S., Yu, C., Sun, Y. et al. Specular reflection removal
            of ocean surface remote sensing images from UAVs. Multimedia Tools
            Appl 77, 11363–11379 (2018). https://doi.org/10.1007/s11042-017-5551-7

        Parameters
        ----------
        img: numpy.ndarray, shape=(H,W,C)
            A numpy ndarray of an RGB image.
        percent_diffuse
            An estimate of the % of pixels that show purely diffuse reflection.

        Returns
        -------
        numpy.ndarray, shape=(H,W)
            1D image with values being an estimate of specular reflectance.
        """
        # Calculate the pixel-wise max intensity and intensity range over RGB channels
        i_max = np.amax(img, axis=2)
        i_min = np.amin(img, axis=2)
        i_range = i_max - i_min

        # Calculate intensity ratio
        q = np.divide(i_max, i_range + EPSILON)

        # Select diffuse only pixels using the PERCENTILE_THRESH
        # i.e. A percentage of PERCENTILE_THRESH pixels are supposed to have no
        #     specular reflection
        num_thresh = math.ceil(percent_diffuse * q.size)

        # Get intensity ratio by separating the image into diffuse and specular sections
        q_x_hat = np.partition(q.ravel(), num_thresh)[num_thresh]

        # Estimate the spectral component of each pixel
        spec_ref_est = np.clip(i_max - (q_x_hat * i_range), 0, None)

        return spec_ref_est
