"""Band alignment module for multi-band imagery using phase correlation.

This module provides automatic alignment of image bands from multi-sensor cameras
where each band may have a slight spatial offset due to sensor positioning.

Created by: Taylor Denouden
Organization: Hakai Institute
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@dataclass
class BandOffsets:
    """Calibrated offsets for each band relative to reference band (index 0)."""

    x_offsets: tuple[int, ...]
    y_offsets: tuple[int, ...]

    @property
    def num_bands(self) -> int:
        """Return the number of bands."""
        return len(self.x_offsets)

    def has_offset(self) -> bool:
        """Return True if any band has non-zero offset."""
        return any(x != 0 or y != 0 for x, y in zip(self.x_offsets, self.y_offsets))


class BandAligner:
    """Handles band alignment calibration and application using phase correlation."""

    def __init__(
        self,
        calibration_samples: int = 5,
        *,
        enabled: bool = True,
    ) -> None:
        """Create a new BandAligner.

        Parameters
        ----------
        calibration_samples
            Number of images to sample for calibration (default 5).
        enabled
            Whether alignment is enabled (default True).

        """
        self.calibration_samples = calibration_samples
        self.enabled = enabled
        self._offsets: BandOffsets | None = None
        self._calibrated = False

    @property
    def is_calibrated(self) -> bool:
        """Return True if calibration has been performed."""
        return self._calibrated

    @property
    def offsets(self) -> BandOffsets | None:
        """Return the calibrated offsets, or None if not calibrated."""
        return self._offsets

    def calibrate(
        self,
        image_paths: Iterable[list[str] | str],
        load_fn: Callable[[list[str] | str], np.ndarray],
    ) -> BandOffsets | None:
        """Calibrate alignment offsets from a sample of images.

        Parameters
        ----------
        image_paths
            Iterable of image paths (or path groups for multi-file loaders).
        load_fn
            Function to load image data given paths.

        Returns
        -------
        BandOffsets | None
            Calibrated offsets for each band, or None if calibration failed.

        """
        if not self.enabled:
            self._calibrated = True
            return None

        paths_list = list(image_paths)
        if not paths_list:
            logger.warning("No images found for band alignment calibration")
            self._calibrated = True
            return None

        sample_paths = paths_list[: self.calibration_samples]
        logger.info(f"Calibrating band alignment from {len(sample_paths)} sample images")

        all_x_offsets: list[list[int]] = []
        all_y_offsets: list[list[int]] = []

        try:
            for paths in sample_paths:
                img = load_fn(paths)

                if img.ndim != 3:  # noqa: PLR2004
                    logger.warning("Image is not multi-band, skipping alignment calibration")
                    self.enabled = False
                    self._calibrated = True
                    return None

                num_bands = img.shape[2]
                ref_band = img[:, :, 0]

                x_offs = [0]
                y_offs = [0]

                for band_idx in range(1, num_bands):
                    target_band = img[:, :, band_idx]
                    x_off, y_off = self._estimate_offset(ref_band, target_band)
                    x_offs.append(x_off)
                    y_offs.append(y_off)

                all_x_offsets.append(x_offs)
                all_y_offsets.append(y_offs)

        except cv2.error as e:
            logger.warning(f"Band alignment calibration failed: {e}. Alignment disabled.")
            self.enabled = False
            self._calibrated = True
            return None

        if all_x_offsets:
            num_bands = len(all_x_offsets[0])
            median_x = tuple(int(np.median([sample[i] for sample in all_x_offsets])) for i in range(num_bands))
            median_y = tuple(int(np.median([sample[i] for sample in all_y_offsets])) for i in range(num_bands))
            self._offsets = BandOffsets(x_offsets=median_x, y_offsets=median_y)

            if self._offsets.has_offset():
                logger.info(f"Band alignment offsets: x={median_x}, y={median_y}")
            else:
                logger.info("Band alignment calibration found no significant offsets")

        self._calibrated = True
        return self._offsets

    def align(self, img: np.ndarray) -> np.ndarray:
        """Apply calibrated alignment to an image.

        Parameters
        ----------
        img
            Image array with shape (H, W, num_bands).

        Returns
        -------
        np.ndarray
            Aligned image with same shape.

        Raises
        ------
        ValueError
            If image band count doesn't match calibration.

        """
        if not self.enabled:
            return img

        if self._offsets is None:
            return img

        if not self._offsets.has_offset():
            return img

        if img.shape[2] != self._offsets.num_bands:
            msg = f"Image has {img.shape[2]} bands but alignment calibrated for {self._offsets.num_bands} bands"
            raise ValueError(msg)

        aligned_bands = []
        for band_idx in range(img.shape[2]):
            band = img[:, :, band_idx]
            x_off = self._offsets.x_offsets[band_idx]
            y_off = self._offsets.y_offsets[band_idx]
            aligned_band = self._apply_offset(band, x_off, y_off)
            aligned_bands.append(aligned_band)

        return np.stack(aligned_bands, axis=2)

    def unalign_mask(self, mask: np.ndarray) -> np.ndarray:
        """Shift a mask back to original (unaligned) coordinate space for each band.

        Takes a 2D union mask (or 3D per-band masks) and creates shifted versions
        for each band so the mask aligns with the original unaligned band images.

        Parameters
        ----------
        mask
            Mask array with shape (H, W) for union mask or (H, W, num_bands) for per-band.

        Returns
        -------
        np.ndarray
            3D mask array (H, W, num_bands) with each band shifted to its original coordinates.

        """
        if not self.enabled:
            return mask

        if self._offsets is None:
            return mask

        if not self._offsets.has_offset():
            return mask

        # Handle 2D union mask - replicate and shift for each band
        if mask.ndim == 2:  # noqa: PLR2004
            logger.info(
                f"Shifting union mask to per-band coordinates: x={self._offsets.x_offsets}, y={self._offsets.y_offsets}"
            )
            unaligned_masks = []
            for band_idx in range(self._offsets.num_bands):
                x_off = -self._offsets.x_offsets[band_idx]
                y_off = -self._offsets.y_offsets[band_idx]
                unaligned_mask = self._apply_offset(mask.astype(np.float32), x_off, y_off)
                unaligned_masks.append(unaligned_mask)
            return np.stack(unaligned_masks, axis=2)

        # Handle 3D per-band masks
        if mask.shape[2] != self._offsets.num_bands:
            msg = f"Mask has {mask.shape[2]} bands but alignment calibrated for {self._offsets.num_bands} bands"
            raise ValueError(msg)

        logger.info(
            f"Shifting per-band masks to original coordinates: x={self._offsets.x_offsets}, y={self._offsets.y_offsets}"
        )

        unaligned_masks = []
        for band_idx in range(mask.shape[2]):
            mask_band = mask[:, :, band_idx]
            x_off = -self._offsets.x_offsets[band_idx]
            y_off = -self._offsets.y_offsets[band_idx]
            unaligned_mask = self._apply_offset(mask_band.astype(np.float32), x_off, y_off)
            unaligned_masks.append(unaligned_mask)

        return np.stack(unaligned_masks, axis=2)

    @staticmethod
    def _estimate_offset(
        ref_band: np.ndarray,
        target_band: np.ndarray,
    ) -> tuple[int, int]:
        """Estimate offset of target band relative to reference using phase correlation.

        Returns
        -------
        tuple[int, int]
            (x_offset, y_offset) to apply to target band to align with reference.
            This is the negated shift (correction vector).

        """
        shift, _ = cv2.phaseCorrelate(
            ref_band.astype(np.float64),
            target_band.astype(np.float64),
        )
        # Negate to get correction vector (shift needed to align target to ref)
        return round(-shift[0]), round(-shift[1])

    @staticmethod
    def _apply_offset(
        band: np.ndarray,
        x_offset: int,
        y_offset: int,
    ) -> np.ndarray:
        """Apply translation offset to a single band."""
        if x_offset == 0 and y_offset == 0:
            return band

        m = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
        return cv2.warpAffine(
            band.astype(np.float32),
            m,
            (band.shape[1], band.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
