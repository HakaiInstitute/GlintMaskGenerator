"""Band alignment module for multi-band imagery using phase correlation.

This module provides automatic alignment of image bands from multi-sensor cameras
where each band may have a slight spatial offset due to sensor positioning.

Created by: Taylor Denouden
Organization: Hakai Institute
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger


@dataclass
class BandOffsets:
    """Offsets for each band relative to reference band (index 0)."""

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
    """Handles per-image band alignment using phase correlation."""

    def __init__(self, *, enabled: bool = True) -> None:
        """Create a new BandAligner.

        Parameters
        ----------
        enabled
            Whether alignment is enabled (default True).

        """
        self.enabled = enabled

    def align(self, img: np.ndarray) -> tuple[np.ndarray, BandOffsets | None]:
        """Compute alignment offsets and apply them to align an image.

        Parameters
        ----------
        img
            Image array with shape (H, W, num_bands).

        Returns
        -------
        tuple[np.ndarray, BandOffsets | None]
            Tuple of (aligned_image, offsets). Offsets is None if alignment
            is disabled or image is not multi-band.

        """
        if not self.enabled:
            return img, None

        if img.ndim != 3:  # noqa: PLR2004
            return img, None

        num_bands = img.shape[2]
        ref_band = img[:, :, 0]

        x_offs = [0]
        y_offs = [0]

        try:
            for band_idx in range(1, num_bands):
                target_band = img[:, :, band_idx]
                x_off, y_off = self._estimate_offset(ref_band, target_band)
                x_offs.append(x_off)
                y_offs.append(y_off)
        except cv2.error as e:
            logger.warning(f"Band alignment failed for image: {e}")
            return img, None

        offsets = BandOffsets(x_offsets=tuple(x_offs), y_offsets=tuple(y_offs))

        if not offsets.has_offset():
            return img, offsets

        # Apply offsets to align bands
        aligned_bands = []
        for band_idx in range(num_bands):
            band = img[:, :, band_idx]
            x_off = offsets.x_offsets[band_idx]
            y_off = offsets.y_offsets[band_idx]
            aligned_band = self._apply_offset(band, x_off, y_off)
            aligned_bands.append(aligned_band)

        return np.stack(aligned_bands, axis=2), offsets

    def unalign_mask(self, mask: np.ndarray, offsets: BandOffsets | None) -> np.ndarray:
        """Shift a mask back to original (unaligned) coordinate space for each band.

        Takes a 2D union mask (or 3D per-band masks) and creates shifted versions
        for each band so the mask aligns with the original unaligned band images.

        Parameters
        ----------
        mask
            Mask array with shape (H, W) for union mask or (H, W, num_bands) for per-band.
        offsets
            The offsets used during alignment. If None, mask is returned unchanged.

        Returns
        -------
        np.ndarray
            3D mask array (H, W, num_bands) with each band shifted to its original coordinates,
            or the original mask if no offsets were applied.

        """
        if not self.enabled:
            return mask

        if offsets is None:
            return mask

        if not offsets.has_offset():
            return mask

        # Handle 2D union mask - replicate and shift for each band
        if mask.ndim == 2:  # noqa: PLR2004
            logger.debug(f"Shifting union mask to per-band coordinates: x={offsets.x_offsets}, y={offsets.y_offsets}")
            unaligned_masks = []
            for band_idx in range(offsets.num_bands):
                x_off = -offsets.x_offsets[band_idx]
                y_off = -offsets.y_offsets[band_idx]
                unaligned_mask = self._apply_offset(mask.astype(np.float32), x_off, y_off)
                unaligned_masks.append(unaligned_mask)
            return np.stack(unaligned_masks, axis=2)

        # Handle 3D per-band masks
        if mask.shape[2] != offsets.num_bands:
            msg = f"Mask has {mask.shape[2]} bands but alignment computed for {offsets.num_bands} bands"
            raise ValueError(msg)

        logger.debug(f"Shifting per-band masks to original coordinates: x={offsets.x_offsets}, y={offsets.y_offsets}")

        unaligned_masks = []
        for band_idx in range(mask.shape[2]):
            mask_band = mask[:, :, band_idx]
            x_off = -offsets.x_offsets[band_idx]
            y_off = -offsets.y_offsets[band_idx]
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
