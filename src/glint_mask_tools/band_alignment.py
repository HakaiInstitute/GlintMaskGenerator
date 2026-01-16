"""Band alignment module for multi-band imagery using ECC alignment.

This module provides automatic alignment of image bands from multi-sensor cameras
where each band may have a slight spatial offset due to sensor positioning.
Uses Enhanced Correlation Coefficient (ECC) maximization with Euclidean motion
model (rotation + translation) for robust alignment across spectral bands.

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
    """Transformation matrices for each band relative to reference band (index 0).

    Each matrix is a 2x3 Euclidean transform (rotation + translation) designed
    for use with cv2.WARP_INVERSE_MAP.
    """

    warp_matrices: tuple[np.ndarray, ...]

    @property
    def num_bands(self) -> int:
        """Return the number of bands."""
        return len(self.warp_matrices)

    def has_offset(self) -> bool:
        """Return True if any band has non-identity transformation."""
        identity = np.eye(2, 3, dtype=np.float32)
        return any(not np.allclose(m, identity, atol=0.01) for m in self.warp_matrices)


class BandAligner:
    """Handles per-image band alignment using ECC (Enhanced Correlation Coefficient)."""

    def __init__(self, *, enabled: bool = True) -> None:
        """Create a new BandAligner.

        Parameters
        ----------
        enabled
            Whether alignment is enabled (default True).

        """
        self.enabled = enabled

    def align(self, img: np.ndarray) -> tuple[np.ndarray, BandOffsets | None]:
        """Compute alignment transforms and apply them to align an image.

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

        # Identity transform for reference band
        warp_matrices = [np.eye(2, 3, dtype=np.float32)]

        try:
            for band_idx in range(1, num_bands):
                target_band = img[:, :, band_idx]
                warp_matrix = self._estimate_transform(ref_band, target_band)
                warp_matrices.append(warp_matrix)
        except cv2.error as e:
            logger.warning(f"Band alignment failed for image: {e}")
            return img, None

        offsets = BandOffsets(warp_matrices=tuple(warp_matrices))

        if not offsets.has_offset():
            return img, offsets

        # Apply transforms to align bands
        aligned_bands = []
        for band_idx in range(num_bands):
            band = img[:, :, band_idx]
            warp_matrix = offsets.warp_matrices[band_idx]
            aligned_band = self._apply_transform(band, warp_matrix, inverse=False)
            aligned_bands.append(aligned_band)

        return np.stack(aligned_bands, axis=2), offsets

    def unalign_mask(self, mask: np.ndarray, offsets: BandOffsets | None) -> np.ndarray:
        """Transform a mask back to original (unaligned) coordinate space for each band.

        Takes a 2D union mask (or 3D per-band masks) and creates transformed versions
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
            3D mask array (H, W, num_bands) with each band transformed to its original
            coordinates, or the original mask if no offsets were applied.

        """
        if not self.enabled:
            return mask

        if offsets is None:
            return mask

        if not offsets.has_offset():
            return mask

        # Handle 2D union mask - replicate and transform for each band
        if mask.ndim == 2:  # noqa: PLR2004
            logger.debug("Transforming union mask to per-band coordinates")
            unaligned_masks = []
            for band_idx in range(offsets.num_bands):
                warp_matrix = offsets.warp_matrices[band_idx]
                # Apply inverse transform to move mask back to original band coordinates
                unaligned_mask = self._apply_transform(mask.astype(np.float32), warp_matrix, inverse=True)
                unaligned_masks.append(unaligned_mask)
            return np.stack(unaligned_masks, axis=2)

        # Handle 3D per-band masks
        if mask.shape[2] != offsets.num_bands:
            msg = f"Mask has {mask.shape[2]} bands but alignment computed for {offsets.num_bands} bands"
            raise ValueError(msg)

        logger.debug("Transforming per-band masks to original coordinates")

        unaligned_masks = []
        for band_idx in range(mask.shape[2]):
            mask_band = mask[:, :, band_idx]
            warp_matrix = offsets.warp_matrices[band_idx]
            # Apply inverse transform to move mask back to original band coordinates
            unaligned_mask = self._apply_transform(mask_band.astype(np.float32), warp_matrix, inverse=True)
            unaligned_masks.append(unaligned_mask)

        return np.stack(unaligned_masks, axis=2)

    @staticmethod
    def _estimate_transform(
        ref_band: np.ndarray,
        target_band: np.ndarray,
    ) -> np.ndarray:
        """Estimate Euclidean transform of target band relative to reference using ECC.

        Uses Enhanced Correlation Coefficient (ECC) maximization with Euclidean
        motion model (rotation + translation) for robust alignment across
        spectral bands with different intensity distributions.

        Returns
        -------
        np.ndarray
            2x3 warp matrix for use with cv2.WARP_INVERSE_MAP.

        """
        # Initialize warp matrix as identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # ECC criteria: max 1000 iterations or convergence at 1e-6
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)

        _, warp_matrix = cv2.findTransformECC(
            ref_band.astype(np.float32),
            target_band.astype(np.float32),
            warp_matrix,
            motionType=cv2.MOTION_EUCLIDEAN,
            criteria=criteria,
        )

        return warp_matrix

    @staticmethod
    def _apply_transform(
        band: np.ndarray,
        warp_matrix: np.ndarray,
        *,
        inverse: bool = False,
    ) -> np.ndarray:
        """Apply Euclidean transformation to a single band.

        Parameters
        ----------
        band
            2D array to transform.
        warp_matrix
            2x3 Euclidean warp matrix from ECC estimation.
        inverse
            If False, apply transform with WARP_INVERSE_MAP (aligns band to reference).
            If True, apply transform without WARP_INVERSE_MAP (inverse transformation).

        """
        # Check if identity transform
        identity = np.eye(2, 3, dtype=np.float32)
        if np.allclose(warp_matrix, identity, atol=0.01):
            return band

        flags = cv2.INTER_LINEAR
        if not inverse:
            # Forward alignment: use WARP_INVERSE_MAP as ECC warp matrices expect
            flags |= cv2.WARP_INVERSE_MAP

        return cv2.warpAffine(
            band.astype(np.float32),
            warp_matrix,
            (band.shape[1], band.shape[0]),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
