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


def _compose_warp_matrices(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compose two 2x3 Euclidean warp matrices.

    Given transforms m1 (A → B) and m2 (B → C), returns the combined
    transform (A → C).

    Parameters
    ----------
    m1
        2x3 warp matrix for first transformation.
    m2
        2x3 warp matrix for second transformation.

    Returns
    -------
    np.ndarray
        2x3 composed warp matrix.

    """
    # Convert to 3x3 homogeneous matrices
    m1_h = np.vstack([m1, [0, 0, 1]])
    m2_h = np.vstack([m2, [0, 0, 1]])
    # Compose: m2 @ m1 applies m1 first, then m2
    composed = m2_h @ m1_h
    # Return as 2x3
    return composed[:2, :].astype(np.float32)


class BandAligner:
    """Handles per-image band alignment using ECC (Enhanced Correlation Coefficient)."""

    def __init__(self, *, enabled: bool = True, max_workers: int | None = None) -> None:
        """Create a new BandAligner.

        Parameters
        ----------
        enabled
            Whether alignment is enabled (default True).
        max_workers
            Deprecated parameter, kept for backwards compatibility.
            Previously controlled parallel band processing, but chained
            alignment requires sequential processing.

        """
        self.enabled = enabled
        self.max_workers = max_workers  # Kept for backwards compatibility
        # Cache of previous local warp matrices per band for faster convergence
        # Each entry maps band_idx -> local transform (band_idx -> band_idx-1)
        self._prev_warp_matrices: dict[int, np.ndarray] = {}

    def align(self, img: np.ndarray) -> tuple[np.ndarray, BandOffsets | None]:
        """Compute alignment transforms and apply them to align an image.

        Uses chained/sequential alignment where each band is aligned to its
        previous neighbor (band 1 to band 0, band 2 to band 1, etc.). This
        improves alignment success for bands that are spectrally distant from
        the reference, since adjacent bands typically share more similar features.

        The final warp matrices are composed so each represents the cumulative
        transform from that band to band 0's coordinate space.

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

        # Identity transform for reference band (band 0)
        cumulative_warp_matrices = [np.eye(2, 3, dtype=np.float32)]

        try:
            # Sequential/chained alignment: each band aligned to its previous neighbor
            # This improves success rate for spectrally distant bands (e.g., NIR to Blue)
            # since adjacent bands share more similar features
            cumulative_warp = np.eye(2, 3, dtype=np.float32)

            for band_idx in range(1, num_bands):
                ref_band = img[:, :, band_idx - 1]
                target_band = img[:, :, band_idx]

                # Use previous local warp matrix as initial guess if available
                initial_warp = self._prev_warp_matrices.get(band_idx)
                local_warp = self._estimate_transform(ref_band, target_band, initial_warp)

                # Cache the local transform for next image
                self._prev_warp_matrices[band_idx] = local_warp

                # Compose with previous cumulative transform to get band → band 0
                cumulative_warp = _compose_warp_matrices(local_warp, cumulative_warp)
                cumulative_warp_matrices.append(cumulative_warp.copy())

        except cv2.error as e:
            logger.warning(f"Band alignment failed for image: {e}")
            # Clear cache on failure to prevent bad cached values from persisting
            self._prev_warp_matrices.clear()
            return img, None

        offsets = BandOffsets(warp_matrices=tuple(cumulative_warp_matrices))

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
        initial_warp: np.ndarray | None = None,
    ) -> np.ndarray:
        """Estimate Euclidean transform of target band relative to reference using ECC.

        Uses Enhanced Correlation Coefficient (ECC) maximization with Euclidean
        motion model (rotation + translation) for robust alignment across
        spectral bands with different intensity distributions.

        Parameters
        ----------
        ref_band
            Reference band to align to.
        target_band
            Band to be aligned.
        initial_warp
            Optional initial guess for warp matrix. Using the warp matrix from
            the previous image of the same band can significantly speed up convergence
            for sequential images with similar alignment parameters. If ECC fails to
            converge with the initial guess, it will automatically retry with an
            identity matrix.

        Returns
        -------
        np.ndarray
            2x3 warp matrix for use with cv2.WARP_INVERSE_MAP.

        """
        # Initialize warp matrix from previous result or as identity
        warp_matrix = initial_warp.copy() if initial_warp is not None else np.eye(2, 3, dtype=np.float32)

        # Use fewer iterations when we have a good initial guess from cache
        # This significantly speeds up convergence for sequential images
        max_iterations = 400 if initial_warp is not None else 1000
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, 1e-6)

        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_band.astype(np.float32),
                target_band.astype(np.float32),
                warp_matrix,
                motionType=cv2.MOTION_EUCLIDEAN,
                criteria=criteria,
            )
        except cv2.error as e:
            # If we had an initial guess and it failed, retry with identity matrix
            if initial_warp is not None:
                logger.debug(f"ECC failed with cached initial guess, retrying with identity: {e}")
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                try:
                    _, warp_matrix = cv2.findTransformECC(
                        ref_band.astype(np.float32),
                        target_band.astype(np.float32),
                        warp_matrix,
                        motionType=cv2.MOTION_EUCLIDEAN,
                        criteria=criteria,
                    )
                except cv2.error as retry_error:
                    # Retry with identity also failed, re-raise the original error
                    # This ensures we get the same failure behavior as before caching
                    logger.debug(f"ECC retry with identity also failed: {retry_error}")
                    raise e from retry_error
            else:
                # No initial guess was used, so re-raise the error
                raise

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
