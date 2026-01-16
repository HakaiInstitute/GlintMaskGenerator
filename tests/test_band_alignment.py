"""Tests for band alignment module.

Created by: Taylor Denouden
Organization: Hakai Institute
"""

import cv2
import numpy as np
import pytest

from glint_mask_tools.band_alignment import BandAligner, BandOffsets


def make_translation_matrix(tx: float, ty: float) -> np.ndarray:
    """Create a 2x3 translation warp matrix."""
    return np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)


def make_identity_offsets(num_bands: int) -> BandOffsets:
    """Create BandOffsets with identity transforms for all bands."""
    matrices = tuple(np.eye(2, 3, dtype=np.float32) for _ in range(num_bands))
    return BandOffsets(warp_matrices=matrices)


def make_translation_offsets(translations: list[tuple[float, float]]) -> BandOffsets:
    """Create BandOffsets with translation-only transforms.

    Parameters
    ----------
    translations
        List of (tx, ty) translation pairs for each band.

    """
    matrices = tuple(make_translation_matrix(tx, ty) for tx, ty in translations)
    return BandOffsets(warp_matrices=matrices)


class TestBandOffsets:
    """Tests for BandOffsets dataclass."""

    def test_has_offset_with_identity(self):
        """Identity transforms should return False for has_offset."""
        offsets = make_identity_offsets(3)
        assert not offsets.has_offset()

    def test_has_offset_with_translation(self):
        """Non-identity transform should return True."""
        offsets = make_translation_offsets([(0, 0), (5, 0), (0, 0)])
        assert offsets.has_offset()

    def test_has_offset_with_small_rotation(self):
        """Transform with small rotation should return True."""
        # Small rotation (1 degree)
        theta = np.radians(1)
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]],
            dtype=np.float32,
        )
        offsets = BandOffsets(
            warp_matrices=(
                np.eye(2, 3, dtype=np.float32),
                rotation_matrix,
            )
        )
        assert offsets.has_offset()

    def test_num_bands(self):
        """num_bands should return correct count."""
        offsets = make_identity_offsets(5)
        assert offsets.num_bands == 5


class TestBandAlignerDisabled:
    """Tests for disabled BandAligner."""

    def test_disabled_aligner_returns_input_unchanged(self):
        """Disabled aligner should return input unchanged."""
        aligner = BandAligner(enabled=False)
        rng = np.random.default_rng(42)
        img = rng.random((100, 100, 5)).astype(np.float32)
        result, offsets = aligner.align(img)
        assert np.array_equal(result, img)
        assert offsets is None

    def test_disabled_aligner_unalign_returns_input(self):
        """Disabled aligner should return mask unchanged in unalign."""
        aligner = BandAligner(enabled=False)
        masks = np.ones((50, 50, 3), dtype=np.uint8)
        offsets = make_translation_offsets([(0, 0), (1, 0), (2, 0)])
        result = aligner.unalign_mask(masks, offsets)
        assert np.array_equal(result, masks)


class TestBandAlignerTransformEstimation:
    """Tests for transform estimation using ECC."""

    def test_estimate_transform_identical_bands(self):
        """Identical bands should have identity transform."""
        rng = np.random.default_rng(42)
        band = rng.random((100, 100)).astype(np.float32)
        warp = BandAligner._estimate_transform(band, band)
        identity = np.eye(2, 3, dtype=np.float32)
        assert np.allclose(warp, identity, atol=0.1)

    def test_estimate_transform_shifted_band(self):
        """ECC should detect known shifts."""
        # Create reference band with distinct pattern
        ref = np.zeros((100, 100), dtype=np.float32)
        ref[40:60, 40:60] = 1.0  # Bright square in center

        # Create shifted band (shifted by +5 in x, +3 in y)
        target = np.zeros((100, 100), dtype=np.float32)
        target[43:63, 45:65] = 1.0

        warp = BandAligner._estimate_transform(ref, target)

        # Extract translation from warp matrix
        tx, ty = warp[0, 2], warp[1, 2]
        # The warp should represent the shift needed (around +5, +3 for WARP_INVERSE_MAP usage)
        # Allow tolerance for ECC precision
        assert abs(tx - 5) <= 2
        assert abs(ty - 3) <= 2


class TestBandAlignerApplyTransform:
    """Tests for transform application."""

    def test_apply_transform_identity(self):
        """Identity transform should return unchanged band."""
        rng = np.random.default_rng(42)
        band = rng.random((50, 50)).astype(np.float32)
        identity = np.eye(2, 3, dtype=np.float32)
        result = BandAligner._apply_transform(band, identity)
        assert np.array_equal(result, band)

    def test_apply_transform_translates_image(self):
        """Translation transform should translate the image."""
        band = np.zeros((50, 50), dtype=np.float32)
        band[20:30, 20:30] = 1.0  # Square in center

        # Create translation matrix (tx=5, ty=3)
        warp = make_translation_matrix(5, 3)
        result = BandAligner._apply_transform(band, warp, inverse=False)

        # With WARP_INVERSE_MAP (inverse=False): dst(x,y) = src(x+tx, y+ty)
        # So the image content shifts LEFT by tx and UP by ty
        # Original position should have less content
        assert result[20:30, 20:30].sum() < 100
        # New position (shifted left/up) should have the square
        assert result[17:27, 15:25].sum() > 50


class TestBandAlignerAlign:
    """Tests for per-image alignment."""

    def test_align_2d_image_returns_unchanged(self):
        """2D image should return unchanged with None offsets."""
        aligner = BandAligner()
        rng = np.random.default_rng(42)
        img = rng.random((100, 100)).astype(np.float32)
        result, offsets = aligner.align(img)
        assert np.array_equal(result, img)
        assert offsets is None

    def test_align_computes_transforms_per_image(self):
        """Alignment should compute and apply transforms for each image."""
        aligner = BandAligner()

        # Create image with known offset between bands
        base = np.zeros((100, 100), dtype=np.float32)
        base[40:60, 40:60] = 1.0

        shifted = np.zeros((100, 100), dtype=np.float32)
        shifted[43:63, 45:65] = 1.0  # shifted by +5 x, +3 y

        img = np.stack([base, shifted], axis=2)
        _, offsets = aligner.align(img)

        assert offsets is not None
        assert offsets.num_bands == 2

        # Reference band should have identity transform
        identity = np.eye(2, 3, dtype=np.float32)
        assert np.allclose(offsets.warp_matrices[0], identity, atol=0.1)

        # Band 1 should have non-identity transform
        assert not np.allclose(offsets.warp_matrices[1], identity, atol=0.5)

    def test_align_identical_bands_identity_transform(self):
        """Identical bands should produce identity transforms."""
        aligner = BandAligner()
        rng = np.random.default_rng(42)
        base = rng.random((100, 100)).astype(np.float32)
        img = np.stack([base, base, base], axis=2)

        aligned, offsets = aligner.align(img)

        assert offsets is not None
        assert not offsets.has_offset()
        # Image should be unchanged when no offsets needed
        assert np.allclose(aligned, img, atol=0.01)

    def test_align_returns_aligned_image(self):
        """Alignment should return properly aligned image."""
        aligner = BandAligner()

        # Create image with offset
        base = np.zeros((100, 100), dtype=np.float32)
        base[40:60, 40:60] = 1.0

        shifted = np.zeros((100, 100), dtype=np.float32)
        shifted[40:60, 50:70] = 1.0  # shifted by +10 in x

        img = np.stack([base, shifted], axis=2)
        aligned, offsets = aligner.align(img)

        assert offsets is not None
        assert offsets.has_offset()
        # After alignment, band 1 should be shifted to match band 0
        assert not np.array_equal(aligned[:, :, 1], img[:, :, 1])


class TestBandAlignerUnalignMask:
    """Tests for unaligning masks back to original coordinates."""

    def test_unalign_disabled_returns_input(self):
        """Disabled aligner should return masks unchanged."""
        aligner = BandAligner(enabled=False)
        masks = np.ones((50, 50, 3), dtype=np.uint8)
        offsets = make_translation_offsets([(0, 0), (1, 0), (2, 0)])
        result = aligner.unalign_mask(masks, offsets)
        assert np.array_equal(result, masks)

    def test_unalign_none_offsets_returns_input(self):
        """None offsets should return masks unchanged."""
        aligner = BandAligner()
        masks = np.ones((50, 50, 3), dtype=np.uint8)
        result = aligner.unalign_mask(masks, None)
        assert np.array_equal(result, masks)

    def test_unalign_identity_transforms_returns_input(self):
        """Identity transforms should return masks unchanged."""
        aligner = BandAligner()
        offsets = make_identity_offsets(3)
        masks = np.ones((50, 50, 3), dtype=np.uint8)
        result = aligner.unalign_mask(masks, offsets)
        assert np.array_equal(result, masks)

    def test_unalign_2d_union_mask_creates_shifted_per_band(self):
        """2D union mask should be replicated and shifted for each band."""
        aligner = BandAligner()
        # Translations in warp matrix. During alignment (WARP_INVERSE_MAP):
        # - tx=20 shifts image LEFT by 20
        # - tx=-10 shifts image RIGHT by 10
        # During unalignment (no WARP_INVERSE_MAP):
        # - tx=20 shifts mask RIGHT by 20 (opposite direction)
        # - tx=-10 shifts mask LEFT by 10 (opposite direction)
        offsets = make_translation_offsets([(0, 0), (20, 0), (-10, 0)])

        # Create 2D union mask with a square
        mask_2d = np.zeros((100, 100), dtype=np.float32)
        mask_2d[40:60, 40:60] = 255

        result = aligner.unalign_mask(mask_2d, offsets)

        # Should now be 3D with one mask per band
        assert result.ndim == 3
        assert result.shape[2] == 3

        # Band 0: no shift, square at cols 40-60
        _rows0, cols0 = np.where(result[:, :, 0] > 0)
        assert cols0.min() == 40
        assert cols0.max() == 59

        # Band 1: unalign shifts RIGHT by 20, square at cols 60-80
        _rows1, cols1 = np.where(result[:, :, 1] > 0)
        assert cols1.min() == 60
        assert cols1.max() == 79

        # Band 2: unalign shifts LEFT by 10, square at cols 30-50
        _rows2, cols2 = np.where(result[:, :, 2] > 0)
        assert cols2.min() == 30
        assert cols2.max() == 49

    def test_unalign_3d_applies_inverse_transforms(self):
        """Unalign should apply inverse of alignment transforms to 3D masks."""
        aligner = BandAligner()
        offsets = make_translation_offsets([(0, 0), (5, 2), (-3, -1)])

        # Create masks with a pattern
        masks = np.zeros((50, 50, 3), dtype=np.float32)
        masks[20:30, 20:30, :] = 255  # Square in center of all bands

        result = aligner.unalign_mask(masks, offsets)

        # Band 0 (identity) should be unchanged
        assert np.array_equal(result[:, :, 0], masks[:, :, 0])

        # Band 1 (offset 5, 2) unalign shifts mask by (-5, -2)
        assert result[20:30, 20:30, 1].sum() < 255 * 100  # Original position has less

        # Band 2 (offset -3, -1) unalign shifts mask by (3, 1)
        assert result[20:30, 20:30, 2].sum() < 255 * 100  # Original position has less

    def test_unalign_band_count_mismatch_raises(self):
        """Mismatched band count should raise ValueError."""
        aligner = BandAligner()
        offsets = make_translation_offsets([(0, 0), (1, 0), (2, 0)])
        masks = np.ones((50, 50, 5), dtype=np.float32)  # 5 bands, offsets for 3

        with pytest.raises(ValueError, match="computed for"):
            aligner.unalign_mask(masks, offsets)


class TestBandAlignerIntegration:
    """Integration tests for complete alignment workflow."""

    def test_full_align_and_unalign_workflow(self):
        """Test complete workflow: align image then unalign mask."""
        aligner = BandAligner()

        # Create image with known offset
        base = np.zeros((100, 100), dtype=np.float32)
        base[40:60, 40:60] = 1.0

        shifted = np.zeros((100, 100), dtype=np.float32)
        shifted[40 + 2 : 60 + 2, 40 + 3 : 60 + 3] = 1.0

        img = np.stack([base, shifted], axis=2)

        # Align the image
        _, offsets = aligner.align(img)

        assert offsets is not None
        assert offsets.has_offset()

        # Create a mask in aligned space
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[45:55, 45:55] = 1  # Glint detected at center

        # Unalign the mask
        result = aligner.unalign_mask(mask, offsets)

        assert result.ndim == 3
        assert result.shape == (100, 100, 2)

        # Each band should have the mask at its original coordinate position
        for i in range(2):
            rows, _cols = np.where(result[:, :, i] > 0)
            assert len(rows) > 0, f"Band {i} should have masked pixels"

    def test_align_then_unalign_restores_original_coordinates(self):
        """Aligning and then unaligning masks should restore original coordinate system."""
        aligner = BandAligner()

        # Create a mask in original coordinates with distinct patterns per band
        original_mask = np.zeros((100, 100, 3), dtype=np.float32)
        original_mask[40:50, 40:50, 0] = 255  # Band 0 pattern
        original_mask[30:40, 60:70, 1] = 255  # Band 1 pattern
        original_mask[70:80, 20:30, 2] = 255  # Band 2 pattern

        # Manually set offsets for this test
        offsets = make_translation_offsets([(0, 0), (3, 1), (-2, 2)])

        # Simulate aligning, processing, then unaligning
        # First "align" the mask (shift to aligned space)
        aligned = aligner.align(original_mask)

        # Then unalign (shift back)
        unaligned = aligner.unalign_mask(aligned[0].astype(np.float32), offsets)

        # Band 0 should match exactly (no offset)
        assert np.allclose(unaligned[:, :, 0], original_mask[:, :, 0], atol=1)

    def test_union_mask_workflow_with_alignment(self):
        """Test complete workflow: union mask with per-band shifting."""
        aligner = BandAligner()
        # During unalignment (no WARP_INVERSE_MAP):
        # - tx=10 shifts mask RIGHT by 10
        # - tx=-5 shifts mask LEFT by 5
        offsets = make_translation_offsets([(0, 0), (10, 5), (-5, -3)])

        # Simulate a 2D union mask (glint detected at one location in aligned space)
        union_mask = np.zeros((100, 100), dtype=np.float32)
        union_mask[45:55, 45:55] = 1  # Glint detected at center

        # Unalign creates per-band shifted versions
        result = aligner.unalign_mask(union_mask, offsets)

        assert result.ndim == 3
        assert result.shape == (100, 100, 3)

        # Each band should have the mask at a different position
        for i in range(3):
            rows, _cols = np.where(result[:, :, i] > 0)
            assert len(rows) > 0, f"Band {i} should have masked pixels"

        # Verify band 0 is at original position (no shift)
        rows0, _cols0 = np.where(result[:, :, 0] > 0)
        assert 45 <= rows0.min() <= 46
        assert 54 <= rows0.max() <= 55

        # Band 1: unalign shifts RIGHT by 10 (mask at cols 55-65)
        _rows1, cols1 = np.where(result[:, :, 1] > 0)
        assert cols1.min() > 50  # Shifted right

        # Band 2: unalign shifts LEFT by 5 (mask at cols 40-50)
        _rows2, cols2 = np.where(result[:, :, 2] > 0)
        assert cols2.max() < 55  # Shifted left

    def test_euclidean_rotation_alignment(self):
        """Test that rotation is detected and applied correctly."""
        aligner = BandAligner()

        # Create reference with a distinctive pattern
        ref = np.zeros((100, 100), dtype=np.float32)
        # Create an L-shaped pattern to detect rotation
        ref[30:70, 40:50] = 1.0  # Vertical bar
        ref[60:70, 40:70] = 1.0  # Horizontal bar

        # Create slightly rotated version
        theta = np.radians(2)  # 2 degree rotation
        center = (50, 50)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), center[0] * (1 - np.cos(theta)) + center[1] * np.sin(theta)],
                [np.sin(theta), np.cos(theta), center[1] * (1 - np.cos(theta)) - center[0] * np.sin(theta)],
            ],
            dtype=np.float32,
        )

        rotated = cv2.warpAffine(ref, rotation_matrix, (100, 100), flags=cv2.INTER_LINEAR)

        img = np.stack([ref, rotated], axis=2)
        _, offsets = aligner.align(img)

        assert offsets is not None
        # The warp matrix for band 1 should have rotation components
        warp = offsets.warp_matrices[1]
        # Check that it's not pure translation (has rotation)
        # For pure translation: warp[0,0] == 1, warp[0,1] == 0, warp[1,0] == 0, warp[1,1] == 1
        # For rotation: warp[0,0] = cos(theta), warp[0,1] = -sin(theta), etc.
        is_pure_translation = (
            np.isclose(warp[0, 0], 1, atol=0.01)
            and np.isclose(warp[0, 1], 0, atol=0.01)
            and np.isclose(warp[1, 0], 0, atol=0.01)
            and np.isclose(warp[1, 1], 1, atol=0.01)
        )
        # With MOTION_EUCLIDEAN, rotation should be detected
        assert not is_pure_translation or offsets.has_offset()


class TestBandAlignerCaching:
    """Tests for warp matrix caching optimization."""

    def test_cache_initialized_empty(self):
        """Test that cache starts empty."""
        aligner = BandAligner()
        assert len(aligner._prev_warp_matrices) == 0

    def test_cache_populated_after_alignment(self):
        """Test that cache is populated after processing an image."""
        aligner = BandAligner()

        # Create a 3-band image with known shift
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80, 0] = 255  # Reference band
        img[25:85, 25:85, 1] = 255  # Band 1 shifted
        img[22:82, 22:82, 2] = 255  # Band 2 shifted differently

        _, _offsets = aligner.align(img)

        # Cache should have entries for bands 1 and 2 (not 0, which is reference)
        assert len(aligner._prev_warp_matrices) == 2
        assert 1 in aligner._prev_warp_matrices
        assert 2 in aligner._prev_warp_matrices
        assert 0 not in aligner._prev_warp_matrices

    def test_cache_used_for_subsequent_images(self):
        """Test that cached matrices are used as initial guesses for next images."""
        aligner = BandAligner()

        # Create first image with shift
        img1 = np.zeros((100, 100, 2), dtype=np.uint8)
        img1[20:80, 20:80, 0] = 255
        img1[25:85, 25:85, 1] = 255  # 5-pixel shift

        # Process first image to populate cache
        _, offsets1 = aligner.align(img1)
        cached_warp = aligner._prev_warp_matrices[1].copy()

        # Create second image with different shift
        img2 = np.zeros((100, 100, 2), dtype=np.uint8)
        img2[30:90, 30:90, 0] = 255
        img2[40:100, 40:100, 1] = 255  # 10-pixel shift (different from first)

        # Process second image - should use cached warp as starting point
        _, offsets2 = aligner.align(img2)

        # Both should have computed valid transforms
        assert offsets1 is not None
        assert offsets2 is not None
        assert offsets1.has_offset()
        assert offsets2.has_offset()

        # Cache should be updated with new warp (different shift amount)
        assert not np.allclose(aligner._prev_warp_matrices[1], cached_warp, atol=0.5)

    def test_cache_persists_across_multiple_images(self):
        """Test that cache persists and updates across a sequence of images."""
        aligner = BandAligner()

        # Process a sequence of 5 images with similar band alignment
        for i in range(5):
            offset = 20 + i  # Gradually changing shift
            img = np.zeros((100, 100, 2), dtype=np.uint8)
            img[offset : offset + 60, offset : offset + 60, 0] = 255
            img[offset + 5 : offset + 65, offset + 5 : offset + 65, 1] = 255

            _, offsets = aligner.align(img)

            # Cache should always have band 1
            assert 1 in aligner._prev_warp_matrices
            # Each image should compute valid offsets
            assert offsets is not None

    def test_cache_independent_per_band(self):
        """Test that different bands have independent cached matrices."""
        aligner = BandAligner()

        # Create 4-band image with different shifts per band
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[20:80, 20:80, 0] = 255  # Reference
        img[22:82, 22:82, 1] = 255  # Small shift
        img[30:90, 30:90, 2] = 255  # Larger shift
        img[25:85, 25:85, 3] = 255  # Medium shift

        _, _offsets = aligner.align(img)

        # Should have independent cache entries for bands 1, 2, 3
        assert len(aligner._prev_warp_matrices) == 3

        # Each cached matrix should be different
        warp1 = aligner._prev_warp_matrices[1]
        warp2 = aligner._prev_warp_matrices[2]
        warp3 = aligner._prev_warp_matrices[3]

        # Matrices should differ (different shifts)
        assert not np.array_equal(warp1, warp2)
        assert not np.array_equal(warp2, warp3)
        assert not np.array_equal(warp1, warp3)

    def test_fallback_to_identity_on_convergence_failure(self):
        """Test that ECC falls back to identity matrix when cached guess fails."""
        import unittest.mock  # noqa: PLC0415

        aligner = BandAligner()

        # Create first image with a specific pattern
        img1 = np.zeros((100, 100, 2), dtype=np.uint8)
        img1[20:80, 20:80, 0] = 255
        img1[25:85, 25:85, 1] = 255

        # Process first image to populate cache
        _, offsets1 = aligner.align(img1)
        assert offsets1 is not None

        # Create second image where we'll force the first ECC call to fail
        # but the second call (with identity) will succeed
        img2 = np.zeros((100, 100, 2), dtype=np.uint8)
        img2[30:90, 30:90, 0] = 255
        img2[38:98, 38:98, 1] = 255  # Larger offset

        # Mock cv2.findTransformECC to fail first time, succeed second time
        original_ecc = cv2.findTransformECC
        call_count = [0]

        def mock_ecc(*args, **kwargs):
            call_count[0] += 1
            # Fail on first call (with cached warp), succeed on second (with identity)
            if call_count[0] == 1:
                msg = "Mock convergence failure"
                raise cv2.error(msg)
            return original_ecc(*args, **kwargs)

        with unittest.mock.patch("cv2.findTransformECC", side_effect=mock_ecc):
            _, offsets2 = aligner.align(img2)

        # Should successfully return offsets after fallback to identity
        assert offsets2 is not None
        # Should have called ECC twice (once with cache, once with identity)
        assert call_count[0] == 2

    def test_cache_cleared_on_alignment_failure(self):
        """Test that cache is cleared when alignment fails completely."""
        import unittest.mock  # noqa: PLC0415

        aligner = BandAligner()

        # Create and process first image to populate cache
        img1 = np.zeros((100, 100, 2), dtype=np.uint8)
        img1[20:80, 20:80, 0] = 255
        img1[25:85, 25:85, 1] = 255

        _, offsets1 = aligner.align(img1)
        assert offsets1 is not None
        # Cache should be populated
        assert len(aligner._prev_warp_matrices) == 1

        # Create second image and force both attempts to fail
        img2 = np.zeros((100, 100, 2), dtype=np.uint8)
        img2[30:90, 30:90, 0] = 255
        img2[38:98, 38:98, 1] = 255

        # Mock to always fail
        with unittest.mock.patch("cv2.findTransformECC", side_effect=cv2.error("Mock failure")):
            _, offsets2 = aligner.align(img2)

        # Alignment should have failed
        assert offsets2 is None
        # Cache should be cleared after failure
        assert len(aligner._prev_warp_matrices) == 0


class TestBandAlignerParallel:
    """Tests for parallel band processing."""

    def test_parallel_processing_enabled_by_default(self):
        """Test that parallel processing is used by default for multi-band images."""
        aligner = BandAligner()
        # Default max_workers should be None
        assert aligner.max_workers is None

    def test_parallel_processing_produces_same_results_as_sequential(self):
        """Test that parallel and sequential processing produce identical results."""
        # Create a 5-band image with known shifts
        img = np.zeros((100, 100, 5), dtype=np.uint8)
        img[20:80, 20:80, 0] = 255  # Reference
        for band_idx in range(1, 5):
            shift = band_idx * 2
            img[20 + shift : 80 + shift, 20 + shift : 80 + shift, band_idx] = 255

        # Process with parallel
        aligner_parallel = BandAligner(max_workers=4)
        aligned_parallel, offsets_parallel = aligner_parallel.align(img)

        # Process sequentially
        aligner_sequential = BandAligner(max_workers=1)
        aligned_sequential, offsets_sequential = aligner_sequential.align(img)

        # Results should be identical
        assert offsets_parallel is not None
        assert offsets_sequential is not None
        assert offsets_parallel.num_bands == offsets_sequential.num_bands

        # Compare warp matrices (should be very close, allowing for numerical precision)
        for i in range(offsets_parallel.num_bands):
            assert np.allclose(offsets_parallel.warp_matrices[i], offsets_sequential.warp_matrices[i], atol=1e-4)

        # Compare aligned images
        assert np.allclose(aligned_parallel, aligned_sequential, atol=1.0)

    def test_parallel_disabled_for_few_bands(self):
        """Test that parallel processing is skipped for 2-band images."""
        # Create a 2-band image
        img = np.zeros((100, 100, 2), dtype=np.uint8)
        img[20:80, 20:80, 0] = 255
        img[25:85, 25:85, 1] = 255

        # Even with max_workers > 1, should use sequential for only 2 bands
        aligner = BandAligner(max_workers=4)
        _, offsets = aligner.align(img)

        # Should still work correctly
        assert offsets is not None

    def test_explicit_max_workers(self):
        """Test setting explicit max_workers parameter."""
        aligner = BandAligner(max_workers=2)
        assert aligner.max_workers == 2

        # Create and process a multi-band image
        img = np.zeros((100, 100, 5), dtype=np.uint8)
        img[20:80, 20:80, 0] = 255
        for band_idx in range(1, 5):
            shift = band_idx * 3
            img[20 + shift : 80 + shift, 20 + shift : 80 + shift, band_idx] = 255

        _, offsets = aligner.align(img)
        assert offsets is not None
        assert offsets.num_bands == 5

    def test_single_worker_equals_sequential(self):
        """Test that max_workers=1 uses sequential processing path."""
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[20:80, 20:80, 0] = 255
        for band_idx in range(1, 4):
            shift = band_idx * 4
            img[20 + shift : 80 + shift, 20 + shift : 80 + shift, band_idx] = 255

        aligner = BandAligner(max_workers=1)
        _, offsets = aligner.align(img)

        assert offsets is not None
        assert offsets.num_bands == 4
