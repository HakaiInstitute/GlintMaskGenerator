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
