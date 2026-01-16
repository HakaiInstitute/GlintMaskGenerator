"""Tests for band alignment module.

Created by: Taylor Denouden
Organization: Hakai Institute
"""

import numpy as np
import pytest

from glint_mask_tools.band_alignment import BandAligner, BandOffsets


class TestBandOffsets:
    """Tests for BandOffsets dataclass."""

    def test_has_offset_with_zeros(self):
        """Zero offsets should return False for has_offset."""
        offsets = BandOffsets(x_offsets=(0, 0, 0), y_offsets=(0, 0, 0))
        assert not offsets.has_offset()

    def test_has_offset_with_nonzero_x(self):
        """Non-zero x offset should return True."""
        offsets = BandOffsets(x_offsets=(0, 1, 0), y_offsets=(0, 0, 0))
        assert offsets.has_offset()

    def test_has_offset_with_nonzero_y(self):
        """Non-zero y offset should return True."""
        offsets = BandOffsets(x_offsets=(0, 0, 0), y_offsets=(0, 0, -1))
        assert offsets.has_offset()

    def test_num_bands(self):
        """num_bands should return correct count."""
        offsets = BandOffsets(x_offsets=(0, 1, 2, 3, 4), y_offsets=(0, 0, 0, 0, 0))
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
        offsets = BandOffsets(x_offsets=(0, 1, 2), y_offsets=(0, 0, 0))
        result = aligner.unalign_mask(masks, offsets)
        assert np.array_equal(result, masks)


class TestBandAlignerOffsetEstimation:
    """Tests for offset estimation using phase correlation."""

    def test_estimate_offset_identical_bands(self):
        """Identical bands should have zero offset."""
        rng = np.random.default_rng(42)
        band = rng.random((100, 100)).astype(np.float64)
        x, y = BandAligner._estimate_offset(band, band)
        assert x == 0
        assert y == 0

    def test_estimate_offset_shifted_band(self):
        """Phase correlation should detect known shifts."""
        # Create reference band with distinct pattern
        ref = np.zeros((100, 100), dtype=np.float64)
        ref[40:60, 40:60] = 1.0  # Bright square in center

        # Create shifted band (shifted by +5 in x, +3 in y)
        target = np.zeros((100, 100), dtype=np.float64)
        target[43:63, 45:65] = 1.0

        x, y = BandAligner._estimate_offset(ref, target)
        # Offset to align target TO ref is negative of the shift
        # Allow tolerance of 1 pixel for phase correlation precision
        assert abs(x - (-5)) <= 1
        assert abs(y - (-3)) <= 1


class TestBandAlignerApplyOffset:
    """Tests for offset application."""

    def test_apply_offset_no_change(self):
        """Zero offset should return unchanged band."""
        rng = np.random.default_rng(42)
        band = rng.random((50, 50)).astype(np.float32)
        result = BandAligner._apply_offset(band, 0, 0)
        assert np.array_equal(result, band)

    def test_apply_offset_translates_image(self):
        """Non-zero offset should translate the image."""
        band = np.zeros((50, 50), dtype=np.float32)
        band[20:30, 20:30] = 1.0  # Square in center

        result = BandAligner._apply_offset(band, 5, 3)

        # Original position should be empty (filled with border value 0)
        assert result[20:30, 20:30].sum() < 100  # Not all 1s anymore
        # New position should have the square (shifted by +5 x, +3 y)
        assert result[23:33, 25:35].sum() > 50


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

    def test_align_computes_offsets_per_image(self):
        """Alignment should compute and apply offsets for each image."""
        aligner = BandAligner()

        # Create image with known offset between bands
        base = np.zeros((100, 100), dtype=np.float64)
        base[40:60, 40:60] = 1.0

        shifted = np.zeros((100, 100), dtype=np.float64)
        shifted[43:63, 45:65] = 1.0  # shifted by +5 x, +3 y

        img = np.stack([base, shifted], axis=2)
        aligned, offsets = aligner.align(img)

        assert offsets is not None
        assert offsets.num_bands == 2
        assert offsets.x_offsets[0] == 0  # Reference band has no offset
        # Band 1 should have offset detected (approximately -5, -3)
        assert abs(offsets.x_offsets[1] - (-5)) <= 1
        assert abs(offsets.y_offsets[1] - (-3)) <= 1

    def test_align_identical_bands_zero_offset(self):
        """Identical bands should produce zero offsets."""
        aligner = BandAligner()
        rng = np.random.default_rng(42)
        base = rng.random((100, 100))
        img = np.stack([base, base, base], axis=2)

        aligned, offsets = aligner.align(img)

        assert offsets is not None
        assert not offsets.has_offset()
        # Image should be unchanged when no offsets needed
        assert np.array_equal(aligned, img)

    def test_align_returns_aligned_image(self):
        """Alignment should return properly aligned image."""
        aligner = BandAligner()

        # Create image with offset
        base = np.zeros((100, 100), dtype=np.float64)
        base[40:60, 40:60] = 1.0

        shifted = np.zeros((100, 100), dtype=np.float64)
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
        offsets = BandOffsets(x_offsets=(0, 1, 2), y_offsets=(0, 0, 0))
        result = aligner.unalign_mask(masks, offsets)
        assert np.array_equal(result, masks)

    def test_unalign_none_offsets_returns_input(self):
        """None offsets should return masks unchanged."""
        aligner = BandAligner()
        masks = np.ones((50, 50, 3), dtype=np.uint8)
        result = aligner.unalign_mask(masks, None)
        assert np.array_equal(result, masks)

    def test_unalign_zero_offsets_returns_input(self):
        """Zero offsets should return masks unchanged."""
        aligner = BandAligner()
        offsets = BandOffsets(x_offsets=(0, 0, 0), y_offsets=(0, 0, 0))
        masks = np.ones((50, 50, 3), dtype=np.uint8)
        result = aligner.unalign_mask(masks, offsets)
        assert np.array_equal(result, masks)

    def test_unalign_2d_union_mask_creates_shifted_per_band(self):
        """2D union mask should be replicated and shifted for each band."""
        aligner = BandAligner()
        offsets = BandOffsets(x_offsets=(0, 20, -10), y_offsets=(0, 0, 0))

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

        # Band 1: shifted by -20, square at cols 20-40
        _rows1, cols1 = np.where(result[:, :, 1] > 0)
        assert cols1.min() == 20
        assert cols1.max() == 39

        # Band 2: shifted by +10, square at cols 50-70
        _rows2, cols2 = np.where(result[:, :, 2] > 0)
        assert cols2.min() == 50
        assert cols2.max() == 69

    def test_unalign_3d_applies_inverse_offsets(self):
        """Unalign should apply inverse of alignment offsets to 3D masks."""
        aligner = BandAligner()
        offsets = BandOffsets(x_offsets=(0, 5, -3), y_offsets=(0, 2, -1))

        # Create masks with a pattern
        masks = np.zeros((50, 50, 3), dtype=np.float32)
        masks[20:30, 20:30, :] = 255  # Square in center of all bands

        result = aligner.unalign_mask(masks, offsets)

        # Band 0 (no offset) should be unchanged
        assert np.array_equal(result[:, :, 0], masks[:, :, 0])

        # Band 1 (offset 5, 2) should be shifted by (-5, -2)
        # The square should move from (20:30, 20:30) to (18:28, 15:25)
        assert result[20:30, 20:30, 1].sum() < 255 * 100  # Original position has less

        # Band 2 (offset -3, -1) should be shifted by (3, 1)
        # The square should move from (20:30, 20:30) to (21:31, 23:33)
        assert result[20:30, 20:30, 2].sum() < 255 * 100  # Original position has less

    def test_unalign_band_count_mismatch_raises(self):
        """Mismatched band count should raise ValueError."""
        aligner = BandAligner()
        offsets = BandOffsets(x_offsets=(0, 1, 2), y_offsets=(0, 0, 0))
        masks = np.ones((50, 50, 5), dtype=np.float32)  # 5 bands, offsets for 3

        with pytest.raises(ValueError, match="computed for"):
            aligner.unalign_mask(masks, offsets)


class TestBandAlignerIntegration:
    """Integration tests for complete alignment workflow."""

    def test_full_align_and_unalign_workflow(self):
        """Test complete workflow: align image then unalign mask."""
        aligner = BandAligner()

        # Create image with known offset
        base = np.zeros((100, 100), dtype=np.float64)
        base[40:60, 40:60] = 1.0

        shifted = np.zeros((100, 100), dtype=np.float64)
        shifted[40 + 2 : 60 + 2, 40 + 3 : 60 + 3] = 1.0

        img = np.stack([base, shifted], axis=2)

        # Align the image
        aligned, offsets = aligner.align(img)

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
        offsets = BandOffsets(x_offsets=(0, 3, -2), y_offsets=(0, 1, 2))

        # Simulate aligning, processing, then unaligning
        # First "align" the mask (shift to aligned space)
        aligned = aligner.align(original_mask.astype(np.float64))

        # Then unalign (shift back)
        unaligned = aligner.unalign_mask(aligned[0].astype(np.float32), offsets)

        # Band 0 should match exactly (no offset)
        assert np.allclose(unaligned[:, :, 0], original_mask[:, :, 0], atol=1)

    def test_union_mask_workflow_with_alignment(self):
        """Test complete workflow: union mask with per-band shifting."""
        aligner = BandAligner()
        offsets = BandOffsets(x_offsets=(0, 10, -5), y_offsets=(0, 5, -3))

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

        # Band 1 should be shifted by (-10, -5)
        _rows1, cols1 = np.where(result[:, :, 1] > 0)
        assert cols1.max() < 50  # Shifted left

        # Band 2 should be shifted by (5, 3)
        _rows2, cols2 = np.where(result[:, :, 2] > 0)
        assert cols2.min() > 45  # Shifted right
