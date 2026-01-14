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

    def test_disabled_aligner_returns_input(self):
        """Disabled aligner should return input unchanged."""
        aligner = BandAligner(enabled=False)
        rng = np.random.default_rng(42)
        img = rng.random((100, 100, 5)).astype(np.float32)
        result = aligner.align(img)
        assert np.array_equal(result, img)

    def test_disabled_aligner_is_calibrated_after_calibrate(self):
        """Disabled aligner should mark as calibrated after calibrate call."""
        aligner = BandAligner(enabled=False)
        assert not aligner.is_calibrated

        aligner.calibrate([], lambda _: None)
        assert aligner.is_calibrated


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


class TestBandAlignerCalibration:
    """Tests for calibration workflow."""

    def test_calibrate_creates_offsets(self):
        """Calibration should create valid offsets."""
        aligner = BandAligner(calibration_samples=2)

        def mock_load(_paths):
            """Create synthetic multi-band image with no offset."""
            rng = np.random.default_rng(42)
            base = rng.random((100, 100))
            return np.stack([base, base, base], axis=2)

        mock_paths = [["path1"], ["path2"], ["path3"]]
        offsets = aligner.calibrate(mock_paths, mock_load)

        assert aligner.is_calibrated
        assert offsets is not None
        assert offsets.num_bands == 3
        # All bands identical, so offsets should be zero
        assert not offsets.has_offset()

    def test_calibrate_empty_paths(self):
        """Calibration with empty paths should handle gracefully."""
        aligner = BandAligner(calibration_samples=5)
        offsets = aligner.calibrate([], lambda _: None)

        assert aligner.is_calibrated
        assert offsets is None

    def test_calibrate_2d_image_disables_alignment(self):
        """Non-multi-band image should disable alignment."""
        aligner = BandAligner(calibration_samples=1)

        def mock_load(_paths):
            rng = np.random.default_rng(42)
            return rng.random((100, 100))  # 2D, not 3D

        offsets = aligner.calibrate([["path1"]], mock_load)

        assert aligner.is_calibrated
        assert not aligner.enabled
        assert offsets is None


class TestBandAlignerAlign:
    """Tests for alignment application."""

    def test_align_applies_offsets(self):
        """Alignment should apply calibrated offsets."""
        aligner = BandAligner()
        aligner._offsets = BandOffsets(
            x_offsets=(0, 2, -1),
            y_offsets=(0, 1, -2),
        )
        aligner._calibrated = True

        img = np.ones((50, 50, 3), dtype=np.float32)
        result = aligner.align(img)

        assert result.shape == img.shape
        # Band 0 should be unchanged (zero offset)
        assert np.array_equal(result[:, :, 0], img[:, :, 0])

    def test_align_no_offset_returns_original(self):
        """Alignment with zero offsets should return original."""
        aligner = BandAligner()
        aligner._offsets = BandOffsets(
            x_offsets=(0, 0, 0),
            y_offsets=(0, 0, 0),
        )
        aligner._calibrated = True

        rng = np.random.default_rng(42)
        img = rng.random((50, 50, 3)).astype(np.float32)
        result = aligner.align(img)

        assert np.array_equal(result, img)

    def test_align_band_count_mismatch_raises(self):
        """Mismatched band count should raise ValueError."""
        aligner = BandAligner()
        aligner._offsets = BandOffsets(
            x_offsets=(0, 1, 2),
            y_offsets=(0, 0, 0),
        )
        aligner._calibrated = True

        img = np.ones((50, 50, 5), dtype=np.float32)  # 5 bands, calibrated for 3

        with pytest.raises(ValueError, match="calibrated for"):
            aligner.align(img)

    def test_align_uncalibrated_returns_original(self):
        """Uncalibrated aligner should return original image."""
        aligner = BandAligner()
        rng = np.random.default_rng(42)
        img = rng.random((50, 50, 3)).astype(np.float32)
        result = aligner.align(img)

        assert np.array_equal(result, img)


class TestBandAlignerUnalignMask:
    """Tests for unaligning masks back to original coordinates."""

    def test_unalign_disabled_returns_input(self):
        """Disabled aligner should return masks unchanged."""
        aligner = BandAligner(enabled=False)
        masks = np.ones((50, 50, 3), dtype=np.uint8)
        result = aligner.unalign_mask(masks)
        assert np.array_equal(result, masks)

    def test_unalign_no_offsets_returns_input(self):
        """No offsets should return masks unchanged."""
        aligner = BandAligner()
        masks = np.ones((50, 50, 3), dtype=np.uint8)
        result = aligner.unalign_mask(masks)
        assert np.array_equal(result, masks)

    def test_unalign_zero_offsets_returns_input(self):
        """Zero offsets should return masks unchanged."""
        aligner = BandAligner()
        aligner._offsets = BandOffsets(x_offsets=(0, 0, 0), y_offsets=(0, 0, 0))
        aligner._calibrated = True

        masks = np.ones((50, 50, 3), dtype=np.uint8)
        result = aligner.unalign_mask(masks)
        assert np.array_equal(result, masks)

    def test_unalign_2d_union_mask_creates_shifted_per_band(self):
        """2D union mask should be replicated and shifted for each band."""
        aligner = BandAligner()
        aligner._offsets = BandOffsets(x_offsets=(0, 20, -10), y_offsets=(0, 0, 0))
        aligner._calibrated = True

        # Create 2D union mask with a square
        mask_2d = np.zeros((100, 100), dtype=np.float32)
        mask_2d[40:60, 40:60] = 255

        result = aligner.unalign_mask(mask_2d)

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
        aligner._offsets = BandOffsets(x_offsets=(0, 5, -3), y_offsets=(0, 2, -1))
        aligner._calibrated = True

        # Create masks with a pattern
        masks = np.zeros((50, 50, 3), dtype=np.float32)
        masks[20:30, 20:30, :] = 255  # Square in center of all bands

        result = aligner.unalign_mask(masks)

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
        aligner._offsets = BandOffsets(x_offsets=(0, 1, 2), y_offsets=(0, 0, 0))
        aligner._calibrated = True

        masks = np.ones((50, 50, 5), dtype=np.float32)  # 5 bands, calibrated for 3

        with pytest.raises(ValueError, match="calibrated for"):
            aligner.unalign_mask(masks)


class TestBandAlignerIntegration:
    """Integration tests for complete alignment workflow."""

    def test_full_calibrate_and_align_workflow(self):
        """Test complete workflow: calibrate then align."""
        aligner = BandAligner(calibration_samples=3)

        # Create images with known offset
        def create_shifted_image(shift_x, shift_y):
            base = np.zeros((100, 100), dtype=np.float64)
            base[40:60, 40:60] = 1.0

            shifted = np.zeros((100, 100), dtype=np.float64)
            shifted[40 + shift_y : 60 + shift_y, 40 + shift_x : 60 + shift_x] = 1.0

            return np.stack([base, shifted], axis=2)

        def mock_load(_paths):
            return create_shifted_image(3, 2)

        mock_paths = [["p1"], ["p2"], ["p3"]]
        aligner.calibrate(mock_paths, mock_load)

        assert aligner.is_calibrated
        assert aligner.offsets is not None
        assert aligner.offsets.has_offset()

        # Now align an image
        test_img = create_shifted_image(3, 2)
        aligned = aligner.align(test_img)

        assert aligned.shape == test_img.shape
        # After alignment, bands should be more similar
        # (not a perfect test but verifies something happened)
        assert not np.array_equal(aligned[:, :, 1], test_img[:, :, 1])

    def test_align_then_unalign_restores_original_coordinates(self):
        """Aligning and then unaligning masks should restore original coordinate system."""
        aligner = BandAligner()
        aligner._offsets = BandOffsets(x_offsets=(0, 3, -2), y_offsets=(0, 1, 2))
        aligner._calibrated = True

        # Create a mask in original coordinates with distinct patterns per band
        original_mask = np.zeros((100, 100, 3), dtype=np.float32)
        original_mask[40:50, 40:50, 0] = 255  # Band 0 pattern
        original_mask[30:40, 60:70, 1] = 255  # Band 1 pattern
        original_mask[70:80, 20:30, 2] = 255  # Band 2 pattern

        # Simulate aligning, processing, then unaligning
        # First "align" the mask (shift to aligned space)
        aligned = aligner.align(original_mask.astype(np.float64))

        # Then unalign (shift back)
        unaligned = aligner.unalign_mask(aligned.astype(np.float32))

        # Band 0 should match exactly (no offset)
        assert np.allclose(unaligned[:, :, 0], original_mask[:, :, 0], atol=1)

        # Other bands should be close to original (some edge artifacts from double translation)
        # Check that the main pattern is in approximately the right place
        assert unaligned[30:40, 60:70, 1].sum() > 200 * 50  # Pattern roughly restored
        assert unaligned[70:80, 20:30, 2].sum() > 200 * 50  # Pattern roughly restored

    def test_union_mask_workflow_with_alignment(self):
        """Test complete workflow: union mask with per-band shifting."""
        aligner = BandAligner()
        aligner._offsets = BandOffsets(x_offsets=(0, 10, -5), y_offsets=(0, 5, -3))
        aligner._calibrated = True

        # Simulate a 2D union mask (glint detected at one location in aligned space)
        union_mask = np.zeros((100, 100), dtype=np.float32)
        union_mask[45:55, 45:55] = 1  # Glint detected at center

        # Unalign creates per-band shifted versions
        result = aligner.unalign_mask(union_mask)

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
