"""Tests for threshold maskers with different sensor types.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2024-01-01
"""

import numpy as np
import pytest
from PIL import Image

from glint_mask_generator.glint_algorithms import ThresholdAlgorithm
from glint_mask_generator.image_loaders import (
    CIRLoader,
    MicasenseRedEdgeLoader,
    P4MSLoader,
    RGBLoader,
)
from glint_mask_generator.maskers import Masker


def create_test_image_8bit(height=32, width=32, channels=3, add_glint=False):
    """Create a test image with 8-bit depth."""
    rng = np.random.default_rng(42)
    if add_glint:
        # Create image with some high-intensity "glint" pixels
        img = rng.uniform(50, 150, (height, width, channels)).astype(np.uint8)
        # Add some bright spots to simulate glint
        img[height // 4 : height // 2, width // 4 : width // 2] = 255
    else:
        img = rng.uniform(50, 200, (height, width, channels)).astype(np.uint8)
    return Image.fromarray(img)


def create_test_image_16bit(height=32, width=32, channels=1, add_glint=False):
    """Create a test image with 16-bit depth."""
    rng = np.random.default_rng(42)
    if add_glint:
        # Create image with some high-intensity "glint" pixels
        img = rng.uniform(5000, 25000, (height, width)).astype(np.uint16)
        # Add some bright spots to simulate glint
        img[height // 4 : height // 2, width // 4 : width // 2] = 65535
    else:
        img = rng.uniform(5000, 40000, (height, width)).astype(np.uint16)

    if channels > 1:
        img = np.stack([img] * channels, axis=2)
    return Image.fromarray(img)


@pytest.fixture
def rgb_test_data(tmp_path):
    """Create test RGB images with known glint patterns."""
    # Create RGB images with glint
    for filename in ["rgb1.jpg", "rgb2.png"]:
        img = create_test_image_8bit(height=64, width=64, channels=3, add_glint=True)
        img.save(tmp_path / filename)

    return tmp_path


@pytest.fixture
def cir_test_data(tmp_path):
    """Create test CIR image with known glint patterns."""
    img = create_test_image_8bit(height=256, width=256, channels=4, add_glint=True)
    img.save(tmp_path / "cir_image.tif")
    return tmp_path


@pytest.fixture
def micasense_test_data(tmp_path):
    """Create test MicaSense RedEdge images with known glint patterns."""
    # Create 5-band MicaSense images
    for capture_id in [1234, 5678]:
        for band in range(1, 6):
            # Band 1 (blue) gets glint, others don't
            img = create_test_image_16bit(height=64, width=64, add_glint=(band == 1))
            img.save(tmp_path / f"IMG_{capture_id}_{band}.tif")

    return tmp_path


@pytest.fixture
def micasense_dual_test_data(tmp_path):
    """Create test MicaSense RedEdge Dual images with known glint patterns."""
    # Create 10-band MicaSense Dual images
    for capture_id in [9999, 8888]:
        for band in range(1, 11):
            # Band 1 (blue) gets glint, others don't
            img = create_test_image_16bit(height=64, width=64, add_glint=(band == 1))
            img.save(tmp_path / f"IMG_{capture_id}_{band}.tif")

    return tmp_path


@pytest.fixture
def p4ms_test_data(tmp_path):
    """Create test P4MS images with known glint patterns."""
    # Create P4MS images - P4MS has 5 bands with specific naming pattern
    # Pattern is DJI_###1.TIF where ### is 3-digit capture ID and last digit is band
    for capture_id in [101, 201]:  # 3-digit capture IDs
        for band in range(1, 6):
            # Band 1 (blue) gets glint, others don't
            img = create_test_image_16bit(height=64, width=64, add_glint=(band == 1))
            img.save(tmp_path / f"DJI_{capture_id}{band}.TIF")

    return tmp_path


def test_threshold_algorithm_8bit():
    """Test threshold algorithm with 8-bit data."""
    # Create test image with known values
    img = np.array([[[100, 150, 200], [50, 75, 100]], [[200, 250, 255], [25, 50, 75]]], dtype=float)

    # Test with threshold that should mask some pixels
    thresholds = [180, 180, 180]  # Should mask pixels > 180 in any channel
    algorithm = ThresholdAlgorithm(thresholds)
    mask = algorithm(img)

    # Check expected results
    expected = np.array([[True, False], [True, False]])  # Pixel (0,0) and (1,0) should be masked
    assert np.array_equal(mask, expected)


def test_threshold_algorithm_16bit():
    """Test threshold algorithm with 16-bit data."""
    # Create test image with known values (normalized to 0-1 range)
    img = np.array([[[0.3, 0.5, 0.7], [0.1, 0.2, 0.3]], [[0.8, 0.9, 1.0], [0.05, 0.1, 0.15]]], dtype=float)

    # Test with threshold
    thresholds = [0.75, 0.75, 0.75]  # Should mask pixels > 0.75 in any channel
    algorithm = ThresholdAlgorithm(thresholds)
    mask = algorithm(img)

    # Check expected results
    expected = np.array([[False, False], [True, False]])  # Only pixel (1,0) should be masked
    assert np.array_equal(mask, expected)


def test_rgb_threshold_masker(rgb_test_data):
    """Test threshold masker with RGB sensor."""
    mask_dir = rgb_test_data / "masks"
    mask_dir.mkdir()

    # Setup loader and masker
    loader = RGBLoader(rgb_test_data, mask_dir)
    algorithm = ThresholdAlgorithm([0.8, 0.8, 0.8])  # High threshold for 8-bit normalized data
    masker = Masker(algorithm, loader)

    assert len(masker) == 2  # Two RGB images
    assert loader._bit_depth == 8

    # Test single image processing
    img_paths = list(loader.paths)
    img = loader.load_image(img_paths[0])
    normalized_img = loader.preprocess_image(img)

    # Check that image is properly normalized
    assert normalized_img.min() >= 0
    assert normalized_img.max() <= 1

    # Apply algorithm
    mask = algorithm(normalized_img)
    assert mask.dtype == bool
    assert mask.shape == normalized_img.shape[:2]


def test_cir_threshold_masker(cir_test_data):
    """Test threshold masker with CIR sensor."""
    mask_dir = cir_test_data / "masks"
    mask_dir.mkdir()

    # Setup loader and masker
    loader = CIRLoader(cir_test_data, mask_dir)
    algorithm = ThresholdAlgorithm([0.8, 0.8, 0.8, 0.8])  # Threshold for 4 bands
    masker = Masker(algorithm, loader)

    assert len(masker) == 1
    assert loader._bit_depth == 8

    # Test image processing
    img_paths = list(loader.paths)
    img = loader.load_image(img_paths[0])
    normalized_img = loader.preprocess_image(img)

    assert normalized_img.shape[2] == 4  # 4 CIR bands
    assert normalized_img.min() >= 0
    assert normalized_img.max() <= 1


def test_micasense_threshold_masker(micasense_test_data):
    """Test threshold masker with MicaSense RedEdge sensor."""
    mask_dir = micasense_test_data / "masks"
    mask_dir.mkdir()

    # Setup loader and masker
    loader = MicasenseRedEdgeLoader(micasense_test_data, mask_dir)
    algorithm = ThresholdAlgorithm([0.8] * 5)  # Threshold for 5 bands
    masker = Masker(algorithm, loader)

    assert len(masker) == 2  # Two capture sets
    assert loader._bit_depth == 16

    # Test image processing
    img_paths = list(loader.paths)
    img = loader.load_image(img_paths[0])
    normalized_img = loader.preprocess_image(img)

    assert normalized_img.shape[2] == 5  # 5 MicaSense bands
    assert normalized_img.min() >= 0
    assert normalized_img.max() <= 1


def test_p4ms_threshold_masker(p4ms_test_data):
    """Test threshold masker with P4MS sensor."""
    mask_dir = p4ms_test_data / "masks"
    mask_dir.mkdir()

    # Setup loader and masker
    loader = P4MSLoader(p4ms_test_data, mask_dir)
    algorithm = ThresholdAlgorithm([0.8] * 5)  # Threshold for 5 bands
    masker = Masker(algorithm, loader)

    assert len(masker) == 2  # Two capture sets
    assert loader._bit_depth == 16

    # Test image processing
    img_paths = list(loader.paths)
    img = loader.load_image(img_paths[0])
    normalized_img = loader.preprocess_image(img)

    assert normalized_img.shape[2] == 5  # 5 P4MS bands
    assert normalized_img.min() >= 0
    assert normalized_img.max() <= 1


@pytest.mark.parametrize(
    ("bit_depth", "thresholds"),
    [
        (8, [0.5, 0.6, 0.7]),  # 8-bit thresholds
        (16, [0.7, 0.8, 0.9]),  # 16-bit thresholds
    ],
)
def test_threshold_values_respect_bit_depth(bit_depth, thresholds):
    """Test that threshold values work correctly with different bit depths."""
    rng = np.random.default_rng(42)
    if bit_depth == 8:
        # Create 8-bit image data (0-255) normalized to 0-1
        raw_img = rng.uniform(0, 256, (32, 32, 3)).astype(np.uint8)
        normalized_img = raw_img.astype(float) / 255.0
    else:
        # Create 16-bit image data (0-65535) normalized to 0-1
        raw_img = rng.uniform(0, 65536, (32, 32, 3)).astype(np.uint16)
        normalized_img = raw_img.astype(float) / 65535.0

    algorithm = ThresholdAlgorithm(thresholds)
    mask = algorithm(normalized_img)

    # Ensure mask is boolean and has correct shape
    assert mask.dtype == bool
    assert mask.shape == normalized_img.shape[:2]

    # Test that some pixels are masked when image has high values
    high_value_img = np.ones_like(normalized_img) * 0.95
    high_mask = algorithm(high_value_img)
    assert high_mask.any()  # Should mask some pixels

    # Test that no pixels are masked when image has low values
    low_value_img = np.ones_like(normalized_img) * 0.1
    low_mask = algorithm(low_value_img)
    assert not low_mask.any()  # Should not mask any pixels


def test_bit_depth_specific_glint_detection():
    """Test that glint detection works appropriately for each bit depth."""
    # Test 8-bit case - glint at high pixel values (near 255)
    img_8bit = np.zeros((10, 10, 3), dtype=float)
    img_8bit[2:5, 2:5] = 1.0  # Normalized max value for 8-bit

    threshold_8bit = ThresholdAlgorithm([0.9, 0.9, 0.9])
    mask_8bit = threshold_8bit(img_8bit)

    # Should detect glint in the bright region
    assert mask_8bit[3, 3]
    assert not mask_8bit[0, 0]

    # Test 16-bit case - similar pattern but with different threshold
    img_16bit = np.zeros((10, 10, 5), dtype=float)  # 5 bands like MicaSense
    img_16bit[2:5, 2:5] = 1.0  # Normalized max value for 16-bit

    threshold_16bit = ThresholdAlgorithm([0.85] * 5)
    mask_16bit = threshold_16bit(img_16bit)

    # Should detect glint in the bright region
    assert mask_16bit[3, 3]
    assert not mask_16bit[0, 0]
