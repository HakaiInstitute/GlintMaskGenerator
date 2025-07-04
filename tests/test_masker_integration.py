"""Integration tests for complete masker workflow.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2024-01-01
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from glint_mask_tools.glint_algorithms import IntensityRatioAlgorithm, ThresholdAlgorithm
from glint_mask_tools.image_loaders import (
    DJIM3MLoader,
    MicasenseRedEdgeDualLoader,
    MicasenseRedEdgeLoader,
    P4MSLoader,
    SingleFileImageLoader,
)
from glint_mask_tools.maskers import Masker
from glint_mask_tools.utils import normalize_8bit_img, normalize_16bit_img


def create_realistic_test_image(height, width, channels, bit_depth, add_glint_pattern=True):
    """Create a realistic test image with optional glint patterns."""
    rng = np.random.default_rng(42)
    if bit_depth == 8:
        if add_glint_pattern:
            # Create varied background with bright glint spots
            img = rng.uniform(20, 180, (height, width, channels)).astype(np.uint8)
            # Add glint pattern (bright circular spots)
            center_y, center_x = height // 3, width // 3
            y, x = np.ogrid[:height, :width]
            glint_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(height, width) // 8) ** 2
            img[glint_mask] = rng.uniform(240, 256, (np.sum(glint_mask), channels))
        else:
            img = rng.uniform(30, 200, (height, width, channels)).astype(np.uint8)
    else:  # 16-bit
        if add_glint_pattern:
            img = rng.uniform(1000, 30000, (height, width)).astype(np.uint16)
            # Add glint pattern
            center_y, center_x = height // 3, width // 3
            y, x = np.ogrid[:height, :width]
            glint_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= (min(height, width) // 8) ** 2
            img[glint_mask] = rng.uniform(55000, 65536)
        else:
            img = rng.uniform(2000, 40000, (height, width)).astype(np.uint16)

        if channels > 1:
            img = np.stack([img] * channels, axis=2)

    return Image.fromarray(img)


@pytest.fixture
def complete_sensor_suite(tmp_path):
    """Create a complete test suite with all sensor types."""
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    # RGB images
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    for i in range(2):
        img = create_realistic_test_image(128, 128, 3, 8, add_glint_pattern=True)
        img.save(rgb_dir / f"rgb_{i}.jpg")

    # CIR image
    cir_dir = tmp_path / "cir"
    cir_dir.mkdir()
    img = create_realistic_test_image(512, 512, 4, 8, add_glint_pattern=True)
    img.save(cir_dir / "large_cir.tif")

    # MicaSense RedEdge images
    micasense_dir = tmp_path / "micasense"
    micasense_dir.mkdir()
    for capture_id in [1001, 1002]:
        for band in range(1, 6):
            img = create_realistic_test_image(128, 128, 1, 16, add_glint_pattern=(band == 1))
            img.save(micasense_dir / f"IMG_{capture_id}_{band}.tif")
    # MicaSense RedEdge Dual images
    dual_dir = tmp_path / "dual"
    dual_dir.mkdir()
    for capture_id in [2001]:
        for band in range(1, 11):
            img = create_realistic_test_image(128, 128, 1, 16, add_glint_pattern=(band == 1))
            img.save(dual_dir / f"IMG_{capture_id}_{band}.tif")

    # P4MS images
    p4ms_dir = tmp_path / "p4ms"
    p4ms_dir.mkdir()
    for capture_id in [301, 302]:  # 3-digit capture IDs for P4MS
        for band in range(1, 6):
            img = create_realistic_test_image(128, 128, 1, 16, add_glint_pattern=(band == 1))
            img.save(p4ms_dir / f"DJI_{capture_id}{band}.TIF")

    # DJI M3M images
    djim3m_dir = tmp_path / "djim3m"
    djim3m_dir.mkdir()
    for capture_id in ["20221208115250_0001", "20221208115253_0002"]:
        for band in ["G", "R", "RE", "NIR"]:
            img = create_realistic_test_image(128, 128, 1, 16, add_glint_pattern=(band == "G"))
            img.save(djim3m_dir / f"DJI_{capture_id}_MS_{band}.TIF")

    return {
        "rgb_dir": rgb_dir,
        "cir_dir": cir_dir,
        "micasense_dir": micasense_dir,
        "dual_dir": dual_dir,
        "p4ms_dir": p4ms_dir,
        "djim3m_dir": djim3m_dir,
        "mask_dir": mask_dir,
    }


def test_rgb_complete_workflow(complete_sensor_suite):
    """Test complete RGB masking workflow."""
    dirs = complete_sensor_suite

    # Setup loader and masker
    loader = SingleFileImageLoader(dirs["rgb_dir"], dirs["mask_dir"])
    algorithm = ThresholdAlgorithm([0.85, 0.85, 0.85])
    masker = Masker(algorithm, loader, normalize_8bit_img, pixel_buffer=2)

    # Test workflow
    assert len(masker) == 2

    # Process one image manually to test the workflow
    img_paths = list(loader.paths)
    img_path = img_paths[0]

    # Load and preprocess
    img = loader.load_image(img_path)
    assert img.shape[2] == 3
    assert img.dtype == float

    normalized_img = masker.image_preprocessor(img)
    assert normalized_img.min() >= 0
    assert normalized_img.max() <= 1

    # Apply algorithm
    mask = algorithm(normalized_img)
    assert mask.dtype == bool
    assert mask.shape == normalized_img.shape[:2]

    # Post-process mask
    processed_mask = masker.postprocess_mask(mask)
    metashape_mask = masker.to_metashape_mask(processed_mask)

    assert metashape_mask.dtype == np.uint8
    assert metashape_mask.max() <= 255
    assert metashape_mask.min() >= 0

    # Test mask saving
    mask_paths = list(loader.get_mask_save_paths(img_path))
    assert len(mask_paths) == 1

    loader.save_masks(metashape_mask, img_path)
    assert Path(mask_paths[0]).exists()

    # Verify saved mask
    saved_mask = np.array(Image.open(mask_paths[0]))
    assert saved_mask.shape == metashape_mask.shape
    assert np.array_equal(saved_mask, metashape_mask)


def test_micasense_complete_workflow(complete_sensor_suite):
    """Test complete MicaSense masking workflow."""
    dirs = complete_sensor_suite

    # Setup loader and masker
    loader = MicasenseRedEdgeLoader(dirs["micasense_dir"], dirs["mask_dir"])
    algorithm = ThresholdAlgorithm([0.8] * 5)
    masker = Masker(algorithm, loader, normalize_16bit_img, pixel_buffer=3)

    # Test workflow
    assert len(masker) == 2

    # Process one capture set
    img_paths = list(loader.paths)
    capture_paths = img_paths[0]

    # Load multi-band image
    img = loader.load_image(capture_paths)
    assert img.shape[2] == 5  # 5 bands
    assert img.dtype == float

    normalized_img = masker.image_preprocessor(img)
    assert normalized_img.min() >= 0
    assert normalized_img.max() <= 1

    # Apply algorithm
    mask = algorithm(normalized_img)
    assert mask.dtype == bool

    # Post-process
    processed_mask = masker.postprocess_mask(mask)
    metashape_mask = masker.to_metashape_mask(processed_mask)

    # Test mask generation and saving
    mask_paths = list(loader.get_mask_save_paths(capture_paths))
    assert len(mask_paths) == 5  # One mask per band file

    for mask_path in mask_paths:
        loader.save_masks(metashape_mask, capture_paths)
        assert Path(mask_path).exists()


def test_micasense_dual_complete_workflow(complete_sensor_suite):
    """Test complete MicaSense Dual masking workflow."""
    dirs = complete_sensor_suite

    # Setup loader and masker
    loader = MicasenseRedEdgeDualLoader(dirs["dual_dir"], dirs["mask_dir"])
    algorithm = ThresholdAlgorithm([0.8] * 10)
    masker = Masker(algorithm, loader, normalize_16bit_img)

    # Test workflow
    assert len(masker) == 1  # One capture set

    img_paths = list(loader.paths)
    capture_paths = img_paths[0]

    img = loader.load_image(capture_paths)
    assert img.shape[2] == 10  # 10 bands for dual sensor

    mask_paths = list(loader.get_mask_save_paths(capture_paths))
    assert len(mask_paths) == 10  # One mask per band file


def test_p4ms_complete_workflow(complete_sensor_suite):
    """Test complete P4MS masking workflow."""
    dirs = complete_sensor_suite

    # Setup loader and masker
    loader = P4MSLoader(dirs["p4ms_dir"], dirs["mask_dir"])
    algorithm = ThresholdAlgorithm([0.8] * 5)
    masker = Masker(algorithm, loader, normalize_8bit_img)

    # Test workflow
    assert len(masker) == 2

    # Test image processing
    img_paths = list(loader.paths)
    capture_paths = img_paths[0]

    img = loader.load_image(capture_paths)
    assert img.shape[2] == 5

    mask_paths = list(loader.get_mask_save_paths(capture_paths))
    assert len(mask_paths) == 5


def test_intensity_ratio_algorithm_workflow(complete_sensor_suite):
    """Test complete workflow with IntensityRatioAlgorithm on RGB data."""
    dirs = complete_sensor_suite

    # Setup with intensity ratio algorithm
    loader = SingleFileImageLoader(dirs["rgb_dir"], dirs["mask_dir"])
    algorithm = IntensityRatioAlgorithm(percent_diffuse=0.9, threshold=0.8)
    masker = Masker(algorithm, loader, normalize_8bit_img)

    # Test workflow
    img_paths = list(loader.paths)
    img_path = img_paths[0]

    img = loader.load_image(img_path)
    normalized_img = masker.image_preprocessor(img)

    # Apply intensity ratio algorithm
    mask = algorithm(normalized_img)
    assert mask.dtype == bool

    # Test full processing
    processed_mask = masker.postprocess_mask(mask)
    metashape_mask = masker.to_metashape_mask(processed_mask)

    assert metashape_mask.dtype == np.uint8


def test_pixel_buffer_effects():
    """Test that pixel buffer correctly dilates masks."""
    # Create a simple image with a single bright pixel
    img = np.zeros((20, 20, 3), dtype=float)
    img[10, 10] = 1.0  # Single bright pixel

    algorithm = ThresholdAlgorithm([0.5, 0.5, 0.5])

    # Test without buffer
    masker_no_buffer = Masker(algorithm, None, normalize_8bit_img, pixel_buffer=0)
    mask_no_buffer = algorithm(img)
    processed_no_buffer = masker_no_buffer.postprocess_mask(mask_no_buffer)

    # Only the center pixel should be masked
    assert processed_no_buffer[10, 10] == 1
    assert processed_no_buffer[9, 10] == 0
    assert processed_no_buffer[11, 10] == 0

    # Test with buffer
    masker_with_buffer = Masker(algorithm, None, normalize_8bit_img, pixel_buffer=2)
    processed_with_buffer = masker_with_buffer.postprocess_mask(mask_no_buffer)

    # Center and surrounding pixels should be masked due to buffer
    assert processed_with_buffer[10, 10] == 1
    assert processed_with_buffer[9, 10] == 1  # Should be dilated
    assert processed_with_buffer[11, 10] == 1  # Should be dilated


def test_metashape_mask_conversion():
    """Test conversion to Metashape-compatible mask format."""
    # Create test mask
    mask = np.array([[True, False], [False, True]], dtype=bool)

    masker = Masker(None, None, None)
    metashape_mask = masker.to_metashape_mask(mask)

    # Metashape expects inverted mask (0 for masked, 255 for unmasked)
    expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    assert np.array_equal(metashape_mask, expected)
    assert metashape_mask.dtype == np.uint8


@pytest.mark.parametrize(
    ("sensor_type", "expected_bands"),
    [
        ("rgb", 3),
        ("micasense", 5),
        ("micasense_dual", 10),
        ("p4ms", 5),
        ("djim3m", 4),
    ],
)
def test_all_sensors_produce_valid_masks(complete_sensor_suite, sensor_type, expected_bands):
    """Test that all sensor types produce valid mask outputs."""
    dirs = complete_sensor_suite

    # Setup based on sensor type
    if sensor_type == "rgb":
        loader = SingleFileImageLoader(dirs["rgb_dir"], dirs["mask_dir"])
        algorithm = ThresholdAlgorithm([0.8] * 3)
        pre_processor = normalize_8bit_img
    elif sensor_type == "micasense":
        loader = MicasenseRedEdgeLoader(dirs["micasense_dir"], dirs["mask_dir"])
        algorithm = ThresholdAlgorithm([0.8] * 5)
        pre_processor = normalize_16bit_img
    elif sensor_type == "micasense_dual":
        loader = MicasenseRedEdgeDualLoader(dirs["dual_dir"], dirs["mask_dir"])
        algorithm = ThresholdAlgorithm([0.8] * 10)
        pre_processor = normalize_16bit_img
    elif sensor_type == "p4ms":
        loader = P4MSLoader(dirs["p4ms_dir"], dirs["mask_dir"])
        algorithm = ThresholdAlgorithm([0.8] * 5)
        pre_processor = normalize_16bit_img
    elif sensor_type == "djim3m":
        loader = DJIM3MLoader(dirs["djim3m_dir"], dirs["mask_dir"])
        algorithm = ThresholdAlgorithm([0.8] * 4)
        pre_processor = normalize_16bit_img

    masker = Masker(algorithm, loader, pre_processor)

    if len(masker) > 0:
        # Test one capture
        img_paths = list(loader.paths)
        capture_paths = img_paths[0]

        img = loader.load_image(capture_paths)
        assert img.shape[2] == expected_bands

        normalized_img = masker.image_preprocessor(img)
        mask = algorithm(normalized_img)

        # Validate mask properties
        assert mask.dtype == bool
        assert mask.shape == normalized_img.shape[:2]

        # Test post-processing
        processed_mask = masker.postprocess_mask(mask)
        metashape_mask = masker.to_metashape_mask(processed_mask)

        assert metashape_mask.dtype == np.uint8
        assert metashape_mask.min() >= 0
        assert metashape_mask.max() <= 255


def test_djim3m_complete_workflow(complete_sensor_suite):
    """Test complete DJI M3M masking workflow."""
    dirs = complete_sensor_suite

    # Setup loader and masker
    loader = DJIM3MLoader(dirs["djim3m_dir"], dirs["mask_dir"])
    algorithm = ThresholdAlgorithm([0.85, 0.85, 0.85, 0.85])  # G, R, RE, NIR
    masker = Masker(algorithm, loader, normalize_16bit_img, pixel_buffer=2)

    # Test workflow
    assert len(masker) == 2  # Two captures

    # Process one image manually to test the workflow
    img_paths = list(loader.paths)
    capture_paths = img_paths[0]  # First capture (4 bands)

    # Load and preprocess
    img = loader.load_image(capture_paths)
    assert img.shape[2] == 4  # G, R, RE, NIR bands
    assert img.dtype == float

    normalized_img = masker.image_preprocessor(img)
    assert normalized_img.min() >= 0
    assert normalized_img.max() <= 1

    # Apply algorithm
    mask = algorithm(normalized_img)
    assert mask.dtype == bool
    assert mask.shape == normalized_img.shape[:2]

    # Post-process mask
    processed_mask = masker.postprocess_mask(mask)
    assert processed_mask.dtype == int  # postprocess_mask returns int when pixel_buffer > 0
    assert processed_mask.shape == mask.shape

    # Convert to Metashape format
    metashape_mask = masker.to_metashape_mask(processed_mask)
    assert metashape_mask.dtype == np.uint8
    assert metashape_mask.min() >= 0
    assert metashape_mask.max() <= 255

    # Test that we can process all images
    mask_count = 0
    for capture_paths in loader.paths:
        img = loader.load_image(capture_paths)
        normalized_img = masker.image_preprocessor(img)
        mask = algorithm(normalized_img)
        processed_mask = masker.postprocess_mask(mask)
        metashape_mask = masker.to_metashape_mask(processed_mask)
        mask_count += 1

    assert mask_count == 2  # Both captures processed successfully


def test_djim3m_with_intensity_ratio_algorithm(complete_sensor_suite):
    """Test DJI M3M with intensity ratio algorithm."""
    dirs = complete_sensor_suite

    # Setup loader and masker with intensity ratio algorithm
    loader = DJIM3MLoader(dirs["djim3m_dir"], dirs["mask_dir"])
    algorithm = IntensityRatioAlgorithm(threshold=0.8)  # Uses default percent_diffuse=0.95
    masker = Masker(algorithm, loader, normalize_16bit_img)

    # Test workflow
    assert len(masker) == 2

    # Process one image
    img_paths = list(loader.paths)
    capture_paths = img_paths[0]

    img = loader.load_image(capture_paths)
    assert img.shape[2] == 4  # G, R, RE, NIR bands

    normalized_img = masker.image_preprocessor(img)
    mask = algorithm(normalized_img)
    assert mask.dtype == bool
    assert mask.shape == normalized_img.shape[:2]

    # Test post-processing
    processed_mask = masker.postprocess_mask(mask)
    metashape_mask = masker.to_metashape_mask(processed_mask)
    assert metashape_mask.dtype == np.uint8
