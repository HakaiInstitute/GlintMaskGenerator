"""Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-06-12
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from glint_mask_generator.image_loaders import (
    CIRLoader,
    DJIM3MLoader,
    MicasenseRedEdgeLoader,
    P4MSLoader,
    RGBLoader,
)


def create_test_image_8bit(height=32, width=32, channels=3, add_glint=False):
    """Create a test image with 8-bit depth."""
    rng = np.random.default_rng(42)
    if add_glint:
        # Create image with some high-intensity "glint" pixels
        img = rng.uniform(
            50,
            150,
            (height, width, channels),
        ).astype(np.uint8)

        # Add some bright spots to simulate glint
        img[height // 4 : height // 2, width // 4 : width // 2] = 255
    else:
        img = rng.uniform(
            50,
            200,
            (height, width, channels),
        ).astype(np.uint8)
    return Image.fromarray(img)


def create_test_image_16bit(height=32, width=32, channels=1, add_glint=False):
    """Create a test image with 16-bit depth."""
    rng = np.random.default_rng(42)
    if add_glint:
        # Create image with some high-intensity "glint" pixels
        img = rng.uniform(
            5000,
            25000,
            (height, width),
        ).astype(np.uint16)

        # Add some bright spots to simulate glint
        img[height // 4 : height // 2, width // 4 : width // 2] = 65535
    else:
        img = rng.uniform(
            5000,
            40000,
            (height, width),
        ).astype(np.uint16)

    if channels > 1:
        img = np.stack([img] * channels, axis=2)
    return Image.fromarray(img)


IMG_CONTENT = create_test_image_8bit()
TEST_FILE_NAMES = [
    "1.jpg",
    "2.JPG",
    "3.tif",
    "4.tiff",
    "5.TIF",
    "6.TIFF",
    "7.png",
    "8.PNG",
    "9.jpeg",
    "10.JPEG",
    "IMG_1234_1.tif",
    "IMG_1234_2.tif",
    "IMG_1234_3.tif",
    "IMG_1234_4.tif",
    "IMG_1234_5.tif",
    "IMG_4321_1.TIF",
    "IMG_4321_2.TIF",
    "IMG_4321_3.TIF",
    "IMG_4321_4.TIF",
    "IMG_4321_5.TIF",
    "DJI_1021.TIF",
    "DJI_1022.TIF",
    "DJI_1023.TIF",
    "DJI_1024.TIF",
    "DJI_1025.TIF",
    "DJI_2021.tif",
    "DJI_2022.tif",
    "DJI_2023.tif",
    "DJI_2024.tif",
    "DJI_2025.tif",
]


def test_p4ms_loader(tmp_path):
    """Test that DJI Multispectral masker finds the correct files."""
    # Create dummy files
    all_paths = sorted([str(tmp_path.joinpath(n)) for n in TEST_FILE_NAMES])
    for name in all_paths:
        IMG_CONTENT.save(name)

    # Test the class methods
    image_loader = P4MSLoader(tmp_path, str(Path(tmp_path).joinpath("masks")))

    valid_paths = [
        [
            "DJI_1021.TIF",
            "DJI_1022.TIF",
            "DJI_1023.TIF",
            "DJI_1024.TIF",
            "DJI_1025.TIF",
        ],
        [
            "DJI_2021.TIF",
            "DJI_2022.TIF",
            "DJI_2023.TIF",
            "DJI_2024.TIF",
            "DJI_2025.TIF",
        ],
    ]
    valid_paths = [[str(Path(tmp_path) / p) for p in paths] for paths in valid_paths]
    valid_paths = sorted(valid_paths, key=lambda a: a[0])

    masker_paths = sorted(image_loader.paths, key=lambda a: a[0])
    assert len(image_loader) == len(valid_paths)
    assert np.array_equal(np.array(masker_paths), np.array(valid_paths))

    assert image_loader._is_blue_band_path("DJI_0011.TIF") is True
    assert image_loader._is_blue_band_path("DJI_2221.TIF") is True
    assert image_loader._is_blue_band_path("dji_0011.tif") is True
    assert image_loader._is_blue_band_path("DJI_0013.TIF") is False
    assert image_loader._is_blue_band_path("DJI_0015.TIF") is False
    assert image_loader._is_blue_band_path("DJI_00011.TIF") is False
    assert image_loader._is_blue_band_path("IMG_1234_2.TIF") is False
    assert image_loader._is_blue_band_path("IMG_1234_3.TIF") is False
    assert image_loader._is_blue_band_path("IMG_1234_4.TIF") is False
    assert image_loader._is_blue_band_path("IMG_1234_5.TIF") is False

    assert image_loader._is_blue_band_path(Path("DJI_0011.TIF")) is True
    assert image_loader._is_blue_band_path(Path("DJI_2221.TIF")) is True
    assert image_loader._is_blue_band_path(Path("dji_0011.tif")) is True
    assert image_loader._is_blue_band_path(Path("dji_0101.tif")) is True
    assert image_loader._is_blue_band_path(Path("DJI_0013.TIF")) is False
    assert image_loader._is_blue_band_path(Path("DJI_0015.TIF")) is False
    assert image_loader._is_blue_band_path(Path("DJI_00011.TIF")) is False
    assert image_loader._is_blue_band_path(Path("IMG_1234_2.TIF")) is False
    assert image_loader._is_blue_band_path(Path("IMG_1234_3.TIF")) is False
    assert image_loader._is_blue_band_path(Path("IMG_1234_4.TIF")) is False
    assert image_loader._is_blue_band_path(Path("IMG_1234_5.TIF")) is False

    assert image_loader._is_blue_band_path(Path("/home/dir/dji_0011.tif")) is True
    assert image_loader._is_blue_band_path(Path("/home/dir/DJI_0013.TIF")) is False
    assert image_loader._is_blue_band_path("/home/wherever/dji_0011.tif") is True
    assert image_loader._is_blue_band_path("/home/wherever/DJI_0013.TIF") is False

    assert image_loader._is_blue_band_path(Path("C:\\Users\\some\\dir\\dji_0011.tif")) is True
    assert not image_loader._is_blue_band_path(
        Path("C:\\Users\\some\\dir\\DJI_0013.TIF"),
    )
    assert image_loader._is_blue_band_path("C:\\Users\\some\\dir\\dji_0011.tif") is True
    assert image_loader._is_blue_band_path("C:\\Users\\some\\dir\\DJI_0013.TIF") is False


def test_micasense_red_edge_masker(tmp_path):
    """Test that the Micasense masker finds the correct files."""
    # Create dummy files
    all_paths = sorted([str(tmp_path.joinpath(n)) for n in TEST_FILE_NAMES])
    for name in all_paths:
        IMG_CONTENT.save(name)

    # Test the class methods
    image_loader = MicasenseRedEdgeLoader(tmp_path, Path(tmp_path).joinpath("masks"))

    valid_paths = [
        [
            "IMG_1234_1.tif",
            "IMG_1234_2.tif",
            "IMG_1234_3.tif",
            "IMG_1234_4.tif",
            "IMG_1234_5.tif",
        ],
        [
            "IMG_4321_1.tif",
            "IMG_4321_2.tif",
            "IMG_4321_3.tif",
            "IMG_4321_4.tif",
            "IMG_4321_5.tif",
        ],
    ]
    valid_paths = [[str(Path(tmp_path) / p) for p in paths] for paths in valid_paths]
    valid_paths = sorted(valid_paths, key=lambda a: a[0])

    masker_paths = sorted(image_loader.paths, key=lambda a: a[0])
    assert len(image_loader) == len(valid_paths)
    assert np.array_equal(np.array(masker_paths), np.array(valid_paths))

    assert image_loader._is_blue_band_path("DJI_0014.TIF") is False
    assert image_loader._is_blue_band_path("DJI_2224.TIF") is False
    assert image_loader._is_blue_band_path("dji_0014.tif") is False
    assert image_loader._is_blue_band_path("DJI_0013.TIF") is False
    assert image_loader._is_blue_band_path("DJI_0015.TIF") is False
    assert image_loader._is_blue_band_path("DJI_00014.TIF") is False
    assert image_loader._is_blue_band_path("IMG_1234_1.TIF") is True
    assert image_loader._is_blue_band_path("IMG_9999_1.TIF") is True
    assert image_loader._is_blue_band_path("IMG_0000_1.TIF") is True
    assert image_loader._is_blue_band_path("img_3332_1.tif") is True
    assert image_loader._is_blue_band_path("img_3332_2.tif") is False
    assert image_loader._is_blue_band_path("img_3332_3.tif") is False
    assert image_loader._is_blue_band_path("img_3332_4.tif") is False
    assert image_loader._is_blue_band_path("img_3332_5.tif") is False

    assert image_loader._is_blue_band_path(Path("DJI_0014.TIF")) is False
    assert image_loader._is_blue_band_path(Path("DJI_2224.TIF")) is False
    assert image_loader._is_blue_band_path(Path("dji_0014.tif")) is False
    assert image_loader._is_blue_band_path(Path("DJI_0013.TIF")) is False
    assert image_loader._is_blue_band_path(Path("DJI_0015.TIF")) is False
    assert image_loader._is_blue_band_path(Path("DJI_00014.TIF")) is False
    assert image_loader._is_blue_band_path(Path("IMG_1234_1.TIF")) is True
    assert image_loader._is_blue_band_path(Path("IMG_9999_1.TIF")) is True
    assert image_loader._is_blue_band_path(Path("IMG_0000_1.TIF")) is True
    assert image_loader._is_blue_band_path(Path("img_3332_1.tif")) is True
    assert image_loader._is_blue_band_path(Path("img_3332_2.tif")) is False
    assert image_loader._is_blue_band_path(Path("img_3332_3.tif")) is False
    assert image_loader._is_blue_band_path(Path("img_3332_4.tif")) is False
    assert image_loader._is_blue_band_path(Path("img_3332_5.tif")) is False

    assert image_loader._is_blue_band_path(Path("/home/dir/img_3332_1.tif")) is True
    assert image_loader._is_blue_band_path(Path("/home/dir/img_3332_5.tif")) is False
    assert image_loader._is_blue_band_path("/home/dir/img_3332_1.tif") is True
    assert image_loader._is_blue_band_path("/home/dir/img_3332_5.tif") is False

    assert image_loader._is_blue_band_path(Path("C:\\Users\\some\\dir\\img_3332_1.tif")) is True
    assert image_loader._is_blue_band_path(Path("C:\\Users\\some\\dir\\img_3332_5.tif")) is False
    assert image_loader._is_blue_band_path("C:\\Users\\some\\dir\\img_3332_1.tif") is True
    assert image_loader._is_blue_band_path("C:\\Users\\some\\dir\\img_3332_5.tif") is False


def test_rgb_loader(tmp_path):
    """Test that RGB loader works with 8-bit images."""
    # Create test RGB images
    test_files = ["image1.jpg", "image2.png", "image3.tiff"]

    for filename in test_files:
        img = create_test_image_8bit(height=64, width=64, channels=3)
        img.save(tmp_path / filename)

    # Test the class
    image_loader = RGBLoader(tmp_path, tmp_path / "masks")

    assert len(image_loader) == len(test_files)
    assert image_loader._bit_depth == 8

    # Test image loading
    img_path = str(tmp_path / test_files[0])
    loaded_img = image_loader.load_image(img_path)
    assert loaded_img.shape[2] == 3  # RGB channels
    assert loaded_img.dtype == float


def test_cir_loader(tmp_path):
    """Test that CIR loader works with 4-band images."""
    # Create test CIR image (4-band)
    img = create_test_image_8bit(height=128, width=128, channels=4)
    test_file = "cir_image.tif"
    img.save(tmp_path / test_file)

    # Test the class
    image_loader = CIRLoader(tmp_path, tmp_path / "masks")

    assert len(image_loader) == 1
    assert image_loader._bit_depth == 8
    assert image_loader._crop_size == 256

    # Test image loading
    img_path = str(tmp_path / test_file)
    loaded_img = image_loader.load_image(img_path)
    assert loaded_img.shape[2] == 4  # CIR channels


def test_djim3m_loader(tmp_path):
    """Test DJI M3M image loader functionality."""
    # Create test files with DJI M3M naming pattern
    test_files = [
        "DJI_20221208115250_0001_G.TIF",
        "DJI_20221208115250_0001_R.TIF",
        "DJI_20221208115250_0001_RE.TIF",
        "DJI_20221208115250_0001_NIR.TIF",
        "DJI_20221208115253_0002_G.TIF",
        "DJI_20221208115253_0002_R.TIF",
        "DJI_20221208115253_0002_RE.TIF",
        "DJI_20221208115253_0002_NIR.TIF",
        "other_file.txt",  # Should be ignored
    ]

    for filename in test_files:
        if filename.endswith(".TIF"):
            img = create_test_image_16bit(height=32, width=32, add_glint=True)
            img.save(tmp_path / filename)
        else:
            (tmp_path / filename).write_text("not an image")

    mask_dir = tmp_path / "masks"
    image_loader = DJIM3MLoader(tmp_path, mask_dir)

    # Test bit depth
    assert image_loader._bit_depth == 16

    # Test green band pattern matching
    assert image_loader._is_green_band_path("DJI_20221208115250_0001_G.TIF")
    assert image_loader._is_green_band_path("/path/to/DJI_20221208115250_0001_G.TIF")
    assert image_loader._is_green_band_path(Path("DJI_20221208115250_0001_G.TIF"))
    assert not image_loader._is_green_band_path("DJI_20221208115250_0001_R.TIF")
    assert not image_loader._is_green_band_path("other_file.txt")

    # Test green band path extraction
    green_band_paths = list(image_loader._green_band_paths)
    expected_green_paths = [
        str(tmp_path / "DJI_20221208115250_0001_G.TIF"),
        str(tmp_path / "DJI_20221208115253_0002_G.TIF"),
    ]
    assert sorted(green_band_paths) == sorted(expected_green_paths)

    # Test band path generation from green band path
    green_path = tmp_path / "DJI_20221208115250_0001_G.TIF"
    band_paths = DJIM3MLoader._green_band_path_to_band_paths(green_path)
    expected_band_paths = [
        str(tmp_path / "DJI_20221208115250_0001_G.TIF"),
        str(tmp_path / "DJI_20221208115250_0001_R.TIF"),
        str(tmp_path / "DJI_20221208115250_0001_RE.TIF"),
        str(tmp_path / "DJI_20221208115250_0001_NIR.TIF"),
    ]
    assert band_paths == expected_band_paths

    # Test complete path grouping
    all_paths = list(image_loader.paths)
    assert len(all_paths) == 2  # Two captures
    assert len(all_paths[0]) == 4  # Four bands per capture
    assert len(all_paths[1]) == 4  # Four bands per capture

    # Test image loading with all 4 bands
    capture_paths = [
        str(tmp_path / "DJI_20221208115250_0001_G.TIF"),
        str(tmp_path / "DJI_20221208115250_0001_R.TIF"),
        str(tmp_path / "DJI_20221208115250_0001_RE.TIF"),
        str(tmp_path / "DJI_20221208115250_0001_NIR.TIF"),
    ]
    loaded_img = image_loader.load_image(capture_paths)
    assert loaded_img.shape[2] == 4  # Four bands (G, R, RE, NIR)


@pytest.fixture
def sensor_test_images(tmp_path):
    """Fixture that creates test images for all sensor types."""
    # RGB images
    rgb_files = ["rgb1.jpg", "rgb2.png"]
    for filename in rgb_files:
        img = create_test_image_8bit(height=64, width=64, channels=3, add_glint=True)
        img.save(tmp_path / filename)

    # CIR image
    cir_img = create_test_image_8bit(height=256, width=256, channels=4, add_glint=True)
    cir_img.save(tmp_path / "cir_large.tif")

    # MicaSense RedEdge files (5 bands)
    for capture_id in [1234, 5678]:
        for band in range(1, 6):
            img = create_test_image_16bit(height=32, width=32, add_glint=(band == 1))
            img.save(tmp_path / f"IMG_{capture_id}_{band}.tif")

    # P4MS files - Use 3-digit capture IDs and proper band naming
    for capture_id in [101, 201]:  # 3-digit capture IDs for P4MS
        for band in range(1, 6):
            img = create_test_image_16bit(height=32, width=32, add_glint=(band == 1))
            img.save(tmp_path / f"DJI_{capture_id}{band}.TIF")

    # DJI M3M files
    for capture_id in ["20221208115250_0001", "20221208115253_0002"]:
        for band in ["G", "R", "RE", "NIR"]:
            img = create_test_image_16bit(height=32, width=32, add_glint=(band == "G"))
            img.save(tmp_path / f"DJI_{capture_id}_{band}.TIF")

    return tmp_path


def test_all_sensor_bit_depths(sensor_test_images):
    """Test that all sensors report correct bit depths."""
    tmp_path = sensor_test_images
    mask_dir = tmp_path / "masks"

    loaders = [
        (RGBLoader(tmp_path, mask_dir), 8),
        (CIRLoader(tmp_path, mask_dir), 8),
        (MicasenseRedEdgeLoader(tmp_path, mask_dir), 16),
        (P4MSLoader(tmp_path, mask_dir), 16),
        (DJIM3MLoader(tmp_path, mask_dir), 16),
    ]

    for loader, expected_bit_depth in loaders:
        assert loader._bit_depth == expected_bit_depth
