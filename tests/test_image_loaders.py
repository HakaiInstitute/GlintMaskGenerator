"""Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-06-12
"""

from pathlib import Path

import numpy as np
from PIL import Image

from glint_mask_tools.image_loaders import (
    BigTiffLoader,
    DJIM3MLoader,
    MicasenseRedEdgeDualLoader,
    MicasenseRedEdgeLoader,
    P4MSLoader,
    SingleFileImageLoader,
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
            "DJI_2021.tif",
            "DJI_2022.tif",
            "DJI_2023.tif",
            "DJI_2024.tif",
            "DJI_2025.tif",
        ],
    ]
    valid_paths = [[str(Path(tmp_path) / p) for p in paths] for paths in valid_paths]
    valid_paths = sorted(valid_paths, key=lambda a: a[0])

    masker_paths = sorted(image_loader.paths, key=lambda a: a[0])
    assert len(image_loader) == len(valid_paths)
    assert np.array_equal(np.array(masker_paths), np.array(valid_paths))


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
            "IMG_4321_1.TIF",
            "IMG_4321_2.TIF",
            "IMG_4321_3.TIF",
            "IMG_4321_4.TIF",
            "IMG_4321_5.TIF",
        ],
    ]
    valid_paths = [[str(Path(tmp_path) / p) for p in paths] for paths in valid_paths]
    valid_paths = sorted(valid_paths, key=lambda a: a[0])

    masker_paths = sorted(image_loader.paths, key=lambda a: a[0])
    assert len(image_loader) == len(valid_paths)
    assert np.array_equal(np.array(masker_paths), np.array(valid_paths))


def test_micasense_red_edge_dual_loader(tmp_path):
    """Test that Micasense Red Edge Dual loader finds the correct files and has 10 bands."""
    # Create dummy files for dual sensor (10 bands)
    test_files = [f"IMG_1234_{i}.tif" for i in range(1, 11)] + [f"IMG_5678_{i}.tif" for i in range(1, 11)]

    for filename in test_files:
        img = create_test_image_16bit(height=32, width=32)
        img.save(tmp_path / filename)

    # Test the class
    image_loader = MicasenseRedEdgeDualLoader(tmp_path, tmp_path / "masks")

    assert len(image_loader) == 2  # Two capture sets
    assert image_loader._num_bands == 10

    # Test paths grouping
    paths = list(image_loader.paths)
    assert len(paths) == 2
    assert len(paths[0]) == 10  # 10 bands per capture
    assert len(paths[1]) == 10

    # Test image loading
    loaded_img = image_loader.load_image(paths[0])
    assert loaded_img.shape[2] == 10  # 10 bands


def test_rgb_loader(tmp_path):
    """Test that RGB loader works with 8-bit images."""
    # Create test RGB images
    test_files = ["image1.jpg", "image2.png", "image3.tiff"]

    for filename in test_files:
        img = create_test_image_8bit(height=64, width=64, channels=3)
        img.save(tmp_path / filename)

    # Test the class
    image_loader = SingleFileImageLoader(tmp_path, tmp_path / "masks")

    assert len(image_loader) == len(test_files)

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
    image_loader = BigTiffLoader(tmp_path, tmp_path / "masks")

    assert len(image_loader) == 1

    # Test image loading
    img_path = str(tmp_path / test_file)
    loaded_img = image_loader.load_image(img_path)
    assert loaded_img.shape[2] == 4  # CIR channels


def test_djim3m_loader(tmp_path):
    """Test DJI M3M image loader functionality."""
    # Create test files with DJI M3M naming pattern
    test_files = [
        "DJI_20221208115250_0001_MS_G.TIF",
        "DJI_20221208115250_0001_MS_R.TIF",
        "DJI_20221208115250_0001_MS_RE.TIF",
        "DJI_20221208115250_0001_MS_NIR.TIF",
        "DJI_20221208115253_0002_MS_G.TIF",
        "DJI_20221208115253_0002_MS_R.TIF",
        "DJI_20221208115253_0002_MS_RE.TIF",
        "DJI_20221208115253_0002_MS_NIR.TIF",
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

    # Test complete path grouping
    all_paths = list(image_loader.paths)
    assert len(all_paths) == 2  # Two captures
    assert len(all_paths[0]) == 4  # Four bands per capture
    assert len(all_paths[1]) == 4  # Four bands per capture

    # Test image loading with all 4 bands
    capture_paths = [
        str(tmp_path / "DJI_20221208115250_0001_MS_G.TIF"),
        str(tmp_path / "DJI_20221208115250_0001_MS_R.TIF"),
        str(tmp_path / "DJI_20221208115250_0001_MS_RE.TIF"),
        str(tmp_path / "DJI_20221208115250_0001_MS_NIR.TIF"),
    ]
    loaded_img = image_loader.load_image(capture_paths)
    assert loaded_img.shape[2] == 4  # Four bands (G, R, RE, NIR)
