"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-06-12
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

from core.image_loaders import P4MSLoader, MicasenseRedEdgeLoader

IMG_CONTENT = Image.fromarray(np.ones((32, 32, 3)).astype(np.uint8) * 255)


def test_p4ms_loader(tmp_path):
    """Test that DJI Multispectral masker finds the correct files."""
    # Create dummy files
    all_paths = ["1.jpg", "2.JPG", "3.tif", "4.tiff", "5.TIF", "6.TIFF", "7.png", "8.PNG", "9.jpeg", "10.JPEG",
                 "IMG_1234_1.tif", "IMG_1234_2.tif", "IMG_1234_3.tif", "IMG_1234_4.tif", "IMG_1234_5.tif",
                 "IMG_4321_1.TIF", "IMG_4321_2.TIF", "IMG_4321_3.TIF", "IMG_4321_4.TIF", "IMG_4321_5.TIF",
                 "DJI_1021.TIF", "DJI_1022.TIF", "DJI_1023.TIF", "DJI_1024.TIF", "DJI_1025.TIF",
                 "DJI_2021.tif", "DJI_2022.tif", "DJI_2023.tif", "DJI_2024.tif", "DJI_2025.tif"]
    all_paths = sorted([str(tmp_path.joinpath(n)) for n in all_paths])
    for name in all_paths:
        IMG_CONTENT.save(name)

    # Test the class methods
    image_loader = P4MSLoader(tmp_path, str(Path(tmp_path).joinpath("masks")))

    valid_paths = [
        ["DJI_1021.TIF", "DJI_1022.TIF", "DJI_1023.TIF", "DJI_1024.TIF", "DJI_1025.TIF"],
        ["DJI_2021.TIF", "DJI_2022.TIF", "DJI_2023.TIF", "DJI_2024.TIF", "DJI_2025.TIF"]
    ]
    valid_paths = [[os.path.join(tmp_path, p) for p in paths] for paths in valid_paths]
    valid_paths = sorted(valid_paths, key=lambda a: a[0])

    masker_paths = sorted(list(image_loader.paths), key=lambda a: a[0])
    assert len(image_loader) == len(valid_paths)
    assert (np.array(masker_paths) == np.array(valid_paths)).all()

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
    assert not image_loader._is_blue_band_path(Path("C:\\Users\\some\\dir\\DJI_0013.TIF"))
    assert image_loader._is_blue_band_path("C:\\Users\\some\\dir\\dji_0011.tif") is True
    assert image_loader._is_blue_band_path("C:\\Users\\some\\dir\\DJI_0013.TIF") is False


def test_micasense_red_edge_masker(tmp_path):
    """Test that the Micasense masker finds the correct files."""
    # Create dummy files
    all_paths = ["1.jpg", "2.JPG", "3.tif", "4.tiff", "5.TIF", "6.TIFF", "7.png", "8.PNG", "9.jpeg", "10.JPEG",
                 "IMG_1234_1.tif", "IMG_1234_2.tif", "IMG_1234_3.tif", "IMG_1234_4.tif", "IMG_1234_5.tif",
                 "IMG_4321_1.TIF", "IMG_4321_2.TIF", "IMG_4321_3.TIF", "IMG_4321_4.TIF", "IMG_4321_5.TIF",
                 "DJI_1021.TIF", "DJI_1022.TIF", "DJI_1023.TIF", "DJI_1024.TIF", "DJI_1025.TIF",
                 "DJI_2021.tif", "DJI_2022.tif", "DJI_2023.tif", "DJI_2024.tif", "DJI_2025.tif"]
    all_paths = sorted([str(tmp_path.joinpath(n)) for n in all_paths])
    for name in all_paths:
        IMG_CONTENT.save(name)

    # Test the class methods
    image_loader = MicasenseRedEdgeLoader(tmp_path, Path(tmp_path).joinpath("masks"))

    valid_paths = [
        ["IMG_1234_1.tif", "IMG_1234_2.tif", "IMG_1234_3.tif", "IMG_1234_4.tif", "IMG_1234_5.tif"],
        ["IMG_4321_1.tif", "IMG_4321_2.tif", "IMG_4321_3.tif", "IMG_4321_4.tif", "IMG_4321_5.tif"]
    ]
    valid_paths = [[os.path.join(tmp_path, p) for p in paths] for paths in valid_paths]
    valid_paths = sorted(valid_paths, key=lambda a: a[0])

    masker_paths = sorted(list(image_loader.paths), key=lambda a: a[0])
    assert len(image_loader) == len(valid_paths)
    assert (np.array(masker_paths) == np.array(valid_paths)).all()

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
