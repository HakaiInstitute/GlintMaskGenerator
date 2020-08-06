"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-06-12
"""

from abc import ABC
from pathlib import Path

import numpy as np
from PIL import Image

from core.abstract_masker import Masker
from core.specular_maskers import RGBSpecularMasker

IMG_CONTENT = Image.fromarray(np.ones((32, 32, 3)).astype(np.uint8) * 255)


def test_class_inheritance():
    """Test that the classes have the correct superclasses."""
    assert issubclass(Masker, ABC)
    assert issubclass(RGBSpecularMasker, Masker)


def test_rgb_specular_masker(tmp_path):
    """Test that the RGB specular masker finds the correct files."""
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
    masker = RGBSpecularMasker(tmp_path, Path(tmp_path).joinpath("masks"))

    valid_paths = all_paths
    valid_paths = sorted([str(tmp_path.joinpath(n)) for n in valid_paths])
    assert len(masker) == len(valid_paths)
    assert all(np.array(sorted(masker.img_paths)) == np.array(sorted(valid_paths)))
