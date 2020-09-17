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
from core.bin_maskers import P4MSRedEdgeBinMasker, MicasenseRedEdgeBinMasker, BinMasker, RGBBinMasker

IMG_CONTENT = Image.fromarray(np.ones((32, 32, 3)).astype(np.uint8) * 255)


def test_class_inheritance():
    """Ensure that the maskers have the correct superclasses."""
    assert issubclass(Masker, ABC)
    assert issubclass(BinMasker, Masker)
    assert issubclass(P4MSRedEdgeBinMasker, BinMasker)
    assert issubclass(MicasenseRedEdgeBinMasker, BinMasker)
    assert issubclass(RGBBinMasker, BinMasker)


def test_dji_multispectral_masker(tmp_path):
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
    masker = P4MSRedEdgeBinMasker(tmp_path, Path(tmp_path).joinpath("masks"))

    valid_paths = ["DJI_1024.TIF", "DJI_2024.tif"]
    valid_paths = sorted([str(tmp_path.joinpath(n)) for n in valid_paths])
    assert len(masker) == len(valid_paths)
    assert all(np.array(sorted(masker.img_paths)) == np.array(sorted(valid_paths)))

    assert masker.is_dji_red_edge("DJI_0014.TIF") is True
    assert masker.is_dji_red_edge("DJI_2224.TIF") is True
    assert masker.is_dji_red_edge("dji_0014.tif") is True
    assert masker.is_dji_red_edge("DJI_0013.TIF") is False
    assert masker.is_dji_red_edge("DJI_0015.TIF") is False
    assert masker.is_dji_red_edge("DJI_00014.TIF") is False
    assert masker.is_dji_red_edge("IMG_1234_1.TIF") is False
    assert masker.is_dji_red_edge("IMG_1234_2.TIF") is False
    assert masker.is_dji_red_edge("IMG_1234_3.TIF") is False
    assert masker.is_dji_red_edge("IMG_1234_5.TIF") is False

    assert masker.is_dji_red_edge(Path("DJI_0014.TIF")) is True
    assert masker.is_dji_red_edge(Path("DJI_2224.TIF")) is True
    assert masker.is_dji_red_edge(Path("dji_0014.tif")) is True
    assert masker.is_dji_red_edge(Path("dji_0104.tif")) is True
    assert masker.is_dji_red_edge(Path("DJI_0013.TIF")) is False
    assert masker.is_dji_red_edge(Path("DJI_0015.TIF")) is False
    assert masker.is_dji_red_edge(Path("DJI_00014.TIF")) is False
    assert masker.is_dji_red_edge(Path("IMG_1234_1.TIF")) is False
    assert masker.is_dji_red_edge(Path("IMG_1234_2.TIF")) is False
    assert masker.is_dji_red_edge(Path("IMG_1234_3.TIF")) is False
    assert masker.is_dji_red_edge(Path("IMG_1234_5.TIF")) is False

    assert masker.is_dji_red_edge(Path("/home/dir/dji_0014.tif")) is True
    assert masker.is_dji_red_edge(Path("/home/dir/DJI_0013.TIF")) is False
    assert masker.is_dji_red_edge("/home/wherever/dji_0014.tif") is True
    assert masker.is_dji_red_edge("/home/wherever/DJI_0013.TIF") is False

    assert masker.is_dji_red_edge(Path("C:\\Users\\some\\dir\\dji_0014.tif")) is True
    assert not masker.is_dji_red_edge(Path("C:\\Users\\some\\dir\\DJI_0013.TIF"))
    assert masker.is_dji_red_edge("C:\\Users\\some\\dir\\dji_0014.tif") is True
    assert masker.is_dji_red_edge("C:\\Users\\some\\dir\\DJI_0013.TIF") is False


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
    masker = MicasenseRedEdgeBinMasker(tmp_path, Path(tmp_path).joinpath("masks"))

    valid_paths = ["IMG_1234_5.tif", "IMG_4321_5.TIF"]
    valid_paths = sorted([str(tmp_path.joinpath(n)) for n in valid_paths])
    assert len(masker) == len(valid_paths)
    assert all(np.array(sorted(masker.img_paths)) == np.array(sorted(valid_paths)))

    assert masker.is_micasense_red_edge("DJI_0014.TIF") is False
    assert masker.is_micasense_red_edge("DJI_2224.TIF") is False
    assert masker.is_micasense_red_edge("dji_0014.tif") is False
    assert masker.is_micasense_red_edge("DJI_0013.TIF") is False
    assert masker.is_micasense_red_edge("DJI_0015.TIF") is False
    assert masker.is_micasense_red_edge("DJI_00014.TIF") is False
    assert masker.is_micasense_red_edge("IMG_1234_5.TIF") is True
    assert masker.is_micasense_red_edge("IMG_9999_5.TIF") is True
    assert masker.is_micasense_red_edge("IMG_0000_5.TIF") is True
    assert masker.is_micasense_red_edge("img_3332_5.tif") is True
    assert masker.is_micasense_red_edge("img_3332_1.tif") is False
    assert masker.is_micasense_red_edge("img_3332_2.tif") is False
    assert masker.is_micasense_red_edge("img_3332_3.tif") is False
    assert masker.is_micasense_red_edge("img_3332_4.tif") is False

    assert masker.is_micasense_red_edge(Path("DJI_0014.TIF")) is False
    assert masker.is_micasense_red_edge(Path("DJI_2224.TIF")) is False
    assert masker.is_micasense_red_edge(Path("dji_0014.tif")) is False
    assert masker.is_micasense_red_edge(Path("DJI_0013.TIF")) is False
    assert masker.is_micasense_red_edge(Path("DJI_0015.TIF")) is False
    assert masker.is_micasense_red_edge(Path("DJI_00014.TIF")) is False
    assert masker.is_micasense_red_edge(Path("IMG_1234_5.TIF")) is True
    assert masker.is_micasense_red_edge(Path("IMG_9999_5.TIF")) is True
    assert masker.is_micasense_red_edge(Path("IMG_0000_5.TIF")) is True
    assert masker.is_micasense_red_edge(Path("img_3332_5.tif")) is True
    assert masker.is_micasense_red_edge(Path("img_3332_1.tif")) is False
    assert masker.is_micasense_red_edge(Path("img_3332_2.tif")) is False
    assert masker.is_micasense_red_edge(Path("img_3332_3.tif")) is False
    assert masker.is_micasense_red_edge(Path("img_3332_4.tif")) is False

    assert masker.is_micasense_red_edge(Path("/home/dir/img_3332_5.tif")) is True
    assert masker.is_micasense_red_edge(Path("/home/dir/img_3332_1.tif")) is False
    assert masker.is_micasense_red_edge("/home/dir/img_3332_5.tif") is True
    assert masker.is_micasense_red_edge("/home/dir/img_3332_1.tif") is False

    assert masker.is_micasense_red_edge(Path("C:\\Users\\some\\dir\\img_3332_5.tif")) is True
    assert masker.is_micasense_red_edge(Path("C:\\Users\\some\\dir\\img_3332_1.tif")) is False
    assert masker.is_micasense_red_edge("C:\\Users\\some\\dir\\img_3332_5.tif") is True
    assert masker.is_micasense_red_edge("C:\\Users\\some\\dir\\img_3332_1.tif") is False
