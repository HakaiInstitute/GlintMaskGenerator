import numpy as np
from PIL import Image

from core.common import get_img_paths

IMG_CONTENT = Image.fromarray(np.ones((32, 32, 3)).astype(np.uint8) * 255)


def _clean_dir(dir_path):
    for f in dir_path.glob('*'):
        f.unlink()


def test_get_img_paths(tmp_path):
    # Test detection of RGB files
    _clean_dir(tmp_path)
    true_paths = ["1.jpg", "2.JPG", "3.tif", "4.tiff", "5.TIF", "6.TIFF", "7.png", "8.PNG", "9.jpeg", "10.JPEG"]

    true_paths = sorted([str(tmp_path.joinpath(n)) for n in true_paths])
    for name in true_paths:
        IMG_CONTENT.save(name)

    img_paths = sorted(get_img_paths(str(tmp_path), str(tmp_path), red_edge=False))
    assert len(true_paths) == len(img_paths)
    assert all(np.array(true_paths) == np.array(img_paths))

    # Test detection of RedEdge files
    _clean_dir(tmp_path)
    true_paths = [
        "IMG_1234_1.tif", "IMG_1234_2.tif", "IMG_1234_3.tif", "IMG_1234_4.tif", "IMG_1234_5.tif",
        "IMG_4321_1.TIF", "IMG_4321_2.TIF", "IMG_4321_3.TIF", "IMG_4321_4.TIF", "IMG_4321_5.TIF",
        "DJI_1021.TIF", "DJI_1022.TIF", "DJI_1023.TIF", "DJI_1024.TIF", "DJI_1025.TIF",
        "DJI_2021.tif", "DJI_2022.tif", "DJI_2023.tif", "DJI_2024.tif", "DJI_2025.tif",
    ]
    true_paths = sorted([str(tmp_path.joinpath(n)) for n in true_paths])
    for name in true_paths:
        IMG_CONTENT.save(name)

    valid_paths = ["IMG_1234_5.tif", "IMG_4321_5.TIF", "DJI_1024.TIF", "DJI_2024.tif"]
    valid_paths = sorted([str(tmp_path.joinpath(n)) for n in valid_paths])

    img_paths = sorted(get_img_paths(str(tmp_path), str(tmp_path), red_edge=True))
    print(img_paths)
    assert len(valid_paths) == len(img_paths)
    assert all(np.array(valid_paths) == np.array(img_paths))
