import pytest
from pathlib import Path
from src.lazyloader.lazyloader import lazyload, configure_load_options, LoadOptions
from src.datautils.dummy_data_generator import generate_test_data
import shutil
import numpy as np
import os


@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """Generate test data and return path to test directory."""
    test_dir = tmp_path_factory.mktemp("test_data")
    generate_test_data(test_dir)
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


def test_load_2d_tiff(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_2d_XY_512x512.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (512, 512)
    assert dim_order == "XY"


def test_load_3d_tiff_time_series(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_3d_TXY_50x512x512.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 512, 512)
    assert dim_order in ["TXY", "ZXY"]


def test_load_3d_tiff_rgb(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_3d_XYC_512x512x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (512, 512, 3)
    assert dim_order == "XYC"


def test_load_4d_tiff(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_TXYC_50x512x512x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 512, 512, 3)
    assert dim_order in ["TXYC", "ZXYC"]


def test_load_2d_hdf5(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_2d_XY_512x512.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (512, 512)
    assert dim_order == "XY"


def test_load_3d_hdf5_time_series(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_3d_TXY_50x512x512.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 512, 512)
    assert dim_order in ["TXY", "ZXY"]


def test_load_3d_hdf5_rgb(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_3d_XYC_512x512x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (512, 512, 3)
    assert dim_order == "XYC"


def test_load_4d_hdf5(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_TXYC_50x512x512x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 512, 512, 3)
    assert dim_order in ["TXYC", "ZXYC"]


def test_load_video(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_TXYC_50x512x512x3.avi"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 512, 512, 3)
    assert dim_order == "TXYC"


def test_load_with_options(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_TXYC_50x512x512x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    options = configure_load_options(
        t_range=(10, 30), y_range=(100, 300), x_range=(200, 400)
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (20, 200, 200, 3)
    assert dim_order in ["TXYC", "ZXYC"]


def test_load_partial_video(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_TXYC_50x512x512x3.avi"
    assert file_path.exists(), f"File not found: {file_path}"
    options = configure_load_options(
        t_range=(5, 25), y_range=(100, 300), x_range=(200, 400)
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (20, 200, 200, 3)
    assert dim_order == "TXYC"


def test_load_5d_tiff(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_5d_TZCYX_50x50x512x512x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 50, 512, 512, 3)
    assert dim_order == "TZXYC"


def test_load_5d_hdf5(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_5d_TZCYX_50x50x512x512x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 50, 512, 512, 3)
    assert dim_order == "TZXYC"


def test_load_3d_tiff_z_stack(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_3d_ZXY_50x512x512.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 512, 512)
    assert dim_order in ["ZXY", "TXY"]


def test_load_4d_tiff_z_stack_time_series(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_TZXY_50x50x512x512.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 50, 512, 512)
    assert dim_order == "TZXY"


def test_load_4d_tiff_z_stack_rgb(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_ZXYC_50x512x512x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert data.shape == (50, 512, 512, 3)
    assert dim_order in ["ZXYC", "TXYC"]


def test_load_with_options_5d(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_5d_TZCYX_50x50x512x512x3.tiff"
    assert file_path.exists(), f"File not found: {file_path}"
    options = configure_load_options(
        t_range=(5, 15), z_range=(2, 8), y_range=(100, 300), x_range=(200, 400)
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 6, 200, 200, 3)
    assert dim_order == "TZXYC"


def test_load_2d_grayscale_video(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_3d_TXY_50x512x512_grayscale.avi"
    assert file_path.exists(), f"File not found: {file_path}"
    options = configure_load_options(target_order="TXY")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 512, 512)
    assert dim_order == "TXY"


def test_load_partial_5d_hdf5(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_5d_TZCYX_50x50x512x512x3.h5"
    assert file_path.exists(), f"File not found: {file_path}"
    options = configure_load_options(
        t_range=(5, 15),
        z_range=(2, 8),
        y_range=(100, 300),
        x_range=(200, 400),
        c_range=(0, 2),
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 6, 200, 200, 2)
    assert dim_order == "TZXYC"


def test_load_2d_zarr(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_2d_XY_512x512.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (512, 512)
    assert dim_order == "XY"


def test_load_3d_zarr_time_series(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_3d_TXY_50x512x512.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 512, 512)
    assert dim_order in ["TXY", "ZXY"]


def test_load_4d_zarr(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_TXYC_50x512x512x3.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 512, 512, 3)
    assert dim_order in ["TXYC", "ZXYC"]


def test_load_5d_zarr(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_5d_TZCYX_50x50x512x512x3.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 50, 512, 512, 3)
    assert dim_order == "TZXYC"


def test_load_3d_zarr_z_stack(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_3d_ZXY_50x512x512.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 512, 512)
    assert dim_order in ["ZXY", "TXY"]


def test_load_4d_zarr_z_stack_time_series(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_TZXY_50x50x512x512.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 50, 512, 512)
    assert dim_order == "TZXY"


def test_load_4d_zarr_z_stack_rgb(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_4d_ZXYC_50x512x512x3.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    data, dim_order, metadata = lazyload(file_path)
    assert isinstance(data, np.ndarray)
    assert data.shape == (50, 512, 512, 3)
    assert dim_order in ["ZXYC", "TXYC"]


def test_load_zarr_with_options(test_data_dir: Path) -> None:
    file_path = test_data_dir / "test_5d_TZCYX_50x50x512x512x3.zarr"
    assert file_path.exists(), f"File not found: {file_path}"
    options = configure_load_options(
        t_range=(5, 15),
        z_range=(2, 8),
        y_range=(100, 300),
        x_range=(200, 400),
        c_range=(0, 2),
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert isinstance(data, np.ndarray)
    assert data.shape == (10, 6, 200, 200, 2)
    assert dim_order == "TZXYC"


def test_load_2d_grayscale_image_folder(test_data_dir: Path) -> None:
    folder_path = test_data_dir / "test_2d_XY_512x512_folder"
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    options = configure_load_options(target_order="TXY")
    data, dim_order, metadata = lazyload(folder_path, options)
    assert data.shape == (50, 512, 512)
    assert dim_order == "TXY"
    assert len(os.listdir(folder_path)) == 50


def test_load_3d_rgb_image_folder(test_data_dir: Path) -> None:
    folder_path = test_data_dir / "test_3d_XYC_512x512x3_folder"
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    options = configure_load_options(target_order="TXYC")
    data, dim_order, metadata = lazyload(folder_path, options)
    assert data.shape == (50, 512, 512, 3)
    assert dim_order == "TXYC"
    assert len(os.listdir(folder_path)) == 50


def test_load_partial_2d_grayscale_image_folder(test_data_dir: Path) -> None:
    folder_path = test_data_dir / "test_2d_XY_512x512_folder"
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    options = configure_load_options(
        t_range=(10, 30), y_range=(100, 300), x_range=(200, 400), target_order="TXY"
    )
    data, dim_order, metadata = lazyload(folder_path, options)
    assert data.shape == (20, 200, 200)
    assert dim_order == "TXY"


def test_load_partial_3d_rgb_image_folder(test_data_dir: Path) -> None:
    folder_path = test_data_dir / "test_3d_XYC_512x512x3_folder"
    assert folder_path.exists(), f"Folder not found: {folder_path}"
    options = configure_load_options(
        t_range=(10, 30), y_range=(100, 300), x_range=(200, 400), c_range=(1, 3), target_order="TXYC"
    )
    data, dim_order, metadata = lazyload(folder_path, options)
    assert data.shape == (20, 200, 200, 2)
    assert dim_order == "TXYC"


def test_load_hdf5_multi_dataset(test_data_dir: Path) -> None:
    file_path = test_data_dir / "multi_dataset.h5"
    assert file_path.exists(), f"File not found: {file_path}"

    # Test loading the first dataset using integer selector
    options = LoadOptions(dataset=0)
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 512, 512)
    assert dim_order in ["TXY", "ZXY"]

    # Test loading the second dataset using integer selector
    options = LoadOptions(dataset=1)
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 512, 512, 3)
    assert dim_order in ["TXYC", "ZXYC"]

    # Test loading a dataset from a group using string selectors
    options = LoadOptions(group="group1", dataset="dataset3")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 10, 10, 10)
    assert dim_order == "TZXY"

    # Test loading a large dataset from a group using string selectors
    options = LoadOptions(group="group1", dataset="dataset4")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 50, 512, 512, 3)
    assert dim_order == "TZXYC"


def test_load_zarr_multi_dataset(test_data_dir: Path) -> None:
    file_path = test_data_dir / "multi_dataset.zarr"
    assert file_path.exists(), f"File not found: {file_path}"

    # Test loading the first dataset using integer selector
    options = LoadOptions(dataset=0)
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 512, 512)
    assert dim_order in ["TXY", "ZXY"]

    # Test loading the second dataset using integer selector
    options = LoadOptions(dataset=1)
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 512, 512, 3)
    assert dim_order in ["TXYC", "ZXYC"]

    # Test loading a dataset from a group using string selectors
    options = LoadOptions(group="group1", dataset="dataset3")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 10, 10, 10)
    assert dim_order == "TZXY"

    # Test loading a large dataset from a group using string selectors
    options = LoadOptions(group="group2", dataset="dataset6")
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (50, 50, 512, 512, 3)
    assert dim_order == "TZXYC"


def test_load_partial_hdf5_multi_dataset(test_data_dir: Path) -> None:
    file_path = test_data_dir / "multi_dataset.h5"
    assert file_path.exists(), f"File not found: {file_path}"

    options = configure_load_options(
        group="group1",
        dataset="dataset4",
        t_range=(5, 15),
        z_range=(2, 8),
        y_range=(100, 300),
        x_range=(200, 400),
        c_range=(0, 2)
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 6, 200, 200, 2)
    assert dim_order == "TZXYC"


def test_load_partial_zarr_multi_dataset(test_data_dir: Path) -> None:
    file_path = test_data_dir / "multi_dataset.zarr"
    assert file_path.exists(), f"File not found: {file_path}"

    options = configure_load_options(
        group="group2",
        dataset="dataset6",
        t_range=(5, 15),
        z_range=(2, 8),
        y_range=(100, 300),
        x_range=(200, 400),
        c_range=(0, 2)
    )
    data, dim_order, metadata = lazyload(file_path, options)
    assert data.shape == (10, 6, 200, 200, 2)
    assert dim_order == "TZXYC"

# Add more tests as needed for different scenarios and edge cases
