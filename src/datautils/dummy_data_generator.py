import numpy as np
import tifffile
import h5py
import cv2
import zarr
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def generate_2d_array(shape: Tuple[int, int]) -> np.ndarray:
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def generate_3d_array(shape: Tuple[int, int, int]) -> np.ndarray:
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def generate_4d_array(shape: Tuple[int, int, int, int]) -> np.ndarray:
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def generate_5d_array(shape: Tuple[int, int, int, int, int]) -> np.ndarray:
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def save_tiff(data: np.ndarray, output_path: Path) -> None:
    tifffile.imwrite(str(output_path), data)


def save_hdf5(data: np.ndarray, output_path: Path, dataset_name: str = "data") -> None:
    with h5py.File(output_path, "w") as f:
        f.create_dataset(dataset_name, data=data)


def save_video(data: np.ndarray, output_path: Path, fps: int = 30) -> None:
    """
    Save a numpy array as a video file, supporting both grayscale and color videos.

    :param data: Input numpy array with shape (T, H, W) for grayscale or (T, H, W, 3) for color
    :param output_path: Path to save the output video file
    :param fps: Frames per second for the output video
    """
    if data.ndim not in (3, 4):
        raise ValueError(f"Unsupported data shape for video: {data.shape}")

    is_color = data.ndim == 4
    height, width = data.shape[1:3]

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height), isColor=is_color
    )

    try:
        for frame in data:
            out.write(frame.astype(np.uint8))
    finally:
        out.release()


def save_zarr(data: np.ndarray, output_path: Path, dataset_name: str = "data") -> None:
    """
    Save a numpy array as a Zarr array.

    :param data: Input numpy array
    :param output_path: Path to save the output Zarr array
    :param dataset_name: Name of the dataset within the Zarr array
    """
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset(dataset_name, data=data, chunks=True)


def save_image_folder(
    data: np.ndarray, output_path: Path, prefix: str = "image", file_format: str = "tiff"
) -> None:
    """
    Save a numpy array as a folder of images.

    :param data: Input numpy array with shape (T, ...) where T is the number of images
    :param output_path: Path to save the output folder
    :param prefix: Prefix for the image filenames
    :param file_format: Format to save the images (tiff or png)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(data):
        if file_format == "tiff":
            tifffile.imwrite(str(output_path / f"{prefix}_{i:04d}.tiff"), img)
        elif file_format == "png":
            cv2.imwrite(str(output_path / f"{prefix}_{i:04d}.png"), img)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")


def generate_multi_dataset_hdf5(output_path: Path) -> None:
    """
    Generate an HDF5 file with multiple datasets and groups.

    :param output_path: Path to save the HDF5 file
    """
    logger.info(f"Generating multi-dataset HDF5 file: {output_path}")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("dataset1", data=np.random.rand(50, 512, 512))
        f.create_dataset("dataset2", data=np.random.rand(50, 512, 512, 3))
        
        group1 = f.create_group("group1")
        group1.create_dataset("dataset3", data=np.random.rand(10, 10, 10, 10))
        group1.create_dataset("dataset4", data=np.random.rand(50, 50, 512, 512, 3))
        
        group2 = f.create_group("group2")
        group2.create_dataset("dataset5", data=np.random.rand(10, 10, 10, 10))
        group2.create_dataset("dataset6", data=np.random.rand(10, 10, 10, 10, 3))


def generate_multi_dataset_zarr(output_path: Path) -> None:
    """
    Generate a Zarr file with multiple datasets and groups.

    :param output_path: Path to save the Zarr file
    """
    logger.info(f"Generating multi-dataset Zarr file: {output_path}")
    root = zarr.open(str(output_path), mode="w")
    root.create_dataset("dataset1", data=np.random.rand(50, 512, 512))
    root.create_dataset("dataset2", data=np.random.rand(50, 512, 512, 3))
    
    group1 = root.create_group("group1")
    group1.create_dataset("dataset3", data=np.random.rand(10, 10, 10, 10))
    group1.create_dataset("dataset4", data=np.random.rand(10, 10, 10, 10, 3))
    
    group2 = root.create_group("group2")
    group2.create_dataset("dataset5", data=np.random.rand(10, 10, 10, 10))
    group2.create_dataset("dataset6", data=np.random.rand(50, 50, 512, 512, 3))


def generate_test_data(
    output_dir: Path,
    shape: Tuple[int, int] = (512, 512),
    z_size: int = 50,
    t_size: int = 50,
    c_size: int = 3,
) -> None:
    """
    Generate test data with various dimensions and save in different formats.

    :param output_dir: Path to the directory where test data will be saved
    :param shape: Tuple of (height, width) for 2D arrays
    :param z_size: Size of the Z dimension for 3D and higher dimensional arrays
    :param t_size: Size of the T (time) dimension for time series data
    :param c_size: Size of the C (channel) dimension for multi-channel data
    """
    logger.info(f"Generating test data in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_XY = f"{shape[0]}x{shape[1]}"
    shape_TXY = f"{t_size}x{shape[0]}x{shape[1]}"
    shape_XYC = f"{shape[0]}x{shape[1]}x{c_size}"
    shape_TXYC = f"{t_size}x{shape[0]}x{shape[1]}x{c_size}"
    shape_TZCYX = f"{t_size}x{z_size}x{shape[0]}x{shape[1]}x{c_size}"
    shape_ZXY = f"{z_size}x{shape[0]}x{shape[1]}"
    shape_TZXY = f"{t_size}x{z_size}x{shape[0]}x{shape[1]}"
    shape_ZXYC = f"{z_size}x{shape[0]}x{shape[1]}x{c_size}"

    # Generate and save 2D array (single image)
    array_2d = generate_2d_array(shape)
    save_tiff(array_2d, output_dir / f"test_2d_XY_{shape_XY}.tiff")
    save_hdf5(array_2d, output_dir / f"test_2d_XY_{shape_XY}.h5")
    save_zarr(array_2d, output_dir / f"test_2d_XY_{shape_XY}.zarr")

    # Generate and save 3D array (time series or RGB image)
    array_3d_time = generate_3d_array((t_size, *shape))
    save_tiff(array_3d_time, output_dir / f"test_3d_TXY_{shape_TXY}.tiff")
    save_hdf5(array_3d_time, output_dir / f"test_3d_TXY_{shape_TXY}.h5")
    save_zarr(array_3d_time, output_dir / f"test_3d_TXY_{shape_TXY}.zarr")
    save_image_folder(array_3d_time, output_dir / f"test_3d_TXY_{shape_TXY}_folder")

    array_3d_rgb = generate_3d_array((*shape, c_size))
    save_tiff(array_3d_rgb, output_dir / f"test_3d_XYC_{shape_XYC}.tiff")
    save_hdf5(array_3d_rgb, output_dir / f"test_3d_XYC_{shape_XYC}.h5")

    # Generate and save 4D array (time series with channels)
    array_4d = generate_4d_array((t_size, *shape, c_size))
    save_tiff(array_4d, output_dir / f"test_4d_TXYC_{shape_TXYC}.tiff")
    save_hdf5(array_4d, output_dir / f"test_4d_TXYC_{shape_TXYC}.h5")
    save_zarr(array_4d, output_dir / f"test_4d_TXYC_{shape_TXYC}.zarr")
    save_video(array_4d, output_dir / f"test_4d_TXYC_{shape_TXYC}.avi")
    save_image_folder(array_4d, output_dir / f"test_4d_TXYC_{shape_TXYC}_folder")

    # Generate and save 5D array
    array_5d = generate_5d_array((t_size, z_size, *shape, c_size))
    save_tiff(array_5d, output_dir / f"test_5d_TZCYX_{shape_TZCYX}.tiff")
    save_hdf5(array_5d, output_dir / f"test_5d_TZCYX_{shape_TZCYX}.h5")
    save_zarr(array_5d, output_dir / f"test_5d_TZCYX_{shape_TZCYX}.zarr")

    # Generate and save 3D Z-stack
    array_3d_z = generate_3d_array((z_size, *shape))
    save_tiff(array_3d_z, output_dir / f"test_3d_ZXY_{shape_ZXY}.tiff")
    save_hdf5(array_3d_z, output_dir / f"test_3d_ZXY_{shape_ZXY}.h5")
    save_zarr(array_3d_z, output_dir / f"test_3d_ZXY_{shape_ZXY}.zarr")

    # Generate and save 4D Z-stack time series
    array_4d_zt = generate_4d_array((t_size, z_size, *shape))
    save_tiff(array_4d_zt, output_dir / f"test_4d_TZXY_{shape_TZXY}.tiff")
    save_hdf5(array_4d_zt, output_dir / f"test_4d_TZXY_{shape_TZXY}.h5")
    save_zarr(array_4d_zt, output_dir / f"test_4d_TZXY_{shape_TZXY}.zarr")

    # Generate and save 4D Z-stack RGB
    array_4d_zrgb = generate_4d_array((z_size, *shape, c_size))
    save_tiff(array_4d_zrgb, output_dir / f"test_4d_ZXYC_{shape_ZXYC}.tiff")
    save_hdf5(array_4d_zrgb, output_dir / f"test_4d_ZXYC_{shape_ZXYC}.h5")
    save_zarr(array_4d_zrgb, output_dir / f"test_4d_ZXYC_{shape_ZXYC}.zarr")

    # Generate and save 2D grayscale video
    array_3d_gray = generate_3d_array((t_size, *shape))
    save_video(array_3d_gray, output_dir / f"test_3d_TXY_{shape_TXY}_grayscale.avi")

    # Generate and save image folder (2D grayscale)
    array_2d_folder = generate_3d_array((t_size, *shape))
    save_image_folder(array_2d_folder, output_dir / f"test_2d_XY_{shape_XY}_folder", file_format="png")

    # Generate and save image folder (2D RGB)
    array_3d_rgb_folder = generate_4d_array((t_size, *shape, c_size))
    save_image_folder(array_3d_rgb_folder, output_dir / f"test_3d_XYC_{shape_XYC}_folder", file_format="png")

    # Generate multi-dataset HDF5 and Zarr files
    generate_multi_dataset_hdf5(output_dir / "multi_dataset.h5")
    generate_multi_dataset_zarr(output_dir / "multi_dataset.zarr")

    logger.info(f"Test data generated and saved in {output_dir}")


if __name__ == "__main__":
    output_dir = Path("test_data")
    generate_test_data(output_dir)
    print(f"Test data generated and saved in {output_dir}")