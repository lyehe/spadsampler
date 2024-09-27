import numpy as np
import tifffile
import h5py
import cv2
from pathlib import Path
from typing import Tuple

def generate_2d_array(shape: Tuple[int, int]) -> np.ndarray:
    return np.random.randint(0, 256, shape, dtype=np.uint8)

def generate_3d_array(shape: Tuple[int, int, int]) -> np.ndarray:
    return np.random.randint(0, 256, shape, dtype=np.uint8)

def generate_4d_array(shape: Tuple[int, int, int, int]) -> np.ndarray:
    return np.random.randint(0, 256, shape, dtype=np.uint8)

def save_tiff(data: np.ndarray, output_path: Path) -> None:
    tifffile.imwrite(str(output_path), data)

def save_hdf5(data: np.ndarray, output_path: Path, dataset_name: str = "data") -> None:
    with h5py.File(output_path, "w") as f:
        f.create_dataset(dataset_name, data=data)

def save_video(data: np.ndarray, output_path: Path, fps: int = 30) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (data.shape[2], data.shape[1]))
    
    for frame in data:
        out.write(frame)
    
    out.release()

def generate_test_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save 2D array (single image)
    array_2d = generate_2d_array((512, 512))
    save_tiff(array_2d, output_dir / "test_2d_XY_512x512.tiff")
    save_hdf5(array_2d, output_dir / "test_2d_XY_512x512.h5")

    # Generate and save 3D array (time series or RGB image)
    array_3d_time = generate_3d_array((100, 512, 512))
    save_tiff(array_3d_time, output_dir / "test_3d_TXY_10x512x512.tiff")
    save_hdf5(array_3d_time, output_dir / "test_3d_TXY_10x512x512.h5")

    array_3d_rgb = generate_3d_array((512, 512, 3))
    save_tiff(array_3d_rgb, output_dir / "test_3d_XYC_512x512x3.tiff")
    save_hdf5(array_3d_rgb, output_dir / "test_3d_XYC_512x512x3.h5")
    save_video(array_3d_rgb, output_dir / "test_3d_XYC_512x512x3.avi")

    # Generate and save 4D array (time series with channels)
    array_4d = generate_4d_array((100, 512, 512, 3))
    save_tiff(array_4d, output_dir / "test_4d_TXYC_10x512x512x3.tiff")
    save_hdf5(array_4d, output_dir / "test_4d_TXYC_10x512x512x3.h5")
    save_video(array_4d, output_dir / "test_4d_TXYC_10x512x512x3.avi")
if __name__ == "__main__":
    output_dir = Path("test_data")
    generate_test_data(output_dir)
    print(f"Test data generated and saved in {output_dir}")