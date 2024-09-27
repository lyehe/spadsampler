import cv2
import numpy as np
import tifffile
import h5py
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
import sys


def load_input(
    input_path: Path, frame_size: tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Load input data from various formats (video, array, TIFF, HDF5).

    :param input_path: Path to the input file
    :type input_path: Path
    :param frame_size: Desired frame size for resizing, defaults to (512, 512)
    :type frame_size: tuple[int, int], optional
    :return: Loaded and processed data
    :rtype: np.ndarray
    :raises ValueError: If the input format is unsupported
    """
    if input_path.suffix.lower() in (".mov", ".mp4", ".avi"):
        return load_video(input_path, frame_size)
    elif input_path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(input_path)
    elif input_path.suffix.lower() in (".h5", ".hdf5"):
        return load_first_array_hdf5(input_path)
    else:
        raise ValueError("Unsupported input format")


def load_video(
    video_path: Path, frame_size: tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Load video and convert it to a numpy array.

    :param video_path: Path to the video file
    :type video_path: Path
    :param frame_size: Desired frame size for resizing, defaults to (512, 512)
    :type frame_size: tuple[int, int], optional
    :return: Video data as a numpy array
    :rtype: np.ndarray
    """
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_array = np.empty(
        (total_frames, frame_size[1], frame_size[0], 3), dtype=np.uint8
    )

    for i in range(total_frames):
        ret, frame = video.read()
        if ret:
            video_array[i] = cv2.resize(frame, frame_size)
        else:
            break

    video.release()
    return video_array


def load_first_array_hdf5(file_path: Path) -> np.ndarray:
    """
    Load the first array from an HDF5 file.

    :param file_path: Path to the HDF5 file
    :type file_path: Path
    :return: The first array in the HDF5 file
    :rtype: np.ndarray
    :raises ValueError: If no dataset is found in the HDF5 file
    """
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                return np.array(f[key])
    raise ValueError("No dataset found in the HDF5 file")


def determine_input_structure(data: np.ndarray) -> dict:
    """
    Determine the input size and dimension orders.

    :param data: Input data
    :type data: np.ndarray
    :return: Dictionary containing information about the input structure
    :rtype: dict
    """
    structure = {
        "shape": data.shape,
        "ndim": data.ndim,
        "dtype": data.dtype,
    }

    if data.ndim == 2:
        structure["order"] = "XY"
    elif data.ndim == 3:
        if data.shape[-1] in (3, 4):  # Assume RGB or RGBA
            structure["order"] = "XYC"
        else:
            structure["order"] = "TXY"  # Assume time series
    elif data.ndim == 4:
        if data.shape[-1] in (3, 4):  # Assume RGB or RGBA
            structure["order"] = "TXYC"
        elif data.shape[1] in (3, 4):
            structure["order"] = "TCXY"  # Best practice for 4D data
        elif data.shape[-2] == 3:
            structure["order"] = "XYCT"  # Best practice for 4D data

    return structure


def convert_to_grayscale(data: np.ndarray) -> np.ndarray:
    """
    Convert input data to grayscale.

    :param data: Input data
    :type data: np.ndarray
    :return: Grayscale data
    :rtype: np.ndarray
    """
    if len(data.shape) == 4:
        if data.shape[-1] == 3:  # Channel is at the end
            return data.mean(axis=-1).astype(np.uint8)
        elif data.shape[1] == 3:  # Channel is at the beginning
            return data.mean(axis=1).astype(np.uint8)
    elif len(data.shape) == 3:
        if data.shape[-1] == 3:  # Single image with channel at the end
            return cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        elif data.shape[0] == 3:  # Single image with channel at the beginning
            return cv2.cvtColor(data.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    return data


def resample_matrix(matrix: np.ndarray, p: float) -> np.ndarray:
    """
    Resample the input matrix using binomial distribution.

    :param matrix: Input matrix
    :type matrix: np.ndarray
    :param p: Probability for binomial distribution
    :type p: float
    :return: Resampled matrix
    :rtype: np.ndarray
    """
    z, x, y = matrix.shape
    prob_matrix = np.full((z, x, y), p)
    return np.random.binomial(matrix, prob_matrix)


def plot_input_output(
    input_data: np.ndarray, output_data: np.ndarray, p: float, title: None | str = None
) -> None:
    """
    Plot input and output images side by side.

    :param input_data: Input image data
    :type input_data: np.ndarray
    :param output_data: Output image data
    :type output_data: np.ndarray
    :param p: Probability used for resampling
    :type p: float
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    if len(input_data.shape) == 3 and input_data.shape[-1] == 3:
        ax1.imshow(input_data[0])
    else:
        ax1.imshow(input_data[0], cmap="gray")
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2.imshow(output_data[0], cmap="gray")
    ax2.set_title(f"Output Image (p={p:.5f})")
    ax2.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def process_and_save(
    input_data: np.ndarray,
    output_prefix: Path,
    start_range: int = -7,
    end_range: int = -2,
    process_by_frame: bool = False,
) -> None:
    """
    Process input data and save resampled images.

    This function resamples the input data using binomial distribution for a range of probabilities,
    and saves the resulting images in TIFF format. It also creates binary samples and saves them separately.

    :param input_data: Input data to be processed
    :type input_data: np.ndarray
    :param output_prefix: Prefix for output file names
    :type output_prefix: Path
    :param start_range: Start of the range for probability calculation, defaults to -7
    :type start_range: int, optional
    :param end_range: End of the range for probability calculation, defaults to -2
    :type end_range: int, optional
    :param process_by_frame: Whether to process data frame by frame, defaults to False
    :type process_by_frame: bool, optional
    """
    if process_by_frame:
        for frame_idx in range(input_data.shape[0]):
            frame = input_data[frame_idx]
            process_frame(frame, output_prefix, start_range, end_range, frame_idx)
    else:
        process_frame(input_data, output_prefix, start_range, end_range)


def process_frame(
    frame: np.ndarray,
    output_prefix: Path,
    start_range: int,
    end_range: int,
    frame_idx: int | None = None,
) -> None:
    """
    Process a single frame or the entire time series.

    :param frame: Frame or time series data to be processed
    :type frame: np.ndarray
    :param output_prefix: Prefix for output file names
    :type output_prefix: Path
    :param start_range: Start of the range for probability calculation
    :type start_range: int
    :param end_range: End of the range for probability calculation
    :type end_range: int
    :param frame_idx: Index of the frame being processed, or None if processing entire time series
    :type frame_idx: int | None
    """
    if frame.ndim == 3 and frame.shape[-1] in (3, 4):  # Multiple channels
        for channel in range(frame.shape[-1]):
            channel_data = frame[..., channel]
            process_channel(channel_data, output_prefix, start_range, end_range, frame_idx, channel)
    else:  # Single channel
        process_channel(frame, output_prefix, start_range, end_range, frame_idx)


def process_channel(
    channel_data: np.ndarray,
    output_prefix: Path,
    start_range: int,
    end_range: int,
    frame_idx: int | None,
    channel: int | None = None,
) -> None:
    """
    Process a single channel of data.

    :param channel_data: Channel data to be processed
    :type channel_data: np.ndarray
    :param output_prefix: Prefix for output file names
    :type output_prefix: Path
    :param start_range: Start of the range for probability calculation
    :type start_range: int
    :param end_range: End of the range for probability calculation
    :type end_range: int
    :param frame_idx: Index of the frame being processed, or None if processing entire time series
    :type frame_idx: int | None
    :param channel: Index of the channel being processed, or None if single channel
    :type channel: int | None
    """
    mean = channel_data.mean()
    print(f"0.125/mean: {0.125/mean}")

    for i in range(start_range, end_range + 1):
        p = 2**i / mean
        sample = resample_matrix(channel_data, p).astype(np.uint8)
        
        frame_suffix = f"_frame{frame_idx}" if frame_idx is not None else ""
        channel_suffix = f"_channel{channel}" if channel is not None else ""
        
        tifffile.imwrite(
            output_prefix.with_name(f"{output_prefix.stem}{frame_suffix}{channel_suffix}_p={2**i:.5f}_noclip.tif"),
            sample,
        )

        binary_sample = (sample > 0).astype(np.uint8)
        mean_signal = binary_sample.mean()
        tifffile.imwrite(
            output_prefix.with_name(
                f"{output_prefix.stem}{frame_suffix}{channel_suffix}_p={2**i:.5f}_m={mean_signal:.5f}.tif"
            ),
            binary_sample,
        )
        print(f"p: {p}, mean_signal: {mean_signal}")
        plot_input_output(channel_data, sample, p, title=f"p={p:.5f}")


def main(
    input_path: Path,
    output_prefix: Path,
    frame_size: tuple[int, int] = (512, 512),
    start_range: int = -7,
    end_range: int = -2,
    process_by_frame: bool = False,
) -> None:
    """
    Main function to load, process, and save input data.

    This function loads input data from a file, converts it to grayscale,
    and then processes and saves the resulting images.

    :param input_path: Path to the input file
    :type input_path: Path
    :param output_prefix: Prefix for output file names
    :type output_prefix: Path
    :param frame_size: Size of each frame, defaults to (512, 512)
    :type frame_size: tuple[int, int], optional
    :param start_range: Start of the range for probability calculation, defaults to -7
    :type start_range: int, optional
    :param end_range: End of the range for probability calculation, defaults to -2
    :type end_range: int, optional
    :param process_by_frame: Whether to process data frame by frame, defaults to False
    :type process_by_frame: bool, optional
    """
    input_data = load_input(input_path, frame_size)
    input_structure = determine_input_structure(input_data)
    print("Input data structure:", input_structure)
    process_and_save(input_data, output_prefix, start_range, end_range, process_by_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and resample input data.")
    parser.add_argument("input_path", type=str, help="Path to the input file")
    parser.add_argument("output_prefix", type=Path, help="Prefix for output file names")
    parser.add_argument(
        "--frame_size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Frame size for resizing (width height)",
    )
    parser.add_argument(
        "--start_range",
        type=int,
        default=-7,
        help="Start of the range for probability calculation",
    )
    parser.add_argument(
        "--end_range",
        type=int,
        default=-2,
        help="End of the range for probability calculation",
    )
    parser.add_argument(
        "--process_by_frame",
        action="store_true",
        help="Process data frame by frame",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    try:
        main(
            input_path,
            args.output_prefix,
            tuple(args.frame_size),
            args.start_range,
            args.end_range,
            args.process_by_frame,
        )
    except FileNotFoundError:
        print(f"Error: Unable to read input file '{input_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
