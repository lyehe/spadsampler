import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from .io_utils.io_utils import load_input, save_tiff
from .processing import determine_input_structure, process_frame
from .visualization import plot_input_output

def process_and_save(
    input_data: np.ndarray,
    output_prefix: Path,
    start_range: int = -7,
    end_range: int = -2,
    process_by_frame: bool = False,
) -> None:
    if process_by_frame:
        for frame_idx in range(input_data.shape[0]):
            frame = input_data[frame_idx]
            process_single(frame, output_prefix, start_range, end_range, frame_idx)
    else:
        process_single(input_data, output_prefix, start_range, end_range)

def process_single(
    data: np.ndarray,
    output_prefix: Path,
    start_range: int,
    end_range: int,
    frame_idx: Optional[int] = None,
) -> None:
    mean = data.mean()
    print(f"0.125/mean: {0.125/mean}")

    for i in range(start_range, end_range + 1):
        p = 2**i / mean
        sample, binary_sample = process_frame(data, p)
        
        frame_suffix = f"_frame{frame_idx}" if frame_idx is not None else ""
        
        save_tiff(sample, output_prefix.with_name(f"{output_prefix.stem}{frame_suffix}_p={2**i:.5f}_noclip.tif"))

        mean_signal = binary_sample.mean()
        save_tiff(binary_sample, output_prefix.with_name(f"{output_prefix.stem}{frame_suffix}_p={2**i:.5f}_m={mean_signal:.5f}.tif"))
        
        print(f"p: {p}, mean_signal: {mean_signal}")
        plot_input_output(data, sample, p, title=f"p={p:.5f}")

def main(
    input_path: Path,
    output_prefix: Path,
    frame_size: Tuple[int, int] = (512, 512),
    start_range: int = -7,
    end_range: int = -2,
    process_by_frame: bool = False,
) -> None:
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