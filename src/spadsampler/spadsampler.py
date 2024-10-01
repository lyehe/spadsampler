"""Main module of spadsampler."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from logging import getLogger
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lazyimread import imread, imsave, predict_dimension_order
from tqdm import tqdm

logger = getLogger(__name__)

SamplingRange = tuple[int, int] | tuple[float, ...] | int | float
PathVar = Path | str


class MeanAxis(Enum):
    """Enum for processing the stack."""

    XY = "XY"
    TZXY = "TZXY"
    ZXY = "ZXY"
    TYX = "TYX"
    TZXYC = "TZXYC"

    def __str__(self) -> str:
        """Return the string representation of the processing method."""
        return self.value


class SamplingMethod(Enum):
    """Enum for sampling method."""

    BINOMIAL = "binomial"
    BERNOULLI = "bernoulli"


def compute_histogram(
    data: np.ndarray, max_size: int = 128, scale: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the histogram of the input array, subsampling if necessary.

    :param data: Input numpy array
    :param max_size: Maximum size for each dimension when subsampling, defaults to 128
    :return: Histogram and bin edges
    """
    subsample_factors = (np.array(data.shape) // max_size).astype(int)
    if np.any(subsample_factors > 1):
        slices = tuple(slice(None, None, factor) for factor in subsample_factors)
        subsampled_data = data[slices].ravel()
        logger.info(f"Subsampled data with factors: {subsample_factors}")
    else:
        subsampled_data = data.ravel()
        logger.info("No subsampling required")

    max_value = np.max(subsampled_data)
    bins = np.linspace(0, 65535 if max_value > 255 else 255, 257)
    logger.debug(f"Computing histogram with {len(bins)} bins")
    return np.histogram(subsampled_data * scale if scale else subsampled_data, bins=bins)


def plot_histogram(
    data: np.ndarray,
    max_size: int = 128,
    figsize: tuple[int, int] = (10, 6),
    scale: float | None = None,
) -> None:
    """Plot the histogram."""
    if matplotlib.get_backend().lower() in ["agg", "cairo", "pdf", "pgf", "ps", "svg", "template"]:
        logger.warning("Non-interactive backend detected. Skipping plot generation.")
        return

    hist, bin_edges = compute_histogram(data, max_size=max_size, scale=scale)
    plt.figure(figsize=figsize)
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pixel Values")
    plt.show()
    logger.info("Histogram plot displayed")


def _process_single_p(args: tuple[float, np.ndarray, np.ndarray]) -> tuple[str, np.ndarray, float]:
    """Process a single probability."""
    i, data, mean = args
    p_str = f"P{i:.5f}".replace(".", "d")
    p = i / (mean + 1e-6)
    if p > 1:
        p = 1
        logger.warning(f"Probability {p} is greater than 1, setting to 1")
    sampled_array = np.random.binomial(data, p).astype(np.uint8)
    return p_str, sampled_array, p


def binomial_sampling(
    data: np.ndarray,
    axis: tuple[int, ...] | None = None,
    p_range: SamplingRange = (-7, -2),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute the binomial sampling for the input data using parallel processing.

    :param data: Input numpy array
    :param axis: Axis along which to compute the mean, defaults to None
    :param p_range: Range of probabilities to sample, defaults to (-7, -2) ~= (0.0078, 0.125)
    :return: Dicts of probabilities and their corresponding samples and p
    """
    mean = data.mean(axis=axis, keepdims=True)
    logger.info(f"Mean value of data: {mean}")
    if data.dtype not in (np.uint8, np.uint16):
        data = data.astype(np.uint16)
        logger.info("Data converted to uint16")

    match p_range:
        case int(p):  # single int
            p = p if p < 0 else -p
            p_range = (2**p,)
        case (int(p), int(q)):  # tuple of ints
            p = min(p, -p)
            q = min(q, -q)
            p, q = min(p, q), max(p, q)
            p_range = tuple([2**i for i in range(p, q)])  # Convert range to tuple
        case float(p):  # single float
            if 0 < p < 1:
                p_range = (p,)
            else:
                logger.error(f"Invalid p_range: {p_range}")
                raise ValueError(f"Invalid p_range: {p_range}")
        case (float(p), *rest):  # tuple of floats
            valid_range = [float(i) for i in (p, *rest) if 0 < i < 1]
            if not valid_range:
                raise ValueError(f"Invalid p_range: {p_range}. No values between 0 and 1.")
            p_range = tuple(sorted(valid_range))
        case _:  # invalid
            logger.error(f"Invalid p_range: {p_range}")
            raise ValueError(f"Invalid p_range: {p_range}")

    logger.debug(f"Probability range: {p_range}")

    output_v = {}
    output_p = {}

    max_workers = max(min(os.cpu_count() - 1, len(p_range)), 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_p, (i, data, mean)) for i in p_range]
        for future in tqdm(
            as_completed(futures), total=len(futures), colour="green", desc="Sampling"
        ):
            p_str, sampled_array, p = future.result()
            output_v[p_str] = sampled_array
            output_p[p_str] = p
            logger.debug(f"Sampled array for {p_str} with shape {sampled_array.shape}")

    return output_v, output_p


def bernoulli_sampling(
    data: np.ndarray,
    axis: tuple[int, ...] | None = None,
    p_range: SamplingRange = (-7, -2),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute the bernoulli sampling for the input data.

    :param data: Input numpy array
    :param axis: Axis along which to compute the mean, defaults to None
    :param p_range: Range of probabilities to sample, defaults to (-7, -2) ~= (0.0078, 0.125)
    :return: Dicts of probabilities and their corresponding samples and p
    """
    output_v, output_p = binomial_sampling(data, axis=axis, p_range=p_range)
    new_output_v = {}
    new_output_p = {}
    key_list = list(output_v.keys())
    for key in key_list:
        data = output_v.pop(key)
        p = output_p.pop(key)
        data = data > 0
        observed_p = data.mean()
        new_key = f"P{observed_p:.5f}".replace(".", "d")
        new_output_v[new_key] = data
        new_output_p[new_key] = p
    return new_output_v, new_output_p


def _save_single_file(args: tuple[str, np.ndarray, PathVar, str]) -> None:
    """Save a single file."""
    key, value, output, file_name = args
    output_path = output / f"{file_name}_{key}.tif"
    if not output_path.exists():
        imsave(value, output_path)
        logger.info(f"Saved resampled data to: {output_path}")
    else:
        logger.warning(f"Output file {output_path} already exists, skipping")


def sample_data(
    input: np.ndarray | PathVar,
    scale_down: float | None = None,
    dim_order: str | None = None,
    output: PathVar | None = None,
    p_range: SamplingRange = (-7, -2),
    process_by_frame: MeanAxis = MeanAxis.TZXY,
    sampling_method: SamplingMethod = SamplingMethod.BINOMIAL,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Main function for processing and saving the input data.

    :param input: Input data as numpy array or path to input file
    :param scale_down: Scale down factor, defaults to None
    :param dim_order: Dimension order of the input data, defaults to None
    :param output: Path to save output, defaults to None
    :param p_range: Range of probabilities to sample, defaults to (-7, -2)
    :param process_by_frame: Axis along which to compute the mean, defaults to MeanAxis.XY
    :return: Dicts of probabilities and their corresponding samples and p
    """
    # If the input is a path and no output is provided, set the output to the input path
    if output is None and isinstance(input, PathVar):
        input = Path(input)
        output = input.parent / f"{input.stem}_resampled"
        file_name = input.stem
        logger.info(f"Output directory set to: {output}")
    elif output is not None:
        file_name = output.stem
        logger.info(f"Output file name set to: {file_name}")

    # Read the input data if it is a path
    input = input if isinstance(input, np.ndarray) else imread(input)
    logger.info(f"Input data shape: {input.shape}")
    if scale_down is not None:
        input /= scale_down
        logger.info(f"Data scaled down by factor: {scale_down}")

    # Predict the dimension order if not provided
    dim_order = dim_order or predict_dimension_order(input)
    logger.info(f"Dimension order: {dim_order}")
    if len(dim_order) != input.ndim:
        logger.error("Dimension order does not match data dimensions")
        raise ValueError("Dimension order does not match data dimensions")

    # Determine the axis to process by
    axis = tuple([dim_order.index(i) for i in dim_order if i in str(process_by_frame)])
    logger.info(f"Processing axis: {axis}")

    # KEY LOGIC
    if sampling_method == SamplingMethod.BINOMIAL:
        resampled_data, sampling_p = binomial_sampling(input, axis=axis, p_range=p_range)
    elif sampling_method == SamplingMethod.BERNOULLI:
        resampled_data, sampling_p = bernoulli_sampling(input, axis=axis, p_range=p_range)

    if output is not None:
        output = Path(output)
        if not output.exists():
            output.mkdir(parents=True)

        max_workers = max(min(os.cpu_count() - 1, len(resampled_data)), 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_save_single_file, (key, value, output, file_name))
                for key, value in resampled_data.items()
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), colour="blue", desc="Saving"
            ):
                future.result()

    return resampled_data, sampling_p


def _determine_channel_and_slice(
    image: np.ndarray, dim_order: str
) -> tuple[int | None, bool, list[slice]]:
    """Determine channel index, if the image is RGB, and create mid-slice.

    :param image: Input image array
    :param dim_order: Dimension order of the image
    :return: Tuple of (channel_index, is_rgb, mid_slice)
    """
    c_index = dim_order.index("C") if "C" in dim_order else None
    is_rgb = c_index is not None and image.shape[c_index] == 3
    logger.debug(f"Channel index: {c_index}, Is RGB: {is_rgb}")

    other_indices = [i for i, dim in enumerate(dim_order) if dim not in ("X", "Y", "C")]
    mid_slice = [slice(None)] * image.ndim
    for idx in other_indices:
        mid_slice[idx] = slice(image.shape[idx] // 2, image.shape[idx] // 2 + 1)
    if c_index is not None and not is_rgb:
        mid_slice[c_index] = slice(image.shape[c_index] // 2, image.shape[c_index] // 2 + 1)
    logger.debug(f"Mid-slice: {mid_slice}")

    return c_index, is_rgb, mid_slice


def imshow_pairs(
    images: list[np.ndarray] | dict[str, np.ndarray], cmap: str | None = "gray"
) -> None:
    """Print the XY plane of the middle of all other dimensions for multiple images.

    :param images: List of image arrays or dictionary of named image arrays
    :param cmap: Colormap to use for non-RGB images, defaults to "gray"
    """
    if not images:
        logger.error("No images provided")
        raise ValueError("No images provided")

    _, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    axes = [axes] if len(images) == 1 else axes
    items = images.items() if isinstance(images, dict) else enumerate(images)

    for i, (ax, (key, image)) in enumerate(zip(axes, items, strict=True)):
        dim_order = predict_dimension_order(image)
        logger.debug(f"Image {i+1} dimension order: {dim_order}")
        c_index, is_rgb, mid_slice = _determine_channel_and_slice(image, dim_order)
        image_slice = image[tuple(mid_slice)].astype(np.uint8)
        current_cmap = None if is_rgb else cmap
        if is_rgb:
            image_slice = np.moveaxis(image_slice, c_index, -1)
            logger.debug(f"Image {i+1} is RGB, moved channel axis to last dimension")

        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())

        ax.imshow(image_slice.squeeze(), vmin=0, vmax=1, cmap=current_cmap)
        ax.set_title(key if isinstance(images, dict) else f"Image {i+1}")
        ax.axis("off")
        logger.debug(f"Plotted image {i+1}")

    plt.tight_layout()
    plt.show()
    logger.info("All images displayed")
