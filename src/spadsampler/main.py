"""Main module of spadsampler."""

from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lazyimread import imread, imsave, predict_dimension_order

SamplingRange = tuple[int, int] | tuple[float, ...]
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


def compute_histogram(
    data: np.ndarray, max_size: int = 128, scale: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the histogram of the input array, subsampling if necessary.

    :param data: Input numpy array
    :param max_size: Maximum size for each dimension when subsampling, defaults to 128
    :return: Histogram and bin edges
    """
    subsample_factors = np.ceil(np.array(data.shape) / max_size).astype(int)
    if np.any(subsample_factors > 1):
        slices = tuple(slice(None, None, factor) for factor in subsample_factors)
        subsampled_data = data[slices].ravel()
    else:
        subsampled_data = data.ravel()

    max_value = np.max(subsampled_data)
    bins = np.linspace(0, 65535 if max_value > 255 else 255, 257)
    return np.histogram(subsampled_data * scale if scale else subsampled_data, bins=bins)


def plot_histogram(
    data: np.ndarray,
    bins: int = 256,
    figsize: tuple[int, int] = (10, 6),
    scale: float | None = None,
) -> None:
    """Plot the histogram.

    :param data: Input numpy array
    :param bins: Number of histogram bins, defaults to 256
    :param figsize: Figure size, defaults to (10, 6)
    :param scale: Scale factor, defaults to None
    :return: None
    """
    hist, bin_edges = compute_histogram(data, scale, bins=bins)
    plt.figure(figsize=figsize)
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Pixel Values")
    plt.show()


def binomial_sampling(
    data: np.ndarray,
    axis: tuple[int, ...] | None = None,
    p_range: SamplingRange = (-7, -2),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute the binomial sampling for the input data.

    :param data: Input numpy array
    :param axis: Axis along which to compute the mean, defaults to None
    :param p_range: Range of probabilities to sample, defaults to (-7, -2) ~= (0.0078, 0.125)
    :return: Dictionary of probabilities and their corresponding samples and p
    """
    mean = data.mean(axis=axis, keepdims=True)
    p_range = p_range if 0 < p_range[0] < 1 else tuple(2**i for i in range(p_range[0], p_range[1]))
    p_tqdm = tqdm(p_range, colour="green")
    output = {}
    for i in p_tqdm:
        p_str = f"P{i:.5f}".replace(".", "d")
        p_tqdm.set_description(f"Sampling {p_str}")
        p = i / mean
        sampled_array = np.random.binomial(data, p).astype(np.uint8)
        output.update({p_str: (sampled_array, p)})
    return output


def sample_data(
    input: np.ndarray | PathVar,
    scale_down: float | None = None,
    dim_order: str | None = None,
    output: PathVar | None = None,
    range: SamplingRange = (-7, -2),
    process_by_frame: MeanAxis = MeanAxis.XY,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Main function for processing and saving the input data.

    :param input: Input data as numpy array or path to input file
    :param scale_down: Scale down factor, defaults to None
    :param dim_order: Dimension order of the input data, defaults to None
    :param output: Path to save output, defaults to None
    :param range: Range of probabilities to sample, defaults to (-7, -2)
    :param process_by_frame: Axis along which to compute the mean, defaults to MeanAxis.XY
    :return: Dictionary of probabilities and their corresponding samples and p
    """
    if output is None and isinstance(input, PathVar):
        input = Path(input)
        output = input.parent / f"{input.stem}_resampled"
        name = input.stem

    input = imread(input) if isinstance(input, Path) else input
    if scale_down is not None:
        input /= scale_down

    dim_order = dim_order or predict_dimension_order(input)
    if len(dim_order) != input.ndim:
        raise ValueError("Dimension order does not match data dimensions")
    axis = tuple([dim_order.index(i) for i in dim_order if i in str(process_by_frame)])

    resampled_data = binomial_sampling(input, axis=axis, p_range=range)
    if output is not None:
        for key, value in resampled_data.items():
            imsave(output / f"{name}_{key}.tif", value[0])
    return resampled_data


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

    other_indices = [i for i, dim in enumerate(dim_order) if dim not in ("X", "Y", "C")]
    mid_slice = [slice(None)] * image.ndim
    for idx in other_indices:
        mid_slice[idx] = slice(image.shape[idx] // 2, image.shape[idx] // 2 + 1)
    if c_index is not None and not is_rgb:
        mid_slice[c_index] = slice(image.shape[c_index] // 2, image.shape[c_index] // 2 + 1)

    return c_index, is_rgb, mid_slice


def imshow_pairs(
    images: list[np.ndarray] | dict[str, np.ndarray], cmap: str | None = "gray"
) -> None:
    """Print the XY plane of the middle of all other dimensions for multiple images.

    :param images: List of image arrays or dictionary of named image arrays
    :param cmap: Colormap to use for non-RGB images, defaults to "gray"
    """
    if not images:
        raise ValueError("No images provided")

    _, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    axes = [axes] if len(images) == 1 else axes
    items = images.items() if isinstance(images, dict) else enumerate(images)

    for i, (ax, (key, image)) in enumerate(zip(axes, items, strict=True)):
        dim_order = predict_dimension_order(image)
        c_index, is_rgb, mid_slice = _determine_channel_and_slice(image, dim_order)
        image_slice = image[tuple(mid_slice)].astype(np.uint8)
        current_cmap = None if is_rgb else cmap
        if is_rgb:
            image_slice = np.moveaxis(image_slice, c_index, -1)

        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())

        ax.imshow(image_slice.squeeze(), vmin=0, vmax=1, cmap=current_cmap)
        ax.set_title(key if isinstance(images, dict) else f"Image {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
