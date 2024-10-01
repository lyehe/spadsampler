"""Test the spadsampler module."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tifffile

from spadsampler.spadsampler import (
    MeanAxis,
    PathVar,
    _determine_channel_and_slice,
    bernoulli_sampling,
    binomial_sampling,
    compute_histogram,
    imshow_pairs,
    plot_histogram,
    sample_data,
)


@pytest.fixture
def sample_data_2d() -> np.ndarray:
    """Generate a 2D numpy array of random integers between 0 and 255."""
    return np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)


@pytest.fixture
def sample_data_3d() -> np.ndarray:
    """Generate a 3D numpy array of random integers between 0 and 255."""
    return np.random.randint(0, 256, size=(10, 100, 100), dtype=np.uint8)


@pytest.fixture(scope="session")
def mpl_backend():
    """Set the matplotlib backend to the current backend."""
    return plt.get_backend()


@pytest.fixture(scope="session")
def has_gui(mpl_backend):
    """Check if the current backend is a GUI backend."""
    return mpl_backend.lower() not in ["agg", "cairo", "pdf", "pgf", "ps", "svg", "template"]


def test_compute_histogram(sample_data_2d: np.ndarray) -> None:
    """Test if compute_histogram returns correct output types and sizes."""
    hist, bin_edges = compute_histogram(sample_data_2d)
    assert isinstance(hist, np.ndarray)
    assert isinstance(bin_edges, np.ndarray)
    assert len(hist) == 256
    assert len(bin_edges) == 257


def test_compute_histogram_with_scale(sample_data_2d: np.ndarray) -> None:
    """Test if compute_histogram correctly applies scaling."""
    hist, bin_edges = compute_histogram(sample_data_2d, scale=0.5)
    assert np.max(hist) <= 128


@pytest.mark.skipif("not has_gui")
def test_plot_histogram(sample_data_2d: np.ndarray) -> None:
    """Test if plot_histogram runs without errors."""
    plot_histogram(sample_data_2d)
    plt.close("all")


def test_binomial_sampling(sample_data_2d: np.ndarray) -> None:
    """Test if binomial_sampling returns correct output structure and types."""
    result_v, result_p = binomial_sampling(sample_data_2d)
    assert isinstance(result_v, dict)
    assert isinstance(result_p, dict)
    assert len(result_v) == 5  # Default range is (-7, -2), so 5 samples
    assert len(result_p) == 5
    for key, sampled_array in result_v.items():
        assert isinstance(key, str)
        assert isinstance(sampled_array, np.ndarray)
        assert sampled_array.shape == sample_data_2d.shape
    for key, p in result_p.items():
        assert isinstance(key, str)
        assert isinstance(p, np.ndarray)


def test_binomial_sampling_custom_range(sample_data_2d: np.ndarray) -> None:
    """Test if binomial_sampling works with custom probability ranges."""
    result_v, result_p = binomial_sampling(sample_data_2d, p_range=(0.1, 0.2, 0.3))
    assert len(result_v) == 3
    assert len(result_p) == 3


def test_sample_data_with_array(sample_data_3d: np.ndarray) -> None:
    """Test if sample_data works correctly with numpy array input."""
    result_v, result_p = sample_data(sample_data_3d)
    assert isinstance(result_v, dict)
    assert isinstance(result_p, dict)
    assert len(result_v) == 5
    assert len(result_p) == 5
    for key, sampled_array in result_v.items():
        assert isinstance(key, str)
        assert isinstance(sampled_array, np.ndarray)
        assert sampled_array.shape == sample_data_3d.shape
    for key, p in result_p.items():
        assert isinstance(key, str)
        assert isinstance(p, np.ndarray)


def test_sample_data_with_file(tmp_path: Path) -> None:
    """Test if sample_data works correctly with file input and output."""
    input_path = tmp_path / "test_input.tif"
    data = np.random.randint(0, 256, size=(10, 100, 100), dtype=np.uint8)

    tifffile.imwrite(input_path, data)

    result_v, result_p = sample_data(input_path, output=tmp_path)
    assert isinstance(result_v, dict)
    assert isinstance(result_p, dict)
    assert len(result_v) == 5
    assert len(result_p) == 5
    assert len(list(tmp_path.glob("*.tif"))) == 6


def test_sample_data_custom_process_by_frame(sample_data_3d: np.ndarray) -> None:
    """Test if sample_data works with custom process_by_frame parameter."""
    result_v, result_p = sample_data(sample_data_3d, process_by_frame=MeanAxis.ZXY)
    assert isinstance(result_v, dict)
    assert isinstance(result_p, dict)
    assert len(result_v) == 5
    assert len(result_p) == 5


@pytest.mark.skipif("not has_gui")
@pytest.mark.parametrize("dim_order", ["YX", "CYX", "ZCYX", "TZCYX"])
def test_imshow_pairs(dim_order: str) -> None:
    """Test if imshow_pairs handles different dimension orders correctly."""
    shape = [10 if dim in dim_order else 100 for dim in "TZCYX"]
    data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    imshow_pairs({"Test": data})
    plt.close("all")


@pytest.mark.skipif("not has_gui")
def test_imshow_pairs_rgb() -> None:
    """Test if imshow_pairs correctly handles RGB data."""
    data_rgb = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    imshow_pairs({"RGB Test": data_rgb})
    plt.close("all")


@pytest.mark.skipif("not has_gui")
def test_imshow_pairs_multiple_images(sample_data_2d: np.ndarray) -> None:
    """Test if imshow_pairs correctly handles multiple input images."""
    images = {
        "Image 1": sample_data_2d,
        "Image 2": sample_data_2d * 2,
    }
    imshow_pairs(images)
    plt.close("all")


def test_mean_axis_enum() -> None:
    """Test the MeanAxis enum."""
    assert str(MeanAxis.XY) == "XY"
    assert str(MeanAxis.TZXY) == "TZXY"
    assert MeanAxis("ZXY") == MeanAxis.ZXY


def test_path_var_type() -> None:
    """Test the PathVar type."""
    assert isinstance("path/to/file", PathVar)
    assert isinstance(Path("path/to/file"), PathVar)


def test_determine_channel_and_slice() -> None:
    """Test the _determine_channel_and_slice function."""
    image = np.random.rand(5, 3, 10, 10)
    dim_order = "ZCYX"
    c_index, is_rgb, mid_slice = _determine_channel_and_slice(image, dim_order)
    assert c_index == 1
    assert is_rgb
    assert mid_slice == [slice(2, 3), slice(None), slice(None), slice(None)]


def test_sample_data_invalid_dim_order(sample_data_3d: np.ndarray) -> None:
    """Test if sample_data raises an error with invalid dimension order."""
    with pytest.raises(ValueError, match="Dimension order does not match data dimensions"):
        sample_data(sample_data_3d, dim_order="TXYC")  # Invalid dim_order


def test_imshow_pairs_empty_input() -> None:
    """Test if imshow_pairs raises an error with empty input."""
    with pytest.raises(ValueError, match="No images provided"):
        imshow_pairs([])


def test_binomial_sampling_invalid_range() -> None:
    """Test if binomial_sampling raises an error with invalid range."""
    with pytest.raises(ValueError, match="p < 0, p > 1 or p contains NaNs"):
        binomial_sampling(np.random.rand(10, 10), p_range=(1, 2))  # Invalid range


def test_bernoulli_sampling(sample_data_2d: np.ndarray) -> None:
    """Test if bernoulli_sampling returns correct output structure and types."""
    result_v, result_p = bernoulli_sampling(sample_data_2d)
    assert isinstance(result_v, dict)
    assert isinstance(result_p, dict)
    assert len(result_v) == 5  # Default range is (-7, -2), so 5 samples
    assert len(result_p) == 5
    for key, sampled_array in result_v.items():
        assert isinstance(key, str)
        assert isinstance(sampled_array, np.ndarray)
        assert sampled_array.shape == sample_data_2d.shape
        assert sampled_array.dtype == bool  # Check if the output is boolean
    for key, p in result_p.items():
        assert isinstance(key, str)
        assert isinstance(p, np.ndarray)

    # Check if the keys are formatted correctly
    for key in result_v.keys():
        assert key.startswith("P")
        assert "d" in key  # Decimal point is replaced with 'd'

    # Test with custom range
    custom_result_v, custom_result_p = bernoulli_sampling(sample_data_2d, p_range=(0.1, 0.2, 0.3))
    assert len(custom_result_v) == 3
    assert len(custom_result_p) == 3
