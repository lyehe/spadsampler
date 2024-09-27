import numpy as np
import pytest
from src.processing import determine_input_structure, convert_to_grayscale, resample_matrix, process_channel, process_frame

def test_determine_input_structure():
    # Test 2D array
    data_2d = np.zeros((100, 100))
    structure = determine_input_structure(data_2d)
    assert structure["order"] == "XY"
    assert structure["shape"] == (100, 100)
    assert structure["ndim"] == 2

    # Test 3D array (RGB)
    data_3d_rgb = np.zeros((100, 100, 3))
    structure = determine_input_structure(data_3d_rgb)
    assert structure["order"] == "XYC"
    assert structure["shape"] == (100, 100, 3)
    assert structure["ndim"] == 3

    # Test 4D array (Time series RGB)
    data_4d = np.zeros((10, 100, 100, 3))
    structure = determine_input_structure(data_4d)
    assert structure["order"] == "TXYC"
    assert structure["shape"] == (10, 100, 100, 3)
    assert structure["ndim"] == 4

def test_convert_to_grayscale():
    # Test RGB to grayscale
    rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    gray_image = convert_to_grayscale(rgb_image)
    assert gray_image.shape == (100, 100)
    assert gray_image.dtype == np.uint8

    # Test 4D array to grayscale
    rgb_video = np.random.randint(0, 256, (10, 100, 100, 3), dtype=np.uint8)
    gray_video = convert_to_grayscale(rgb_video)
    assert gray_video.shape == (10, 100, 100)
    assert gray_video.dtype == np.uint8

def test_resample_matrix():
    matrix = np.ones((100, 100), dtype=np.uint8) * 100
    p = 0.5
    resampled = resample_matrix(matrix, p)
    assert resampled.shape == matrix.shape
    assert resampled.dtype == matrix.dtype
    assert np.all(resampled <= matrix)

def test_process_channel():
    channel_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    p = 0.5
    sample, binary_sample = process_channel(channel_data, p)
    assert sample.shape == channel_data.shape
    assert binary_sample.shape == channel_data.shape
    assert np.all(binary_sample <= 1)

def test_process_frame():
    # Test single-channel frame
    frame_1ch = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    p = 0.5
    sample_1ch, binary_sample_1ch = process_frame(frame_1ch, p)
    assert sample_1ch.shape == frame_1ch.shape
    assert binary_sample_1ch.shape == frame_1ch.shape

    # Test multi-channel frame
    frame_3ch = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    sample_3ch, binary_sample_3ch = process_frame(frame_3ch, p)
    assert sample_3ch.shape == frame_3ch.shape
    assert binary_sample_3ch.shape == frame_3ch.shape