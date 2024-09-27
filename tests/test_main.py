import pytest
from pathlib import Path
from src.main import process_single, process_and_save
import numpy as np

@pytest.fixture
def sample_data():
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)

@pytest.fixture
def output_prefix(tmp_path):
    return tmp_path / "output"

def test_process_single(sample_data, output_prefix, monkeypatch):
    # Mock save_tiff and plot_input_output functions
    def mock_save_tiff(data, path):
        pass

    def mock_plot_input_output(input_data, output_data, p, title):
        pass

    monkeypatch.setattr("src.main.save_tiff", mock_save_tiff)
    monkeypatch.setattr("src.main.plot_input_output", mock_plot_input_output)

    process_single(sample_data, output_prefix, -7, -2)

def test_process_and_save(sample_data, output_prefix, monkeypatch):
    # Mock process_single function
    def mock_process_single(data, prefix, start, end, frame_idx=None):
        pass

    monkeypatch.setattr("src.main.process_single", mock_process_single)

    # Test without process_by_frame
    process_and_save(sample_data, output_prefix, -7, -2, False)

    # Test with process_by_frame
    process_and_save(np.stack([sample_data, sample_data]), output_prefix, -7, -2, True)

# Add more tests for the main function if needed