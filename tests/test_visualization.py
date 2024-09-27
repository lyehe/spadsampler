import pytest
import numpy as np
from src.visualization import plot_input_output

def test_plot_input_output(monkeypatch):
    # Mock plt.subplots and plt.show
    def mock_subplots(*args, **kwargs):
        class MockAxes:
            def imshow(self, *args, **kwargs):
                pass
            def set_title(self, *args, **kwargs):
                pass
            def axis(self, *args, **kwargs):
                pass
        return None, (MockAxes(), MockAxes())

    def mock_show():
        pass

    monkeypatch.setattr("matplotlib.pyplot.subplots", mock_subplots)
    monkeypatch.setattr("matplotlib.pyplot.show", mock_show)

    # Test with 2D grayscale input
    input_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    output_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    plot_input_output(input_data, output_data, 0.5)

    # Test with 3D RGB input
    input_data_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    output_data_rgb = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    plot_input_output(input_data_rgb, output_data_rgb, 0.5)

    # Test with title
    plot_input_output(input_data, output_data, 0.5, title="Test Plot")