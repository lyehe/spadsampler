# Binomial Sampler

Binomial Sampler is a Python tool for resampling image and video data using binomial distribution. It supports various input formats and provides visualization of input and output data.

## Features

- Support for various input formats (TIFF, HDF5, video)
- Binomial resampling with adjustable parameters
- Visualization of input and output data
- Test data generation for different scenarios

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/binomial_sampler.git
   cd binomial_sampler
   ```

2. Install dependencies:

   Choose one of the following methods to install the required dependencies:

   ### Using conda/mamba (recommended)

   ```bash
   conda env create -f environment.yml
   conda activate binomial_sampler
   ```

   Or if you're using mamba:

   ```bash
   mamba env create -f environment.yml
   mamba activate binomial_sampler
   ```

   ### Using pip

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Main Functionality

The Binomial Sampler can be used through the command-line interface or by importing the module in your Python script.

#### Command-line Usage:

```bash
python -m src.main input_path output_prefix [--frame_size WIDTH HEIGHT] [--start_range START] [--end_range END] [--process_by_frame]
```

Parameters:
- `input_path`: Path to the input file (TIFF, HDF5, or video)
- `output_prefix`: Prefix for output file names
- `--frame_size`: Frame size for resizing (default: 512 512)
- `--start_range`: Start of the range for probability calculation (default: -7)
- `--end_range`: End of the range for probability calculation (default: -2)
- `--process_by_frame`: Process data frame by frame (optional)

#### Python Script Usage:

```python
from pathlib import Path
from src.main import main

input_path = Path("path/to/your/input/file.tiff")
output_prefix = Path("output/prefix")

main(
    input_path=input_path,
    output_prefix=output_prefix,
    frame_size=(512, 512),
    start_range=-7,
    end_range=-2,
    process_by_frame=False
)
```

### Key Features and Input Selection

1. **Input Formats**: The tool supports TIFF, HDF5, and video input formats. Select your input file accordingly.

2. **Probability Range**: The `start_range` and `end_range` parameters determine the range of probabilities used for binomial sampling. The actual probability `p` is calculated as `2^i / mean` for `i` in the range `[start_range, end_range]`. Adjust these values to control the sampling intensity.

3. **Frame Size**: Use the `frame_size` parameter to resize input frames. This is particularly useful for video inputs or when working with large images.

4. **Process by Frame**: The `process_by_frame` option allows processing of multi-frame inputs (like videos or time series data) frame by frame. This can be useful for handling large datasets or applying different processing to each frame.

5. **Visualization**: The tool automatically generates visualizations of the input and output data for each probability value, allowing for easy comparison of results.

### Generating Test Data

To generate test data for various scenarios:

```python
from pathlib import Path
from data.test_data_generator import generate_test_data

output_dir = Path("test_data")
generate_test_data(output_dir)
```

This will create test data in the specified output directory, including 2D, 3D, and 4D arrays in TIFF, HDF5, and video formats.

### Running Tests

To run the test suite:

```bash
pytest tests/
```

This will run all tests in the `tests/` directory, including tests for I/O operations, processing functions, and visualization.

## Module Structure

- `src/io.py`: Functions for loading and saving data in various formats
- `src/processing.py`: Core processing functions for binomial sampling
- `src/visualization.py`: Functions for visualizing input and output data
- `src/main.py`: Main script for running the binomial sampling process
- `data/test_data_generator.py`: Script for generating test data
- `tests/`: Directory containing test files for each module

## License

This project is unlicensed and dedicated to the public domain. For more information, see the [LICENSE](LICENSE) file.