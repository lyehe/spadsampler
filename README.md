### SPADSampler

SPADSampler is a simple tool for creating binomial sampled data mimicking SPAD (Single-Photon Avalanche Diode) images. It provides tools for sampling, histogram computation, and visualization of image data. It has two sampling methods: Binomial and Bernoulli. The image is sampled using the integer value as the trails. A target range of mean photon counts are specified, and the image is sampled to match the specified range. If the sampling method is Bernoulli, the image is truncated to 0 or 1 to mimic the SPAD cut-off.

### Installation

You can clone this repository or install SPADSampler using pip:

```bash
# Install directly from GitHub
pip install git+https://github.com/lyehe/spadsampler.git

# Install from PyPI
pip install spadsampler
```

### Quick Start with Google Colab

You can try out SPADSampler quickly using our Google Colab notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/lyehe/spadsampler/blob/master/examples.ipynb)

### Features

1. Binomial and Bernoulli Sampling: Sample data using either binomial or Bernoulli distribution.
2. Histogram Computation: Compute and visualize histograms of image data.
3. Flexible Input: Accept both numpy arrays and file paths as input.
4. Customizable Sampling: Adjust sampling parameters such as probability range and processing axis.
5. Visualization: Easily visualize original and sampled data side by side.

### Usage

Here's a basic example of how to use SPADSampler:

```python
import numpy as np
from spadsampler import sample_data, imshow_pairs, SamplingMethod

# Generate random data
data = np.random.randint(0, 256, (40, 128, 128, 5), dtype=np.uint8)

# Sample data using binomial sampling
output, = sample_data(data, range=(-6, -3))

# Visualize the results
imshow_pairs({"Original": data, output})

# Sample data using Bernoulli sampling
output_b, = sample_data(data, range=(-6, -3), sampling_method=SamplingMethod.BERNOULLI)

# Visualize the Bernoulli sampling results
imshow_pairs({"Original": data, output_b})
```
