import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def plot_input_output(
    input_data: np.ndarray,
    output_data: np.ndarray,
    p: float,
    title: Optional[str] = None
) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    if input_data.ndim == 3 and input_data.shape[-1] == 3:
        ax1.imshow(input_data)
    else:
        ax1.imshow(input_data, cmap="gray")
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2.imshow(output_data, cmap="gray")
    ax2.set_title(f"Output Image (p={p:.5f})")
    ax2.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()