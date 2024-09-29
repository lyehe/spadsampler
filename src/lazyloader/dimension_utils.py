from typing import Tuple, Optional
import numpy as np


def translate_dimension_names(order: str) -> str:
    """
    Translate F, D, W, H to T, Z, X, Y respectively.
    Ensure the output contains only one each of X, Y, Z, C, T.

    :param order: Input dimension order string
    :return: Translated dimension order string
    """
    translation = {"F": "T", "D": "Z", "W": "X", "H": "Y"}
    seen = set()
    translated = []

    for char in order:
        new_char = translation.get(char, char)
        if new_char in "XYZCT":
            if new_char in seen:
                raise ValueError(
                    f"Duplicate dimension '{new_char}' found in order string"
                )
            seen.add(new_char)
            translated.append(new_char)

    return "".join(translated)


def predict_dimension_order(data: np.ndarray | Tuple[int, ...]) -> str:
    """Predict the original dimension order based on array shape."""
    shape = data.shape if isinstance(data, np.ndarray) else data
    dims = len(shape)

    if dims == 2:
        return "XY"
    elif dims == 3:
        if shape[-1] in (3, 4):  # Likely RGB or RGBA
            return "XYC"
        else:
            return "TXY"  # Assume time series by default for 3D
    elif dims == 4:
        if shape[-1] in (3, 4):  # Likely RGB or RGBA
            return "TXYC"
        else:
            return "TZXY"
    elif dims == 5:
        return "TZXYC"
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}")


def rearrange_dimensions(
    data: np.ndarray,
    current_order: str,
    target_order: Optional[str] = None,
    return_order: bool = True,
) -> np.ndarray | Tuple[np.ndarray, Optional[str]]:
    """
    Rearrange the dimensions of the input data to the desired order.

    Args:
        data (np.ndarray): Input data array.
        current_order (str): Current dimension order of the data.
        target_order (str, optional): Desired dimension order. If None, predict order.
        return_order (bool): Whether to return the final order string.

    Returns:
        Tuple[np.ndarray, Optional[str]]: Rearranged data array and final order (if return_order is True).
    """
    current_dims = data.ndim
    print(f"The shape of the input data is {data.shape}")

    if target_order:
        target_order = translate_dimension_names(target_order)
    else:
        target_order = current_order
        print(f"No target order specified, using current order: {target_order}")

    valid_orders = ["TZXYC", "ZXYC", "TXYC", "TXY", "ZXY", "XYC", "XY"]
    if target_order not in valid_orders:
        raise ValueError(
            f"Invalid target order: {target_order}. Must be one of {valid_orders}"
        )

    if len(target_order) != current_dims:
        raise ValueError(
            f"Target order '{target_order}' does not match input dimensions {current_dims}"
        )

    print(f"The current order is {current_order} and the target is {target_order}")
    transpose_axes = [current_order.index(dim) for dim in target_order]
    print(f"Transposing axes: {transpose_axes}")

    rearranged_data = np.transpose(data, transpose_axes)
    print(f"The shape of the rearranged data is {rearranged_data.shape}")

    return (rearranged_data, target_order) if return_order else rearranged_data
