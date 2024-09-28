from typing import Tuple, Optional, List
import numpy as np

def translate_dimension_names(order: str) -> str:
    """
    Translate F, D, W, H to T, Z, X, Y respectively.
    Ensure the output contains only one each of X, Y, Z, C, T.
    """
    translation = {"F": "T", "D": "Z", "W": "X", "H": "Y"}
    translated = "".join(translation.get(char, char) for char in order)

    # Check for duplicates and raise an error if found
    seen = set()
    duplicates = [char for char in translated if char in "XYZCT" and not seen.add(char)]
    if duplicates:
        raise ValueError(f"Duplicate dimension '{duplicates[0]}' found in order string")

    return translated

def predict_dimension_order(data: np.ndarray | Tuple[int, ...]) -> Optional[str]:
    """Predict the original dimension order based on array shape."""
    shape = data.shape if isinstance(data, np.ndarray) else data
    dims = len(shape)
    
    if dims == 5 and all(size >= 32 for size in shape):
        print("Warning: Cannot determine order for 5D arrays with all dims >= 32. User must specify order.")
        return None

    small_dims = [i for i, size in enumerate(shape) if size < 32]
    
    order = _predict_order_based_on_dims(dims, small_dims, shape)
    
    predicted_order = ''.join(order)
    print(f"Predicted order: {predicted_order}")
    return predicted_order

def _predict_order_based_on_dims(dims: int, small_dims: List[int], shape: Tuple[int, ...]) -> List[str]:
    if small_dims:
        return _predict_order_with_small_dims(dims, small_dims, shape)
    else:
        return _predict_order_without_small_dims(dims)

def _predict_order_with_small_dims(dims: int, small_dims: List[int], shape: Tuple[int, ...]) -> List[str]:
    if len(small_dims) == 1:
        return _predict_order_with_one_small_dim(dims, small_dims[0], shape)
    else:
        return _predict_order_with_multiple_small_dims(dims, small_dims, shape)

def _predict_order_with_one_small_dim(dims: int, channel_dim: int, shape: Tuple[int, ...]) -> List[str]:
    order = list('TZXY')
    order.insert(channel_dim, 'C')
    print(f"Warning: Assuming dimension {channel_dim} (size {shape[channel_dim]}) is the channel dimension.")
    return order[:dims]

def _predict_order_with_multiple_small_dims(dims: int, small_dims: List[int], shape: Tuple[int, ...]) -> List[str]:
    channel_dim, time_dim = small_dims[:2]
    order = ['X', 'Y', 'Z']
    order.insert(channel_dim, 'C')
    order.insert(time_dim, 'T')
    print(f"Warning: Assuming dimension {channel_dim} (size {shape[channel_dim]}) is the channel dimension "
          f"and dimension {time_dim} (size {shape[time_dim]}) is the time dimension.")
    return order[:dims]

def _predict_order_without_small_dims(dims: int) -> List[str]:
    if dims == 2:
        return ['X', 'Y']
    elif dims == 3:
        return ['Z', 'X', 'Y']
    elif dims == 4:
        return ['T', 'Z', 'X', 'Y']
    else:
        return ['T', 'Z', 'X', 'Y', 'C'][:dims]

def rearrange_dimensions(
    data: np.ndarray, target_order: Optional[str] = None, return_order: bool = True
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Rearrange the dimensions of the input data to the desired order.

    Args:
        data (np.ndarray): Input data array.
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
        target_order = predict_dimension_order(data.shape)
        if target_order is None:
            raise ValueError(
                "Cannot determine dimension order. Please specify target_order."
            )
        print(f"Predicted order: {target_order}")

    valid_orders = ["TZXYC", "ZXYC", "TXYC", "TXY", "ZXY", "XYC", "XY"]
    if target_order not in valid_orders:
        raise ValueError(
            f"Invalid target order: {target_order}. Must be one of {valid_orders}"
        )

    if len(target_order) != current_dims:
        raise ValueError(
            f"Target order '{target_order}' does not match input dimensions {current_dims}"
        )

    current_order = "TZXYC"[:current_dims]

    print(f"The current order is {current_order} and the target is {target_order}")
    transpose_axes = [current_order.index(dim) for dim in target_order]
    print(f"Transposing axes: {transpose_axes}")

    rearranged_data = np.transpose(data, transpose_axes)
    print(f"The shape of the rearranged data is {rearranged_data.shape}")

    return (rearranged_data, target_order) if return_order else (rearranged_data, None)