import numpy as np
import cv2
from typing import Dict, Tuple, Union

def determine_input_structure(data: np.ndarray) -> Dict[str, Union[Tuple[int, ...], int, np.dtype, str]]:
    structure: Dict[str, Union[Tuple[int, ...], int, np.dtype, str]] = {
        "shape": data.shape,
        "ndim": data.ndim,
        "dtype": data.dtype,
    }

    if data.ndim == 2:
        structure["order"] = "XY"
    elif data.ndim == 3:
        if data.shape[-1] in (3, 4):
            structure["order"] = "XYC"
        else:
            structure["order"] = "TXY"
    elif data.ndim == 4:
        if data.shape[-1] in (3, 4):
            structure["order"] = "TXYC"
        elif data.shape[1] in (3, 4):
            structure["order"] = "TCXY"
        elif data.shape[-2] == 3:
            structure["order"] = "XYCT"
    else:
        structure["order"] = "Unknown"

    return structure

def convert_to_grayscale(data: np.ndarray) -> np.ndarray:
    if data.ndim == 4:
        if data.shape[-1] == 3:
            return data.mean(axis=-1).astype(np.uint8)
        elif data.shape[1] == 3:
            return data.mean(axis=1).astype(np.uint8)
    elif data.ndim == 3:
        if data.shape[-1] == 3:
            return cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
        elif data.shape[0] == 3:
            return cv2.cvtColor(data.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    return data

def resample_matrix(matrix: np.ndarray, p: float) -> np.ndarray:
    prob_matrix = np.full(matrix.shape, p)
    return np.random.binomial(matrix, prob_matrix).astype(matrix.dtype)

def process_channel(channel_data: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
    sample = resample_matrix(channel_data, p).astype(np.uint8)
    binary_sample = (sample > 0).astype(np.uint8)
    return sample, binary_sample

def process_frame(frame: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
    if frame.ndim == 3 and frame.shape[-1] in (3, 4):
        sample = np.zeros_like(frame)
        binary_sample = np.zeros_like(frame)
        for channel in range(frame.shape[-1]):
            sample[..., channel], binary_sample[..., channel] = process_channel(frame[..., channel], p)
    else:
        sample, binary_sample = process_channel(frame, p)
    return sample, binary_sample