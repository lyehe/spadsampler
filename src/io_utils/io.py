from __future__ import annotations
from abc import ABC, abstractmethod
import cv2
import numpy as np
import tifffile
import h5py
import zarr
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, TypeVar, Literal
from dataclasses import dataclass
from .dimension_utils import rearrange_dimensions, predict_dimension_order
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define complex types
DimensionOrder = Literal["T", "Z", "C", "Y", "X"]
RangeType = Union[int, Tuple[int, int]]
DatasetType = Union[str, int, None]
GroupType = Optional[str]
FilePathType = Union[str, Path]
DataType = TypeVar("DataType", np.ndarray, h5py.Dataset, zarr.Array)


@dataclass
class LoadOptions:
    """
    Options for loading data, including ranges, dataset, and group.
    """

    ranges: Dict[DimensionOrder, RangeType] = None
    dataset: DatasetType = None
    group: GroupType = None

    def __post_init__(self):
        self.ranges = self.ranges or {}


def configure_load_options(
    t_range: Union[int, tuple[int, int]] | None = None,
    z_range: Union[int, tuple[int, int]] | None = None,
    c_range: Union[int, tuple[int, int]] | None = None,
    y_range: Union[int, tuple[int, int]] | None = None,
    x_range: Union[int, tuple[int, int]] | None = None,
    dataset: str | int | None = None,
    group: str | None = None,
) -> LoadOptions:
    """
    Alias for creating a LoadOptions instance with the given parameters.

    Args:
        t_range: Range for the T dimension.
        z_range: Range for the Z dimension.
        c_range: Range for the C dimension.
        y_range: Range for the Y dimension.
        x_range: Range for the X dimension.
        dataset: Dataset name or index.
        group: Group name.

    Returns:
        LoadOptions instance.
    """
    ranges = {
        dim: range_val
        for dim, range_val in zip(
            "TZCYX", (t_range, z_range, c_range, y_range, x_range)
        )
        if range_val is not None
    }

    logger.debug(
        f"Creating LoadOptions with ranges={ranges}, dataset={dataset}, group={group}"
    )
    return LoadOptions(ranges=ranges, dataset=dataset, group=group)


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(
        self, file_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """Load data from a file."""

    @staticmethod
    def process_range(range_val: Optional[RangeType], max_val: int) -> Tuple[int, int]:
        """Process and validate a range value."""
        if isinstance(range_val, int):
            return 0, min(range_val, max_val)
        if isinstance(range_val, tuple) and len(range_val) == 2:
            start, end = range_val
            return max(0, start), min(end, max_val)

        start, end = 0, max_val
        if end > max_val or start < 0:
            logger.warning(f"Range {range_val} out of bounds. Using {start}:{end}")
        return start, end

    def apply_slices(
        self,
        data: DataType,
        dim_order: str,
        options: LoadOptions,
    ) -> np.ndarray:
        """Apply slices to the data based on the dimension order and options."""
        slices = [
            slice(
                *self.process_range(options.ranges.get(dim, slice(None)), data.shape[i])
            )
            for i, dim in enumerate(dim_order)
        ]
        logger.debug(f"Applying slices: {slices}")
        return data[tuple(slices)]


def load_data(
    file_path: FilePathType, options: Optional[LoadOptions] = None
) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
    """
    Functional interface to load data from a file or folder.

    Args:
        file_path (FilePathType): Path to the file or folder to load.
        options (LoadOptions, optional): Options for loading the data. Defaults to None.

    Returns:
        np.ndarray | Tuple[np.ndarray, Optional[str], Optional[Dict]]: Loaded data, dimension order, and metadata.
    """
    file_path = Path(file_path)
    options = options or LoadOptions()
    logger.info(f"Loading data from {file_path}")

    if file_path.is_dir():
        logger.debug("Detected directory, using ImageFolderLoader")
        return ImageFolderLoader().load(file_path, options)

    loaders = {
        (".tif", ".tiff"): TiffLoader(),
        (".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"): VideoLoader(),
        (".h5", ".hdf5"): HDF5Loader(),
        (".zarr",): ZarrLoader(),
    }

    for extensions, loader in loaders.items():
        if file_path.suffix.lower() in extensions:
            logger.debug(f"Using {loader.__class__.__name__} for {file_path.suffix}")
            return loader.load(file_path, options)

    logger.error(f"Unsupported file format: {file_path.suffix}")
    raise ValueError(f"Unsupported file format: {file_path.suffix}")


class ImageFolderLoader(DataLoader):
    """
    Loader for image folders.
    """

    def load(
        self, folder_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """
        Load images from a folder.
        """
        logger.info(f"Loading images from folder: {folder_path}")
        image_files = sorted(
            [
                f
                for f in folder_path.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            ]
        )
        if not image_files:
            logger.error(f"No image files found in folder: {folder_path}")
            raise ValueError(f"No image files found in folder: {folder_path}")

        t_start, t_end = self.process_range(
            options.ranges.get("T", len(image_files)), len(image_files)
        )
        logger.debug(f"Processing images from {t_start} to {t_end}")
        images = []

        for img_file in image_files[t_start:t_end]:
            logger.debug(f"Loading image: {img_file}")
            img = cv2.imread(str(img_file))
            if img is None:
                logger.error(f"Unable to open image file: {img_file}")
                raise IOError(f"Unable to open image file: {img_file}")

            img = self.apply_slices(img, "YXC", options)
            images.append(img)

        logger.info(f"Loaded {len(images)} images")
        return np.array(images), "TYXC", None


class VideoLoader(DataLoader):
    """
    Loader for video files.
    """

    def load(
        self, video_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """
        Load frames from a video file.
        """
        logger.info(f"Loading video from: {video_path}")
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            logger.error(f"Unable to open video file: {video_path}")
            raise IOError(f"Unable to open video file: {video_path}")

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        t_start, t_end = self.process_range(
            options.ranges.get("T", frame_count), frame_count
        )
        logger.debug(f"Processing frames from {t_start} to {t_end}")

        frames = []
        for i in range(t_end):
            ret, frame = video.read()
            if not ret:
                logger.warning(f"End of video reached at frame {i}")
                break
            if i >= t_start:
                frame = self.apply_slices(frame, "YXC", options)
                frames.append(frame)

        video.release()
        logger.info(f"Loaded {len(frames)} frames")
        return np.array(frames), "TYXC", None


class TiffLoader(DataLoader):
    """
    Loader for TIFF files.
    """

    def load(
        self, file_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """
        Load data from a TIFF file.
        """
        logger.info(f"Loading TIFF file: {file_path}")
        with tifffile.TiffFile(str(file_path)) as tif:
            full_data = tif.asarray()
            metadata = tif.imagej_metadata or tif.metadata or {}

        dim_order = metadata.get("axes", "")
        if dim_order:
            dim_order = "".join(
                char.upper() for char in dim_order if char.upper() in "TZCYX"
            )
        else:
            dim_order = "TZCYX"[: full_data.ndim]
        logger.debug(f"Dimension order: {dim_order}")

        data = self.apply_slices(full_data, dim_order, options)

        # Output metadata as JSON if available
        if metadata:
            json_path = file_path.with_suffix(".json")
            logger.info(f"Writing metadata to {json_path}")
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return data, dim_order or None, metadata


class HDF5Loader(DataLoader):
    """
    Loader for HDF5 files.
    """

    def load(
        self, file_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """
        Load data from an HDF5 file.
        """
        logger.info(f"Loading HDF5 file: {file_path}")
        with h5py.File(file_path, "r") as f:
            dataset = self._get_dataset(f, options.dataset)
            full_data = f[dataset]
            dim_order = full_data.attrs.get("dim_order", "")
            if dim_order:
                dim_order = "".join(
                    char.upper() for char in dim_order if char.upper() in "TZCYX"
                )
            else:
                dim_order = "TZCYX"[: full_data.ndim]
            logger.debug(f"Dimension order: {dim_order}")

            data = self.apply_slices(full_data, dim_order, options)

            # Collect metadata
            metadata = dict(full_data.attrs)

            # Output metadata as JSON if available
            if metadata:
                json_path = file_path.with_suffix(".json")
                logger.info(f"Writing metadata to {json_path}")
                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=2)

        return data, dim_order or None, metadata

    def _get_dataset(self, f: h5py.File, dataset: DatasetType) -> str:
        """
        Get the dataset name from an HDF5 file.
        """
        if isinstance(dataset, int):
            return list(f.keys())[dataset]
        elif dataset is None:
            return list(f.keys())[0]
        elif dataset not in f:
            logger.error(f"Dataset '{dataset}' not found in the HDF5 file.")
            raise ValueError(f"Dataset '{dataset}' not found in the HDF5 file.")
        return dataset


class ZarrLoader(DataLoader):
    """
    Loader for Zarr files.
    """

    def load(
        self, file_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """
        Load data from a Zarr file.
        """
        logger.info(f"Loading Zarr file: {file_path}")
        root = zarr.open(str(file_path), mode="r")
        data = root[options.group] if options.group else root

        dim_order = data.attrs.get("dim_order", "")
        if dim_order:
            dim_order = "".join(
                char.upper() for char in dim_order if char.upper() in "TZCYX"
            )
        else:
            dim_order = "TZCYX"[: data.ndim]
        logger.debug(f"Dimension order: {dim_order}")

        sliced_data = self.apply_slices(data, dim_order, options)

        # Collect metadata
        metadata = dict(data.attrs)

        # Output metadata as JSON if available
        if metadata:
            json_path = file_path.with_suffix(".json")
            logger.info(f"Writing metadata to {json_path}")
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return sliced_data, dim_order or None, metadata


class DataLoaderFactory:
    """
    Factory class for creating appropriate DataLoader instances.
    """

    @staticmethod
    def get_loader(file_path: Path) -> DataLoader:
        """
        Get the appropriate loader based on the file path.
        """
        logger.debug(f"Getting loader for {file_path}")
        if file_path.is_dir():
            return ImageFolderLoader()
        elif file_path.suffix.lower() in (".mov", ".mp4", ".avi", ".webm", ".mkv"):
            return VideoLoader()
        elif file_path.suffix.lower() in (".tif", ".tiff"):
            return TiffLoader()
        elif file_path.suffix.lower() in (".h5", ".hdf5"):
            return HDF5Loader()
        elif file_path.suffix.lower() == ".zarr":
            return ZarrLoader()
        else:
            logger.error(f"Unsupported input format: {file_path}")
            raise ValueError(f"Unsupported input format: {file_path}")


def load_input(
    input_path: Path,
    dim_order: Optional[str] = None,
    target_order: Optional[str] = None,
    ranges: Optional[Dict[DimensionOrder, RangeType]] = None,
    hdf5_dataset: DatasetType = None,
    zarr_group: GroupType = None,
) -> Tuple[np.ndarray, str, Optional[Dict]]:
    """
    Load input data from various file formats.
    """
    logger.info(f"Loading input from {input_path}")
    options = LoadOptions(ranges=ranges, dataset=hdf5_dataset, group=zarr_group)
    loader = DataLoaderFactory.get_loader(input_path)
    data, current_order, metadata = loader.load(input_path, options)

    if dim_order:
        current_order = dim_order
    elif not current_order:
        current_order = predict_dimension_order(data)
        logger.info(f"Predicted input dimension order: {current_order}")

    if target_order:
        logger.info(f"Rearranging dimensions from {current_order} to {target_order}")
        data, final_order = rearrange_dimensions(data, target_order)
    else:
        final_order = current_order

    return data, final_order, metadata


def save_tiff(data: np.ndarray, output_path: Path, dim_order: str) -> None:
    """
    Save data as a TIFF file.
    """
    logger.info(f"Saving TIFF file to {output_path}")
    metadata = {"axes": dim_order}
    tifffile.imwrite(str(output_path), data, metadata=metadata)


def save_hdf5(
    data: np.ndarray, output_path: Path, dataset_name: str = "data", dim_order: str = ""
) -> None:
    """
    Save data as an HDF5 file.
    """
    logger.info(f"Saving HDF5 file to {output_path}")
    with h5py.File(output_path, "w") as f:
        dataset = f.create_dataset(dataset_name, data=data)
        if dim_order:
            dataset.attrs["dim_order"] = dim_order


def save_zarr(
    data: np.ndarray, output_path: Path, group_name: str = "data", dim_order: str = ""
) -> None:
    """
    Save data as a Zarr file.
    """
    logger.info(f"Saving Zarr file to {output_path}")
    root = zarr.open(str(output_path), mode="w")
    dataset = root.create_dataset(group_name, data=data)
    if dim_order:
        dataset.attrs["dim_order"] = dim_order
