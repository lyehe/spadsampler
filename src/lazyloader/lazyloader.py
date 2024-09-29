from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from json import dump
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np
from h5py import Dataset, File
from tifffile import TiffFile, imwrite
from xmltodict import parse
from zarr import Array, Group, open as zarr_open

from .dimension_utils import predict_dimension_order, rearrange_dimensions

# Set up logging
logger = getLogger(__name__)

# Define complex types
DimensionOrder = Literal["T", "Z", "X", "Y", "C"]
RangeType = Union[int, Tuple[int, int]]
DimensionRangeType = Dict[DimensionOrder, RangeType]
DatasetType = Union[str, int, None]
GroupType = Optional[Union[str, int]]
FilePathType = Union[str, Path]
DataType = TypeVar("DataType", np.ndarray, Dataset, Array)


@dataclass
class LoadOptions:
    """
    Options for loading data, including ranges, dataset, and group.
    """

    ranges: DimensionRangeType = field(default_factory=dict)
    dataset: Optional[DatasetType] = None
    group: Optional[GroupType] = None
    dim_order: Optional[str] = None
    target_order: Optional[str] = None


def configure_load_options(
    t_range: Optional[RangeType] = None,
    z_range: Optional[RangeType] = None,
    x_range: Optional[RangeType] = None,
    y_range: Optional[RangeType] = None,
    c_range: Optional[RangeType] = None,
    dataset: Optional[DatasetType] = None,
    group: Optional[GroupType] = None,
    dim_order: Optional[str] = None,
    target_order: Optional[str] = None,
) -> LoadOptions:
    """
    Create a LoadOptions instance with the given parameters.

    :param t_range: Range for the T dimension
    :param z_range: Range for the Z dimension
    :param x_range: Range for the X dimension
    :param y_range: Range for the Y dimension
    :param c_range: Range for the C dimension
    :param dataset: Dataset name or index
    :param group: Group name or index
    :param dim_order: Input dimension order
    :param target_order: Target dimension order
    :return: LoadOptions instance
    """
    ranges = {
        dim: range_val
        for dim, range_val in zip(
            "TZXYC", (t_range, z_range, x_range, y_range, c_range)
        )
        if range_val is not None
    }

    logger.debug(
        f"Creating LoadOptions with ranges={ranges}, dataset={dataset}, group={group}"
    )
    return LoadOptions(
        ranges=ranges,
        dataset=dataset,
        group=group,
        dim_order=dim_order,
        target_order=target_order,
    )


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(
        self, file_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """
        Load data from a file.

        :param file_path: Path to the file to load
        :param options: LoadOptions instance with loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """

    @staticmethod
    def _process_range(range_val: Optional[RangeType], max_val: int) -> Tuple[int, int]:
        """
        Process and validate a range value.

        :param range_val: The range value to process
        :param max_val: The maximum allowed value
        :return: Tuple of (start, end) values
        """
        if isinstance(range_val, int):
            return 0, min(range_val, max_val)
        if isinstance(range_val, tuple) and len(range_val) == 2:
            start, end = range_val
            return max(0, start), min(end, max_val)
        return 0, max_val

    def _calculate_slices(
        self, shape: Tuple[int, ...], options: LoadOptions
    ) -> Tuple[slice, ...]:
        """
        Calculate slices for all dimensions based on the shape and options.

        :param shape: The shape of the data
        :param options: LoadOptions instance with loading parameters
        :return: Tuple of slices for each dimension
        """
        slices = []
        dim_order = options.dim_order or predict_dimension_order(shape)

        for i, dim in enumerate(dim_order):
            if dim in options.ranges:
                start, end = self._process_range(options.ranges[dim], shape[i])
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))

        logger.debug(f"Calculated slices: {slices}")
        return tuple(slices)


class ImageFolderLoader(DataLoader):
    """
    Loader for image folders.
    """

    def load(
        self, folder_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """
        Load images from a folder.

        :param folder_path: Path to the folder containing images
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
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

        first_image = cv2.imread(str(image_files[0]))
        if first_image is None:
            logger.error(f"Unable to open first image file: {image_files[0]}")
            raise IOError(f"Unable to open first image file: {image_files[0]}")

        is_color = len(first_image.shape) == 3 and first_image.shape[2] == 3
        shape = (len(image_files),) + first_image.shape
        dim_order = "TXYC" if is_color else "TXY"

        slices = self._calculate_slices(shape, options)
        t_slice, y_slice, x_slice = slices[:3]
        c_slice = slices[3] if is_color else None

        t_start, t_stop = t_slice.start or 0, t_slice.stop or len(image_files)
        y_start, y_stop = y_slice.start or 0, y_slice.stop or first_image.shape[0]
        x_start, x_stop = x_slice.start or 0, x_slice.stop or first_image.shape[1]

        if is_color:
            c_start, c_stop = c_slice.start or 0, c_slice.stop or first_image.shape[2]
            data_shape = (
                t_stop - t_start,
                y_stop - y_start,
                x_stop - x_start,
                c_stop - c_start,
            )
        else:
            data_shape = (t_stop - t_start, y_stop - y_start, x_stop - x_start)

        data = np.empty(data_shape, dtype=np.uint8)

        for i, img_file in enumerate(image_files[t_start:t_stop]):
            logger.debug(f"Loading image: {img_file}")
            img = cv2.imread(str(img_file))
            if img is None:
                logger.error(f"Unable to open image file: {img_file}")
                raise IOError(f"Unable to open image file: {img_file}")
            if is_color:
                data[i] = img[y_start:y_stop, x_start:x_stop, c_start:c_stop]
            else:
                data[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[
                    y_start:y_stop, x_start:x_stop
                ]

        logger.info(f"Loaded {len(data)} images")

        # Convert to grayscale if target order is TXY
        if options.target_order == "TXY" and is_color:
            logger.info("Converting color images to grayscale")
            data = np.mean(data, axis=-1).astype(np.uint8)
            dim_order = "TXY"

        return data, dim_order, None


class VideoLoader(DataLoader):
    """Loader for video files."""

    def load(
        self, video_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, str, Optional[Dict[str, Any]]]:
        """
        Load frames from a video file.

        :param video_path: Path to the video file
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        logger.info(f"Loading video from: {video_path}")
        is_grayscale: bool = options.target_order == "TXY"

        cap = cv2.VideoCapture(str(video_path))
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, first_frame = cap.read()
            if not ret:
                logger.error("Failed to read the first frame of the video.")
                raise ValueError("Failed to read the first frame of the video.")

            shape = (frame_count,) + first_frame.shape
            slices = self._calculate_slices(shape, options)

            t_slice, y_slice, x_slice, *c_slice = slices + (slice(None),)
            t_start = t_slice.start if t_slice.start is not None else 0
            t_stop = t_slice.stop if t_slice.stop is not None else frame_count
            data_shape = (
                t_stop - t_start,
                y_slice.stop - y_slice.start
                if y_slice.stop is not None
                else first_frame.shape[0],
                x_slice.stop - x_slice.start
                if x_slice.stop is not None
                else first_frame.shape[1],
            )
            if not is_grayscale:
                data_shape += (
                    c_slice[0].stop - c_slice[0].start
                    if c_slice[0].stop is not None
                    else first_frame.shape[2],
                )

            data = np.empty(data_shape, dtype=np.uint8)
            dim_order = "TXY" if is_grayscale else "TXYC"

            cap.set(cv2.CAP_PROP_POS_FRAMES, t_start)
            for i in range(t_start, t_stop):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Reached end of video at frame {i}")
                    break
                processed_frame = frame[y_slice, x_slice]
                if is_grayscale:
                    data[i - t_start] = cv2.cvtColor(
                        processed_frame, cv2.COLOR_BGR2GRAY
                    )
                else:
                    data[i - t_start] = processed_frame[:, :, c_slice[0]]

            logger.info(f"Loaded {len(data)} frames from video")
            return data, dim_order, None
        finally:
            cap.release()


class TiffLoader(DataLoader):
    """Loader for TIFF files."""

    def load(
        self, file_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """
        Load data from a TIFF or OME-TIFF file.

        :param file_path: Path to the TIFF file
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        logger.info(f"Loading TIFF file: {file_path}")
        with TiffFile(str(file_path)) as tif:
            return (
                self._load_ome_tiff(tif, options)
                if tif.is_ome
                else self._load_regular_tiff(tif, options)
            )

    def _load_regular_tiff(
        self, tif: TiffFile, options: LoadOptions
    ) -> Tuple[np.ndarray, str, Optional[Dict]]:
        """
        Load data from a regular TIFF file.

        :param tif: TiffFile object
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        metadata: Dict[str, Any] = {}
        if tif.imagej_metadata:
            metadata.update(tif.imagej_metadata)
        metadata.update({tag.name: tag.value for tag in tif.pages[0].tags})

        dim_order = predict_dimension_order(tif.series[0].shape)
        logger.debug(f"Predicted dimension order for TIFF: {dim_order}")

        slices = self._calculate_slices(tif.series[0].shape, options)
        data = tif.asarray()[slices]

        # Handle grayscale videos
        if data.ndim == 3 and data.shape[-1] == 3:
            if np.all(data[:, :, 0] == data[:, :, 1]) and np.all(
                data[:, :, 1] == data[:, :, 2]
            ):
                logger.info("Detected grayscale video, removing color dimension")
                data = data[:, :, 0]
                dim_order = dim_order.replace("C", "")

        return data, dim_order, metadata

    def _load_ome_tiff(
        self, tif: TiffFile, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict]]:
        """
        Load data from an OME-TIFF file.

        :param tif: TiffFile object
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        metadata = tif.ome_metadata

        dim_order = predict_dimension_order(tif.series[0].shape)
        if isinstance(metadata, dict) and "Image" in metadata:
            ome_dim_order = metadata["Image"].get("DimensionOrder", "")
            dim_order = "".join(
                char
                for char in ome_dim_order
                if char in "TZXYC"[: len(tif.series[0].shape)]
            )

        logger.debug(f"Dimension order: {dim_order}")

        slices = self._calculate_slices(tif.series[0].shape, options)
        data = tif.asarray()[slices]

        # Handle grayscale videos
        if data.ndim == 3 and data.shape[-1] == 3:
            if np.all(data[:, :, 0] == data[:, :, 1]) and np.all(
                data[:, :, 1] == data[:, :, 2]
            ):
                logger.info("Detected grayscale video, removing color dimension")
                data = data[:, :, 0]
                dim_order = dim_order.replace("C", "")

        metadata = parse(metadata) if isinstance(metadata, str) else metadata

        return data, dim_order, metadata


class HDF5Loader(DataLoader):
    """Loader for HDF5 files."""

    def load(
        self, file_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, str, Optional[Dict]]:
        """
        Load data from an HDF5 file.

        :param file_path: Path to the HDF5 file
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        logger.info(f"Loading HDF5 file: {file_path}")
        with File(file_path, "r") as f:
            data_array = self._get_data(f, options)
            metadata = dict(data_array.attrs)
            dim_order = metadata.get(
                "dim_order", predict_dimension_order(data_array.shape)
            )

            slices = self._calculate_slices(data_array.shape, options)
            data = data_array[slices]

        return data, dim_order, metadata

    def _get_data(self, root: File, options: LoadOptions) -> Dataset:
        """
        Get the data array from an HDF5 file.

        This method handles both group and dataset selection in a consistent manner.
        If a group is specified, it first selects the group, then looks for the dataset within that group.
        If no group is specified, it looks for the dataset in the root.
        If no dataset is specified, it returns the first dataset found in the selected group or root.

        :param root: HDF5 File object
        :param options: LoadOptions object containing loading parameters
        :return: HDF5 Dataset object
        """
        group = self._get_group(root, options.group)
        return self._get_dataset(group, options.dataset)

    def _get_group(self, file: File, group: Optional[GroupType]) -> Union[File, Group]:
        if group is None:
            return file
        if isinstance(group, int):
            groups = [v for k, v in file.items() if isinstance(v, Group)]
            if group < 0 or group >= len(groups):
                raise ValueError(f"Invalid group index: {group}")
            return groups[group]
        elif isinstance(group, str):
            if group not in file:
                raise ValueError(f"Group '{group}' not found in file")
            return file[group]
        else:
            raise ValueError(f"Invalid group type: {type(group)}")

    def _get_dataset(self, group: Union[File, Group], dataset: DatasetType) -> Dataset:
        if dataset is None:
            # If no dataset is specified, return the first dataset found
            for name, item in group.items():
                if isinstance(item, Dataset):
                    logger.debug(f"Using first dataset found: {name}")
                    return item
            raise ValueError("No dataset found in the HDF5 file")
        elif isinstance(dataset, int):
            datasets = [v for k, v in group.items() if isinstance(v, Dataset)]
            if dataset < 0 or dataset >= len(datasets):
                raise ValueError(f"Invalid dataset index: {dataset}")
            return datasets[dataset]
        elif isinstance(dataset, str):
            if dataset not in group:
                raise ValueError(f"Dataset '{dataset}' not found in group")
            return group[dataset]
        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")


class ZarrLoader(DataLoader):
    """Loader for Zarr files."""

    def load(
        self, file_path: Path, options: LoadOptions
    ) -> Tuple[np.ndarray, Optional[str], Optional[Dict[str, Any]]]:
        """
        Load data from a Zarr file.

        :param file_path: Path to the Zarr file
        :param options: LoadOptions object containing loading parameters
        :return: Tuple of (data, dimension order, metadata)
        """
        logger.info(f"Loading Zarr file: {file_path}")
        root = zarr_open(str(file_path), mode="r")

        data_array = self._get_data(root, options)
        metadata = dict(data_array.attrs)
        dim_order = metadata.get("dim_order") or predict_dimension_order(
            data_array.shape
        )
        dim_order = "".join(
            char.upper() for char in dim_order if char.upper() in "TZXYC"
        )

        logger.debug(f"Dimension order: {dim_order}")

        slices = self._calculate_slices(data_array.shape, options)
        data = data_array[slices]

        return data, dim_order, metadata or None

    def _get_data(self, root: Group, options: LoadOptions) -> Array:
        """
        Get the data array from a Zarr file.

        This method handles both group and dataset selection in a consistent manner.
        If a group is specified, it first selects the group, then looks for the dataset within that group.
        If no group is specified, it looks for the dataset in the root.
        If no dataset is specified, it returns the first array found in the selected group or root.

        :param root: Zarr root group
        :param options: LoadOptions object containing loading parameters
        :return: Zarr Array object
        """
        group = self._get_group(root, options.group)
        return self._get_dataset(group, options.dataset)

    def _get_group(self, root: Group, group: Optional[GroupType]) -> Group:
        if group is None:
            return root
        if isinstance(group, int):
            group_names = list(root.group_keys())
            if 0 <= group < len(group_names):
                return root[group_names[group]]
            logger.error(f"Group index {group} is out of range")
            raise ValueError(f"Group index {group} is out of range")
        if group not in root:
            logger.error(f"Group '{group}' not found in the Zarr file.")
            raise ValueError(f"Group '{group}' not found in the Zarr file.")
        return root[group]

    def _get_dataset(self, group: Group, dataset: DatasetType) -> Array:
        if isinstance(dataset, int):
            arrays = list(group.arrays())
            if 0 <= dataset < len(arrays):
                return arrays[dataset][1]
            logger.error(f"Dataset index {dataset} is out of range")
            raise ValueError(f"Dataset index {dataset} is out of range")
        if dataset is None:
            for name, value in group.arrays():
                if isinstance(value, Array):
                    logger.debug(f"Using first array found: {name}")
                    return value
            logger.error("No array found in Zarr group")
            raise ValueError("No array found in Zarr group")
        if dataset not in group or not isinstance(group[dataset], Array):
            logger.error(f"Dataset '{dataset}' not found in the Zarr group.")
            raise ValueError(f"Dataset '{dataset}' not found in the Zarr group.")
        return group[dataset]


class DataLoaderFactory:
    """
    Factory class for creating appropriate DataLoader instances.
    """

    @staticmethod
    def get_loader(file_path: Path) -> DataLoader:
        """
        Get the appropriate loader based on the file path.

        :param file_path: Path to the file or directory
        :return: Appropriate DataLoader instance
        """
        logger.debug(f"Getting loader for {file_path}")

        if file_path.suffix.lower() == ".zarr" or DataLoaderFactory._is_zarr_directory(
            file_path
        ):
            logger.info("Using ZarrLoader")
            return ZarrLoader()
        elif file_path.is_dir():
            logger.info("Using ImageFolderLoader")
            return ImageFolderLoader()
        elif file_path.suffix.lower() in (".mov", ".mp4", ".avi", ".webm", ".mkv"):
            logger.info("Using VideoLoader")
            return VideoLoader()
        elif file_path.suffix.lower() in (".tif", ".tiff"):
            logger.info("Using TiffLoader")
            return TiffLoader()
        elif file_path.suffix.lower() in (".h5", ".hdf5"):
            logger.info("Using HDF5Loader")
            return HDF5Loader()
        else:
            logger.error(f"Unsupported input format: {file_path}")
            raise ValueError(f"Unsupported input format: {file_path}")

    @staticmethod
    def _is_zarr_directory(path: Path) -> bool:
        """
        Check if the given path is a Zarr directory.

        :param path: Path to check
        :return: True if the path is a Zarr directory, False otherwise
        """
        return path.is_dir() and any(
            f.name in (".zarray", ".zgroup") for f in path.iterdir()
        )


def lazyload(
    input_path: Path,
    options: Optional[LoadOptions] = None,
) -> Tuple[np.ndarray, str, Optional[Dict]]:
    """
    Load input data from various file formats.

    :param input_path: Path to the input file
    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (data, dimension order, metadata)
    """
    logger.info(f"Loading input from {input_path}")
    options = options or LoadOptions()

    loader = DataLoaderFactory.get_loader(input_path)
    data, current_order, metadata = loader.load(input_path, options)

    if options.dim_order:
        current_order = options.dim_order
    elif not current_order:
        current_order = predict_dimension_order(data)
        logger.info(f"Predicted input dimension order: {current_order}")

    if options.target_order:
        logger.info(
            f"Rearranging dimensions from {current_order} to {options.target_order}"
        )
        data, final_order = rearrange_dimensions(
            data, current_order, options.target_order
        )
    else:
        final_order = current_order

    return data, final_order, metadata


def load(
    input_path: Path,
    options: Optional[LoadOptions] = None,
) -> Tuple[np.ndarray, str, Optional[Dict]]:
    """
    Alias for lazyload function, providing a shorter name for convenience.

    :param input_path: Path to the input file
    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (data, dimension order, metadata)
    """
    return lazyload(input_path, options)


def imread(
    input_path: Path,
    options: Optional[LoadOptions] = None,
) -> Tuple[np.ndarray, str, Optional[Dict]]:
    """
    Alias for lazyload function, mimicking the common imread function name.
    This allows for easier transition from other image reading libraries.

    :param input_path: Path to the input file
    :param options: LoadOptions instance with loading parameters
    :return: Tuple of (data, dimension order, metadata)
    """
    return lazyload(input_path, options)


def save_tiff(
    data: np.ndarray,
    output_path: Path,
    dim_order: str,
    save_metadata: bool = False,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save data as a TIFF file.

    :param data: numpy array to save
    :param output_path: Path to save the TIFF file
    :param dim_order: Dimension order of the data
    :param save_metadata: Whether to save metadata
    :param metadata: Optional metadata to save
    """
    logger.info(f"Saving TIFF file to {output_path}")
    imwrite(str(output_path), data, metadata={"axes": dim_order})

    if save_metadata and metadata:
        metadata_path = output_path.with_suffix(".json")
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            dump(metadata, f, indent=2)


def save_hdf5(
    data: np.ndarray,
    output_path: Path,
    dataset_name: str = "data",
    dim_order: str = "",
    save_metadata: bool = False,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save data as an HDF5 file.

    :param data: numpy array to save
    :param output_path: Path to save the HDF5 file
    :param dataset_name: Name of the dataset in the HDF5 file
    :param dim_order: Dimension order of the data
    :param save_metadata: Whether to save metadata
    :param metadata: Optional metadata to save
    """
    logger.info(f"Saving HDF5 file to {output_path}")
    with File(output_path, "w") as f:
        dataset = f.create_dataset(dataset_name, data=data)
        if dim_order:
            dataset.attrs["dim_order"] = dim_order

        if save_metadata and metadata:
            for key, value in metadata.items():
                dataset.attrs[key] = value


def save_zarr(
    data: np.ndarray,
    output_path: Path,
    group_name: str = "data",
    dim_order: str = "",
    save_metadata: bool = False,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save data as a Zarr file.

    :param data: numpy array to save
    :param output_path: Path to save the Zarr file
    :param group_name: Name of the group in the Zarr file
    :param dim_order: Dimension order of the data
    :param save_metadata: Whether to save metadata
    :param metadata: Optional metadata to save
    """
    logger.info(f"Saving Zarr file to {output_path}")
    root = zarr_open(str(output_path), mode="w")
    dataset = root.create_dataset(group_name, data=data)
    if dim_order:
        dataset.attrs["dim_order"] = dim_order

    if save_metadata and metadata:
        for key, value in metadata.items():
            dataset.attrs[key] = value


def save_folder(
    data: np.ndarray,
    output_path: Path,
    save_metadata: bool = False,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save data as a folder of images.

    :param data: numpy array to save
    :param output_path: Path to save the folder of images
    :param save_metadata: Whether to save metadata
    :param metadata: Optional metadata to save
    """
    logger.info(f"Saving folder of images to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(data):
        img_path = output_path / f"{i:04d}.tiff"
        imwrite(str(img_path), img)

    if save_metadata and metadata:
        metadata_path = output_path / "metadata.json"
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            dump(metadata, f, indent=2)


class SaveFactory:
    @staticmethod
    def get_saver(
        output_path: Path,
    ) -> Callable[[np.ndarray, Path, str, bool, Optional[Dict]], None]:
        """
        Get the appropriate saver based on the output path.

        :param output_path: Path to save the output
        :return: Appropriate saver function
        """
        logger.debug(f"Getting saver for {output_path}")
        if output_path.suffix.lower() in (".tif", ".tiff"):
            logger.info("Using TIFF saver")
            return save_tiff
        elif output_path.suffix.lower() in (".h5", ".hdf5"):
            logger.info("Using HDF5 saver")
            return save_hdf5
        elif output_path.suffix.lower() == ".zarr":
            logger.info("Using Zarr saver")
            return save_zarr
        else:
            logger.info("Using folder saver")
            return save_folder
