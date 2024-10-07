import copy
import os
import socket
import time
from abc import ABC, abstractmethod
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cifutils import CIFParser
from torch.utils.data import ConcatDataset, Dataset

from datahub.common import default, exists
from datahub.datasets import logger
from datahub.datasets.dataframe_parsers import MetadataRowParser, load_example_from_metadata_row
from datahub.preprocessing.constants import NA_VALUES
from datahub.transforms.base import Compose, Transform, TransformedDict
from datahub.utils.debug import save_failed_example_to_disk
from datahub.utils.rng import capture_rng_states

_USER = default(os.getenv("USER"), "")


class BaseDataset(ABC):
    """
    Abstract base class for datasets. All dataset types (e.g., Pandas, Polars) should inherit from this class
    and implement its methods.

    In addition to the standard PyTorch Dataset methods (`__getitem__`, `__len__`), this class requires
    implementations for converting between example IDs and indices, which is necessary for our nested dataset structure.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        pass

    @abstractmethod
    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert an example ID or list of example IDs to the corresponding index or indices."""
        pass

    @abstractmethod
    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert an index or list of indices to the corresponding example ID or IDs."""
        pass


class StructuralDatasetWrapper(BaseDataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_parser: MetadataRowParser,
        cif_parser: CIFParser | None = None,
        cif_parser_args: dict | None = None,
        transform: Transform | Compose | None = None,
        return_key: str | None = None,
        save_failed_examples_to_dir: PathLike | str | None = f"/net/scratch/{_USER}/failures" if _USER else None,
    ):
        """
        Decorator (wrapper) for an arbitrary Dataset (e.g., PandasDataset, PolarsDataset, etc.) to handle loading of structural data from PDB or CIF files,
        parsing, and applying a Transformation pipeline to the data.

        Designed to be used with a Transforms pipeline to process the data and a MetadataRowParser to convert the dataset rows into a common dictionary format.

        For more detail, see the README in the `datasets` directory.

        Args:
            dataset (Dataset): The dataset to wrap. For example, a PandasDataset, PolarsDataset, or standard PyTorch Dataset.
            dataset_parser (MetadataRowParser): Parser to convert dataset metadata rows into a common dictionary format. See `datahub.datasets.dataframe_parsers`.
            cif_parser (CIFParser, optional): Parser for CIF files. If None, a new CIFParser will be created.
            cif_parser_args (dict, optional): Arguments to pass to the CIFParser (will override the defaults). Defaults to None.
            transform (Transform | Compose, optional): Transformation pipeline to apply to the data. See `datahub.transforms.base`.
            return_key (str, optional): Key to return from the data dictionary instead of the entire dictionary.
            save_failed_examples_to_dir (PathLike | str | None, optional): Directory to save failed examples. Defaults to f"/net/scratch/{_USER}/failures".

        Example usage:
            ```python
            dataset = StructuralDatasetDecorator(dataset=PandasDataset(data="path/to/data.csv"), ...)
            dataset[0]  # Returns the processed data for the first example.
            ```
        """
        # ...basic assignments
        self.transform = transform
        self.return_key = return_key
        self.save_failed_examples_to_dir = (
            Path(save_failed_examples_to_dir) if exists(save_failed_examples_to_dir) else None
        )
        self.cif_parser_args = cif_parser_args
        self.dataset_parser = dataset_parser
        self.dataset = dataset

        # ...initialize the CIFParser, if not provided
        self.cif_parser = cif_parser if cif_parser else CIFParser()

        # ...carry forward the data
        self.data = self.dataset.data

        # ...carry forward the name
        self.name = self.dataset.name if hasattr(self.dataset, "name") else repr(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        """
        Performs the following steps:
            (1) Retrieve the row at the specified index from the dataset using the __getitem__ method.
            (2) Parse the row into a common dictionary format using the dataset parser.
            (3) Load the CIF file from the information in the common dictionary format (i.e., the "path" key).
            (4) Apply the transformation pipeline to the data which, at a minimum, contains the output of the CIFParser.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Any: The processed item.
        """

        # Capture example ID & current rng state (for reproducibility & debugging)
        if hasattr(self, "idx_to_id"):
            # ...if the dataset has a custom idx_to_id method, use it (e.g., for a PandasDataset)
            example_id = self.idx_to_id(idx)
        else:
            # ...otherwise, fallback to a the `id_column` or a string representation of the index
            example_id = self.dataset[idx][self.id_column] if self.id_column else f"row_{idx}"

        # Get process id and hostname (for debugging)
        logger.debug(f"({socket.gethostname()}:{os.getpid()}) Processing example ID: {example_id}")

        # Load the row, using the __getitem__ method of the dataset
        row = self.dataset[idx]

        # Process the row into a transform-ready dictionary with the given CIF and dataset parsers
        # We require the "data" dictionary output from `load_example_from_metadata_row` to contain, at a minimum:
        #   (a) An "id" key, which uniquely identifies the example within the dataframe; and,
        #   (b) The "path" key, which is the path to the CIF file
        _start_parse_time = time.time()
        data = load_example_from_metadata_row(
            row, self.dataset_parser, cif_parser=self.cif_parser, cif_parser_args=self.cif_parser_args
        )
        _stop_parse_time = time.time()

        # Manually add timing for cif-parsing
        data = TransformedDict(data)
        data.__transform_history__.append(
            dict(
                name="load_example_from_metadata_row",
                instance=hex(id(load_example_from_metadata_row)),
                start_time=_start_parse_time,
                end_time=_stop_parse_time,
                processing_time=_stop_parse_time - _start_parse_time,
            )
        )

        # Apply the transformation pipeline to the data
        if exists(self.transform):
            try:
                rng_state_dict = capture_rng_states(include_cuda=False)
                data = self.transform(data)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                # Log the error and save the failed example to disk (optional)
                logger.info(f"Error processing row {idx} ({example_id}): {e}")

                if exists(self.save_failed_examples_to_dir):
                    save_failed_example_to_disk(
                        example_id=example_id,
                        error_msg=e,
                        rng_state_dict=rng_state_dict,
                        data={},  # We do not save the data, since it may be large.
                        fail_dir=self.save_failed_examples_to_dir,
                    )
                raise e

        # Return the specified key or the entire data dict (i.e., only "feats" key from the Transform dictionary)
        if exists(self.return_key):
            return data[self.return_key]
        else:
            return data

    def __len__(self) -> int:
        """Pass through the length of the wrapped dataset."""
        return len(self.dataset)

    def __contains__(self, example_id: str) -> bool:
        """Pass through the contains method of the wrapped dataset."""
        return example_id in self.dataset

    def id_to_idx(self, example_id: str) -> int:
        """Pass through the id_to_idx method of the wrapped dataset."""
        return self.dataset.id_to_idx(example_id)

    def idx_to_id(self, idx: int) -> str:
        """Pass through the idx_to_id method of the wrapped dataset."""
        return self.dataset.idx_to_id(idx)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped dataset."""
        return getattr(self.dataset, name)


class PandasDataset(BaseDataset):
    """
    A wrapper around PyTorch's Dataset class that allows for easy loading, filtering, and indexing of datasets stored as Pandas DataFrames.
    The underlying DataFrame can be accessed via the `data` property.

    For example usage, see the tests in `tests/datasets/test_datasets.py`.

    Args:
        data (pd.DataFrame | PathLike): The dataset, either as a Pandas DataFrame or a path to a file.
        id_column (str | None, optional): The column to use as the index; must be unique within the DataFrame. Defaults to None.
            For example, we use the `example_id` column as the index in the `PDBDataset`. By setting the dataframe index to the `example_id`
            column, we can retrieve the row corresponding to a specific example ID by calling `dataset.data.loc[example_id]` in O(1) time.
        filters (list[str] | None, optional): A list of query strings to filter the data. Defaults to None. For examples on how to specify filters,
            see the docstring for `_apply_filters`.
        name (str | None, optional): The name of the dataset. Defaults to None. Useful for debugging and logging.
        columns_to_load (list[str] | None, optional): Specific columns to load if data is provided as a file path. Defaults to None. Helpful for
            large datasets where only a subset of columns is needed (if using `parquet` or other columnar storage formats).
        **load_kwargs (Any): Additional keyword arguments for loading the data.

    Attributes:
        data (pd.DataFrame): The underlying DataFrame, accessible via the `data` property.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame | PathLike,
        id_column: str | None = None,
        filters: list[str] | None = None,
        name: str | None = None,
        columns_to_load: list[str] | None = None,
        **load_kwargs: Any,
    ):
        if name is not None:
            self.name = name
        else:
            self.name = repr(self)

        # Load the data from the path, if provided (and load only the specified columns)
        if isinstance(data, (PathLike, str)):
            data = self._load_from_path(data, columns_to_load, **load_kwargs)
        self._data = data

        # Apply filters, if provided
        self.filters = filters
        self._already_filtered = False
        if exists(filters):
            self._apply_filters(filters)
        self._already_filtered = True

        if id_column is not None:
            assert id_column in self._data.columns, f"Column {id_column} not found in dataset."
            self._data.set_index(id_column, inplace=True, drop=False, verify_integrity=True)

    def _load_from_path(
        self, path: PathLike | str, columns_to_load: list[str] | None = None, **load_kwargs: Any
    ) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".csv":
            data = pd.read_csv(path, usecols=columns_to_load, keep_default_na=False, na_values=NA_VALUES, **load_kwargs)
        elif path.suffix == ".parquet":
            data = pd.read_parquet(path, columns=columns_to_load, **load_kwargs)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return data

    @property
    def data(self) -> pd.DataFrame:
        """Expose underlying dataframe as property to discourage changing it (can lead to unexpected behavior with torch ConcatDatasets)."""
        return self._data

    def __getitem__(self, idx: int) -> Any:
        return self._data.iloc[idx]

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        return example_id in self._data.index

    def _id_to_index_single(self, example_id: str) -> int:
        return self._data.index.get_loc(example_id)

    def _id_to_index_multiple(self, example_ids: list[str]) -> list[int]:
        idxs = np.arange(len(self._data))
        return [idxs[self._data.index.get_loc(example_id)] for example_id in example_ids]

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert an example ID to the corresponding local index."""
        if np.isscalar(example_id):
            return self._id_to_index_single(example_id)
        elif isinstance(example_id, (list, np.ndarray, tuple)):
            return self._id_to_index_multiple(example_id)
        else:
            raise ValueError(f"Invalid type for example_id: {type(example_id)}")

    def idx_to_id(self, idx: int | list[int]) -> str | np.ndarray:
        """Convert a local index to the corresponding example ID."""
        _return_single = False
        if isinstance(idx, int):
            _return_single = True
            idx = slice(idx, idx + 1)
        ids = self._data.iloc[idx].index.values
        return ids if not _return_single else ids[0]

    def _apply_filters(self, filters: list[str]) -> pd.DataFrame:
        """
        Apply filters to the data based on the provided list of query strings.
        For documentation on pandas query syntax, see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html

        Args:
            filters (List[str]): List of query strings to apply to the data.

        Raises:
            ValueError: If the data is not initialized or if a query removes all rows.
            Warning: If a query does not remove any rows.

        Exampleelse:
            logger.info(
                f"Query '{query}' filtered dataset from {original_num_rows:,} to {filtered_num_rows:,} rows (dropped {original_num_rows - filtered_num_rows:,} rows)"
            ):
            queries = [
                "deposition_date < '2020-01-01'",
                "resolution < 2.5 and ~method.str.contains('NMR')",
                "cluster.notnull()",
                "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']"
            ]
        """
        assert not self._already_filtered, "Filters cannot be applied after initialization."

        # Apply queries one by one, confirming the impact of each
        for query in filters:
            self._apply_query(query)

    def _apply_query(self, query: str):
        """
        Apply a single query to the data.

        Args:
            query (str): A query string to apply to the data.
        """
        # Filter using query and validate impact
        original_num_rows = len(self._data)
        self._data = self._data.query(query)
        filtered_num_rows = len(self._data)
        self._validate_filter_impact(query, original_num_rows, filtered_num_rows)

    def _validate_filter_impact(self, query: str, original_num_rows: int, filtered_num_rows: int):
        """
        Validate the impact of the filter.

        Args:
            query (str): The query string that was applied.
            original_num_rows (int): The number of rows before applying the filter.
            filtered_num_rows (int): The number of rows after applying the filter.

        Raises:
            Warning: If the filter did not remove any rows.
            ValueError: If the filter removed all rows.
        """
        rows_removed = original_num_rows - filtered_num_rows
        percent_removed = (rows_removed / original_num_rows) * 100
        percent_remaining = (filtered_num_rows / original_num_rows) * 100

        if filtered_num_rows == original_num_rows:
            logger.warning(f"Query '{query}' on dataset {self.name} did not remove any rows.")
        elif filtered_num_rows == 0:
            raise ValueError(f"Query '{query}' on dataset {self.name} removed all rows.")
        else:
            logger.info(
                f"\n+-------------------------------------------+\n"
                f"Query '{query}' on dataset {self.name}:\n"
                f"  - Started with: {original_num_rows:,} rows\n"
                f"  - Removed: {rows_removed:,} rows ({percent_removed:.2f}%)\n"
                f"  - Remaining: {filtered_num_rows:,} rows ({percent_remaining:.2f}%)\n"
                f"+-------------------------------------------+\n"
            )


class ConcatDatasetWithID(ConcatDataset):
    """Equivalent to `torch.utils.data.ConcatDataset` but allows accessing examples by ID."""

    datasets: list[Dataset]

    def __init__(self, datasets: list[Dataset]):
        super().__init__(datasets)

        # Print the length of each dataset
        for i, dataset in enumerate(datasets):
            logger.info(f"Dataset {i} ({type(dataset)}): {len(dataset):,} examples")

    @cached_property
    def _can_convert_ids_and_idx(self) -> bool:
        has_id_to_idx = all(hasattr(sub_dataset, "id_to_idx") for sub_dataset in self.datasets)
        has_idx_to_id = all(hasattr(sub_dataset, "idx_to_id") for sub_dataset in self.datasets)
        return has_id_to_idx and has_idx_to_id and self._can_check_contains

    @cached_property
    def _can_check_contains(self) -> bool:
        return all(hasattr(sub_dataset, "__contains__") for sub_dataset in self.datasets)

    def _raise_if_cannot_check_contains(self):
        if not self._can_check_contains:
            raise ValueError("This dataset cannot check if it contains an example ID.")

    def _raise_if_cannot_convert_ids_and_idx(self):
        if not self._can_convert_ids_and_idx:
            raise ValueError("This dataset cannot convert example IDs to indices.")

    def _raise_if_idx_out_of_bounds(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise ValueError(f"Index {idx} out of bounds for dataset of length {len(self)}.")

    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID."""
        self._raise_if_cannot_check_contains()
        for sub_dataset in self.datasets:
            if example_id in sub_dataset:
                return True
        return False

    def id_to_idx(self, example_id: str) -> int:
        """Retrieves the index corresponding to the example ID.

        WARNING: Assumes that the example ID is unique within the dataset. If not,
            the first occurrence of the example ID is returned.
        """
        # TODO: Generalize to list[str]
        self._raise_if_cannot_convert_ids_and_idx()
        offset = 0
        for sub_dataset in self.datasets:
            if example_id in sub_dataset:
                return offset + sub_dataset.id_to_idx(example_id)
            offset += len(sub_dataset)
        raise ValueError(f"Example ID {example_id} not found in any sub-dataset.")

    def idx_to_id(self, idx: int) -> str:
        """Retrieves the example ID corresponding to the index."""
        # TODO: Generalize to list[int]
        self._raise_if_cannot_convert_ids_and_idx()
        self._raise_if_idx_out_of_bounds(idx)
        for sub_dataset in self.datasets:
            if idx < len(sub_dataset):
                return sub_dataset.idx_to_id(idx)
            idx -= len(sub_dataset)
        # This should never be reached
        raise ValueError(f"Index {idx} out of bounds for any sub-dataset.")

    def get_dataset_by_idx(self, idx: int) -> Dataset:
        """Retrieves the dataset containing the index."""
        self._raise_if_idx_out_of_bounds(idx)
        for sub_dataset in self.datasets:
            if idx < len(sub_dataset):
                return sub_dataset
            idx -= len(sub_dataset)
        # This should never be reached
        raise ValueError(f"Index {idx} out of bounds for any sub-dataset.")

    def get_dataset_by_id(self, example_id: str) -> Dataset:
        """Retrieves the dataset containing the example ID.

        WARNING: Assumes that the example ID is unique within the dataset. If not,
            the first occurrence of the example ID is returned.
        """
        idx = self.id_to_idx(example_id)
        return self.get_dataset_by_idx(idx)


def get_row_and_index_by_example_id(dataset: ConcatDatasetWithID, example_id: str) -> dict:
    """
    Retrieve a row and its index from a nested dataset structure by its example ID.

    Parameters:
        dataset (PandasDataset | ConcatDataset): The dataset or concatenated dataset to search.
            Must have the `id_to_idx` method.
        example_id (str): The example ID to search for.

    Returns:
        tuple: A tuple containing the row (pd.Series) and the (global)index (int) corresponding to the example ID.
    """
    assert hasattr(dataset, "id_to_idx"), "Dataset must have the `id_to_idx` method."
    idx = dataset.id_to_idx(example_id)

    _local_idx = copy.deepcopy(idx)
    while isinstance(dataset, ConcatDatasetWithID):
        dataset = dataset.get_dataset_by_idx(_local_idx)
        _local_idx = dataset.id_to_idx(example_id)

    row = dataset.data.loc[example_id]
    return {"row": row, "index": idx}


class FallbackDatasetWrapper(Dataset):
    """
    A wrapper around a dataset that allows for a fallback dataset to be used when an error occurs.

    Meant to be used with a FallbackSamplerWrapper.
    """

    def __init__(self, dataset: Dataset, fallback_dataset: Dataset):
        """
        FallbackDatasetWrapper is a wrapper around a dataset that provides a fallback mechanism
        to another dataset in case of errors during data retrieval.

        Attributes:
            - dataset (Dataset): The primary dataset to retrieve data from.
            - fallback_dataset (Dataset): The fallback dataset to use when an error occurs. This
                may be the same as the primary dataset, or a different one.
        """
        self.dataset = dataset
        self.fallback_dataset = fallback_dataset

    def __getitem__(self, idxs: tuple[int, ...]) -> Any:
        idx = idxs[0]
        try:
            return self.dataset[idx]
        except KeyboardInterrupt as e:
            raise e
        except StopIteration as e:
            raise e
        except Exception as e:
            example_id = f" ({self.dataset.idx_to_id(idx)})" if hasattr(self.dataset, "idx_to_id") else ""
            logger.error(f"(Primary dataset): Error ({e}) at index {idx}." + example_id)
            for i, fallback_idx in enumerate(idxs[1:]):
                example_id = (
                    f" ({self.fallback_dataset.idx_to_id(fallback_idx)})"
                    if hasattr(self.fallback_dataset, "idx_to_id")
                    else ""
                )
                logger.warning(f"(Fallback {i+1}/{len(idxs)-1}): Trying fallback index {fallback_idx}." + example_id)
                try:
                    return self.fallback_dataset[fallback_idx]
                except KeyboardInterrupt as fallback_e:
                    raise fallback_e
                except StopIteration as fallback_e:
                    raise fallback_e
                except Exception as fallback_e:
                    logger.error(f"(Fallback {i+1}/{len(idxs)-1}): Error at index {idx}: {fallback_e}." + example_id)
            raise e

    def __len__(self):
        return len(self.dataset)
