import copy
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

from datahub.datasets import logger


class PandasDataset(Dataset):
    """
    A base class for datasets that are stored as Pandas DataFrames.
    The underlying dataframe can be accessed via the `data` attribute.
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

        # Load the data from the path, if provided (and load only the specified columns)
        if isinstance(data, (PathLike, str)):
            data = self._load_from_path(data, columns_to_load, **load_kwargs)
        self._data = data

        # Apply filters, if provided
        self.filters = filters
        original_num_rows = len(self._data)
        self._already_filtered = False
        if filters:
            logger.info(f"Applying filters: {filters}")
            self._apply_filters(filters)
            logger.info(
                f"Filtered dataset from {original_num_rows:,} to {len(self._data):,} rows (dropped {original_num_rows - len(self._data):,} rows)"
            )
        self._already_filtered = True

        if id_column is not None:
            assert id_column in self._data.columns, f"Column {id_column} not found in dataset."
            self._data.set_index(id_column, inplace=True, drop=False, verify_integrity=True)

    def _load_from_path(
        self, path: PathLike | str, columns_to_load: list[str] | None = None, **load_kwargs: Any
    ) -> pd.DataFrame:
        path = Path(path)
        if path.suffix == ".csv":
            data = pd.read_csv(path, usecols=columns_to_load, **load_kwargs)
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
        if filtered_num_rows == original_num_rows:
            logger.warning(f"Query '{query}' did not remove any rows.")
        elif filtered_num_rows == 0:
            raise ValueError(f"Query '{query}' removed all rows.")
        else:
            logger.info(
                f"Query '{query}' filtered dataset from {original_num_rows:,} to {filtered_num_rows:,} rows (dropped {original_num_rows - filtered_num_rows:,} rows)"
            )


class NamedConcatDataset(ConcatDataset):
    """Equivalent to `torch.utils.data.ConcatDataset` but allows accessing examples by ID."""

    datasets: list[Dataset]

    def __init__(self, datasets: list[Dataset], name: str | None = None):
        super().__init__(datasets)
        if name is not None:
            self.name = name

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


def get_row_and_index_by_example_id(dataset: NamedConcatDataset, example_id: str) -> dict:
    """
    Retrieve a row and its index from a nested dataset structure by its example ID.
    The constituent Datasets must be named to allow for recursive search.

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
    while isinstance(dataset, NamedConcatDataset):
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
