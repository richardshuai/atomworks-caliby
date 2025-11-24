"""Dataset for loading molecular structures from ASE LMDB databases."""

import logging
import warnings
from collections.abc import Callable
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atomworks.ml.datasets.base import ExampleIDMixin, MolecularDataset
from atomworks.ml.utils.io import read_parquet_with_metadata

try:
    import ase.db
except ImportError:
    warnings.warn(
        "ASE library is required for AseDBDataset. Please install ASE to use this dataset.",
        ImportWarning,
        stacklevel=2,
    )
    ase = None

logger = logging.getLogger(__name__)


class AseDBDataset(MolecularDataset, ExampleIDMixin):
    """Dataset for loading molecular structures from ASE LMDB databases.

    Uses memory-efficient lazy shard loading - only one shard is kept
    open at a time (~50MB), instead of all shards (~50MB × num_shards).
    """

    def __init__(
        self,
        *,
        lmdb_path: str,
        name: str,
        loader: Callable | None = None,
        metadata_parquet: str | None = None,
        id_column: str = "lmdb_idx",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        per_atom_properties: list[str] | None = None,
        global_properties: list[str] | None = None,
        transform: Callable | None = None,
        save_failed_examples_to_dir: str | Path | None = None,
    ):
        """Initialize AseDBDataset.

        Args:
            lmdb_path: Path to the ASE LMDB database directory
            name: Descriptive name for this dataset
            loader: Optional callable to process atoms_row into data dictionary.
                Should accept (atoms_row, example_id, global_idx) and return dict with atom_array.
            metadata_parquet: Optional path to parquet file with metadata for filtering.
                Must contain 'lmdb_idx' column mapping to LMDB row IDs.
            id_column: Column name in metadata_parquet containing LMDB indices (default: "lmdb_idx")
            filters: Optional list of pandas query strings to filter the metadata
            columns_to_load: Optional list of columns to load from metadata parquet
            per_atom_properties: List of per-atom properties to extract from atoms.arrays
                as AtomArray annotations (e.g., forces, charges)
            global_properties: List of global properties to extract from atoms.info
                into extra_info dict (e.g., energy, charge, spin)
            transform: Transform pipeline to apply to loaded data
            save_failed_examples_to_dir: Optional directory to save failed examples for debugging
        """
        super().__init__(
            name=name,
            loader=loader,
            transform=transform,
            save_failed_examples_to_dir=save_failed_examples_to_dir,
        )

        self.lmdb_path = Path(lmdb_path)
        self.metadata_parquet = metadata_parquet
        self.id_column = id_column

        # Set default properties if not provided
        self.per_atom_properties = per_atom_properties or []
        self.global_properties = global_properties or []

        # Discover shard files without connecting (lazy loading)
        self._filepaths = self._discover_shards(self.lmdb_path)
        self._num_shards = len(self._filepaths)
        logger.info(f"Found {self._num_shards} shard(s) at {self.lmdb_path}")

        # Auto-detect entries per shard and total (connects to first+last shards briefly)
        self._entries_per_shard, self._total_entries = self._detect_shard_info()
        logger.info(f"Entries per shard: {self._entries_per_shard:,}, total: {self._total_entries:,}")

        # Lazy shard state - only one shard open at a time
        self._current_shard_idx: int | None = None
        self._current_db = None
        self._current_ids: list | None = None

        # Index mapping: None for unfiltered (identity), numpy array for filtered
        if metadata_parquet is not None:
            logger.info(f"Loading metadata from {metadata_parquet}")
            self.metadata_df = self._load_metadata(metadata_parquet, filters, columns_to_load)
            self.indices: np.ndarray | None = np.array(self.metadata_df[id_column], dtype=np.int64)
            logger.info(f"Filtered dataset contains {len(self.indices):,} examples")
        else:
            self.metadata_df = None
            self.indices = None  # Use identity mapping: indices[i] == i
            logger.info(f"Dataset contains {self._total_entries:,} examples")

    def _discover_shards(self, lmdb_path: Path) -> list[str]:
        """Discover shard files without connecting to them.

        Args:
            lmdb_path: Path to database file or directory

        Returns:
            List of file paths to shard files
        """
        if lmdb_path.is_file():
            return [str(lmdb_path)]
        elif lmdb_path.is_dir():
            filepaths = sorted(glob(str(lmdb_path / "*.aselmdb")))
            if not filepaths:
                raise ValueError(f"No .aselmdb files found in directory: {lmdb_path}")
            return filepaths
        else:
            raise ValueError(f"Path does not exist: {lmdb_path}")

    def _detect_shard_info(self) -> tuple[int, int]:
        """Detect entries per shard and total entries.

        Connects to first and last shards to determine accurate counts.

        Returns:
            Tuple of (entries_per_shard, total_entries)
        """

        def get_shard_count(path: str) -> int:
            connect_args = {"readonly": True, "use_lock_file": False} if "aselmdb" in path else {}
            db = ase.db.connect(path, **connect_args)
            try:
                return len(db.ids) if hasattr(db, "ids") else db.count()
            finally:
                if hasattr(db, "close"):
                    db.close()

        first_count = get_shard_count(self._filepaths[0])

        if self._num_shards == 1:
            return first_count, first_count

        # Check last shard to handle variable shard sizes
        last_count = get_shard_count(self._filepaths[-1])

        # Calculate total: assume all shards except last have first_count entries
        total = (self._num_shards - 1) * first_count + last_count
        return first_count, total

    def _ensure_shard_loaded(self, shard_idx: int) -> None:
        """Load a shard if not already loaded, closing previous shard.

        Args:
            shard_idx: Index of shard to load (0-based)
        """
        if self._current_shard_idx == shard_idx:
            return

        # Close previous shard
        if self._current_db is not None:
            if hasattr(self._current_db, "close"):
                try:
                    self._current_db.close()
                except Exception as e:
                    logger.warning(f"Failed to close shard {self._current_shard_idx}: {e}")
            self._current_db = None
            self._current_ids = None

        # Open new shard
        path = self._filepaths[shard_idx]
        connect_args = {"readonly": True, "use_lock_file": False} if "aselmdb" in path else {}
        self._current_db = ase.db.connect(path, **connect_args)
        self._current_ids = self._current_db.ids if hasattr(self._current_db, "ids") else None
        self._current_shard_idx = shard_idx
        logger.debug(f"Loaded shard {shard_idx}: {Path(path).name}")

    def _load_metadata(
        self, parquet_path: str, filters: list[str] | None, columns_to_load: list[str] | None
    ) -> pd.DataFrame:
        """Load and filter metadata parquet file.

        Args:
            parquet_path: Path to parquet file
            filters: Optional list of pandas query strings
            columns_to_load: Optional list of columns to load

        Returns:
            Filtered DataFrame with metadata
        """
        df = read_parquet_with_metadata(parquet_path, columns=columns_to_load)

        # Ensure id_column exists
        if self.id_column not in df.columns:
            raise ValueError(
                f"Metadata parquet must contain '{self.id_column}' column. " f"Found columns: {list(df.columns)}"
            )

        # Apply filters
        if filters:
            for query in filters:
                original_len = len(df)
                df = df.query(query)
                filtered_len = len(df)
                logger.info(
                    f"Filter '{query}': {original_len:,} → {filtered_len:,} "
                    f"({filtered_len/original_len*100:.1f}% remaining)"
                )

                if len(df) == 0:
                    raise ValueError(f"Filter '{query}' removed all examples")

        return df

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        if self.indices is not None:
            return len(self.indices)
        return self._total_entries

    def __getitem__(self, idx: int) -> Any:
        """Load and transform an example by index.

        Args:
            idx: The dataset index (0 to len-1), supports negative indexing

        Returns:
            Transformed data dictionary

        Raises:
            IndexError: If index is out of bounds
        """
        # Handle negative indexing
        dataset_len = len(self)
        if idx < 0:
            idx = dataset_len + idx
        if idx < 0 or idx >= dataset_len:
            raise IndexError(f"Index {idx} out of range for dataset with {dataset_len} examples")

        # Get the global LMDB index (identity if no filter, otherwise lookup)
        global_idx = idx if self.indices is None else int(self.indices[idx])

        # Calculate which shard this index belongs to
        shard_idx = global_idx // self._entries_per_shard
        local_idx = global_idx % self._entries_per_shard

        if shard_idx >= self._num_shards:
            raise IndexError(f"Global index {global_idx} exceeds dataset bounds ({self._total_entries})")

        # Ensure correct shard is loaded (lazy loading)
        self._ensure_shard_loaded(shard_idx)

        # Get the actual row ID from the currently loaded shard
        # Note: Some shards may have fewer entries than _entries_per_shard
        if self._current_ids is not None:
            actual_shard_len = len(self._current_ids)
            if local_idx >= actual_shard_len:
                # This shard has fewer entries - index is out of bounds for the dataset
                raise IndexError(
                    f"Index {idx} (global {global_idx}) exceeds dataset bounds. "
                    f"Shard {shard_idx} has {actual_shard_len} entries, not {self._entries_per_shard}."
                )
            lmdb_row_id = self._current_ids[local_idx]
        else:
            lmdb_row_id = local_idx + 1  # ASE uses 1-based IDs

        # Fetch atoms from currently loaded shard
        # RATIONALE: ASE's LMDB backend doesn't expose a public method to get rows by ID.
        # _get_row() is the internal method used by ASE's own .get() method.
        atoms_row = self._current_db._get_row(lmdb_row_id)

        # Use example_id for tracking (use source attribute as example_id)
        example_id = atoms_row.data["source"]

        # Apply loader to convert atoms_row to data dictionary
        data = self._apply_loader((atoms_row, example_id, global_idx))

        # Apply transform pipeline
        data = self._apply_transform(data, example_id=example_id, idx=idx)

        return data

    # ExampleIDMixin methods
    def __contains__(self, example_id: str) -> bool:
        """Check if the dataset contains the example ID.

        Args:
            example_id: Global LMDB index as string

        Returns:
            True if the example ID is in this dataset
        """
        try:
            global_idx = int(example_id)
        except (ValueError, TypeError):
            return False

        if self.indices is None:
            # Unfiltered: any valid index is in the dataset
            return 0 <= global_idx < self._total_entries

        # Filtered: check if index is in the filtered set
        pos = np.searchsorted(self.indices, global_idx)
        return pos < len(self.indices) and self.indices[pos] == global_idx

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to dataset index(es).

        Args:
            example_id: Single ID or list of IDs (global LMDB indices as strings)

        Returns:
            Corresponding dataset index or list of indices

        Raises:
            ValueError: If example_id is not found in the dataset
        """
        if isinstance(example_id, list):
            return [self._single_id_to_idx(eid) for eid in example_id]
        return self._single_id_to_idx(example_id)

    def _single_id_to_idx(self, example_id: str) -> int:
        """Convert a single example ID to dataset index."""
        try:
            global_idx = int(example_id)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Example ID '{example_id}' is not a valid integer") from e

        if self.indices is None:
            # Unfiltered: identity mapping
            if 0 <= global_idx < self._total_entries:
                return global_idx
            raise ValueError(f"Example ID '{example_id}' out of range [0, {self._total_entries})")

        # Filtered: binary search
        pos = int(np.searchsorted(self.indices, global_idx))
        if pos < len(self.indices) and self.indices[pos] == global_idx:
            return pos
        raise ValueError(f"Example ID '{example_id}' not found in filtered dataset " f"({len(self.indices)} examples)")

    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert dataset index(es) to example ID(s).

        Args:
            idx: Single index or list of indices (0 to len-1)

        Returns:
            Corresponding example ID(s) as string(s) (global LMDB indices)
        """
        if isinstance(idx, list):
            return [self._single_idx_to_id(i) for i in idx]
        return self._single_idx_to_id(idx)

    def _single_idx_to_id(self, idx: int) -> str:
        """Convert a single dataset index to example ID."""
        if self.indices is None:
            # Unfiltered: identity mapping
            return str(idx)
        return str(self.indices[idx])

    def close(self) -> None:
        """Explicitly close database connection.

        This method should be called when you're done using the dataset to ensure
        the database connection is properly closed. Also called automatically on deletion.
        """
        if self._current_db is not None:
            if hasattr(self._current_db, "close"):
                try:
                    self._current_db.close()
                except Exception as e:
                    logger.warning(f"Failed to close shard {self._current_shard_idx}: {e}")
            self._current_db = None
            self._current_ids = None
            self._current_shard_idx = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> bool:
        """Context manager exit with automatic cleanup."""
        self.close()
        return False

    def __del__(self):
        """Close database connections on deletion."""
        self.close()
