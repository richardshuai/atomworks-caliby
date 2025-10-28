"""Dataset for loading molecular structures from ASE LMDB databases."""

import bisect
import logging
from collections.abc import Callable
from glob import glob
from pathlib import Path
from typing import Any

import ase.db
import numpy as np
import pandas as pd

from atomworks.io.utils.ase_conversions import ase_to_atom_array
from atomworks.ml.datasets.datasets import ExampleIDMixin, MolecularDataset
from atomworks.ml.utils.io import read_parquet_with_metadata

logger = logging.getLogger(__name__)


class AseDBDataset(MolecularDataset, ExampleIDMixin):
    """Dataset for loading molecular structures from ASE LMDB databases.

    This dataset supports:
    - Direct loading from ASE LMDB databases (like OMol25)
    - Optional metadata parquet for efficient filtering
    - Automatic index caching for fast initialization
    - Extraction of per-atom and global properties

    The dataset converts ASE Atoms objects to Biotite AtomArray format and
    extracts DFT properties (energies, forces, charges, etc.) into the standard
    AtomWorks data format.

    Example:
        >>> # Without metadata parquet (uses cached index)
        >>> dataset = AseDBDataset(lmdb_path="/path/to/train_4M", name="omol25_train", transform=my_transform)

        >>> # With metadata parquet for filtering
        >>> dataset = AseDBDataset(
        ...     lmdb_path="/path/to/train_4M",
        ...     metadata_parquet="train_4M_metadata.parquet",
        ...     filters=["data_id == 'elytes'", "num_atoms < 100"],
        ...     name="omol25_filtered",
        ...     transform=my_transform,
        ... )
    """

    def __init__(
        self,
        *,
        lmdb_path: str,
        name: str,
        metadata_parquet: str | None = None,
        id_column: str = "lmdb_idx",
        filters: list[str] | None = None,
        columns_to_load: list[str] | None = None,
        per_atom_properties: list[str] | None = None,
        global_properties: list[str] | None = None,
        transform: Callable | None = None,
        use_cache: bool = True,
        save_failed_examples_to_dir: str | Path | None = None,
    ):
        """Initialize AseDBDataset.

        Args:
            lmdb_path: Path to the ASE LMDB database directory
            name: Descriptive name for this dataset
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
            use_cache: Whether to use/create cached index when no metadata_parquet provided
            save_failed_examples_to_dir: Optional directory to save failed examples for debugging
        """
        super().__init__(
            name=name,
            transform=transform,
            save_failed_examples_to_dir=save_failed_examples_to_dir,
        )

        self.lmdb_path = Path(lmdb_path)
        self.metadata_parquet = metadata_parquet
        self.id_column = id_column
        self.use_cache = use_cache

        # Set default properties if not provided
        self.per_atom_properties = per_atom_properties or []
        self.global_properties = global_properties or []

        # Open LMDB connection(s)
        # Handle both single file and directory with multiple .aselmdb files
        logger.info(f"Connecting to ASE LMDB database at {self.lmdb_path}")
        self.dbs, self.db_ids = self._connect_databases(self.lmdb_path)

        # Build cumulative index for efficient lookup across databases
        # This follows fairchem's approach for sharded databases
        idlens = [len(ids) for ids in self.db_ids]
        total_entries = sum(idlens)

        if total_entries == 0:
            raise ValueError(f"No entries found in LMDB database at {self.lmdb_path}")

        # Keep as numpy array for efficient bisect operations
        self._idlen_cumulative = np.cumsum(idlens)

        # Build index mapping
        if metadata_parquet is not None:
            logger.info(f"Loading metadata from {metadata_parquet}")
            self.metadata_df = self._load_metadata(metadata_parquet, filters, columns_to_load)
            self.indices = self.metadata_df[id_column].tolist()
        else:
            # Use all entries - sequential indices across all databases
            self.metadata_df = None
            self.indices = list(range(total_entries))

        # Build reverse mapping for O(1) id_to_idx lookups
        self._id_to_idx_map = {str(idx): i for i, idx in enumerate(self.indices)}

        logger.info(f"Dataset contains {len(self.indices):,} examples")

    def _connect_databases(self, lmdb_path: Path) -> tuple[list, list]:
        """Connect to ASE database file(s).

        Handles both single .aselmdb files and directories containing multiple files.
        Similar to fairchem's approach for handling sharded LMDB databases.

        Args:
            lmdb_path: Path to database file or directory

        Returns:
            Tuple of (list of database connections, list of id lists per database)
        """
        # Determine file paths to connect to
        if lmdb_path.is_file():
            filepaths = [str(lmdb_path)]
        elif lmdb_path.is_dir():
            # Find all .aselmdb files in directory
            filepaths = sorted(glob(str(lmdb_path / "*.aselmdb")))
            if not filepaths:
                raise ValueError(f"No .aselmdb files found in directory: {lmdb_path}")
        else:
            raise ValueError(f"Path does not exist: {lmdb_path}")

        logger.info(f"Found {len(filepaths)} database file(s)")

        # Connect to each database
        dbs = []
        connection_errors = []
        for path in filepaths:
            try:
                # Connect with readonly and no lock file for multi-process safety
                # Following fairchem's approach
                connect_args = {}
                if "aselmdb" in path:
                    connect_args["readonly"] = True
                    connect_args["use_lock_file"] = False

                db = ase.db.connect(path, **connect_args)
                dbs.append(db)
                logger.info(f"Connected to {Path(path).name}")
            except Exception as e:
                error_msg = f"Failed to connect to {Path(path).name}: {type(e).__name__}: {e}"
                logger.error(error_msg)
                connection_errors.append(error_msg)

        if not dbs:
            error_summary = "\n".join(connection_errors) if connection_errors else "Unknown connection error"
            raise ValueError(
                f"Could not connect to any databases at {lmdb_path}\n"
                f"Expected .aselmdb files or ASE-compatible database files.\n"
                f"Connection errors:\n{error_summary}"
            )

        # Get IDs from each database
        db_ids = []
        for i, db in enumerate(dbs):
            if hasattr(db, "ids"):
                # Fast path: database has ids attribute (LMDB backend)
                db_ids.append(db.ids)
                logger.debug(f"Database {i} has {len(db.ids)} entries")
            else:
                # Slow path: iterate to get ids (SQLite, JSON backends)
                ids = [row.id for row in db.select()]
                db_ids.append(ids)
                logger.debug(f"Database {i} has {len(ids)} entries")

        total_count = sum(len(ids) for ids in db_ids)
        logger.info(f"Total structures across all databases: {total_count:,}")

        return dbs, db_ids

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
        return len(self.indices)

    def __getitem__(self, idx: int) -> Any:
        """Load and transform an example by index.

        Args:
            idx: The dataset index (0 to len-1), supports negative indexing

        Returns:
            Transformed data dictionary

        Raises:
            IndexError: If index is out of bounds
        """
        # Get the global index for this example (Python handles negative indexing)
        try:
            global_idx = self.indices[idx]
        except IndexError as e:
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.indices)} examples") from e

        # Map global index to database and row ID using bisect_right
        # This follows fairchem's approach exactly
        # For _idlen_cumulative = [10, 20, 30]:
        #   - global_idx=9 -> db_idx=0, el_idx=9 (last item of first DB)
        #   - global_idx=10 -> db_idx=1, el_idx=0 (first item of second DB)
        #   - global_idx=20 -> db_idx=2, el_idx=0 (first item of third DB)
        db_idx = bisect.bisect_right(self._idlen_cumulative, global_idx)

        # Calculate the index within that specific database
        el_idx = global_idx if db_idx == 0 else global_idx - self._idlen_cumulative[db_idx - 1]

        if el_idx < 0 or el_idx >= len(self.db_ids[db_idx]):
            raise IndexError(
                f"Invalid element index {el_idx} for database {db_idx} "
                f"(global_idx={global_idx}, db has {len(self.db_ids[db_idx])} entries)"
            )

        # Get the actual row ID from the database
        lmdb_row_id = self.db_ids[db_idx][el_idx]

        # Fetch atoms from appropriate database
        # RATIONALE: ASE's LMDB backend doesn't expose a public method to get rows by ID.
        # _get_row() is the internal method used by ASE's own .get() method.
        # This pattern is used by fairchem and is stable across ASE versions 3.x.
        atoms_row = self.dbs[db_idx]._get_row(lmdb_row_id)
        atoms = atoms_row.toatoms()

        # Update atoms.info with any data from the row
        if isinstance(atoms_row.data, dict):
            atoms.info.update(atoms_row.data)

        # Convert to Biotite AtomArray
        # Note: ase_to_atom_array already transfers positions, elements, and standard arrays
        atom_array = ase_to_atom_array(atoms)

        # Extract additional per-atom properties as annotations
        # Only process properties not already transferred by ase_to_atom_array
        for prop in self.per_atom_properties:
            # Skip properties already handled by the conversion (numbers, positions)
            if prop in ("numbers", "positions"):
                continue
            if prop in atoms.arrays:
                # Only set if not already present to avoid overwriting conversion results
                if not hasattr(atom_array, prop):
                    atom_array.set_annotation(prop, atoms.arrays[prop])
            else:
                logger.warning(
                    f"Requested per-atom property '{prop}' not found in atoms.arrays for example {global_idx}. "
                    f"Available properties: {list(atoms.arrays.keys())}"
                )

        # Extract global properties into extra_info
        extra_info = {}
        for prop in self.global_properties:
            if prop in atoms.info:
                extra_info[prop] = atoms.info[prop]
            else:
                logger.warning(
                    f"Requested global property '{prop}' not found in atoms.info for example {global_idx}. "
                    f"Available properties: {list(atoms.info.keys())}"
                )

        # Build data dictionary
        # Use global_idx as the example_id for consistency
        example_id = str(global_idx)
        data = {
            "example_id": example_id,
            "atom_array": atom_array,
            "extra_info": extra_info,
        }

        # Apply transform
        data = self._apply_transform(data, example_id=data["example_id"], idx=idx)

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
            return global_idx in self.indices
        except (ValueError, TypeError):
            return False

    def id_to_idx(self, example_id: str | list[str]) -> int | list[int]:
        """Convert example ID(s) to index(es).

        Uses O(1) dictionary lookup for efficient conversion.

        Args:
            example_id: Single ID or list of IDs (global LMDB indices as strings)

        Returns:
            Corresponding dataset index or list of indices

        Raises:
            ValueError: If example_id is not found in the dataset
        """
        if isinstance(example_id, list):
            result = []
            for eid in example_id:
                idx = self._id_to_idx_map.get(str(eid))
                if idx is None:
                    # Provide helpful error message
                    if not self.indices:
                        raise ValueError(f"Example ID '{eid}' not found - dataset is empty")
                    raise ValueError(
                        f"Example ID '{eid}' not found in dataset. "
                        f"Dataset contains {len(self.indices)} examples with IDs from "
                        f"{self.indices[0]} to {self.indices[-1]}."
                    )
                result.append(idx)
            return result

        # Single ID lookup
        idx = self._id_to_idx_map.get(str(example_id))
        if idx is None:
            # Provide helpful error message
            if not self.indices:
                raise ValueError(f"Example ID '{example_id}' not found - dataset is empty")
            raise ValueError(
                f"Example ID '{example_id}' not found in dataset. "
                f"Dataset contains {len(self.indices)} examples with IDs from "
                f"{self.indices[0]} to {self.indices[-1]}."
            )
        return idx

    def idx_to_id(self, idx: int | list[int]) -> str | list[str]:
        """Convert index(es) to example ID(s).

        Args:
            idx: Single index or list of indices (0 to len-1)

        Returns:
            Corresponding example ID(s) as string(s) (global LMDB indices)
        """
        if isinstance(idx, list):
            return [str(self.indices[i]) for i in idx]
        return str(self.indices[idx])

    def close(self) -> None:
        """Explicitly close database connections.

        This method should be called when you're done using the dataset to ensure
        database connections are properly closed. Also called automatically on deletion.
        """
        if hasattr(self, "dbs"):
            for i, db in enumerate(self.dbs):
                if hasattr(db, "close"):
                    try:
                        db.close()
                    except Exception as e:
                        logger.warning(f"Failed to close database {i}: {e}")

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
