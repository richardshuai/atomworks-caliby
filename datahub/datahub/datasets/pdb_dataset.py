import os
import socket
import time
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd
from cifutils import CIFParser

from datahub.common import exists
from datahub.datasets import logger
from datahub.datasets.base import PandasDataset
from datahub.datasets.dataframe_parsers import RowParser, load_from_row
from datahub.transforms.base import Compose, Transform, TransformedDict
from datahub.utils.debug import save_failed_example_to_disk
from datahub.utils.rng import capture_rng_states


class PDBDataset(PandasDataset):
    def __init__(
        self,
        name: str,
        dataset_path: PathLike | str | pd.DataFrame,
        dataset_parser: RowParser,
        filters: list[str] | None = None,
        cif_parser: CIFParser | None = None,
        columns_to_load: list[str] | None = None,
        transform: Transform | Compose | None = None,
        id_column: str | None = "example_id",
        return_key: str | None = None,
        save_failed_examples_to_dir: PathLike | str | None = "/net/scratch/failures",
        cif_cache_dir: PathLike | str | None = "/projects/ml/RF2_allatom/cache/cif",
    ):
        """
        Initialize the PDBDataset with paths and optional filters.

        Args:
            name (str): Name of the dataset. Must be unique across all datasets at the same level in the nested dataset hierarchy.
            dataset_path (PathLike | str | pd.DataFrame): Path to DataFrame file or the DataFrame itself.
            protein_msa_dir (PathLike | str): Directory for protein MSA files.
            rna_msa_dir (PathLike | str): Directory for RNA MSA files.
            dataset_parser (RowParser): RowParser object to use for parsing DataFrame rows.
            filters (list[str], optional): List of filter conditions to apply to the data.
            cif_parser (CIFParser, optional): CIFParser object to use for parsing CIF files. If None, a new CIFParser will be created.
            columns_to_load (list[str], optional): List of columns to load from the DataFrame. If None, all columns will be loaded.
                Specify only the columns needed to minimize data transfer to workers.
            transform (Transform | Compose, optional): Transformation pipeline to apply to the data.
            id_column (str, optional): Name of the column containing the example IDs. Defaults to "example_id".
            return_key (str, optional): If provided, returns data[return_key] instead of the entire data dict.
            save_failed_examples_to_dir (PathLike | str | None, optional): Directory to save failed examples to.
                Defaults to "/net/scratch/failures".
            cif_cache_dir (PathLike | str | None, optional): Directory to retrieve cached, processed CIF files from if they are already
                cached, or to cache them in if they are not. If None, CIF files will not be cached.
                Defaults to "/projects/ml/RF2_allatom/cache/cif".
        """
        # Initialize the cif parser, if not provided
        self.cif_parser = cif_parser if cif_parser else CIFParser()

        # Assign parser
        self.dataset_parser = dataset_parser

        # Initialize superclass attributes
        super().__init__(
            data=dataset_path, id_column=id_column, filters=filters, name=name, columns_to_load=columns_to_load
        )

        self.transform = transform
        self.return_key = return_key
        self.cif_cache_dir = Path(cif_cache_dir) if exists(cif_cache_dir) else None
        self.save_failed_examples_to_dir = (
            Path(save_failed_examples_to_dir) if exists(save_failed_examples_to_dir) else None
        )

    def __getitem__(self, idx: int) -> Any:
        # Capture example ID & current rng state (for reproducibility & debugging)
        example_id = self.idx_to_id(idx)
        # Get process id and hostname
        logger.debug(f"({socket.gethostname()}:{os.getpid()}) Processing example ID: {example_id}")

        # Load the row
        row = self._data.iloc[idx]

        # Process the row with the given CIF and dataset parsers
        _start_parse_time = time.time()
        data = load_from_row(row, self.dataset_parser, cif_parser=self.cif_parser, cache_dir=self.cif_cache_dir)
        _stop_parse_time = time.time()
        # ... manually add timing for cif-parsing
        data = TransformedDict(data)
        data.__transform_history__.append(
            dict(
                name="load_from_row",
                instance=hex(id(load_from_row)),
                start_time=_start_parse_time,
                end_time=_stop_parse_time,
                processing_time=_stop_parse_time - _start_parse_time,
            )
        )

        # Featurize the data with the defined pipeline
        if exists(self.transform):
            try:
                rng_state_dict = capture_rng_states(include_cuda=False)
                data = self.transform(data)
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
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

        # Return the specified key or the entire data dict
        if exists(self.return_key):
            return data[self.return_key]
        else:
            return data
