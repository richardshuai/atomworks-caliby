import os
import socket
import time
from os import PathLike
from pathlib import Path
from typing import Any

from cifutils import CIFParser
from torch.utils.data import Dataset

from datahub.common import exists
from datahub.datasets import logger
from datahub.datasets.base import BaseDataset
from datahub.datasets.dataframe_parsers import RowParser, load_from_row
from datahub.transforms.base import Compose, Transform, TransformedDict
from datahub.utils.debug import save_failed_example_to_disk
from datahub.utils.rng import capture_rng_states


class WrappedStructuralDataset(BaseDataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_parser: RowParser,
        cif_parser: CIFParser | None = None,
        cif_parser_args: dict | None = None,
        transform: Transform | Compose | None = None,
        return_key: str | None = None,
        save_failed_examples_to_dir: PathLike | str | None = "/net/scratch/failures",
    ):
        """
        Wrapper around an arbitrary Dataset (e.g., PandasDataset, PolarsDataset, etc.) to handle loading of structural data from PDB or CIF files,
        parsing, and applying a Transformation pipeline to the data.

        Designed to be used with a Transforms pipeline to process the data and a RowParser to convert the dataset rows into a common dictionary format.

        For more detail, see the README in the `datasets` directory.

        Args:
            dataset (Dataset): The dataset to wrap. For example, a PandasDataset, PolarsDataset, or standard PyTorch Dataset.
            dataset_parser (RowParser): Parser to convert dataset rows into a common dictionary format. See `datahub.datasets.dataframe_parsers`.
            cif_parser (CIFParser, optional): Parser for CIF files. If None, a new CIFParser will be created.
            cif_parser_args (dict, optional): Arguments to pass to the CIFParser (will override the defaults). Defaults to None.
            transform (Transform | Compose, optional): Transformation pipeline to apply to the data. See `datahub.transforms.base`.
            return_key (str, optional): Key to return from the data dictionary instead of the entire dictionary.
            save_failed_examples_to_dir (PathLike | str | None, optional): Directory to save failed examples. Defaults to "/net/scratch/failures".
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

        # ...carry forward the data, noting that it shouldn't be modified with "_"
        self.data = self.dataset.data

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
        # We require the "data" dictionary output from `load_from_row` to contain, at a minimum:
        #   (a) An "id" key, which uniquely identifies the example within the dataframe; and,
        #   (b) The "path" key, which is the path to the CIF file
        _start_parse_time = time.time()
        data = load_from_row(row, self.dataset_parser, cif_parser=self.cif_parser, cif_parser_args=self.cif_parser_args)
        _stop_parse_time = time.time()

        # Manually add timing for cif-parsing
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
