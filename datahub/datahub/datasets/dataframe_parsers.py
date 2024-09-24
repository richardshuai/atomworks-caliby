from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from cifutils import CIFParser

from datahub.datasets import logger

"""Default arguments for the CIFParser."""
DEFAULT_CIF_PARSER_ARGS = {
    "add_bonds": True,
    "add_missing_atoms": True,
    "remove_waters": True,
    "patch_symmetry_centers": True,
    "convert_mse_to_met": True,
    "fix_arginines": True,
    "keep_hydrogens": False,
    "model": None,
    "cache_dir": None,
}


class MetadataRowParser(ABC):
    """
    Abstract base class for MetadataRowParsers.

    A MetadataRowParser is a class that parses a row from a DataFrame on disk into a format digestible by the `load_example_from_metadata_row` function.

    In the common case that a model is trained on multiple datasets, each with their own dataframe and base data format,
    we must ensure that the data pipeline receives a consistent input format. By way of example, when training an
    AF-3-style model, we might have a "PDB Chains" dataset of `mmCIF` files, a "PDB Interfaces" dataset of `mmCIF`
    files, and a `distillation` dataset of computationally-generated `PDB` files, and many others.

    We enforce the following common schema for all datasets:
        - "example_id": A unique identifier for the example within the dataset.
        - "path": The path to the data file (which we will load with a CIFParser).

    WARNING: For many transforms, additional keys are required. For example:
        - For cropping, the `query_pn_unit_iids` field is used to center the crop on the interface or pn_unit.
          If not provided, the AF-3-style crop transforms will crop randomly.
        - For loading templates, the "pdb_id" is required to load the correct template from disk (at least with the legacy code).
    """

    required_schema = {
        "example_id": str,
        "path": Path,
    }

    def parse(self, row: pd.Series) -> dict[str, Any]:
        """Wrapper to parse and validate a DataFrame row."""
        output = self._parse(row)

        # Validate output
        self.validate_output(output)
        return output

    @abstractmethod
    def _parse(self, row: pd.Series) -> dict[str, Any]:
        """
        Abstract method to be implemented by subclasses for parsing a DataFrame row.

        Args:
            row (pd.Series): DataFrame row to parse.

        Returns:
            dict[str, Any]: Parsed output dictionary, including required keys.
        """
        pass

    def validate_output(self, output: dict[str, Any]) -> None:
        """Validate the output dictionary for required keys and their types."""
        for key in self.required_schema.keys():
            if key not in output:
                if key == "extra_info":
                    output[key] = {}  # Default to an empty dictionary
                else:
                    raise ValueError(f"Missing key in output: {key}")

        for key, expected_type in self.required_schema.items():
            if not isinstance(output[key], expected_type):
                raise TypeError(f"Key '{key}' has incorrect type: expected {expected_type}, got {type(output[key])}")


class PNUnitsDFParser(MetadataRowParser):
    """
    Parser for pn_units DataFrame rows.

    In addition to standard fields (example_id, path), this parser also includes:
        - The query pn_unit instance ID, which is used to center the crop.
        - The assembly ID, which is used to load the correct assembly from the CIF file.
        - Any extra information from the DataFrame, which is stored in the `extra_info` field.
    """

    def __init__(self, base_dir: Path = Path("/databases/rcsb/cif"), file_extension: str = ".cif.gz"):
        self.base_dir = base_dir
        self.file_extension = file_extension

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        # For the Query DF, the query pn_unit is the only pn_unit in the query
        query_pn_unit_iids = [row["q_pn_unit_iid"]]

        # Build the path to the CIF file
        pdb_id = row["pdb_id"]
        path = Path(f"{self.base_dir}/{pdb_id[1:3]}/{pdb_id}{self.file_extension}")

        # Put the full row in the extra info dictionary
        extra_info = row.to_dict()

        return {
            "example_id": row["example_id"],
            "path": path,
            "pdb_id": pdb_id,
            "assembly_id": row["assembly_id"],
            "query_pn_unit_iids": query_pn_unit_iids,  # Where to center the crop; if more than one, center the crop on the interface
            "extra_info": extra_info,
        }


class InterfacesDFParser(MetadataRowParser):
    """
    Parser for interfaces DataFrame rows.

    In addition to standard fields (example_id, path), this parser also includes:
        - The two query pn_unit instance IDs, as a list, which are used to sample the interface during cropping.
        - The assembly ID, which is used to load the correct assembly from the CIF file.
        - Any extra information from the DataFrame, which is stored in the `extra_info` field.
    """

    def __init__(self, base_dir: Path = Path("/databases/rcsb/cif"), file_extension: str = ".cif.gz"):
        self.base_dir = base_dir
        self.file_extension = file_extension

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        # For the Interfaces DF, there are two query pn_units
        query_pn_unit_iids = [row["pn_unit_1_iid"], row["pn_unit_2_iid"]]

        # Build the path to the CIF file
        pdb_id = row["pdb_id"]
        path = Path(f"{self.base_dir}/{pdb_id[1:3]}/{pdb_id}{self.file_extension}")

        # Put the full row in the extra info dictionary
        extra_info = row.to_dict()

        return {
            "example_id": row["example_id"],
            "path": path,
            "pdb_id": pdb_id,
            "assembly_id": row["assembly_id"],
            "query_pn_unit_iids": query_pn_unit_iids,  # Where to center the crop; if more than one, center the crop on the interface
            "extra_info": extra_info,
        }


class ValidationDFParser(MetadataRowParser):
    """
    Parser for AF-3-style validation DataFrame rows.

    As output, we give:
        - pdb_id: The PDB ID of the structure.
        - assembly_id: The assembly ID of the structure, required to load the correct assembly from the CIF file.
        - path: The path to the CIF file.
        - example_id: An identifier created on-the-fly that combines the pdb_id and assembly_id.
        - ground_truth: A dictionary containing non-feature information for loss and validation. For validation, we initialize with the following:
            - interfaces_to_score: A list of tuples like (pn_unit_iid_1, pn_unit_iid_2, interface_type), which represent low-homology interfaces to score.
            - pn_units_to_score: A list of tuples like (pn_unit_iid, pn_unit_type), which represent low-homology pn_units to score.
    """

    def __init__(self, base_dir: Path = Path("/databases/rcsb/cif"), file_extension: str = ".cif.gz"):
        self.base_dir = base_dir
        self.file_extension = file_extension

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        # Build the path to the CIF file
        pdb_id = row["pdb_id"]
        path = Path(f"{self.base_dir}/{pdb_id[1:3]}/{pdb_id}{self.file_extension}")

        # Extract the interfaces and pn_units to score

        # Example: [(A_1, B_1, "protein-protein"), (B_1, C_1, "protein-ligand")]
        interfaces_to_score = eval(row["interfaces_to_score"])
        # Example: [(A_1, "protein"), (B_1, "DNA")]
        pn_units_to_score = eval(row["pn_units_to_score"])

        # Create an example_id from the pdb_id and assembly_id
        example_id = f"{pdb_id}_{row['assembly_id']}"

        return {
            "example_id": example_id,
            "path": path,
            "pdb_id": pdb_id,
            "assembly_id": row["assembly_id"],
            "ground_truth": {
                "interfaces_to_score": interfaces_to_score,
                "pn_units_to_score": pn_units_to_score,
            },
        }


class AF2FB_DistillationParser(MetadataRowParser):
    """
    Parser for AF2FB distillation metadata.

    The AF2FB distillation dataset is provided courtesy of Meta/Facebook.
    It contains ~7.6 Mio AF2 predicted structures from UniRef50.

    Metadata (i.e. which sequences, which cluster identities @ 30% seq.id,
    whether a sequence has an msa & template, sequence_hash etc.) are stored
    in the `af2_distillation_facebook.parquet` dataframe.

    The parquet has the following columns:
        - example_id
        - n_atoms
        - n_res
        - mean_plddt
        - min_plddt
        - median_plddt
        - sequence_hash
        - has_msa
        - msa_depth
        - has_template
        - cluster_id
        - seq (!WARNING: this is a relatively data-heavy column)
    """

    def __init__(self, base_dir: str = "/squash/af2_distillation_facebook", file_extension: str = ".cif"):
        """
        Initialize the AF2FB_DistillationParser.

        This parser is designed to handle the AF2FB distillation dataset, which contains
        approximately 7.6 million AlphaFold2 predicted structures from UniRef50.

        Args:
            - base_dir (str): The base directory where the AF2FB distillation dataset is stored.
                Defaults to "/squash/af2_distillation_facebook", which is stored on `tukwila` for
                ML model training.
            - file_extension (str): The file extension of the structure files. Defaults to ".cif".

        Raises:
            - AssertionError: If the specified dataset directory does not exist.
        """
        self.dataset_dir = Path(base_dir)
        self.file_extension = file_extension
        assert self.dataset_dir.exists(), f"Dataset directory {self.dataset_dir} does not exist."

    @staticmethod
    def _get_shard_from_hash(hash_value: str) -> str:
        """Due to the size of the AF2FB dataset, we store it with 2-level sharding.

        The two layers of sharding is an optimization technique for faster filesystem
        performance. (Do not put more than 10k files in any directory).

        Example:
            - example_id: UniRef50_A0A1S3ZVX8
            - sequence_hash: f771c39dfbf

        therefore the two level shard is `f7/71/` and the files can be found at
            -  ./cif/f7/71/UniRef50_A0A1S3ZVX8.cif
            -  ./msa/f7/71/f771c39dfbf.a3m
            -  ./template/f7/71/f771c39dfbf.atab
        """
        return f"{hash_value[:2]}/{hash_value[2:4]}/"

    def _parse(self, row: pd.Series) -> dict:
        example_id = row["example_id"]
        sequence_hash = row["sequence_hash"]

        path = (
            self.dataset_dir / "cif" / self._get_shard_from_hash(sequence_hash) / f"{example_id}{self.file_extension}"
        )

        return {
            "example_id": example_id,
            "path": path,
            "assembly_id": "1",  # just default to the first assembly (=identity if none given)
            "sequence_hash": sequence_hash,
        }


def load_example_from_metadata_row(
    metadata_row: pd.Series,
    metadata_row_parser: MetadataRowParser,
    *,
    cif_parser: CIFParser | None = None,
    cif_parser_args: dict | None = None,
) -> dict:
    """
    Load training/validation example from a DataFrame row into a common format using the given metadata row parsing function and CIFParser.

    Performs the following steps:
        (1) Parse the row into a common dictionary format using the provided row parsing function and metadata row.
        (2) Load the CIF file from the information in the common dictionary format (i.e., the "path" key).
        (3) Combine the parsed row data and the loaded CIF data into a single dictionary.

    Args:
        metadata_row (pd.Series): The DataFrame row to parse.
        metadata_row_parser (MetadataRowParser): The parser to use for converting the row into a dictionary format.
        cif_parser (CIFParser, optional): The parser to use for loading CIF data. Defaults to None.
        cif_parser_args (dict, optional): Additional arguments for the CIF parser. Defaults to None.

    Returns:
        dict: A dictionary containing the parsed row data and additional loaded CIF data.
    """
    # Load common outputs from the dataframe using the provided parsing function
    # See the `MetadataRowParser` class for more details on the expected output schema
    parsed_row = metadata_row_parser.parse(metadata_row)

    if cif_parser is None:
        logger.warning("No parser provided. Initializing default parser; this may take a few seconds.")
        cif_parser = CIFParser()

    # Default cif_parser_args to an empty dictionary if not provided
    if cif_parser_args is None:
        cif_parser_args = {}

    # Convenience utilities to default to loading from and saving to cache if a cache_dir is provided, unless explicitly overridden
    if "cache_dir" in cif_parser_args and cif_parser_args["cache_dir"]:
        cif_parser_args.setdefault("load_from_cache", True)
        cif_parser_args.setdefault("save_to_cache", True)

    # Merge DEFAULT_CIF_PARSER_ARGS with cif_parser_args, overriding with the keys present in cif_parser_args
    merged_cif_parser_args = {**DEFAULT_CIF_PARSER_ARGS, **cif_parser_args}

    # Use the parse function with the provided CIFParser and CIFParser arguments
    result_dict = cif_parser.parse(
        filename=parsed_row["path"],
        build_assembly=(parsed_row["assembly_id"],),  # Convert list to tuple (make hashable)
        **merged_cif_parser_args,
    )

    # Combine the PDB output and the parsed output into our clean representation
    data = {
        # ...from the data frame (at a minimum, example_id and path)
        **parsed_row,
        # ...from the CIFParser
        "atom_array": result_dict["assemblies"][parsed_row["assembly_id"]][0],  # First model
        "atom_array_stack": result_dict["assemblies"][parsed_row["assembly_id"]],  # All models
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }

    return data
