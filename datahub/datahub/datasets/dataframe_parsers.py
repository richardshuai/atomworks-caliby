from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

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
    "load_from_cache": False,
    "save_to_cache": False,
}


class RowParser(ABC):
    """
    Abstract base class for RowParsers.

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
    def _parse(self, row: pd.Series) -> Dict[str, Any]:
        """
        Abstract method to be implemented by subclasses for parsing a DataFrame row.

        Args:
            row (pd.Series): DataFrame row to parse.

        Returns:
            Dict[str, Any]: Parsed output dictionary, including required keys.
        """
        pass

    def validate_output(self, output: Dict[str, Any]) -> None:
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


class PNUnitsDFParser(RowParser):
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

    def _parse(self, row: pd.Series) -> Dict[str, Any]:
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


class InterfacesDFParser(RowParser):
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

    def _parse(self, row: pd.Series) -> Dict[str, Any]:
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


def load_from_row(
    row: pd.Series,
    row_parser: RowParser,
    *,
    cif_parser: CIFParser | None = None,
    cif_parser_args: dict | None = None,
) -> dict:
    """
    Load data from a DataFrame row into a common format using a given row parsing function and CIFParser.

    Performs the following steps:
        (1) Parse the row into a common dictionary format using the provided row parsing function.
        (2) Load the CIF file from the information in the common dictionary format (i.e., the "path" key).
        (3) Combine the parsed row data and the loaded CIF data into a single dictionary.

    Args:
        row (pd.Series): The DataFrame row to parse.
        row_parser (RowParser): The parser to use for converting the row into a dictionary format.
        cif_parser (CIFParser, optional): The parser to use for loading CIF data. Defaults to None.
        cif_parser_args (dict, optional): Additional arguments for the CIF parser. Defaults to None.

    Returns:
        dict: A dictionary containing the parsed row data and additional loaded CIF data.
    """
    # Load common outputs from the dataframe using the provided parsing function
    # See the `RowParser` class for more details on the expected output schema
    parsed_row = row_parser.parse(row)

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
