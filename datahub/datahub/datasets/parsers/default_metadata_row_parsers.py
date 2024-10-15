"""MetadataRowParser implementations for the default PDB datasets (supporting AF-3 and RF2AA workflows)."""

from pathlib import Path
from typing import Any

import pandas as pd

from datahub.datasets.parsers import MetadataRowParser


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
