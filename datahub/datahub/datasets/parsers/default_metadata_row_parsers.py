"""MetadataRowParser implementations for the default PDB datasets (supporting AF-3 and RF2AA workflows)."""

from pathlib import Path
from typing import Any

import pandas as pd

from datahub.common import exists
from datahub.datasets.parsers import MetadataRowParser


class PNUnitsDFParser(MetadataRowParser):
    """
    Parser for pn_units DataFrame rows.

    In addition to standard fields (example_id, path), this parser also includes:
        - The query pn_unit instance ID, which is used to center the crop.
        - The assembly ID, which is used to load the correct assembly from the CIF file.
        - Any extra information from the DataFrame, which is stored in the `extra_info` field.
    """

    def __init__(
        self, base_dir: Path = Path("/projects/ml/frozen_pdb_copies/2024_12_01_pdb"), file_extension: str = ".cif.gz"
    ):
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

    def __init__(
        self, base_dir: Path = Path("/projects/ml/frozen_pdb_copies/2024_12_01_pdb"), file_extension: str = ".cif.gz"
    ):
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


class GenericDFParser(MetadataRowParser):
    """
    Generic parser for interfaces and/or pn_units. Any extra columns will be included in "extra_info".
    Note: By convention, `example_id` values are generated with `datahub.common.generate_example_id`.
    Also note: It is important to avoid duplication of interfaces due to order inversion. If not using the preprocessing
        scripts in datahub, ensure that your interfaces dataframe has been checked for this.

    Parameters:
        example_id_colname (str): Name of the column containing a unique identifier for each example across all datasets
        path_colname (str): Name of the column containing paths to the structure files.
        pn_unit_iid_colnames (str | List[str]): The name(s) of the column(s) containing the cifutils pn_unit_iid(s).
            If given as a list, this must contain one element for a monomers dataset and two for an interfaces dataset.
        assembly_id_colname (str | None): Optional parameter giving the name of the column containing the assembly ID.
            If None, the assembly ID will be set to "1" for all examples.

    Example dataframe:
        example_id                      path                      pn_unit_1_iid  pn_unit_2_iid
        {['my-dataset']}{1}{[A_1,B_1]}  /path/to/structure_1.cif  A_1            B_1
        {['my-dataset']}{2}{[C_1,B_1]}  /path/to/structure_2.cif  C_1            B_1
    """

    def __init__(
        self,
        pn_unit_iid_colnames: str | list[str] = ["pn_unit_1_iid", "pn_unit_2_iid"],
        example_id_colname: str = "example_id",
        path_colname: str = "path",
        assembly_id_colname: str | None = None,
    ):
        self.example_id_colname = example_id_colname
        self.path_colname = path_colname
        self.pn_unit_iid_colnames = (
            pn_unit_iid_colnames if isinstance(pn_unit_iid_colnames, list) else [pn_unit_iid_colnames]
        )

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        # Assemble input pn_units
        query_pn_unit_iids = [row[colname] for colname in self.pn_unit_iid_colnames]

        # Get the assembly ID if specified, otherwise default to "1"
        if exists(self.assembly_id_colname):
            assembly_id = row[self.assembly_id_colname]
        else:
            assembly_id = "1"

        return {
            "example_id": row[self.example_id_colname],
            "path": Path(row[self.path_colname]),
            "assembly_id": assembly_id,
            "query_pn_unit_iids": query_pn_unit_iids,
            "extra_info": row.to_dict(),
        }
