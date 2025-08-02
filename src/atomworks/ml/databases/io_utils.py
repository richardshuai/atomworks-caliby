import json
import os
from dataclasses import asdict
from enum import Flag, StrEnum
from typing import Any, Dict, Optional

from cifutils import parse
from cifutils.tools.inference import components_to_atom_array
from cifutils.utils.io_utils import to_cif_file

from datahub.databases.data_source_utils import DATA_SOURCE_DB_PATH, get_data_source
from datahub.databases.dataclasses import BindNoBindMeasurement
from datahub.databases.utils import _smart_cast
from datahub.datasets.parsers.base import DEFAULT_CIF_PARSER_ARGS

DATABASE_DEFAULT_CIF_PARSER_ARGS = DEFAULT_CIF_PARSER_ARGS | {
    "fix_arginines": False,
    "add_missing_atoms": False,
    "remove_ccds": [],
}


def save_measurement_to_cif(
    measurement: BindNoBindMeasurement,
    path: os.PathLike,
) -> None:
    """
    Saves a BindNoBindMeasurement to a CIF file with all metadata stored as extra categories.

    This function takes a BindNoBindMeasurement dataclass and saves it as a CIF file with
    all the measurement metadata (name, bind_no_bind, binary, pH, metadata) stored as
    extra categories in the CIF file. The atom_array is saved as the main structure data.

    Args:
        measurement: The BindNoBindMeasurement to save.
        path: The path to save the CIF file to.
    """
    # Convert atom_array to the appropriate format if it's a list of ChemicalComponents
    if isinstance(measurement.atom_array, list):
        atom_array = components_to_atom_array(measurement.atom_array)
    else:
        atom_array = measurement.atom_array

    # Convert dataclass to dictionary and exclude atom_array
    measurement_dict = asdict(measurement)
    del measurement_dict["atom_array"]

    # Add data source fields
    data_source = get_data_source(measurement.data_source_id)
    if data_source is None:
        raise ValueError(
            f"Data source {measurement.data_source_id} not found in database located at {DATA_SOURCE_DB_PATH}."
        )
    data_source_dict = asdict(data_source)
    del data_source_dict["data_source_id"]  # already exists in measurement_dict
    measurement_dict.update(data_source_dict)

    # Serialize dtypes as necessary
    for key, value in measurement_dict.items():
        if isinstance(value, dict):  # dictionaries are saved as json strings
            measurement_dict[key] = json.dumps(value)
        elif isinstance(value, list):  # lists are saved as strings
            measurement_dict[key] = str(value)
        elif isinstance(value, bool):  # bools are saved as ints
            measurement_dict[key] = int(value)
        elif isinstance(value, StrEnum):  # str enums are saved as strings
            measurement_dict[key] = value.value
        elif isinstance(value, Flag):  # flags are saved as ints
            measurement_dict[key] = int(value.value)

    # Save to CIF file
    to_cif_file(
        atom_array,
        path=path,
        extra_categories={
            "database_fields": measurement_dict,
        },
    )


def load_measurement_from_cif(
    file: os.PathLike,
    *,
    assembly_id: str = "1",
    cif_parser_args: Optional[Dict[str, Any]] = DATABASE_DEFAULT_CIF_PARSER_ARGS,
) -> BindNoBindMeasurement:
    """
    Loads a BindNoBindMeasurement from a CIF file.

    This function loads a CIF file and reconstructs a BindNoBindMeasurement dataclass
    from the structure data and metadata stored in the extra categories.

    Args:
        file: The path to the CIF file to load.
        assembly_id: The assembly ID to load. Defaults to "1".
        cif_parser_args: Additional arguments for CIF parsing. Defaults to None.

    Returns:
        BindNoBindMeasurement: The reconstructed measurement dataclass.

    Raises:
        ValueError: If required metadata fields are missing from the CIF file.
    """
    # Default cif_parser_args to an empty dictionary if not provided
    if cif_parser_args is None:
        cif_parser_args = {}

    extra_fields = BindNoBindMeasurement.get_field_info()
    extra_fields.pop("atom_array")
    extra_fields = list(extra_fields.keys())

    result_dict = parse(
        filename=file,
        build_assembly=(assembly_id,),
        extra_fields=["database_fields"],
        keep_cif_block=True,
        **cif_parser_args,
    )

    # Extract the atom array
    atom_array = result_dict["assemblies"][assembly_id][0]

    # Extract metadata from extra categories
    cif_block = result_dict.get("cif_block", {})

    # Extract all fields from the CIF block
    field_info = BindNoBindMeasurement.get_field_info()
    measurement_dict = {}
    for field_name in extra_fields:
        if field_name in cif_block["database_fields"]:
            value = cif_block["database_fields"][field_name].as_array()[0]
            value = _smart_cast(value, field_info[field_name])
            measurement_dict[field_name] = value

    # validate that all required fields are present
    required_fields = BindNoBindMeasurement.get_required_fields()
    assert all(
        field_name in measurement_dict for field_name in required_fields if field_name != "atom_array"
    ), f"Missing required fields: {set(required_fields) - set(measurement_dict.keys())}"

    # add atom_array to measurement_dict
    measurement_dict["atom_array"] = atom_array

    return BindNoBindMeasurement(**measurement_dict)
