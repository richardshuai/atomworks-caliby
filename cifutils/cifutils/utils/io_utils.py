"""
General utility functions for working with CIF files in Biotite.
"""

__all__ = ["load_any", "get_structure", "read_any", "to_cif_buffer", "to_cif_string", "to_cif_file"]

import gzip
import io
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import pdbx
import biotite.structure.io.pdb as biotite_pdb
import numpy as np
from pathlib import Path
import logging
from typing import Literal
from contextlib import nullcontext
import os
from cifutils.common import default
from datetime import datetime
from cifutils.common import exists

from cifutils.constants import ATOMIC_NUMBER_TO_ELEMENT


logger = logging.getLogger("cifutils")


def _get_logged_in_user():
    """
    Get the logged in user.
    """
    try:
        return os.getlogin()
    except OSError:
        return "unknown_user"


def load_any(
    file_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
    file_type: Literal["cif", "mmcif", "pdbx", "pdb", "pdb1", "bcif"] | None = None,
    *,
    extra_fields: list[str] = [],
    assume_residues_all_resolved: bool = False,
    include_bonds: bool = True,
    model: int | None = None,
    altloc: Literal["first", "occupancy", "all"] = "occupancy",
) -> AtomArrayStack | AtomArray:
    """
    Convenience function for loading a structure from a file or buffer.
    """
    file_obj = read_any(file_or_buffer, file_type=file_type)
    return get_structure(
        file_obj,
        extra_fields=extra_fields,
        assume_residues_all_resolved=assume_residues_all_resolved,
        include_bonds=include_bonds,
        model=model,
        altloc=altloc,
    )


def get_structure(
    file_obj: pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile | pdbx.CIFBlock,
    *,
    extra_fields: list[str] = [],
    assume_residues_all_resolved: bool = False,
    include_bonds: bool = True,
    model: int | None = None,
    altloc: Literal["first", "occupancy", "all"] = "occupancy",
) -> AtomArrayStack | AtomArray:
    """
    Load example structure into Biotite's AtomArrayStack using the specified fields and assumptions.

    Args:
       - file_obj (pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile): The file object to load with Biotite.
       - extra_fields (list): List of extra fields to include as AtomArray annotations.
       - assume_residues_all_resolved (bool): If True, assumes all residues are resolved and sets occupancy to 1.0 for all atoms.
       - model (int): The model number to use for loading the structure.
       - altloc (Literal["first", "occupancy", "all"]): The altloc ID to use for loading the structure.

    Returns:
        AtomArrayStack: The loaded structure with the specified fields and assumptions.

    Reference:
        Biotite documentation (https://www.biotite-python.org/apidoc/biotite.structure.io.pdbx.get_structure.html#biotite.structure.io.pdbx.get_structure)
    """
    match type(file_obj):
        case pdbx.CIFFile | pdbx.BinaryCIFFile | pdbx.CIFBlock:
            # Filter extra annotations to fields that are actually present in the file
            if not isinstance(file_obj, pdbx.CIFBlock):
                cif_block = file_obj.block
            extra_fields = _filter_extra_fields(extra_fields, cif_block["atom_site"])

            atom_array_stack = pdbx.get_structure(
                file_obj,
                model=model,
                extra_fields=extra_fields,
                use_author_fields=False,
                altloc="first" if "occupancy" not in extra_fields else altloc,
                include_bonds=include_bonds,
            )
        case biotite_pdb.PDBFile:
            atom_array_stack = biotite_pdb.get_structure(
                file_obj,
                model=model,
                extra_fields=extra_fields,
                altloc=altloc,
                include_bonds=include_bonds,
            )
        case _:
            raise ValueError(f"Unsupported file type: {type(file_obj)}. Must be a CIFFile, BinaryCIFFile, or PDBFile.")

    if assume_residues_all_resolved:
        # Set the occupancy to 1.0 for all atoms if we're assuming everything is resolved
        atom_array_stack.set_annotation("occupancy", np.ones(atom_array_stack.array_length()))

    return atom_array_stack


def read_any(
    path_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
    file_type: Literal["cif", "mmcif", "pdbx", "pdb", "pdb1", "bcif"] | None = None,
) -> AtomArray:
    """
    Reads any of the allowed file types into the appropriate Biotite file object.

    Args:
        path_or_buffer (PathLike | io.StringIO | io.BytesIO): The path to the file or a buffer to read from.
            If a buffer, it's highly recommended to specify the file_type.
        file_type (Literal["cif", "mmcif", "pdbx", "pdb", "pdb1", "bcif"], optional): Type of the file.
            If None, it will be inferred from the file extension. When using a buffer, the file type must be specified.
        **load_kwargs: Additional keyword arguments to pass to the Biotite loading function.

    Returns:
        AtomArray: The loaded structure.

    Raises:
        ValueError: If the file type is unsupported or cannot be determined.
    """
    # Convert string paths to Path objects
    if isinstance(path_or_buffer, str):
        path_or_buffer = Path(path_or_buffer)

    # Determine file type and open context
    if isinstance(path_or_buffer, io.BytesIO):
        inferred_file_type = "bcif"
        open_context = nullcontext(path_or_buffer)
    elif isinstance(path_or_buffer, io.StringIO):
        # ... if second line starts with '#', it is very likely a cif file
        path_or_buffer.seek(0)
        path_or_buffer.readline()  # Skip the first line
        second_line = path_or_buffer.readline().strip()
        path_or_buffer.seek(0)
        inferred_file_type = "cif" if second_line.startswith("#") else "pdb"
        open_context = nullcontext(path_or_buffer)
    elif isinstance(path_or_buffer, Path):
        if path_or_buffer.suffix in (".gz", ".gzip"):
            open_context = gzip.open(path_or_buffer, "rt")
            inferred_file_type = Path(path_or_buffer.stem).suffix.lstrip(".")
        else:
            open_context = nullcontext(path_or_buffer)
            inferred_file_type = path_or_buffer.suffix.lstrip(".")
    else:
        raise ValueError(f"Unsupported path_or_buffer type: {type(path_or_buffer)}")

    # Determine the appropriate file object based on file type
    file_type = default(file_type, inferred_file_type)
    if file_type in ("cif", "mmcif", "pdbx"):
        file_cls = pdbx.CIFFile
    elif file_type in ("pdb", "pdb1"):
        file_cls = biotite_pdb.PDBFile
    elif file_type == "bcif":
        file_cls = pdbx.BinaryCIFFile
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Load the structure
    with open_context as f:
        file_obj = file_cls.read(f)

    return file_obj


def to_cif_buffer(
    structure: AtomArray,
    *,
    id: str = "unknown_id",
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
) -> io.StringIO:
    """Convert an AtomArray structure to a CIF formatted StringIO buffer.

    Args:
        - structure (AtomArray): The atomic structure to be converted.
        - id (str): The ID of the entry. This will be used as the data block name.
        - author (str): The author of the entry.
        - date (str): The date of the entry.
        - time (str): The time of the entry.
        - extra_categories (dict[str, dict[str, float | int | str | list | np.ndarray]] | None, optional):
            Additional CIF categories to include in data block. These must be a dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"pdbx_reflns_number_d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}

    Returns:
        StringIO: A buffer containing the CIF formatted string representation of the structure.
    """
    structure = structure.copy()
    cif_file = pdbx.CIFFile()

    if not exists(date):
        date = datetime.now().strftime("%Y-%m-%d")
    if not exists(time):
        time = datetime.now().strftime("%H:%M:%S")

    # If elements are given as atomic numbers, convert them to element symbols
    structure.element = np.vectorize(lambda x: ATOMIC_NUMBER_TO_ELEMENT.get(x, x))(structure.element)

    # If altloc information is present but no altloc id is given, set all to "."
    if "altloc_id" in structure.get_annotation_categories() and structure.altloc_id[0].strip() == "":
        structure.altloc_id = ["."] * structure.array_length()

    block = pdbx.convert._get_or_create_block(cif_file, block_name=id)
    Category = block.subcomponent_class()
    Column = Category.subcomponent_class()

    block["entry"] = Category(
        {
            "id": id,
            "author": author,
            "date": date,
            "time": time,
        }
    )

    # Set the structure in the CIF file
    pdbx.set_structure(cif_file, structure, data_block=id, include_bonds=True)

    # Add extra categories if provided
    if extra_categories:
        for category_name, category_data in extra_categories.items():
            category = Category()
            for key, value in category_data.items():
                category[key] = Column(value)
            block[category_name] = category

    # Serialize the CIF file to a string
    buffer = io.StringIO()
    cif_file.write(
        buffer,
    )
    return buffer


def to_cif_string(
    structure: AtomArray,
    *,
    id: str = "unknown_id",
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
) -> str:
    """Convert an AtomArray structure to a CIF formatted string.

    Args:
        - structure (AtomArray): The atomic structure to be converted.
        - id (str): The ID of the entry. This will be used as the data block name.
        - author (str): The author of the entry.
        - date (str): The date of the entry.
        - time (str): The time of the entry.
        - extra_categories (dict[str, dict[str, float | int | str | list | np.ndarray]] | None, optional):
            Additional CIF categories to include in data block. These must be a dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"pdbx_reflns_number_d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}

    Returns:
        str: The CIF formatted string representation of the structure.
    """
    return to_cif_buffer(
        structure, id=id, author=author, date=date, time=time, extra_categories=extra_categories
    ).getvalue()


def to_cif_file(
    structure: AtomArray,
    path: os.PathLike,
    *,
    id: str = "unknown_id",
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
) -> str:
    """Convert an AtomArray structure to a CIF formatted file.

    Args:
        - structure (AtomArray): The atomic structure to be converted.
        - path (os.PathLike): The file path where the CIF formatted structure will be saved.
        - id (str): The ID of the entry. This will be used as the data block name.
        - author (str): The author of the entry.
        - date (str): The date of the entry.
        - time (str): The time of the entry.
        - extra_categories (dict[str, dict[str, float | int | str | list | np.ndarray]] | None, optional):
            Additional CIF categories to include in data block. These must be a dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"pdbx_reflns_number_d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}

    Returns:
        str: The file path where the CIF formatted structure was saved.

    Raises:
        IOError: If there's an issue writing to the specified file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            to_cif_buffer(
                structure, id=id, author=author, date=date, time=time, extra_categories=extra_categories
            ).getvalue()
        )
    return path


def to_pdb_buffer(
    structure: AtomArray,
) -> io.StringIO:
    """Convert an AtomArray structure to a PDB formatted StringIO buffer.

    NOTE: It's recommended to use `to_cif_buffer` instead of this function. That function
    is more flexible and can handle extra annotations and metadata that PDB does not support.

    Args:
        - structure (AtomArray): The atomic structure to be converted.

    Returns:
        StringIO: The PDB formatted StringIO buffer of the structure.
    """
    # Create a PDBFile object
    pdb_file = biotite_pdb.PDBFile()

    # Set the structure and bonds
    pdb_file.set_structure(structure)

    # Convert to string
    buffer = io.StringIO()
    pdb_file.write(buffer)
    return buffer


def to_pdb_string(
    structure: AtomArray,
) -> str:
    """
    Convert an AtomArray structure to a PDB formatted string.

    NOTE: It's recommended to use `to_cif_string` instead of this function. That function
    is more flexible and can handle extra annotations and metadata that PDB does not support.

    Args:
        - structure (AtomArray): The atomic structure to be converted.

    Returns:
        str: The PDB formatted string representation of the structure.
    """
    return to_pdb_buffer(structure).getvalue()


def _filter_extra_fields(extra_fields: list[str], atom_site: pdbx.CIFCategory) -> list[str]:
    """
    Filter the extra fields to only include fields that are actually present in the file.
    """
    _translate_builtin_fields = {
        "atom_id": "id",
        "charge": "pdbx_formal_charge",
        "b_factor": "B_iso_or_equiv",
        "occupancy": "occupancy",
    }
    _fields_with_default = {
        "label_entity_id",
        "auth_seq_id",
    }

    filtered_extra_fields = []
    for field in extra_fields:
        if field in _fields_with_default:
            filtered_extra_fields.append(field)
            continue
        if _translate_builtin_fields.get(field, field) in atom_site:
            filtered_extra_fields.append(field)
        else:
            logger.warning(f"Field {field} not found in file, ignoring.")

    return filtered_extra_fields
