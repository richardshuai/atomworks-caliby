"""
General utility functions for working with CIF files in Biotite.
"""

__all__ = ["get_structure", "load_any", "read_any", "to_cif_buffer", "to_cif_file", "to_cif_string"]

import gzip
import io
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import biotite.structure as struc
import biotite.structure.io.pdb as biotite_pdb
import numpy as np
from biotite.setup_ccd import _concatenate_blocks_into_category
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.io import mol, pdbx

from cifutils.common import exists
from cifutils.constants import ATOMIC_NUMBER_TO_ELEMENT, STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from cifutils.enums import ChainType
from cifutils.utils.sequence import get_1_from_3_letter_code

logger = logging.getLogger("cifutils")


def _get_logged_in_user() -> str:
    """
    Get the logged in user.
    """
    try:
        return os.getlogin()
    except OSError:
        return "unknown_user"


def _has_ambiguous_bond_annotation(atom_array: AtomArray) -> bool:
    """
    Detect if there is ambiguous annotation of the structure that would
    lead to loss of information when writing out the structure.

    This happens because the `struct_conn` category distinguishes bonds
    between different atoms based on the 5-tuple:
        (chain_id, res_id, res_name, atom_id, ins_code)

    To properly save bonds with a structure, make sure that all atoms
    have unique 5-tuples.

    Args:
        atom_array (AtomArray): The atom array to check for ambiguous annotations.

    Returns:
        bool: True if ambiguous annotations are detected, False otherwise.
    """
    # Create a structured array with the 5-tuple elements
    identifier_dtypes = [
        ("chain_id", atom_array.chain_id.dtype if "chain_id" in atom_array.get_annotation_categories() else "U1"),
        ("res_id", atom_array.res_id.dtype if "res_id" in atom_array.get_annotation_categories() else "U1"),
        ("res_name", atom_array.res_name.dtype if "res_name" in atom_array.get_annotation_categories() else "U1"),
        ("atom_name", atom_array.atom_name.dtype if "atom_name" in atom_array.get_annotation_categories() else "U1"),
        ("ins_code", atom_array.ins_code.dtype if "ins_code" in atom_array.get_annotation_categories() else "U1"),
    ]

    structured_array = np.empty(atom_array.array_length(), dtype=identifier_dtypes)
    for category in identifier_dtypes:
        name, dtype = category
        structured_array[name] = (
            atom_array.get_annotation(name)
            if name in atom_array.get_annotation_categories()
            else ["."] * atom_array.array_length()
        )

    # Use numpy's unique function with return_counts=True to find duplicates
    _, counts = np.unique(structured_array, return_counts=True)

    # If any count is greater than 1, we have ambiguous annotations
    return np.any(counts > 1)


def load_any(
    file_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
    file_type: Literal["cif", "mmcif", "pdbx", "pdb", "pdb1", "bcif"] | None = None,
    *,
    extra_fields: list[str] = [],
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
        include_bonds=include_bonds,
        model=model,
        altloc=altloc,
    )


def get_structure(
    file_obj: pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile | pdbx.CIFBlock,
    *,
    extra_fields: list[str] = [],
    include_bonds: bool = True,
    model: int | None = None,
    altloc: Literal["first", "occupancy", "all"] = "occupancy",
) -> AtomArrayStack | AtomArray:
    """
    Load example structure into Biotite's AtomArrayStack using the specified fields and assumptions.

    Args:
       - file_obj (pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile): The file object to load with Biotite.
       - extra_fields (list): List of extra fields to include as AtomArray annotations.
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

    return atom_array_stack


def infer_pdb_file_type(path_or_buffer: os.PathLike | io.StringIO | io.BytesIO) -> Literal["cif", "pdb", "bcif", "sdf"]:
    """
    Infer the file type of a PDB file or buffer.
    """
    # Convert string paths to Path objects
    if isinstance(path_or_buffer, str):
        path_or_buffer = Path(path_or_buffer)

    # Determine file type and open context
    if isinstance(path_or_buffer, io.BytesIO):
        return "bcif"
    elif isinstance(path_or_buffer, io.StringIO):
        # ... if second line starts with '#', it is very likely a cif file
        path_or_buffer.seek(0)
        path_or_buffer.readline()  # Skip the first line
        second_line = path_or_buffer.readline().strip()
        path_or_buffer.seek(0)
        return "cif" if second_line.startswith("#") else "pdb"
    elif isinstance(path_or_buffer, Path):
        if path_or_buffer.suffix in (".gz", ".gzip"):
            inferred_file_type = Path(path_or_buffer.stem).suffix.lstrip(".")
        else:
            inferred_file_type = path_or_buffer.suffix.lstrip(".")

    # Canonicalize the file type
    if inferred_file_type in ("cif", "mmcif", "pdbx"):
        return "cif"
    elif inferred_file_type in ("pdb", "pdb1"):
        return "pdb"
    elif inferred_file_type == "bcif":
        return "bcif"
    elif inferred_file_type == "sdf":
        return "sdf"
    else:
        raise ValueError(f"Unsupported file type: {inferred_file_type}")


def read_any(
    path_or_buffer: os.PathLike | io.StringIO | io.BytesIO,
    file_type: Literal["cif", "pdb", "bcif", "sdf"] | None = None,
) -> pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile:
    """
    Reads any of the allowed file types into the appropriate Biotite file object.

    Args:
        path_or_buffer (PathLike | io.StringIO | io.BytesIO): The path to the file or a buffer to read from.
            If a buffer, it's highly recommended to specify the file_type.
        file_type (Literal["cif", "pdb", "bcif"], optional): Type of the file.
            If None, it will be inferred from the file extension. When using a buffer, the file type must be specified.

    Returns:
        pdbx.CIFFile | biotite_pdb.PDBFile | pdbx.BinaryCIFFile: The loaded file object.

    Raises:
        ValueError: If the file type is unsupported or cannot be determined.
    """
    # Determine file type
    if file_type is None:
        file_type = infer_pdb_file_type(path_or_buffer)
        open_mode = "rb" if file_type == "bcif" else "rt"

    # Convert string paths to Path objects and decompress if necessary
    if isinstance(path_or_buffer, str | Path):
        path_or_buffer = Path(path_or_buffer)
        if path_or_buffer.suffix in (".gz", ".gzip"):
            with gzip.open(path_or_buffer, open_mode) as f:
                buffer_type = io.StringIO if open_mode == "rt" else io.BytesIO
                path_or_buffer = buffer_type(f.read())

    # Determine the appropriate file object based on file type
    if file_type == "cif":
        file_cls = pdbx.CIFFile
    elif file_type == "pdb":
        file_cls = biotite_pdb.PDBFile
    elif file_type == "bcif":
        file_cls = pdbx.BinaryCIFFile
    elif file_type == "sdf":
        file_cls = mol.SDFile
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Load the file content
    file_obj = file_cls.read(path_or_buffer)

    return file_obj


def _build_entity_poly(
    atom_array: struc.AtomArray,
) -> dict[str, dict[str, float | int | str | list | np.ndarray]]:
    """
    Build the entity_poly category for a CIF file from an AtomArray.

    This function processes polymer entities in the structure and generates their sequence information
    in both canonical and non-canonical forms.

    Args:
        - atom_array: AtomArray containing the structure data with polymer chain information.

    Returns:
        A dictionary containing the entity_poly category with the following fields:
        - entity_id: List of entity identifiers
        - type: List of polymer types (polypeptide, polynucleotide, etc.)
        - nstd_linkage: List indicating presence of non-standard linkages
        - nstd_monomer: List indicating presence of non-standard monomers
        - pdbx_seq_one_letter_code: List of sequences in one-letter code
        - pdbx_seq_one_letter_code_can: List of canonical sequences
        - pdbx_strand_id: List of chain identifiers
        - pdbx_target_identifier: List of target identifiers
    """
    _entity_poly_categories = (
        "entity_id",
        "type",
        "nstd_linkage",
        "nstd_monomer",
        "pdbx_seq_one_letter_code",
        "pdbx_seq_one_letter_code_can",
        "pdbx_strand_id",
        "pdbx_target_identifier",
    )

    # ... get index of the first atom of each chain
    chain_starts = struc.get_chain_starts(atom_array)

    # ... get chain ids, iids, entity ids, and chain types
    chain_ids = atom_array.chain_id[chain_starts]
    chain_iids = atom_array.chain_iid[chain_starts]
    entity_ids = atom_array.chain_entity[chain_starts]
    is_polymer = atom_array.is_polymer[chain_starts]
    chain_types = atom_array.chain_type[chain_starts]

    if not np.any(is_polymer):
        return {}

    unique_polymer_entity_ids = np.unique(entity_ids[is_polymer])
    entity_poly = {cat: [] for cat in _entity_poly_categories}
    for entity_id in unique_polymer_entity_ids:
        # ... get all relevant chain ids
        chain_ids = np.unique(chain_ids[entity_ids == entity_id])

        # ... get chain type
        chain_type = ChainType.as_enum(chain_types[entity_ids == entity_id][0])

        # ... get sequence
        example_chain_iid = chain_iids[entity_ids == entity_id][0]
        res_starts = struc.get_residue_starts(atom_array[atom_array.chain_iid == example_chain_iid])
        seq = atom_array.res_name[res_starts]
        wrap_every_n = lambda text, n: "\n".join(text[i : i + n] for i in range(0, len(text), n))  # noqa: E731
        processed_entity_non_canonical_sequence = "".join(
            get_1_from_3_letter_code(ccd_code, chain_type, use_closest_canonical=False) for ccd_code in seq
        )
        processed_entity_non_canonical_sequence = wrap_every_n(processed_entity_non_canonical_sequence, 80)
        processed_entity_canonical_sequence = "".join(
            get_1_from_3_letter_code(ccd_code, chain_type, use_closest_canonical=True) for ccd_code in seq
        )
        processed_entity_canonical_sequence = wrap_every_n(processed_entity_canonical_sequence, 80)

        # ... check for non-standard monomers
        has_non_standard_monomer = ~np.all(np.isin(seq, STANDARD_AA + STANDARD_RNA + STANDARD_DNA))

        # ... add to entity_poly
        entity_poly["entity_id"].append(entity_id)
        entity_poly["type"].append(chain_type.to_string().lower())
        entity_poly["nstd_linkage"].append("no")
        entity_poly["nstd_monomer"].append("yes" if has_non_standard_monomer else "no")
        entity_poly["pdbx_seq_one_letter_code"].append(processed_entity_non_canonical_sequence)
        entity_poly["pdbx_seq_one_letter_code_can"].append(processed_entity_canonical_sequence)
        entity_poly["pdbx_strand_id"].append(",".join(chain_ids))
        entity_poly["pdbx_target_identifier"].append("?")
    return {"entity_poly": entity_poly}


def _write_categories_to_block(
    block: "pdbx.Block", categories: dict[str, dict[str, float | int | str | list | np.ndarray]]
) -> None:
    """Write a set of categories to a CIF block"""
    Category = block.subcomponent_class()  # noqa: N806
    Column = Category.subcomponent_class()  # noqa: N806
    for category_name, category_data in categories.items():
        category = Category()
        if category_data is not None:
            for key, value in category_data.items():
                category[key] = Column(value)
        else:
            raise ValueError(f"Category {category_name} is empty. Do not write empty categories to CIF file.")
        block[category_name] = category


def _cif_to_bcif(cif_file: pdbx.CIFFile | pdbx.BinaryCIFFile) -> pdbx.BinaryCIFFile:
    """Convert a given CIF file to an optimized BCIF file."""
    compressed_file = pdbx.BinaryCIFFile()
    for block_name, block in cif_file.items():
        compressed_block = pdbx.BinaryCIFBlock()
        for category_name in block:
            _tmp_cif_file = pdbx.CIFFile()
            _tmp_cif_file[block_name] = block
            compressed_block[category_name] = pdbx.compress(
                _concatenate_blocks_into_category(_tmp_cif_file, category_name)
            )
        compressed_file[block_name] = compressed_block
    return compressed_file


def _to_cif_or_bcif(
    structure: AtomArray,
    *,
    id: str = "unknown_id",
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    include_entity_poly: bool = False,
    include_nan_coords: bool = True,
    include_bonds: bool = True,
    extra_fields: list[str] | Literal["all"] = [],
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
    as_bcif: bool = False,
    _allow_ambiguous_bond_annotations: bool = False,
) -> pdbx.CIFFile | pdbx.BinaryCIFFile:
    structure = structure.copy()
    cif_file = pdbx.CIFFile()

    if not exists(date):
        date = datetime.now().strftime("%Y-%m-%d")
    if not exists(time):
        time = datetime.now().strftime("%H:%M:%S")

    if not _allow_ambiguous_bond_annotations and _has_ambiguous_bond_annotation(structure):
        raise ValueError(
            "Ambiguous bond annotations detected. This happens when there are atoms that "
            "have the same `(chain_id, res_id, res_name, atom_id, ins_code)` identifier. "
            "This happens for example when you have a bio-assembly with multiple copies "
            "of a chain that only differ by `transformation_id`.\n"
            "You can fix this for example by re-naming the chains to be named uniquely."
            "For more info, see: https://git.ipd.uw.edu/ai/cifutils/-/issues/15"
        )

    # If elements are given as atomic numbers, convert them to (uppercase) element symbols
    structure.element = np.vectorize(lambda x: ATOMIC_NUMBER_TO_ELEMENT.get(x, x))(structure.element)

    # If altloc information is present but no altloc id is given, set all to "."
    if "altloc_id" in structure.get_annotation_categories() and structure.altloc_id[0].strip() == "":
        structure.altloc_id = ["."] * structure.array_length()

    block = pdbx.convert._get_or_create_block(cif_file, block_name=id)

    # Build metadata
    metadata = {"entry": {"id": id, "author": author, "date": date, "time": time}}
    for flag, build_func in [
        (include_entity_poly, _build_entity_poly),
    ]:
        if flag:
            try:
                metadata.update(build_func(structure))
            except Exception as e:
                logger.warning(f"Failed to build `{build_func.__name__}`: {e}")
    # Write metadata to block
    _write_categories_to_block(block, metadata)

    # Set the structure in the CIF file
    if extra_fields == "all":
        _standard_cif_annotations = frozenset(
            {
                "chain_id",
                "res_id",
                "res_name",
                "atom_name",
                "atom_id",
                "element",
                "ins_code",
                "hetero",
                "altloc_id",
                "charge",
                "occupancy",
                "b_factor",
            }
        )
        extra_fields = list(set(structure.get_annotation_categories()) - _standard_cif_annotations)

    if not include_nan_coords:
        has_nan_coords = np.any(np.isnan(structure.coord), axis=tuple(range(1, structure.coord.ndim)))
        structure = structure[~has_nan_coords]

    pdbx.set_structure(cif_file, structure, data_block=id, include_bonds=include_bonds, extra_fields=extra_fields)

    # Add extra categories if provided
    if extra_categories:
        _write_categories_to_block(block, extra_categories)

    if as_bcif:
        cif_file = _cif_to_bcif(cif_file)
    return cif_file


def to_cif_buffer(
    structure: AtomArray,
    *,
    id: str = "unknown_id",
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    include_entity_poly: bool = False,
    include_nan_coords: bool = True,
    include_bonds: bool = True,
    extra_fields: list[str] | Literal["all"] = [],
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
    _allow_ambiguous_bond_annotations: bool = False,
    as_bcif: bool = False,
) -> io.StringIO | io.BytesIO:
    """Convert an AtomArray structure to a CIF formatted StringIO buffer.

    Args:
        structure (AtomArray): The atomic structure to be converted.
        id (str): The ID of the entry. This will be used as the data block name.
        author (str): The author of the entry.
        date (str): The date of the entry.
        time (str): The time of the entry.
        include_entity_poly (bool): Whether to write entity_poly category in the CIF file.
        include_nan_coords (bool): Whether to write NaN coordinates in the CIF file.
        include_bonds (bool): Whether to write bonds in the CIF file.
        extra_fields (list[str] | Literal["all"]): Additional atom_array annotations to include in the CIF file.
        extra_categories (dict[str, dict[str, float | int | str | list | np.ndarray]] | None, optional):
            Additional CIF categories to include in data block. These must be a dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"pdbx_reflns_number_d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}
        _allow_ambiguous_bond_annotations (bool, optional): Private argument, not meant for public use.
            If True, allows ambiguous bond annotations.

    Returns:
        StringIO | BytesIO: A buffer containing the CIF/BCIF formatted string/bytes representation of the structure.
    """
    file_obj = _to_cif_or_bcif(
        structure,
        id=id,
        author=author,
        date=date,
        time=time,
        include_entity_poly=include_entity_poly,
        include_nan_coords=include_nan_coords,
        include_bonds=include_bonds,
        extra_fields=extra_fields,
        extra_categories=extra_categories,
        as_bcif=as_bcif,
        _allow_ambiguous_bond_annotations=_allow_ambiguous_bond_annotations,
    )
    buffer = io.BytesIO() if as_bcif else io.StringIO()
    file_obj.write(buffer)
    buffer.seek(0)
    return buffer


def to_cif_string(
    structure: AtomArray,
    *,
    id: str = "unknown_id",
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    include_entity_poly: bool = False,
    include_nan_coords: bool = True,
    include_bonds: bool = True,
    extra_fields: list[str] | Literal["all"] = [],
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
    as_bcif: bool = False,
    _allow_ambiguous_bond_annotations: bool = False,
) -> str | bytes:
    """Convert an AtomArray structure to a CIF formatted string.

    Args:
        structure (AtomArray): The atomic structure to be converted.
        id (str): The ID of the entry. This will be used as the data block name.
        author (str): The author of the entry.
        date (str): The date of the entry.
        time (str): The time of the entry.
        include_entity_poly (bool): Whether to write entity_poly category in the CIF file.
        include_nan_coords (bool): Whether to write NaN coordinates in the CIF file.
        include_bonds (bool): Whether to write bonds in the CIF file.
        extra_fields (list[str] | Literal["all"]): Additional atom_array annotations to include in the CIF file.
        extra_categories (dict[str, dict[str, float | int | str | list | np.ndarray]] | None, optional):
            Additional CIF categories to include in data block. These must be a dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"pdbx_reflns_number_d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}

    Returns:
        str | bytes: The CIF/BCIF formatted string/bytes representation of the structure.
    """
    return to_cif_buffer(
        structure,
        id=id,
        author=author,
        date=date,
        time=time,
        include_entity_poly=include_entity_poly,
        include_nan_coords=include_nan_coords,
        include_bonds=include_bonds,
        extra_fields=extra_fields,
        extra_categories=extra_categories,
        _allow_ambiguous_bond_annotations=_allow_ambiguous_bond_annotations,
        as_bcif=as_bcif,
    ).getvalue()


def to_cif_file(
    structure: AtomArray,
    path: os.PathLike,
    *,
    file_type: Literal["cif", "bcif", "cif.gz"] | None = None,
    id: str | None = None,
    author: str = _get_logged_in_user(),
    date: str | None = None,
    time: str | None = None,
    include_entity_poly: bool = True,
    include_nan_coords: bool = True,
    include_bonds: bool = True,
    extra_fields: list[str] | Literal["all"] = [],
    extra_categories: dict[str, dict[str, float | int | str | list | np.ndarray]] | None = None,
    _allow_ambiguous_bond_annotations: bool = False,
    gzip_output: bool | None = None,
) -> os.PathLike:
    """Convert an AtomArray structure to a CIF/BCIF formatted file.

    Args:
        structure (AtomArray): The atomic structure to be converted.
        path (os.PathLike): The file path where the CIF formatted structure will be saved.
        file_type (Literal["cif", "bcif", "cif.gz"] | None): The file type to save the structure as.
            If None, the file type will be inferred from the path.
        id (str | None): The ID of the entry. This will be used as the data block name.
            If None, the data block name will be inferred from the path.
        author (str): The author of the entry.
        date (str): The date of the entry.
        time (str): The time of the entry.
        include_entity_poly (bool): Whether to write entity_poly category in the CIF file.
        include_nan_coords (bool): Whether to write NaN coordinates in the CIF file.
        include_bonds (bool): Whether to write bonds in the CIF file.
        extra_fields (list[str] | Literal["all"]): Additional atom_array annotations to include in the CIF file.
        extra_categories (dict[str, dict[str, float | int | str | list | np.ndarray]] | None, optional):
            Additional CIF categories to include in data block. These must be a dict of form {category_name: {column_name: value}}.
            Example: {"reflns": {"pdbx_reflns_number_d_mean": 1.0}, "my_metadata": {"hi": np.arange(10)}}
        gzip_output (bool | None): Whether to gzip the output file. If None, the file type will be inferred from the path.
            WARNING: This option is deprecated. Please use `file_type` instead.

    Returns:
        str: The file path where the CIF formatted structure was saved.

    Raises:
        IOError: If there's an issue writing to the specified file path.
    """
    # turn any relative path into an absolute path
    path = str(os.path.abspath(path))

    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if gzip_output is not None:
        warnings.warn(
            "The `gzip_output` argument is deprecated. Please use `file_type` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if str(path).endswith(".cif"):
            path = path.replace(".cif", ".cif.gz") if gzip_output else path
        elif str(path).endswith(".cif.gz"):
            pass
        else:
            path = path + ".cif.gz"

        file_type = "gz" if gzip_output else "cif"

    _file_type_map = {
        # suffix: (as_bcif, open_func, open_mode, path_suffix)
        "cif": (False, open, "wt", ".cif"),
        "bcif": (True, open, "wb", ".bcif"),
        "cif.gz": (False, gzip.open, "wt", ".cif.gz"),
        "bcif.gz": (True, gzip.open, "wb", ".bcif.gz"),
        # ... default if no suffix is provided
        "": (False, gzip.open, "wt", ".cif.gz"),
    }
    if file_type is None:
        file_name = os.path.basename(path)
        file_type = file_name.split(".")[-1] if "." in file_name else ""
        # ... with gz suffix by fetching the pre-gz suffix
        file_type = file_name.split(".")[-2] + ".gz" if file_type == "gz" else file_type

    as_bcif, open_func, open_mode, path_suffix = _file_type_map[file_type]

    # ... check that the file ends with the correct suffix, otherwise mutate the path to end with the correct suffix
    if not path.endswith(path_suffix):
        path = path + path_suffix
    # ... get the name minus the suffix
    file_name = os.path.basename(path).replace(path_suffix, "")

    file_obj = _to_cif_or_bcif(
        structure,
        id=id or file_name,
        author=author,
        date=date,
        time=time,
        include_entity_poly=include_entity_poly,
        include_nan_coords=include_nan_coords,
        include_bonds=include_bonds,
        extra_fields=extra_fields,
        extra_categories=extra_categories,
        _allow_ambiguous_bond_annotations=_allow_ambiguous_bond_annotations,
        as_bcif=as_bcif,
    )

    with open_func(path, mode=open_mode) as f:
        file_obj.write(f)

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

    if _has_ambiguous_bond_annotation(structure):
        raise ValueError(
            "Ambiguous bond annotations detected. This happens when there are atoms that "
            "have the same `(chain_id, res_id, res_name, atom_id, ins_code)` identifier. "
            "This happens for example when you have a bio-assembly with multiple copies "
            "of a chain that only differ by `transformation_id`.\n"
            "You can fix this for example by re-naming the chains to be named uniquely."
            "For more info, see: https://git.ipd.uw.edu/ai/cifutils/-/issues/15"
        )

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
