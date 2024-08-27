"""
General utility functions for working with CIF files in Biotite.
"""

from __future__ import annotations
import gzip
import numpy as np
from pathlib import Path
from os import PathLike
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx import CIFFile, BinaryCIFFile, CIFBlock
import logging


logger = logging.getLogger(__name__)


def load_structure(cif_block: CIFBlock, common_extra_fields: list, assume_residues_all_resolved: bool, model: int):
    """
    Load example structure into Biotite's AtomArrayStack using the specified fields and assumptions.

    Args:
        cif_block (CIFBlock): The CIF block to load with Biotite. Must contain the ATOM_SITE category.
        common_extra_fields (list): List of extra fields to include as AtomArray annotations.
        assume_residues_all_resolved (bool): If True, assumes all residues are resolved and sets occupancy to 1.0 for all atoms.
        model (int): The model number to use for loading the structure.

    Returns:
        AtomArrayStack: The loaded structure with the specified fields and assumptions.

    Reference:
        Biotite documentation (https://www.biotite-python.org/apidoc/biotite.structure.io.pdbx.get_structure.html#biotite.structure.io.pdbx.get_structure)
    """

    atom_array_stack = pdbx.get_structure(
        cif_block,
        extra_fields=common_extra_fields,
        use_author_fields=False,
        altloc="occupancy"
        if not assume_residues_all_resolved
        else "first",  # If we're assuming residues are all resolved, we only need the first altloc (and we don't have occupancy)
        model=model,
    )
    if assume_residues_all_resolved:
        # Set the occupancy to 1.0 for all atoms if we're assuming everything is resolved
        atom_array_stack.set_annotation("occupancy", np.ones(atom_array_stack.array_length()))
    return atom_array_stack


def read_cif_file(filename: PathLike) -> CIFFile | BinaryCIFFile:
    """Reads a CIF, BCIF, or gzipped CIF/BCIF file and returns its contents."""
    if not isinstance(filename, Path):
        filename = Path(filename)

    file_ext = filename.suffix

    if file_ext == ".gz":
        with gzip.open(filename, "rt") as f:
            # Handle gzipped CIF files
            if filename.name.endswith(".cif.gz"):
                cif_file = pdbx.CIFFile.read(f)
            elif filename.name.endswith(".bcif.gz"):
                with gzip.open(filename, "rb") as bf:
                    cif_file = pdbx.BinaryCIFFile.read(bf)
            else:
                raise ValueError("Unsupported file format for gzip compressed file")
    elif file_ext == ".bcif":
        # Handle BinaryCIF files
        cif_file = pdbx.BinaryCIFFile.read(filename)
    elif file_ext == ".cif":
        # Handle plain CIF files
        cif_file = pdbx.CIFFile.read(filename)
    else:
        raise ValueError(f"Unsupported file format {file_ext} in {filename}")

    return cif_file
