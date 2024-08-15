"""
General utility functions for working with CIF files in Biotite.
"""

from __future__ import annotations
import gzip
from collections import OrderedDict
from pathlib import Path
from os import PathLike
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure.atoms import repeat
from biotite.structure.io.pdbx import CIFFile, BinaryCIFFile
import logging
from Bio.Data.PDBData import (
    protein_letters_3to1_extended,
    nucleic_letters_3to1_extended,
    protein_letters_3to1,
    protein_letters_1to3,
    nucleic_letters_3to1,
)


logger = logging.getLogger(__name__)

# Manually encode mapping from 1 to 3 for nuclelic acids (Bio.Data.PDBData is insufficient)
rna_letters_1to3 = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
}

dna_letters_1to3 = {
    "A": "DA",
    "C": "DC",
    "G": "DG",
    "T": "DT",
}

# Create dictionaries that support byte strings
protein_letters_1to3_bytes = {k.encode("utf-8"): v.encode("utf-8") for k, v in protein_letters_1to3.items()}
rna_letters_1to3_bytes = {k.encode("utf-8"): v.encode("utf-8") for k, v in rna_letters_1to3.items()}
dna_letters_1to3_bytes = {k.encode("utf-8"): v.encode("utf-8") for k, v in dna_letters_1to3.items()}


def get_1_from_3_letter_code(
    res_name: str,
    chain_type: str,
    use_closest_canonical: bool = False,
    gap_three_letter: str = "<G>",
    gap_one_letter: str = "-",
) -> str:
    """
    Converts a 3-letter residue name to its 1-letter code based on the chain type.
    Optionally, the closest canonical mapping can be used.

    Parameters:
    - res_name (str): The 3-letter residue name.
    - chain_type (str): The type of chain, e.g., "polypeptide(D)", "polypeptide(L)", "polydeoxyribonucleotide", or "polyribonucleotide".
    - use_closest_canonical (bool): Whether to use the closest canonical mapping (from BioPython). Defaults to False.
    - gap_three_letter (str): The three-letter code for a gap. Defaults to "<G>".
    - gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).

    Returns:
    - str: The corresponding 1-letter code. Returns "X" if the residue name or chain type is not supported.
    """
    # Convert gaps ("<G>") to "-", or whatever is specified
    if res_name == gap_three_letter:
        return gap_one_letter

    chain_type = chain_type.lower()
    if chain_type == "polypeptide(d)" or chain_type == "polypeptide(l)":
        # Proteins
        if use_closest_canonical:
            return protein_letters_3to1_extended.get(res_name, "X")
        else:
            return protein_letters_3to1.get(res_name, "X")
    elif chain_type == "polydeoxyribonucleotide" or chain_type == "polyribonucleotide":
        # Nucleic acids

        # Pad the residue name to 3 characters
        res_name = res_name.ljust(3)

        if use_closest_canonical:
            return nucleic_letters_3to1_extended.get(res_name, "X")
        else:
            return nucleic_letters_3to1.get(res_name, "X")
    else:
        logger.warning(f"Unsupported chain type: {chain_type}")
        return "X"


def get_3_from_1_letter_code(
    letter: str,
    chain_type: str,
    gap_one_letter: str = "-",
    gap_three_letter: str = "<G>",
    unknown_protein_three_letter: str = "UNK",
    unknown_rna_three_letter: str = "X",
    unknown_dna_three_letter: str = "DX",
) -> str:
    """
    Converts a 1-letter residue name to its 3-letter code based on the chain type.
    NOTE: Converting from a three-letter, to a one-letter, back to a three-letter code is not invertible (i.e., 1:1) and may result in a different three-letter sequence.

    Parameters:
    - letter (str): The 1-letter residue name.
    - chain_type (str): The type of chain, e.g., "polypeptide(D)", "polypeptide(L)", "polydeoxyribonucleotide", or "polyribonucleotide".
    - gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).
    - gap_three_letter (str): The three-letter code for a gap. Defaults to "<G>".
    - unknown_protein_three_letter (str): The three-letter code for an unknown protein residue. Defaults to "UNK_PROT".
    - unknown_rna_three_letter (str): The three-letter code for an unknown RNA residue. Defaults to "X" (which is standard)
    - unknown_dna_three_letter (str): The three-letter code for an unknown DNA residue. Defaults to "DX" (which is standard)

    Returns:
    - str: The corresponding 3-letter code. Returns "UNK" if the letter or chain type is not supported.
    """
    chain_type = chain_type.lower()

    # Convert gaps (-) to "<G>", or whatever is specified
    if letter == gap_one_letter:
        return gap_three_letter

    if chain_type == "polypeptide(d)" or chain_type == "polypeptide(l)":
        # Proteins
        return protein_letters_1to3.get(letter, unknown_protein_three_letter)
    elif chain_type == "polydeoxyribonucleotide":
        # DNA
        return dna_letters_1to3.get(letter, unknown_dna_three_letter)
    elif chain_type == "polyribonucleotide":
        # RNA
        return rna_letters_1to3.get(letter, unknown_rna_three_letter)
    else:
        logger.warning(f"Unsupported chain type: {chain_type}")
        return unknown_protein_three_letter


def deduplicate_iterator(iterator):
    """Deduplicate an iterator while preserving order."""
    return iter(OrderedDict.fromkeys(iterator))


def get_bond_type_from_order_and_is_aromatic(order, is_aromatic):
    """Get the biotite struc.BondType from the bond order and aromaticity."""
    aromatic_bond_types = {
        1: struc.BondType.AROMATIC_SINGLE,
        2: struc.BondType.AROMATIC_DOUBLE,
        3: struc.BondType.AROMATIC_TRIPLE,
    }

    non_aromatic_bond_types = {
        1: struc.BondType.SINGLE,
        2: struc.BondType.DOUBLE,
        3: struc.BondType.TRIPLE,
        4: struc.BondType.QUADRUPLE,
    }

    return (
        aromatic_bond_types.get(order, struc.BondType.ANY)
        if is_aromatic
        else non_aromatic_bond_types.get(order, struc.BondType.ANY)
    )


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


def parse_transformations(struct_oper):
    """
    Get transformation operation in terms of rotation matrix and
    translation for each operation ID in ``pdbx_struct_oper_list``.

    Copied from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398
    """
    transformation_dict = {}
    for index, id in enumerate(struct_oper["id"].as_array(str)):
        rotation_matrix = np.array(
            [[struct_oper[f"matrix[{i}][{j}]"].as_array(float)[index] for j in (1, 2, 3)] for i in (1, 2, 3)]
        )
        translation_vector = np.array([struct_oper[f"vector[{i}]"].as_array(float)[index] for i in (1, 2, 3)])
        transformation_dict[id] = (rotation_matrix, translation_vector)
    return transformation_dict


def apply_assembly_transformation(structure, transformation_dict, operation):
    """
    Get subassembly by applying the given operation to the input
    structure containing affected asym IDs.

    Modified from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398
    """
    coord = structure.coord
    # Execute for each transformation step
    # in the operation expression
    for op_step in operation:
        rotation_matrix, translation_vector = transformation_dict[op_step]
        # Rotate
        coord = matrix_rotate(coord, rotation_matrix)
        # Translate
        coord += translation_vector

    # Add a dimension to coord to match expected shape or `repeat` (first dimension is # repeats)
    coord = coord[np.newaxis, ...]

    return repeat(structure, coord)


def matrix_rotate(v, matrix):
    """
    Perform a rotation using a rotation matrix.

    Parameters
    ----------
    v : ndarray
        The coordinates to rotate.
    matrix : ndarray
        The rotation matrix.

    Returns
    -------
    rotated : ndarray
        The rotated coordinates.
    """
    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = v.ndim
    if orig_ndim > 2:
        orig_shape = v.shape
        v = v.reshape(-1, 3)
    # Apply rotation
    v = np.dot(matrix, v.T).T
    # Reshape back into original shape
    if orig_ndim > 2:
        v = v.reshape(*orig_shape)
    return v


def fix_bonded_atom_charges(atom):
    """
    Fix charges and hydrogen counts for cases when
    charged a atom is connected by an inter-residue bond.

    Args:
        atom (Atom): The atom object to be modified.

    Returns:
        dict: A dictionary with updated 'charge', 'hyb', and 'nhyd' values.
    """
    if atom.element == 7 and atom.charge == 1 and atom.hyb == 3 and atom.nhyd == 2 and atom.hvydeg == 2:  # -(NH2+)-
        return {"charge": 0, "hyb": 2, "nhyd": 0}
    elif (
        atom.element == 7 and atom.charge == 1 and atom.hyb == 3 and atom.nhyd == 3 and atom.hvydeg == 0
    ):  # free NH3+ group
        return {"charge": 0, "hyb": 2, "nhyd": 2}
    elif atom.element == 8 and atom.charge == -1 and atom.hyb == 3 and atom.nhyd == 0:
        return {"charge": 0, "hyb": atom.hyb, "nhyd": atom.nhyd}
    elif atom.element == 8 and atom.charge == -1 and atom.hyb == 2 and atom.nhyd == 0:  # O-linked connections
        return {"charge": 0, "hyb": atom.hyb, "nhyd": atom.nhyd}
    elif atom.charge != 0:
        # Additional logic for other cases if needed
        pass
    return {"charge": atom.charge, "hyb": atom.hyb, "nhyd": atom.nhyd}
