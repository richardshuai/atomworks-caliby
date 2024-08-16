"""
General utility functions for working with CIF files in Biotite.
"""

from __future__ import annotations
import gzip
from collections import OrderedDict
import numpy as np
from pathlib import Path
from os import PathLike
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx import CIFFile, BinaryCIFFile, CIFBlock
import logging
from Bio.Data.PDBData import (
    protein_letters_3to1_extended,
    nucleic_letters_3to1_extended,
    protein_letters_3to1,
    protein_letters_1to3,
    nucleic_letters_3to1,
)
from cifutils.cifutils_biotite.common import exists
from functools import lru_cache


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


@lru_cache(maxsize=10000)
def get_chem_comp_type(res_name: str) -> str:
    """
    Get the chemical component type for a residue name from the Chemical Component Dictionary (CCD).
    Can be combined with CHEM_TYPES from `cifutils_biotite.constants` to determine if a residue is a protein, nucleic acid, or carbohydrate.

    Args:
        res_name (str): The residue name.

    Example:
        >>> get_chem_comp_type("ALA")
        'L-PEPTIDE LINKING'
    """
    chemcomp_type = struc.info.ccd.get_from_ccd("chem_comp", res_name, "type")
    if exists(chemcomp_type):
        return chemcomp_type[0].upper()
    else:
        logger.warning(f"Chemical component type for `{res_name}` not found in CCD. Using 'other'.")
        return "other".upper()


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

    Args:
        res_name (str): The 3-letter residue name.
        chain_type (str): The type of chain, e.g., "polypeptide(D)", "polypeptide(L)", "polydeoxyribonucleotide", or "polyribonucleotide".
        use_closest_canonical (bool): Whether to use the closest canonical mapping (from BioPython). Defaults to False.
        gap_three_letter (str): The three-letter code for a gap. Defaults to "<G>".
        gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).

    Returns:
        str: The corresponding 1-letter code. Returns "X" if the residue name or chain type is not supported.
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

    Args:
        letter (str): The 1-letter residue name.
        chain_type (str): The type of chain, e.g., "polypeptide(D)", "polypeptide(L)", "polydeoxyribonucleotide", or "polyribonucleotide".
        gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).
        gap_three_letter (str): The three-letter code for a gap. Defaults to "<G>".
        unknown_protein_three_letter (str): The three-letter code for an unknown protein residue. Defaults to "UNK_PROT".
        unknown_rna_three_letter (str): The three-letter code for an unknown RNA residue. Defaults to "X" (which is standard)
        unknown_dna_three_letter (str): The three-letter code for an unknown DNA residue. Defaults to "DX" (which is standard)

    Returns:
        str: The corresponding 3-letter code. Returns "UNK" if the letter or chain type is not supported.
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
