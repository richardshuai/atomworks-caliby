"""Utility functions for working with monomer sequences."""

__all__ = [
    "get_1_from_3_letter_code",
    "get_3_from_1_letter_code",
]

import numpy as np
from cifutils.utils.io_utils import logger
from cifutils.enums import ChainType
from Bio.Data.PDBData import (
    nucleic_letters_3to1,
    nucleic_letters_3to1_extended,
    protein_letters_1to3,
    protein_letters_3to1,
    protein_letters_3to1_extended,
)  # TODO: Deprecate these in favour of the direct mappings from the CCD
from cifutils.constants import (
    STANDARD_AA,
    STANDARD_PYRAMIDINE_RESIDUES,
    STANDARD_PURINE_RESIDUES,
    UNKNOWN_AA,
    UNKNOWN_RNA,
    UNKNOWN_DNA,
    GAP,
)


def get_1_from_3_letter_code(
    res_name: str,
    chain_type: ChainType,
    use_closest_canonical: bool = False,
    gap_three_letter: str = GAP,
    gap_one_letter: str = "-",
) -> str:
    """
    Converts a 3-letter residue name to its 1-letter code based on the chain type.
    Optionally, the closest canonical mapping can be used.

    Args:
        res_name (str): The 3-letter residue name.
        chain_type (ChainType): The type of chain, using the ChainType enum.
        use_closest_canonical (bool): Whether to use the closest canonical mapping (from BioPython). Defaults to False.
        gap_three_letter (str): The three-letter code for a gap. Defaults to "<G>".
        gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).

    Returns:
        str: The corresponding 1-letter code. Returns "X" if the residue name or chain type is not supported.
    """
    # ...convert gaps ("<G>") to "-", or whatever is specified
    if res_name == gap_three_letter:
        return gap_one_letter

    if chain_type.is_protein():
        if use_closest_canonical:
            return protein_letters_3to1_extended.get(res_name, "X")
        else:
            return protein_letters_3to1.get(res_name, "X")
    elif chain_type.is_nucleic_acid():
        # ...pad the residue name to 3 characters for consistency
        res_name = res_name.ljust(3)

        if use_closest_canonical:
            return nucleic_letters_3to1_extended.get(res_name, "X")
        else:
            return nucleic_letters_3to1.get(res_name, "X")
    else:
        logger.info(f"Unsupported chain type: {chain_type}")
        return "X"


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


def get_3_from_1_letter_code(
    letter: str,
    chain_type: ChainType,
    gap_one_letter: str = "-",
    gap_three_letter: str = GAP,
    unknown_aa: str = UNKNOWN_AA,
    unknown_rna: str = UNKNOWN_RNA,
    unknown_dna: str = UNKNOWN_DNA,
) -> str:
    """
    Converts a 1-letter residue name to its 3-letter code based on the chain type.
    NOTE: Converting from a three-letter, to a one-letter, back to a three-letter code is not invertible (i.e., 1:1) and may result in a different three-letter sequence.

    Args:
        letter (str): The 1-letter residue name.
        chain_type (str): The type of chain, using the ChainType enum.
        gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).
        gap_three_letter (str): The three-letter code for a gap. Defaults to "<G>".
        unknown_aa (str): The three-letter code for an unknown protein residue. Defaults to "UNK_PROT".
        unknown_rna (str): The three-letter code for an unknown RNA residue. Defaults to "X" (which is standard)
        unknown_dna (str): The three-letter code for an unknown DNA residue. Defaults to "DX" (which is standard)

    Returns:
        str: The corresponding 3-letter code.
    """
    assert len(letter) == 1, "The 1-letter code must be a single character."

    # Convert gaps (-) to "<G>", or whatever is specified
    if letter == gap_one_letter:
        return gap_three_letter

    if chain_type.is_protein():
        # Proteins
        return protein_letters_1to3.get(letter, unknown_aa)
    elif chain_type == ChainType.DNA:
        # DNA
        return dna_letters_1to3.get(letter, unknown_dna)
    elif chain_type == ChainType.RNA:
        # RNA
        return rna_letters_1to3.get(letter, unknown_rna)
    else:
        logger.error(f"Unsupported {chain_type=}, returning unknown protein residue {unknown_aa=}.")
        return unknown_aa


def is_pyramidine(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.isin(ccd_code_array, STANDARD_PYRAMIDINE_RESIDUES)


def is_purine(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.isin(ccd_code_array, STANDARD_PURINE_RESIDUES)


def is_unknown_nucleotide(ccd_code_array: np.ndarray) -> np.ndarray:
    ccd_code_array = np.asarray(ccd_code_array)
    return (ccd_code_array == UNKNOWN_DNA) | (ccd_code_array == UNKNOWN_RNA)


def is_standard_aa(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.isin(ccd_code_array, STANDARD_AA)


def is_glycine(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.asarray(ccd_code_array) == "GLY"


def is_standard_aa_not_glycine(ccd_code_array: np.ndarray) -> np.ndarray:
    _PROTEIN_NOT_GLYCINE_RESIDUES = [res for res in STANDARD_AA if res != "GLY"]
    return np.isin(ccd_code_array, _PROTEIN_NOT_GLYCINE_RESIDUES)


def is_protein_unknown(ccd_code_array: np.ndarray) -> np.ndarray:
    return np.asarray(ccd_code_array) == UNKNOWN_AA
