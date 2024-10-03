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
)


def get_1_from_3_letter_code(
    res_name: str,
    chain_type: ChainType,
    use_closest_canonical: bool = False,
    gap_three_letter: str = "<G>",
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
        chain_type (str): The type of chain, using the ChainType enum.
        gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).
        gap_three_letter (str): The three-letter code for a gap. Defaults to "<G>".
        unknown_protein_three_letter (str): The three-letter code for an unknown protein residue. Defaults to "UNK_PROT".
        unknown_rna_three_letter (str): The three-letter code for an unknown RNA residue. Defaults to "X" (which is standard)
        unknown_dna_three_letter (str): The three-letter code for an unknown DNA residue. Defaults to "DX" (which is standard)

    Returns:
        str: The corresponding 3-letter code.
    """
    # Convert gaps (-) to "<G>", or whatever is specified
    if letter == gap_one_letter:
        return gap_three_letter

    if chain_type.is_protein():
        # Proteins
        return protein_letters_1to3.get(letter, unknown_protein_three_letter)
    elif chain_type == ChainType.DNA:
        # DNA
        return dna_letters_1to3.get(letter, unknown_dna_three_letter)
    elif chain_type == ChainType.RNA:
        # RNA
        return rna_letters_1to3.get(letter, unknown_rna_three_letter)
    else:
        logger.error(f"Unsupported chain type: {chain_type}, returning unknown protein residue (i.e., UNK).")
        return unknown_protein_three_letter


def is_pyramidine(res_names: list | np.ndarray) -> np.ndarray:
    """
    Given a list of 3-letter residue names, returns a boolean array indicating whether each residue is a pyramidine.
    """

    def is_pyramidine_residue(res_name: str) -> bool:
        return res_name in ["C", "U", "DC", "DT"]

    apply = np.vectorize(is_pyramidine_residue)
    return apply(np.asarray(res_names))


def is_purine(res_names: list | np.ndarray) -> np.ndarray:
    """
    Given a list of 3-letter residue names, returns a boolean array indicating whether each residue is a purine.
    """

    def is_purine_residue(res_name: str) -> bool:
        return res_name in ["A", "G", "DA", "DG"]

    apply = np.vectorize(is_purine_residue)
    return apply(np.asarray(res_names))


def is_unknown_nucleotide(res_names: list | np.ndarray) -> np.ndarray:
    """
    Given a list of 3-letter residue names, returns a boolean array indicating whether each residue is an unknown nucleotide.
    """

    def is_unknown_nucleotide_residue(res_name: str) -> bool:
        return res_name in ["X", "DX"]

    apply = np.vectorize(is_unknown_nucleotide_residue)
    return apply(np.asarray(res_names))

def is_protein(res_names: list | np.ndarray) -> np.ndarray:
    """
    Given a list of 3-letter residue names, returns a boolean array indicating whether each residue is a protein residue.
    """

    def is_protein_residue(res_name: str) -> bool:
        return res_name in protein_letters_3to1

    apply = np.vectorize(is_protein_residue)
    return apply(np.asarray(res_names))

def is_glycine(res_names: list | np.ndarray) -> np.ndarray:
    """
    Given a list of 3-letter residue names, returns a boolean array indicating whether each residue is a glycine residue.
    """

    def is_glycine_residue(res_name: str) -> bool:
        return res_name == "GLY"

    apply = np.vectorize(is_glycine_residue)
    return apply(np.asarray(res_names))

def is_protein_not_glycine(res_names: list | np.ndarray) -> np.ndarray:
    """
    Given a list of 3-letter residue names, returns a boolean array indicating whether each residue is a protein residue that is not glycine.
    """

    def is_protein_not_glycine_residue(res_name: str) -> bool:
        return res_name in protein_letters_3to1 and res_name != "GLY"

    apply = np.vectorize(is_protein_not_glycine_residue)
    return apply(np.asarray(res_names))

def is_protein_unknown(res_names: list | np.ndarray) -> np.ndarray:
    """
    Given a list of 3-letter residue names, returns a boolean array indicating whether each residue is an unknown protein residue.
    """

    def is_protein_unknown_residue(res_name: str) -> bool:
        return res_name == "UNK"

    apply = np.vectorize(is_protein_unknown_residue)
    return apply(np.asarray(res_names))
