"""Utility functions for working with monomer sequences."""

__all__ = [
    "get_1_from_3_letter_code",
    "get_3_from_1_letter_code",
    "protein_letters_1to3_bytes",
    "rna_letters_1to3_bytes",
    "dna_letters_1to3_bytes",
]

from cifutils.utils.io_utils import logger
from Bio.Data.PDBData import (
    nucleic_letters_3to1,
    nucleic_letters_3to1_extended,
    protein_letters_1to3,
    protein_letters_3to1,
    protein_letters_3to1_extended,
)


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
        logger.info(f"Unsupported chain type: {chain_type}")
        return unknown_protein_three_letter


# Create dictionaries that support byte strings
protein_letters_1to3_bytes = {k.encode("utf-8"): v.encode("utf-8") for k, v in protein_letters_1to3.items()}
rna_letters_1to3_bytes = {k.encode("utf-8"): v.encode("utf-8") for k, v in rna_letters_1to3.items()}
dna_letters_1to3_bytes = {k.encode("utf-8"): v.encode("utf-8") for k, v in dna_letters_1to3.items()}
