import pytest
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE
from cifutils.cifutils_biotite.cifutils_biotite_utils import (
    get_3_from_1_letter_code,
    get_1_from_3_letter_code,
    get_3_from_1_letter_code_bytes,
)
import numpy as np
import re

TEST_CASES = ["2e2h", "4cpa", "1en2", "1aqc", "1ivo", "3k4a", "1cbn", "133d", "1l2y", "3nez"]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_parser_one_letter_sequence_outputs(pdb_id: str):
    path = get_digs_path(pdb_id)
    result = CIF_PARSER_BIOTITE.parse(
        path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        remove_crystallization_aids=True,
        build_assembly="all",
        patch_symmetry_centers=True,
        fix_arginines=False,
        convert_mse_to_met=False,
    )

    chain_info = result["chain_info"]
    for chain_id, chain_details in chain_info.items():
        chain_type = chain_details["type"]
        # Get the atom array for that specific chain and count the number of unique residues
        atom_array = result["atom_array_stack"][0]  # First model
        chain_atom_array = atom_array[atom_array.chain_id == chain_id]
        num_residues = len(np.unique(chain_atom_array.res_id))

        if (
            chain_type == "polypeptide(D)"
            or chain_type == "polypeptide(L)"
            or chain_type == "polydeoxyribonucleotide"
            or chain_type == "polyribonucleotide"
        ):
            unprocessed_entity_canonical_sequence = chain_details["unprocessed_entity_canonical_sequence"]
            unprocessed_entity_non_canonical_sequence = chain_details["unprocessed_entity_non_canonical_sequence"]
            processed_entity_canonical_sequence = chain_details["processed_entity_canonical_sequence"]
            processed_entity_non_canonical_sequence = chain_details["processed_entity_non_canonical_sequence"]

            # Ensure that the processed canonical and non-canonical sequences have the same length
            assert len(processed_entity_canonical_sequence) == len(processed_entity_non_canonical_sequence)

            # Assert that the unprocessed canonical sequence is at least as long as the processed canonical sequence (due to sequence heterogeneity, or NCAA that map to two AA)
            assert len(unprocessed_entity_canonical_sequence) >= len(processed_entity_canonical_sequence)

            # More concise regex to remove characters: B, Z, X, and also the content within parentheses
            if chain_type == "polypeptide(D)" or chain_type == "polypeptide(L)":
                unprocessed_cleaned = re.sub(r"\(.*?\)|[BZX]", "", unprocessed_entity_non_canonical_sequence)
                processed_cleaned = processed_entity_non_canonical_sequence.replace("X", "")
                assert len(unprocessed_cleaned) == len(processed_cleaned)

            # Ensure that the length of both processed sequences matches the number of residues in the chain
            assert len(processed_entity_canonical_sequence) == num_residues
            assert len(processed_entity_non_canonical_sequence) == num_residues

            # Ensure that the length of the unprocessed entity canonical sequence is >= the length of the processed entity canonical sequence
            assert len(unprocessed_entity_canonical_sequence) >= len(processed_entity_canonical_sequence)

            # If there's no sequence heterogeneity, perform additional checks
            if not chain_details["has_sequence_heterogeneity"] and (
                chain_type == "polypeptide(D)" or chain_type == "polypeptide(L)"
            ):
                assert unprocessed_cleaned == processed_cleaned


# Define test cases for proteins
PROTEIN_TEST_CASES = [
    ("A", "polypeptide(D)", "ALA"),
    ("C", "polypeptide(D)", "CYS"),
    ("D", "polypeptide(D)", "ASP"),
    ("E", "polypeptide(D)", "GLU"),
    ("F", "polypeptide(D)", "PHE"),
    ("G", "polypeptide(D)", "GLY"),
    ("H", "polypeptide(D)", "HIS"),
    ("I", "polypeptide(D)", "ILE"),
    ("K", "polypeptide(D)", "LYS"),
    ("L", "polypeptide(D)", "LEU"),
    ("M", "polypeptide(D)", "MET"),
    ("N", "polypeptide(D)", "ASN"),
    ("P", "polypeptide(D)", "PRO"),
    ("Q", "polypeptide(D)", "GLN"),
    ("R", "polypeptide(D)", "ARG"),
    ("S", "polypeptide(D)", "SER"),
    ("T", "polypeptide(D)", "THR"),
    ("V", "polypeptide(D)", "VAL"),
    ("W", "polypeptide(D)", "TRP"),
    ("Y", "polypeptide(D)", "TYR"),
    ("-", "polypeptide(D)", "<GAP>"),
]

# Define test cases for DNA
DNA_TEST_CASES = [
    ("A", "polydeoxyribonucleotide", "DA"),
    ("C", "polydeoxyribonucleotide", "DC"),
    ("G", "polydeoxyribonucleotide", "DG"),
    ("T", "polydeoxyribonucleotide", "DT"),
    ("-", "polydeoxyribonucleotide", "<GAP>"),
]

# Define test cases for RNA
RNA_TEST_CASES = [
    ("A", "polyribonucleotide", "A"),
    ("C", "polyribonucleotide", "C"),
    ("G", "polyribonucleotide", "G"),
    ("U", "polyribonucleotide", "U"),
    ("-", "polyribonucleotide", "<GAP>"),
]

# Define test cases for unknown letters
UNKNOWN_TEST_CASES = [
    ("B", "polypeptide(D)", "UNK"),
    ("Z", "polypeptide(D)", "UNK"),
    ("X", "polypeptide(D)", "UNK"),
    ("B", "polydeoxyribonucleotide", "DX"),
    ("Z", "polydeoxyribonucleotide", "DX"),
    ("X", "polydeoxyribonucleotide", "DX"),
    ("B", "polyribonucleotide", "RX"),
    ("Z", "polyribonucleotide", "RX"),
    ("X", "polyribonucleotide", "RX"),
]


@pytest.mark.parametrize(
    "letter, chain_type, expected_three_letter",
    PROTEIN_TEST_CASES + DNA_TEST_CASES + RNA_TEST_CASES + UNKNOWN_TEST_CASES,
)
def test_get_3_from_1_letter_code(letter, chain_type, expected_three_letter):
    assert get_3_from_1_letter_code(letter, chain_type) == expected_three_letter


# We can't test the reverse mapping for unknown letters (all map to "X")
@pytest.mark.parametrize(
    "expected_one_letter, chain_type, three_letter_code", PROTEIN_TEST_CASES + DNA_TEST_CASES + RNA_TEST_CASES
)
def test_get_1_from_3_letter_code(three_letter_code, chain_type, expected_one_letter):
    assert get_1_from_3_letter_code(three_letter_code, chain_type) == expected_one_letter


# Convert test cases to byte strings
PROTEIN_TEST_CASES_BYTES = [(x.encode("utf-8"), y, z.encode("utf-8")) for x, y, z in PROTEIN_TEST_CASES]
DNA_TEST_CASES_BYTES = [(x.encode("utf-8"), y, z.encode("utf-8")) for x, y, z in DNA_TEST_CASES]
RNA_TEST_CASES_BYTES = [(x.encode("utf-8"), y, z.encode("utf-8")) for x, y, z in RNA_TEST_CASES]
UNKNOWN_TEST_CASES_BYTES = [(x.encode("utf-8"), y, z.encode("utf-8")) for x, y, z in UNKNOWN_TEST_CASES]


@pytest.mark.parametrize(
    "letter, chain_type, expected_three_letter",
    PROTEIN_TEST_CASES_BYTES + DNA_TEST_CASES_BYTES + RNA_TEST_CASES_BYTES + UNKNOWN_TEST_CASES_BYTES,
)
def test_get_3_from_1_letter_code_bytes(letter, chain_type, expected_three_letter):
    assert get_3_from_1_letter_code_bytes(letter, chain_type) == expected_three_letter
