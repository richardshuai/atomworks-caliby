import pytest
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE
import numpy as np
import re

TEST_CASES = ["2e2h", "4cpa", "1en2", "1aqc", "1ivo", "3k4a", "1cbn", "133d", "1l2y", "3nez"]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_one_letter_sequences(pdb_id: str):
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
