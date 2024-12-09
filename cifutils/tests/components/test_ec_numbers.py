import pytest

from tests.conftest import CIF_PARSER_BIOTITE, get_pdb_path

TEST_CASES = [
    {"pdb_id": "3bdp", "chain_id": "C", "ec_numbers": ["2.7.7.7"]},
    {"pdb_id": "8e1d", "chain_id": "B", "ec_numbers": ["2.3.1.48", "2.3.1.-"]},
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_ec_numbers(test_case: dict):
    pdb_id = test_case["pdb_id"]
    path = get_pdb_path(pdb_id)
    result = CIF_PARSER_BIOTITE.parse(
        filename=path,
        add_missing_atoms=False,
        add_bonds=False,
        remove_waters=False,
        residues_to_remove=[],
        build_assembly=None,
        patch_symmetry_centers=False,
        fix_arginines=False,
        convert_mse_to_met=False,
    )
    assert result["chain_info"][test_case["chain_id"]]["ec_numbers"] == test_case["ec_numbers"]
