import pytest
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE
import numpy as np

LIGAND_AT_SYMMETRY_CENTER_TEST_CASES = [
    {
        "pdbid": "7mub",  # metal ion at symmetry center
        "chain_full_ids_to_include": ["D_1", "E_1"],
        "chain_full_ids_to_exclude": ["D_2", "D_3", "D_4", "E_2", "E_3", "E_4"],
    },
    {
        "pdbid": "1xan",  # symmetric ligand at symmetry center
        "chain_full_ids_to_include": ["C_1"],
        "chain_full_ids_to_exclude": ["C_2"],
    },
]


@pytest.mark.parametrize("test_case", LIGAND_AT_SYMMETRY_CENTER_TEST_CASES)
def test_patch_symmetry_centers(test_case: dict):
    # unpack test case
    pdbid = test_case["pdbid"]

    # Parse the file
    filename = get_digs_path(pdbid)
    out = CIF_PARSER_BIOTITE.parse(filename, build_assembly="first", remove_crystallization_aids=True, remove_waters=True)
    chain_full_ids = np.unique(out["assemblies"]["1"].chain_full_id).tolist()

    # Ensure that we excluded clashing chains
    assert set(chain_full_ids).intersection(set(test_case["chain_full_ids_to_exclude"])) == set()

    # Ensure that we included the correct chains
    assert set(chain_full_ids).intersection(set(test_case["chain_full_ids_to_include"])) == set(
        test_case["chain_full_ids_to_include"]
    )
