import pytest
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE
import numpy as np

TEST_CASES = [
    "2w3o", 
]

@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_prior_bugs(pdb_id: str):
    path = get_digs_path(pdb_id)
    result_no_hydrogens = CIF_PARSER_BIOTITE.parse(
        filename=path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        remove_crystallization_aids=True,
        build_assembly="all",
        patch_symmetry_centers=True,
        fix_arginines=True,
        add_hydrogens=False,
    )
    atom_array_no_hydrogens = result_no_hydrogens['assemblies']['1']

    # Assert that the atom array has no hydrogens
    assert np.any(atom_array_no_hydrogens.element != "1")

if __name__ == "__main__":
    pytest.main([__file__])
