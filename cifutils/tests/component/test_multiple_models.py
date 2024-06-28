import pytest
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE
import numpy as np

NMR_TEST_CASES = [
    {"pdb_id": "1l2y", "num_models": 38},
    {"pdb_id": "1g03", "num_models": 20},
]


@pytest.mark.parametrize("test_case", NMR_TEST_CASES)
def test_multiple_models(test_case: dict):
    pdb_id = test_case["pdb_id"]
    path = get_digs_path(pdb_id)
    result = CIF_PARSER_BIOTITE.parse(
        path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        remove_crystallization_aids=True,
        build_assembly="all",
        patch_symmetry_centers=True,
        fix_arginines=True,
        convert_mse_to_met=True,
        model=None,  # Builds all models
    )

    atom_array_stack = result["atom_array_stack"]
    assert atom_array_stack.stack_depth() == test_case["num_models"]

    # Assert all models have different coordiantes
    for i in range(test_case["num_models"] - 1):
        assert not np.array_equal(atom_array_stack[i].coord, atom_array_stack[i + 1].coord)
