import numpy as np
import pytest

from cifutils.utils.atom_matching_utils import assert_same_atom_array
from tests.conftest import CIF_PARSER_BIOTITE, get_pdb_path

TEST_CASES = [
    "2w3o",
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_remove_hydrogens(pdb_id: str):
    path = get_pdb_path(pdb_id)

    # First, we load without hydrogens...
    result_no_hydrogens = CIF_PARSER_BIOTITE.parse(
        filename=path,
        add_missing_atoms=True,
        remove_waters=True,
        build_assembly="all",
        fix_symmetry_centers=True,
        fix_arginines=True,
        keep_hydrogens=False,
    )
    atom_array_no_hydrogens = result_no_hydrogens["assemblies"]["1"][0]  # First bioassembly, first model

    # ...and assert that there are no hydrogens
    assert np.any(atom_array_no_hydrogens.element != "1")

    # Then, we load with hydrogens...
    result_with_hydrogens = CIF_PARSER_BIOTITE.parse(
        filename=path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        build_assembly="all",
        patch_symmetry_centers=True,
        fix_arginines=True,
        keep_hydrogens=True,
    )

    # ...assert that there are hydrogens
    atom_array_with_hydrogens = result_with_hydrogens["assemblies"]["1"][0]  # First bioassembly, first model
    assert np.any(atom_array_with_hydrogens.element == "1")

    # ...remove the hydrogens
    atom_array_with_hydrogens_filtered = atom_array_with_hydrogens[atom_array_with_hydrogens.element != "1"]

    # ...and assert that the atom arrays are the same
    assert_same_atom_array(atom_array_no_hydrogens, atom_array_with_hydrogens_filtered)


if __name__ == "__main__":
    pytest.main([__file__])
