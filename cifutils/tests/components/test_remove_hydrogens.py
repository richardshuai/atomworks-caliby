import numpy as np
import pytest
from conftest import get_pdb_path

from cifutils.parser import parse
from cifutils.utils.testing import assert_same_atom_array

TEST_CASES = ["2w3o"]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_remove_hydrogens(pdb_id: str):
    path = get_pdb_path(pdb_id)

    # First, we load without hydrogens...
    result_no_hydrogens = parse(
        filename=path,
        build_assembly="all",
        remove_hydrogens=True,
    )
    atom_array_no_hydrogens = result_no_hydrogens["assemblies"]["1"][0]  # First bioassembly, first model

    # ...and assert that there are no hydrogens
    assert np.any(atom_array_no_hydrogens.atomic_number != 1)

    # Then, we load with hydrogens...
    result_with_hydrogens = parse(
        filename=path,
        build_assembly="all",
        remove_hydrogens=False,
    )

    # ...assert that there are hydrogens
    atom_array_with_hydrogens = result_with_hydrogens["assemblies"]["1"][0]  # First bioassembly, first model
    assert np.any(atom_array_with_hydrogens.atomic_number == 1)

    # ...remove the hydrogens
    atom_array_with_hydrogens_filtered = atom_array_with_hydrogens[atom_array_with_hydrogens.atomic_number != 1]

    # ...and assert that the atom arrays are the same
    assert_same_atom_array(
        atom_array_no_hydrogens,
        atom_array_with_hydrogens_filtered,
        annotations_to_compare=["chain_id", "res_name", "res_id", "atom_name", "element"],
    )


if __name__ == "__main__":
    pytest.main([__file__])
