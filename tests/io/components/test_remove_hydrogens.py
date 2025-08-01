import numpy as np
import pytest
from tests.io.conftest import get_pdb_path

from atomworks.io.parser import parse
from atomworks.io.utils.testing import assert_same_atom_array

TEST_CASES = ["2w3o"]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_remove_hydrogens(pdb_id: str):
    path = get_pdb_path(pdb_id)

    # First, we load without hydrogens...
    result_no_hydrogens = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="remove",
    )
    atom_array_no_hydrogens = result_no_hydrogens["assemblies"]["1"][0]  # First bioassembly, first model

    # ...and assert that there are no hydrogens
    assert np.any(atom_array_no_hydrogens.atomic_number != 1)

    # Then, we load with hydrogens...
    result_with_hydrogens = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="keep",
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


# test deprecation warning
def test_remove_hydrogens_deprecation(pdb_id: str = "6lyz"):
    path = get_pdb_path(pdb_id)

    arr1 = parse(filename=path, remove_hydrogens=False)["assemblies"]["1"][0]
    arr2 = parse(filename=path, hydrogen_policy="keep")["assemblies"]["1"][0]
    assert_same_atom_array(arr1, arr2)

    # remove
    arr1 = parse(filename=path, remove_hydrogens=True)["assemblies"]["1"][0]
    arr2 = parse(filename=path, hydrogen_policy="remove")["assemblies"]["1"][0]
    assert_same_atom_array(arr1, arr2)


if __name__ == "__main__":
    pytest.main([__file__])
