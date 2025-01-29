import numpy as np
import pytest
from conftest import get_pdb_path

from cifutils.parser import parse

TEST_CASES = ["2w3o"]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_add_hydrogen_atom_positions(pdb_id: str):
    path = get_pdb_path(pdb_id)

    # First, we load without hydrogens...
    result = parse(
        filename=path,
        build_assembly="all",
    )
    atom_array = result["assemblies"]["1"][0]  # First bioassembly, first model

    # count hydrogens
    original_h_count = np.sum(atom_array.atomic_number == 1)

    # Then, we load with hydrogens...
    result_added_hydrogens = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="infer",
    )

    # ...assert that there are hydrogens
    atom_array_added_hydrogens = result_added_hydrogens["assemblies"]["1"][0]  # First bioassembly, first model
    assert np.any(atom_array_added_hydrogens.atomic_number == 1)

    # ...assert that there are more hydrogens than original
    final_h_count = np.sum(atom_array_added_hydrogens.atomic_number == 1)
    assert final_h_count >= original_h_count
    # ...remove the hydrogens

    # Then, we load with removing hydrogens...
    result_remove_hydrogens = parse(filename=path, build_assembly="all", hydrogen_policy="remove")
    atom_array_removed_hydrogens = result_remove_hydrogens["assemblies"]["1"][0]  # First bioassembly, first model
    # count hydrogens
    removed_h_count = np.sum(atom_array_removed_hydrogens.atomic_number == 1)

    assert final_h_count > removed_h_count

    # check if all hydrogen positions not nan
    hydrogens = atom_array_added_hydrogens[atom_array_added_hydrogens.element == "H"]
    assert np.any(np.isnan(hydrogens.coord))


if __name__ == "__main__":
    pytest.main([__file__])
