from typing import Any

import numpy as np
import pytest
from conftest import get_pdb_path

from cifutils.parser import parse

TEST_CASES = [{"pdb_id": "1jj8", "count": 705}, {"pdb_id": "3kz8", "count": 6246}, {"pdb_id": "2r5z", "count": 1632}]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_add_hydrogen_atom_positions(test_case: dict[str, Any]):
    path = get_pdb_path(test_case["pdb_id"])

    # Load anf infer hydrogens...
    result_added_hydrogens = parse(
        filename=path,
        build_assembly="all",
        hydrogen_policy="infer",
    )

    # ...assert that there are hydrogens
    atom_array_added_hydrogens = result_added_hydrogens["assemblies"]["1"][0]  # First bioassembly, first model
    has_resolved_coordinates = ~np.isnan(atom_array_added_hydrogens.coord).any(axis=-1)
    non_nan_array = atom_array_added_hydrogens[has_resolved_coordinates]
    final_h_count = np.sum(non_nan_array.atomic_number == 1)
    assert final_h_count == test_case["count"]


if __name__ == "__main__":
    pytest.main([__file__])
