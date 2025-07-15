import pytest
from conftest import get_pdb_path

from cifutils.parser import parse

TEST_CASES = [
    {"pdb_id": "4oji", "pH": [8.0, 8.0]},  # _exptl_crystal_grow.pdbx_pH_range   pH8.0
    {"pdb_id": "5hs6", "pH": [7.0, 8.0]},  # _exptl_crystal_grow.pdbx_pH_range   7.0-8.0
    {"pdb_id": "5a93", "pH": [5.9, 6.1]},  # 1 ? ? ? 5.9 ? ?  2 ? ? ? 6.1 ? ? in a loop
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_get_ph(test_case: dict):
    pdb_id = test_case["pdb_id"]
    path = get_pdb_path(pdb_id)
    result = parse(
        filename=path,
        add_missing_atoms=False,
        remove_ccds=[],
        hydrogen_policy="infer",
    )
    assert result["metadata"]["crystallization_details"]["pH"] == test_case["pH"]


if __name__ == "__main__":
    pytest.main([__file__])
