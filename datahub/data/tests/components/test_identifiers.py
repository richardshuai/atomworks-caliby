"""PyTest function to check the assignment of PN unit IDs."""

import pytest

from data.tests.conftest import DATA_PREPROCESSOR
from data.tests.test_cases import PN_UNIT_IID_TEST_CASES


@pytest.mark.parametrize("test_case", PN_UNIT_IID_TEST_CASES)
def test_identifiers(test_case):
    pdb_id = test_case["pdb_id"]
    generated_pn_unit_iids = []
    rows = DATA_PREPROCESSOR.get_rows(pdb_id)

    # Sort to only rows where assembly_id is the same as the one in the test case
    rows = [
        row
        for row in rows
        if row["assembly_id"] == test_case["assembly_id"] and row["q_pn_unit_iid"] == test_case["q_pn_unit_iid"]
    ]

    assert (
        len(rows) == 1
    ), f"Expected one row with PDB id {pdb_id}, assembly_id {test_case}, and q_pn_unit_iid {test_case['q_pn_unit_iid']}"
    row = rows[0]

    generated_pn_unit_iids = sorted(eval(row["q_pn_unit_close_pn_unit_iids"]) + [test_case["q_pn_unit_iid"]])
    reference_pn_unit_iids = sorted(test_case["pn_unit_iids"])

    assert (
        generated_pn_unit_iids == reference_pn_unit_iids
    ), f"Generated PN unit instance IDs do not match reference PN unit IIDs for PDB ID {pdb_id} and assembly_id {test_case['assembly_id']}."


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=WARNING", __file__])
