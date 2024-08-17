"""Pytest function to test the detection and assignment of contacting PN units."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from data.tests.conftest import DATA_PREPROCESSOR
from data.tests.test_cases import FIND_CONTACTS_TEST_CASES


@pytest.mark.parametrize("test_case", FIND_CONTACTS_TEST_CASES)
def test_find_contacts(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]

    rows = DATA_PREPROCESSOR.get_rows(pdb_id)
    df = pd.DataFrame(rows)

    for example in test_case["contact_information"]:
        assembly_id = example["assembly_id"]
        pn_unit_iid = example["pn_unit_iid"]

        # Filter the DataFrame to only the PN unit of interest
        pn_unit_row = df[(df["assembly_id"] == assembly_id) & (df["q_pn_unit_iid"] == pn_unit_iid)]

        # Assert that there is only one row
        assert len(pn_unit_row) == 1

        contacting_pn_unit_iids = json.loads(pn_unit_row["q_pn_unit_contacting_pn_unit_iids"].iloc[0])
        assert len(contacting_pn_unit_iids) == example["num_contacting_pn_units"]

        # Count contacting atoms
        contacting_atoms = 0
        for partner in contacting_pn_unit_iids:
            contacting_atoms += partner["num_contacts"]
        assert example["num_contacts"] == contacting_atoms

        # Count close PN units
        num_close_pn_units = len(json.loads(pn_unit_row["q_pn_unit_close_pn_unit_iids"].iloc[0]))
        assert example["num_close_pn_units"] == num_close_pn_units


if __name__ == "__main__":
    pytest.main(["-v", __file__])
