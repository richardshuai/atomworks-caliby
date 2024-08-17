"""
PyTest function to test filtering criteria.
Tested criteria includes:
- The detection and resolutions of clashes within a structure
- The exclusion of non-polymers bonded to a polymer via a non-biological bond
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from data.tests.conftest import DATA_PREPROCESSOR
from data.tests.test_cases import FILTERING_CRITERIA_TEST_CASES


@pytest.mark.parametrize("test_case", FILTERING_CRITERIA_TEST_CASES)
def test_filtering_criteria(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]

    rows = DATA_PREPROCESSOR.get_rows(pdb_id)
    df = pd.DataFrame(rows)
    pn_unit_iids = set(df["q_pn_unit_iid"].unique().tolist())

    pn_units_to_keep = set(test_case["pn_units_to_keep"])
    pn_units_to_remove = set(test_case["pn_units_to_remove"])

    # Assert that we are keeping the correct PN units
    assert pn_units_to_keep.issubset(pn_unit_iids), f"Missing PN unit to keep in {pdb_id}."

    # Assert we are removing the correct PN units
    assert not pn_units_to_remove.intersection(pn_unit_iids), f"Removing PN unit that should be kept in {pdb_id}."
