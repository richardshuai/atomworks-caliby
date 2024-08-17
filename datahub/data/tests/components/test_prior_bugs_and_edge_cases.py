"""PyTest function to check whether past problematic entries run through without error."""

import pytest

from data.tests.conftest import DATA_PREPROCESSOR, seen_pdb_ids
from data.tests.test_cases import FULL_PDB_EDGE_CASE_LIST

PDBS_TO_TEST = set(FULL_PDB_EDGE_CASE_LIST) - seen_pdb_ids()


@pytest.mark.slow
@pytest.mark.parametrize("test_case", PDBS_TO_TEST)
def test_prior_bugs_and_edge_cases(test_case):
    """Runs data loading for a list of prior problematic entries to ensure they run through without error."""
    rows = DATA_PREPROCESSOR.get_rows(test_case)
    assert rows is not None  # Check if the processing runs through


def examine_specific_case(pdb_id):
    """Used for debugging"""
    rows = DATA_PREPROCESSOR.get_rows(pdb_id)
    assert rows is not None  # Check if the processing runs through
