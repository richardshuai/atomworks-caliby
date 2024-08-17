"""Shared test utils"""

import data.data_preprocessor as data_preprocessor
from data.tests.test_cases import (
    CHAIN_TYPE_TEST_CASES,
    FILTERING_CRITERIA_TEST_CASES,
    FIND_CONTACTS_TEST_CASES,
    FULL_PDB_EDGE_CASE_LIST,
    LOI_EXTRACTION_TEST_CASES,
    PDB_PROCESSING_TEST_CASES,
    PN_UNIT_IID_TEST_CASES,
    UNSUPPORTED_CHAIN_TYPE_TEST_CASES,
)

DATA_PREPROCESSOR = data_preprocessor.DataPreprocessor(
    polymer_pn_unit_limit=50,  # Set to 50 for processing speed during testing
)


def get_pdb_id_from_dict_list(dict_list, key="pdb_id"):
    return [item[key] for item in dict_list]


def seen_pdb_ids():
    """Create a set of PDB IDs that have been tested in the test cases, up through the catch-all edge cases."""
    tested_pdb_id_set = set()

    # Add IDs from UNSUPPORTED_CHAIN_TYPE_TEST_CASES
    tested_pdb_id_set.update(UNSUPPORTED_CHAIN_TYPE_TEST_CASES)

    # Add IDs from FIND_CONTACTS_TEST_CASES
    tested_pdb_id_set.update(get_pdb_id_from_dict_list(FIND_CONTACTS_TEST_CASES))

    # Add IDs from CHAIN_TYPE_TEST_CASES
    tested_pdb_id_set.update(get_pdb_id_from_dict_list(CHAIN_TYPE_TEST_CASES))

    # Add IDs from FILTERING_CRITERIA_TEST_CASES
    tested_pdb_id_set.update(get_pdb_id_from_dict_list(FILTERING_CRITERIA_TEST_CASES))

    # Add IDs from MOLECULAR_ID_TEST_CASES
    tested_pdb_id_set.update(get_pdb_id_from_dict_list(PN_UNIT_IID_TEST_CASES))

    # Add IDs from LOI_EXTRACTION_TEST_CASES
    tested_pdb_id_set.update(get_pdb_id_from_dict_list(LOI_EXTRACTION_TEST_CASES))

    # Add IDs from PDB_PROCESSING_TEST_CASES
    tested_pdb_id_set.update(get_pdb_id_from_dict_list(PDB_PROCESSING_TEST_CASES))

    return set(tested_pdb_id_set)


def aggregate_pdb_ids():
    """Create a set of all PDB IDs that have been tested, including the catch-all edge cases"""
    master_pdb_id_set = seen_pdb_ids()

    # Add IDs from FULL_PDB_EDGE_CASE_LIST
    master_pdb_id_set.update(set(FULL_PDB_EDGE_CASE_LIST))
    return master_pdb_id_set
