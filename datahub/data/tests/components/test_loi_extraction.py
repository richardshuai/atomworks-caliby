"""Pytest for LOI - SOI (subject of investigation) extraction"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.tests.conftest import DATA_PREPROCESSOR
from data.tests.test_cases import LOI_EXTRACTION_TEST_CASES


@pytest.mark.parametrize("test_case", LOI_EXTRACTION_TEST_CASES)
def test_loi_extraction(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]

    # Check that the LOI is extracted correctly from the CIF file
    parsed = DATA_PREPROCESSOR._load_cif(pdb_id)
    loi_set = set(parsed["ligand_info"]["ligand_of_interest"])
    assert loi_set == test_case["loi"]

    # Check that the LOI exmples give the correct molecule
    rows = DATA_PREPROCESSOR.get_rows(pdb_id)
    df = pd.DataFrame(rows)

    loi_seen = {k: 0 for k in loi_set}
    for _, row in df.iterrows():
        if row.q_pn_unit_is_loi:
            assembly_id = row.assembly_id
            chain_ids = row.q_pn_unit_id.split(",")
            structure = parsed["assemblies"][assembly_id][0]
            res_names = np.unique(
                structure[(np.isin(structure.chain_id, chain_ids)) & (structure.occupancy > 0)].res_name
            )
            if test_case.get("has_covalently_bonded_loi", False):
                assert any(
                    res in loi_set for res in res_names
                ), f"No LOI molecule found for {row.q_pn_unit_iid} in {res_names}. LOIs: {loi_set}"
                for res in res_names:
                    if res in loi_set:
                        loi_seen[res] += 1
            else:
                assert len(res_names) == 1, f"Multiple LOI molecules found for {row.q_pn_unit_iid}: {res_names}"
                assert res_names[0] in loi_set, f"LOI molecule {res_names[0]} not found in {loi_set}"
                loi_seen[res_names[0]] += 1

    # Check that all LOI molecules have been extracted
    assert all(count > 0 for count in loi_seen.values()), f"Some LOI molecules have not been extracted: {loi_seen}"
