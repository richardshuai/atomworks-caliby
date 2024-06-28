from __future__ import annotations
import pytest
import numpy as np
from cifutils.cifutils_biotite.constants import CRYSTALLIZATION_AIDS
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE

PDB_IDS_TO_TEST = ["101M", "1AH8", "1ATG", "1ARX"]


@pytest.mark.parametrize("pdbid", PDB_IDS_TO_TEST)
def test_remove_crystallization_aids(pdbid: str):
    # Not excluding crystallization aids
    out1 = CIF_PARSER_BIOTITE.parse(get_digs_path(pdbid), remove_crystallization_aids=False)
    assert np.any(
        np.isin(out1["atom_array"].res_name, CRYSTALLIZATION_AIDS)
    ), "No crystallization aids found when not excluding."

    # Excluding crystallization aids
    out2 = CIF_PARSER_BIOTITE.parse(get_digs_path(pdbid), remove_crystallization_aids=True)
    assert not np.any(
        np.isin(out2["atom_array"].res_name, CRYSTALLIZATION_AIDS)
    ), "Crystallization aids found when excluding."
