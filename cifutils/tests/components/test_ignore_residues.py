from __future__ import annotations

import numpy as np
import pytest

from cifutils.constants import CRYSTALLIZATION_AIDS
from tests.conftest import CIF_PARSER_BIOTITE, get_pdb_path

CRYSTALLIZATION_AIDS_PDB_IDS_TO_TEST = ["101M", "1AH8", "1ATG", "1ARX"]


@pytest.mark.parametrize("pdbid", CRYSTALLIZATION_AIDS_PDB_IDS_TO_TEST)
def test_remove_crystallization_aids(pdbid: str):
    # Not excluding crystallization aids
    out1 = CIF_PARSER_BIOTITE.parse(filename=get_pdb_path(pdbid), residues_to_remove=[])
    assert np.any(
        np.isin(out1["asym_unit"].res_name, CRYSTALLIZATION_AIDS)
    ), "No crystallization aids found when not excluding."

    # Excluding crystallization aids
    out2 = CIF_PARSER_BIOTITE.parse(filename=get_pdb_path(pdbid), residues_to_remove=CRYSTALLIZATION_AIDS)
    assert not np.any(
        np.isin(out2["asym_unit"].res_name, CRYSTALLIZATION_AIDS)
    ), "Crystallization aids found when excluding."
