from __future__ import annotations
import pytest
from cifutils.cifutils_biotite import cifutils_biotite
import numpy as np
from cifutils.cifutils_biotite.constants import CRYSTALLIZATION_AIDS
from tests.conftest import get_digs_path

PDB_IDS_TO_TEST = ["101M", "1AH8", "1ATG", "1ARX"]

cif_parser_no_exclude = cifutils_biotite.CIFParser(exclude_crystallization_aid=False)
cif_parser_do_exclude = cifutils_biotite.CIFParser(exclude_crystallization_aid=True)


@pytest.mark.parametrize("pdbid", PDB_IDS_TO_TEST)
def test_remove_crystallization_aids(pdbid: str):
    # Not excluding crystallization aids
    out1 = cif_parser_no_exclude.parse(get_digs_path(pdbid))
    assert np.any(
        np.isin(out1["atom_array"].res_name, CRYSTALLIZATION_AIDS)
    ), "No crystallization aids found when not excluding."

    # Excluding crystallization aids
    out2 = cif_parser_do_exclude.parse(get_digs_path(pdbid))
    assert not np.any(
        np.isin(out2["atom_array"].res_name, CRYSTALLIZATION_AIDS)
    ), "Crystallization aids found when excluding."
