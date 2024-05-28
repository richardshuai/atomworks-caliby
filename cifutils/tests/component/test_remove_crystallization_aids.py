import pytest
from cifutils.cifutils_biotite import cifutils_biotite
import os
import numpy as np
from cifutils.cifutils_biotite.constants import CRYSTALLIZATION_AIDS

PDB_IDS_TO_TEST = ["101M", "1AH8", "1ATG", "1ARX"]

cif_parser_no_exclude = cifutils_biotite.CIFParser(exclude_crystallization_aid=False)
cif_parser_do_exclude = cifutils_biotite.CIFParser(exclude_crystallization_aid=True)


def get_digs_path(pdbid: str) -> str:  # possibly share across tests
    pdbid = pdbid.lower()
    filename = f"/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz"
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename


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
