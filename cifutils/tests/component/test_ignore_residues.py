from __future__ import annotations
import pytest
import numpy as np
from cifutils.cifutils_biotite.constants import CRYSTALLIZATION_AIDS, AF3_EXCLUDED_LIGANDS
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE

CRYSTALLIZATION_AIDS_PDB_IDS_TO_TEST = ["101M", "1AH8", "1ATG", "1ARX"]


@pytest.mark.parametrize("pdbid", CRYSTALLIZATION_AIDS_PDB_IDS_TO_TEST)
def test_remove_crystallization_aids(pdbid: str):
    # Not excluding crystallization aids
    out1 = CIF_PARSER_BIOTITE.parse(
        filename=get_digs_path(pdbid), 
        residues_to_remove = []
    )
    assert np.any(
        np.isin(out1["atom_array_stack"].res_name, CRYSTALLIZATION_AIDS)
    ), "No crystallization aids found when not excluding."

    # Excluding crystallization aids
    out2 = CIF_PARSER_BIOTITE.parse(
        filename=get_digs_path(pdbid), 
        residues_to_remove = CRYSTALLIZATION_AIDS
    )
    assert not np.any(
        np.isin(out2["atom_array_stack"].res_name, CRYSTALLIZATION_AIDS)
    ), "Crystallization aids found when excluding."


EXCLUDED_LIGAND_PDB_IDS_TO_TEST = ["1A7S", "1AIJ"]


@pytest.mark.parametrize("pdbid", EXCLUDED_LIGAND_PDB_IDS_TO_TEST)
def test_remove_af3_excluded_ligands(pdbid: str):
    # Not excluding AF3 ligands
    out1 = CIF_PARSER_BIOTITE.parse(
        filename=get_digs_path(pdbid), 
        residues_to_remove = []
    )
    assert np.any(
        np.isin(out1["atom_array_stack"].res_name, AF3_EXCLUDED_LIGANDS)
    ), "No AF3 excluded ligands found when not excluding."

    # Excluding AF3 ligands
    out2 = CIF_PARSER_BIOTITE.parse(
        filename=get_digs_path(pdbid), 
        residues_to_remove = AF3_EXCLUDED_LIGANDS
    )
    assert not np.any(
        np.isin(out2["atom_array_stack"].res_name, AF3_EXCLUDED_LIGANDS)
    ), "AF3 excluded ligands found when excluding."
