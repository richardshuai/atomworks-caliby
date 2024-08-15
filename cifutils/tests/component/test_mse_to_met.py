import pytest
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE, assert_same_atom_array
from cifutils.cifutils_biotite.transforms.atom_array import mse_to_met
import biotite.structure as struc
import numpy as np

TEST_CASES = ["1aqc"]


def test_mse_to_met_residue():
    mse = struc.array(CIF_PARSER_BIOTITE._build_residue_atoms("MSE"))
    met = struc.array(CIF_PARSER_BIOTITE._build_residue_atoms("MET"))
    is_heavy = lambda x: ~np.isin(x.element, ["1", "H", "D", "T", 1])  # noqa
    mse_converted = mse_to_met(mse)
    assert_same_atom_array(mse_converted[is_heavy(mse_converted)], met[is_heavy(met)])


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_mse_to_met_pdb(pdb_id: str):
    path = get_digs_path(pdb_id)
    result = CIF_PARSER_BIOTITE.parse(
        filename=path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        remove_crystallization_aids=True,
        build_assembly="all",
        patch_symmetry_centers=True,
        fix_arginines=True,
        convert_mse_to_met=True,
    )
    assert result is not None  # Check if processing runs through
