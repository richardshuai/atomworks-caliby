import pytest
from cifutils.utils.atom_matching_utils import assert_same_atom_array
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE
from cifutils.utils.residue_utils import build_chem_comp_atom_list
from cifutils.transforms.atom_array import mse_to_met
import biotite.structure as struc
from cifutils.common import not_isin
from cifutils.constants import HYDROGEN_LIKE_SYMBOLS

TEST_CASES = ["1aqc"]


def test_mse_to_met_residue():
    mse = struc.array(build_chem_comp_atom_list("MSE", keep_hydrogens=True))
    met = struc.array(build_chem_comp_atom_list("MET", keep_hydrogens=True))
    is_heavy = lambda x: not_isin(x.element, HYDROGEN_LIKE_SYMBOLS)  # noqa
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
        build_assembly="all",
        patch_symmetry_centers=True,
        fix_arginines=True,
        convert_mse_to_met=True,
    )
    assert result is not None  # Check if processing runs through
