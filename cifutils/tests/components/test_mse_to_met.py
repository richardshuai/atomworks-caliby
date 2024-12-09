import os

import pytest

from cifutils.common import not_isin
from cifutils.constants import CCD_MIRROR_PATH, HYDROGEN_LIKE_SYMBOLS
from cifutils.transforms.atom_array import mse_to_met
from cifutils.utils.atom_matching_utils import assert_same_atom_array
from cifutils.utils.ccd import get_ccd_component
from tests.conftest import CIF_PARSER_BIOTITE, get_pdb_path

TEST_CASES = ["1aqc"]


@pytest.mark.parametrize("ccd_mirror_path", [CCD_MIRROR_PATH, None])
def test_mse_to_met_residue(ccd_mirror_path: os.PathLike | None):
    # Test with local CCD data
    mse = get_ccd_component("MSE", ccd_mirror_path=ccd_mirror_path)
    met = get_ccd_component("MET", ccd_mirror_path=ccd_mirror_path)
    is_heavy = lambda x: not_isin(x.element, HYDROGEN_LIKE_SYMBOLS)  # noqa
    mse_converted = mse_to_met(mse)
    assert_same_atom_array(mse_converted[is_heavy(mse_converted)], met[is_heavy(met)])


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_mse_to_met_pdb(pdb_id: str):
    path = get_pdb_path(pdb_id)
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
