from cifutils.utils.atom_matching_utils import assert_same_atom_array
from tests.conftest import get_digs_path, TEST_DATA_DIR, CIF_PARSER_BIOTITE


def test_arginine_renaming():
    correct_file = get_digs_path("101m")
    renamed_file = TEST_DATA_DIR / "101m_arginine_nh1nh2_swapped.cif"  # Manually renamed the arginine atoms.
    result1 = CIF_PARSER_BIOTITE.parse(filename=correct_file, build_assembly=None)
    result2 = CIF_PARSER_BIOTITE.parse(filename=renamed_file, build_assembly=None, fix_arginines=True)
    assert_same_atom_array(result1["asym_unit"][0], result2["asym_unit"][0])
