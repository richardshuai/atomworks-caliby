from conftest import TEST_DATA_DIR, get_pdb_path

from cifutils.parser import parse
from cifutils.utils.testing import assert_same_atom_array


def test_arginine_renaming():
    correct_file = get_pdb_path("101m")
    renamed_file = TEST_DATA_DIR / "101m_arginine_nh1nh2_swapped.cif"  # Manually renamed the arginine atoms.
    result1 = parse(filename=correct_file, build_assembly=None)
    result2 = parse(filename=renamed_file, build_assembly=None, fix_arginines=True)
    assert_same_atom_array(result1["asym_unit"][0], result2["asym_unit"][0])
