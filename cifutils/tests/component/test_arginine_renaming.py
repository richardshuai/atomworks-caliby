from tests.conftest import get_digs_path, TEST_DATA_DIR, assert_same_atom_array, CIF_PARSER


def test_arginine_renaming():
    correct_file = get_digs_path("101m")
    renamed_file = TEST_DATA_DIR / "101m_arginine_nh1nh2_swapped.cif"  # Manually renamed the arginine atoms.
    result1 = CIF_PARSER.parse(correct_file, build_assembly=None)
    result2 = CIF_PARSER.parse(renamed_file, build_assembly=None, fix_arginines=True)
    assert_same_atom_array(result1["atom_array"], result2["atom_array"])
