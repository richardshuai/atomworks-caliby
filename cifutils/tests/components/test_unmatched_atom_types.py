import numpy as np

from tests.conftest import CIF_PARSER, TEST_DATA_DIR


def test_unmatched_atom_types():
    """
    Ensure that unmatched atom types are handled correctly. For CIFUtils, that means masking the residue with the unmatched atom with 0 occupancy.
    """
    filename = TEST_DATA_DIR / "1a8o_modified.cif"

    # Parse with cifutils_biotite
    result_dict = result_dict = CIF_PARSER.parse(filename=filename)

    # Ensure that residue 2 has no occupancy
    atom_array = result_dict["asym_unit"][0]
    residue_2 = atom_array[(atom_array.chain_id == "A") & (atom_array.res_id == 2)]
    assert np.sum(residue_2.occupancy) == 0
