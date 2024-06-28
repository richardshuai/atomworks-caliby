import numpy as np
from tests.conftest import TEST_DATA_DIR, CIF_PARSER_BIOTITE

def test_unmatched_atom_types():
    """
    Ensure that unmatched atom types are handled correctly. For cifutils_biotite, that means masking the residue with the unmathced atom with 0 occupancy.
    """
    filename = TEST_DATA_DIR / "1a8o_modified.cif"

    # Parse with cifutils_biotite
    result_dict = result_dict =  CIF_PARSER_BIOTITE.parse(filename)

    # Ensure that residue 2 has no occupancy
    atom_array = result_dict["atom_array_stack"][0]
    residue_2 = atom_array[(atom_array.chain_id == "A") & (atom_array.res_id == 2)]
    assert np.sum(residue_2.occupancy) == 0