"""
Regression tests for complex cases to ensure consistent behavior.
"""

import pickle
import pytest
from pathlib import Path
from cifutils.utils.atom_matching_utils import assert_same_atom_array
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE
from cifutils.constants import CRYSTALLIZATION_AIDS

from assertpy import assert_that

TEST_CASES = [
    "6mub",  # Symmetry center clash
    "1j8z",  # Contains misordered atoms in a residue
    "1fp7",  # Contains bonds between crystallization aids in struct_conn
    "1twr",  # Residue name not in biotite's CCD
    "6q9t",  # Contains residue `QUK` which uses a mix of `std` and `alt` atom ids
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_regression_against_stored_result(pdb_id: str):
    path = get_digs_path(pdb_id)
    result = CIF_PARSER_BIOTITE.parse(
        filename=path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        residues_to_remove=CRYSTALLIZATION_AIDS,
        patch_symmetry_centers=True,
        build_assembly="all",
        fix_arginines=True,
        convert_mse_to_met=True,
        keep_hydrogens=True,
        model=None,
    )
    assert result is not None  # Check if processing runs through

    regression_dir = Path(__file__).parent / "../data/regression_tests/"
    regression_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = regression_dir / f"{pdb_id}.pkl"

    # Uncomment the following lines to create the pickle file
    # with pickle_path.open("wb") as f:
    #     pickle.dump(result, f)

    with pickle_path.open("rb") as f:
        expected_result = pickle.load(f)

    # Check the asymmetric unit...
    assert_same_atom_array(result["asym_unit"], expected_result["asym_unit"])

    # ...the assemblies
    for assembly_id in result["assemblies"]:
        assert_same_atom_array(result["assemblies"][assembly_id], expected_result["assemblies"][assembly_id])

    # ...the ligand of interest information
    assert_that(result["ligand_info"]).is_equal_to(expected_result["ligand_info"])

    # ...the chain information
    assert_that(result["chain_info"]).is_equal_to(expected_result["chain_info"])

    # ...the extra information
    assert_that(result["extra_info"]).is_equal_to(expected_result["extra_info"])

    # ...the metadata
    assert_that(result["metadata"]).is_equal_to(expected_result["metadata"])


if __name__ == "__main__":
    pytest.main([__file__])
