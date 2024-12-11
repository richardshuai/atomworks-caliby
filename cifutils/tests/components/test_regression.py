"""
Regression tests for complex cases to ensure consistent behavior.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
from assertpy import assert_that
from toolz import keymap

from cifutils.constants import CRYSTALLIZATION_AIDS
from cifutils.enums import ChainType
from cifutils.utils.atom_matching_utils import assert_same_atom_array
from tests.conftest import CIF_PARSER, get_pdb_path

TEST_CASES = [
    "6mub",  # Symmetry center clash
    "1j8z",  # Contains misordered atoms in a residue
    "1fp7",  # Contains bonds between crystallization aids in struct_conn
    "1twr",  # Residue name not in biotite's CCD
    "6q9t",  # Contains residue `QUK` which uses a mix of `std` and `alt` atom ids
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_regression_against_stored_result(pdb_id: str):
    path = get_pdb_path(pdb_id)
    result = CIF_PARSER.parse(
        filename=path,
        add_missing_atoms=True,
        remove_waters=True,
        remove_ccds=CRYSTALLIZATION_AIDS,
        fix_ligands_at_symmetry_centers=True,
        build_assembly="all",
        fix_arginines=True,
        convert_mse_to_met=True,
        remove_hydrogens=False,
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
    assert_same_atom_array(
        result["asym_unit"],
        expected_result["asym_unit"],
        annotations_to_compare=["chain_id", "res_name", "res_id", "atom_name"],
    )

    # ... the assemblies
    for assembly_id in result["assemblies"]:
        assert_same_atom_array(
            result["assemblies"][assembly_id],
            expected_result["assemblies"][assembly_id],
            annotations_to_compare=["chain_id", "res_name", "res_id", "atom_name"],
        )

    # ... the ligand of interest information
    assert_that(result["ligand_info"]).is_equal_to(expected_result["ligand_info"])

    # ... the chain information
    assert set(result["chain_info"].keys()) == set(expected_result["chain_info"].keys())
    for chain in result["chain_info"]:
        # LEGACY HACK
        rename = {"residue_name_list": "res_name", "residue_id_list": "res_id", "type": "chain_type"}
        expected_result["chain_info"][chain] = keymap(lambda x: rename.get(x, x), expected_result["chain_info"][chain])

        got = result["chain_info"][chain]["chain_type"]
        # expected = expected_result["chain_info"][chain]["chain_type"]
        expected = ChainType.as_enum(expected_result["chain_info"][chain]["chain_type"])
        assert got == expected, f"Chain info for {chain=} does not match: {got} != {expected}"

        got = result["chain_info"][chain]["res_name"]
        expected = expected_result["chain_info"][chain]["res_name"]
        assert np.array_equal(got, expected), f"Chain info for {chain=} does not match: {got} != {expected}"

        got = result["chain_info"][chain]["res_id"]
        expected = expected_result["chain_info"][chain]["res_id"]
        assert np.array_equal(got, expected), f"Chain info for {chain=} does not match: {got} != {expected}"

        got = result["chain_info"][chain]["is_polymer"]
        expected = expected_result["chain_info"][chain]["is_polymer"]
        assert got == expected, f"Chain info for {chain=} does not match: {got} != {expected}"

    # ... the extra information
    assert_that(result["extra_info"]).is_equal_to(expected_result["extra_info"])

    # ... the metadata
    assert_that(result["metadata"]).is_equal_to(expected_result["metadata"])


if __name__ == "__main__":
    pytest.main([__file__])
