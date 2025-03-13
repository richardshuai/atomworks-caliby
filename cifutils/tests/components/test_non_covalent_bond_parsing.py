import re

import biotite.structure as struc
import pytest
from conftest import get_pdb_path

from cifutils.constants import STRUCT_CONN_BOND_TYPES
from cifutils.parser import parse

# (pdb_id, assembly_id, expected_num_coord_bonds)
TEST_CASE_COORD = [
    ("3u8v", "3", 6),  # Contains some coordination bonds
]

TEST_CASE_HYDROG = [
    ("2k33", "1"),  # Structure doesn't matter here, an error will be thrown regardless
]


@pytest.mark.parametrize("test_case", TEST_CASE_COORD)
def test_coordination_bond_parsing(test_case: tuple):
    pdb_id, assembly_id, expected_num_coord_bonds = test_case
    path = get_pdb_path(pdb_id)

    # Parse AtomArray from CIF
    result = parse(
        filename=path,
        build_assembly=[assembly_id],
        add_bond_types_from_struct_conn=[
            "covale",
            "metalc",
            "disulf",
        ],
    )
    atom_array = result["assemblies"][assembly_id][0]

    num_coord_bonds = sum(atom_array.bonds.as_array()[:, 2] == struc.BondType.COORDINATION)

    assert num_coord_bonds == expected_num_coord_bonds


@pytest.mark.parametrize("test_case", TEST_CASE_HYDROG)
def test_hydrogen_bond_parsing(test_case: tuple):
    pdb_id, assembly_id = test_case
    path = get_pdb_path(pdb_id)

    expected_error_msg = re.escape(
        f"Invalid bond type(s) provided: { {'hydrog'} }! Valid bond types are: {STRUCT_CONN_BOND_TYPES}"
    )

    with pytest.raises(ValueError, match=expected_error_msg):
        parse(
            filename=path,
            build_assembly=[assembly_id],
            add_bond_types_from_struct_conn=[
                "covale",
                "metalc",
                "disulf",
                "hydrog",
            ],
        )
