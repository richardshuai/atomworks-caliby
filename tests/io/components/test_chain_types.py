"""
Tests for chain type assignment and annotation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from tests.io.conftest import get_pdb_path

from atomworks.io.enums import ChainType
from atomworks.io.parser import parse

# General Enum tests


def test_chain_type_to_int():
    assert ChainType.DNA.value == 3


def test_chain_type_from_int():
    assert ChainType(3) == ChainType.DNA


def test_chain_type_from_string():
    assert ChainType.as_enum("polydeoxyribonucleotide") == ChainType.DNA
    assert ChainType.as_enum("POLYDEOXYRIBONUCLEOTIDE") == ChainType.DNA
    assert ChainType.as_enum("PolyDeoxyRibonucleotide") == ChainType.DNA
    with pytest.raises(ValueError):
        ChainType.as_enum("invalid_chain_type")


def test_chain_type_get_chain_type_strings():
    assert "POLYDEOXYRIBONUCLEOTIDE" in ChainType.get_chain_type_strings()


def test_chain_type_equality():
    assert ChainType(3) == ChainType.DNA
    assert ChainType.DNA == 3
    assert ChainType.DNA == "POLYDEOXYRIBONUCLEOTIDE"


CHAIN_TYPE_TEST_CASES = [
    {
        # Simple polymer & non-polymers
        "pdb_id": "6qhp",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.POLYPEPTIDE_L,
            "C": ChainType.NON_POLYMER,  # fluoroacetic acid
            "D": ChainType.NON_POLYMER,  # fluoroacetic acid
        },
    },
    {
        # DNA and RNA, separately
        "pdb_id": "1fix",
        "chain_types": {
            "A": ChainType.RNA,
            "B": ChainType.DNA,
        },
    },
    {
        # DNA and RNA hybrid
        "pdb_id": "1d9d",
        "chain_types": {
            "A": ChainType.DNA_RNA_HYBRID,
            "B": ChainType.POLYPEPTIDE_L,
            "C": ChainType.NON_POLYMER,  # Zinc ion
            "D": ChainType.NON_POLYMER,
            "E": ChainType.NON_POLYMER,
            "F": ChainType.NON_POLYMER,  # Magnesium ion
            "G": ChainType.NON_POLYMER,
            "H": ChainType.NON_POLYMER,  # Solvent
            "I": ChainType.NON_POLYMER,  # Solvent
            "J": ChainType.NON_POLYMER,  # Solvent
            "K": ChainType.NON_POLYMER,  # Solvent
        },
    },
    {
        # Oligosaccharides
        "pdb_id": "1ivo",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.POLYPEPTIDE_L,
            "C": ChainType.POLYPEPTIDE_L,
            "D": ChainType.POLYPEPTIDE_L,
            "E": ChainType.BRANCHED,  # Oligosaccharide
            "F": ChainType.NON_POLYMER,  # Monosaccharide
            "G": ChainType.NON_POLYMER,  # Monosaccharide
            "H": ChainType.NON_POLYMER,
            "I": ChainType.NON_POLYMER,
            "J": ChainType.NON_POLYMER,
            "K": ChainType.NON_POLYMER,
            "L": ChainType.NON_POLYMER,
            "M": ChainType.NON_POLYMER,
        },
    },
    {
        # Covalently bonded ligands
        "pdb_id": "3ne7",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.NON_POLYMER,  # Nickel
            "C": ChainType.NON_POLYMER,  # CoA
            "D": ChainType.NON_POLYMER,
            "E": ChainType.NON_POLYMER,
            "F": ChainType.NON_POLYMER,
            "G": ChainType.NON_POLYMER,
            "H": ChainType.NON_POLYMER,
        },
    },
]


@pytest.mark.parametrize("test_case", CHAIN_TYPE_TEST_CASES)
def test_chain_types(test_case: dict[str, Any]):
    path = get_pdb_path(test_case["pdb_id"])
    result = parse(
        filename=path,
        build_assembly="all",
    )

    atom_array = result["assemblies"]["1"][0]  # Choose first model
    for pn_unit_id in np.unique(atom_array.pn_unit_id):
        pn_unit_atom_array = atom_array[atom_array.pn_unit_id == pn_unit_id]
        # ...check if all chains in a PN unit have the same type
        assert np.unique(pn_unit_atom_array.chain_type).size == 1

        # ...check that the type matches the expected type for chains that we care about
        if pn_unit_id.astype(str) in test_case["chain_types"]:
            # Check ChainType
            got_chain_type = ChainType.as_enum(pn_unit_atom_array.chain_type[0])
            expected_chain_type = test_case["chain_types"][pn_unit_id]
            assert (
                got_chain_type == expected_chain_type
            ), f"Mismatch for {pn_unit_id=}: {got_chain_type=}, {expected_chain_type=}"

            # Check is_polymer
            got_is_polymer = pn_unit_atom_array.is_polymer[0]
            expected_is_polymer = expected_chain_type.is_polymer()
            assert (
                got_is_polymer == expected_is_polymer
            ), f"Mismatch for {pn_unit_id=}: {got_is_polymer=}, {expected_is_polymer=}"
