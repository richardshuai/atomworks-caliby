"""
Tests for chain type assignment and annotation.
"""

from __future__ import annotations

from typing import Any

import pytest
from cifutils.enums import ChainType
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE
import numpy as np

# General Enum tests


def test_chain_type_to_int():
    assert ChainType.DNA.value == 3


def test_chain_type_from_int():
    assert ChainType(3) == ChainType.DNA


def test_chain_type_from_string():
    assert ChainType.from_string("polydeoxyribonucleotide") == ChainType.DNA
    assert ChainType.from_string("POLYDEOXYRIBONUCLEOTIDE") == ChainType.DNA
    assert ChainType.from_string("PolyDeoxyRibonucleotide") == ChainType.DNA
    with pytest.raises(ValueError):
        ChainType.from_string("invalid_chain_type")


def test_chain_type_get_chain_type_strings():
    assert "polydeoxyribonucleotide" in ChainType.get_chain_type_strings()


def test_chain_type_equality():
    assert ChainType.DNA == ChainType(3)
    assert ChainType.DNA == 3
    assert ChainType.DNA == "polydeoxyribonucleotide"


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
            "F": ChainType.NON_POLYMER,  # Magnesium ion
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
            "E": ChainType.NON_POLYMER,  # Oligosaccharide
            "F": ChainType.NON_POLYMER,  # Monosaccharide
            "G": ChainType.NON_POLYMER,  # Monosaccharide
        },
    },
    {
        # Covalently bonded ligands
        "pdb_id": "3ne7",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.NON_POLYMER,  # Nickel
            "C": ChainType.NON_POLYMER,  # CoA
        },
    },
]


@pytest.mark.parametrize("test_case", CHAIN_TYPE_TEST_CASES)
def test_chain_types(test_case: dict[str, Any]):
    path = get_digs_path(test_case["pdb_id"])
    result = CIF_PARSER_BIOTITE.parse(
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
            assert ChainType(pn_unit_atom_array.chain_type[0]) == test_case["chain_types"][pn_unit_id.astype(str)]
