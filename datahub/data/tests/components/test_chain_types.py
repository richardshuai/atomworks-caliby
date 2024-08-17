"""
Tests for chain type processing.
"""

from __future__ import annotations

from typing import Any

import pytest

from data.data_constants import ChainType
from data.tests.conftest import DATA_PREPROCESSOR
from data.tests.test_cases import CHAIN_TYPE_TEST_CASES

# General Enum tests


def test_chain_type_to_int():
    assert ChainType.to_int(ChainType.DNA) == 3


def test_chain_type_from_int():
    assert ChainType.from_int(3) == ChainType.DNA


def test_chain_type_from_string():
    assert ChainType.from_string("polydeoxyribonucleotide") == ChainType.DNA
    assert ChainType.from_string("POLYDEOXYRIBONUCLEOTIDE") == ChainType.DNA
    assert ChainType.from_string("PolyDeoxyRibonucleotide") == ChainType.DNA
    with pytest.raises(ValueError):
        ChainType.from_string("invalid_chain_type")


def test_chain_type_get_chain_type_strings():
    assert "polydeoxyribonucleotide" in ChainType.get_chain_type_strings()


def test_chain_type_equality():
    assert ChainType.DNA == ChainType.from_int(3)
    assert ChainType.DNA == 3
    assert not (ChainType.DNA == "DNA")


def test_chain_type_hash():
    assert hash(ChainType.DNA) == hash("ChainType.dna")


@pytest.mark.parametrize("test_case", CHAIN_TYPE_TEST_CASES)
def test_chain_types(test_case: dict[str, Any]):
    pdb_id = test_case["pdb_id"]

    rows = DATA_PREPROCESSOR.get_rows(pdb_id)

    for row in rows:
        chain_id = row["q_pn_unit_id"].split(",")[0]  # all chains in a PN unit have the same type
        chain_type = ChainType.from_int(row["q_pn_unit_type"])
        if chain_id in test_case["chain_types"]:
            assert chain_type == test_case["chain_types"][chain_id]
