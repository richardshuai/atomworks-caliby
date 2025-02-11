import biotite.structure as struc
import numpy as np
import pytest

from cifutils.utils.selection import ChainIdxSlice, ResIdxSlice, get_residue_starts


@pytest.fixture
def basic_atom_array() -> struc.AtomArray:
    """Creates a basic atom array with multiple residues across different chains."""
    return struc.array(
        [
            # Residue 1, Chain A
            struc.Atom(np.array([1, 1, 1]), chain_id="A", res_id=1, res_name="ALA", atom_name="N"),
            struc.Atom(np.array([1, 1, 2]), chain_id="A", res_id=1, res_name="ALA", atom_name="CA"),
            # Residue 2, Chain A
            struc.Atom(np.array([2, 1, 1]), chain_id="A", res_id=2, res_name="GLY", atom_name="N"),
            struc.Atom(np.array([2, 1, 2]), chain_id="A", res_id=2, res_name="GLY", atom_name="CA"),
            # Residue 3, Chain B
            struc.Atom(np.array([3, 1, 1]), chain_id="B", res_id=3, res_name="VAL", atom_name="N"),
            struc.Atom(np.array([3, 1, 2]), chain_id="B", res_id=3, res_name="VAL", atom_name="CA"),
        ]
    )


def test_get_residue_starts_basic(basic_atom_array: struc.AtomArray) -> None:
    """Test that get_residue_starts correctly identifies the start of each residue."""
    starts = get_residue_starts(basic_atom_array)
    assert len(starts) == 3
    assert list(starts) == [0, 2, 4]


def test_get_residue_starts_complex():
    # fmt: off
    atom_array = struc.array([
        struc.Atom(np.array([44.869,     8.188,    36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  element="7",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([45.024,     7.456,    34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([44.142,     6.714,    34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", element="8",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.669,     8.171,    36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.812,     8.982,    38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.152,     8.296,    39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.479,     9.3  ,    40.792 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="SD", element="16", charge=0,  transformation_id="1"),
        struc.Atom(np.array([43.232,     8.184,    42.102 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CE", element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.46 ,     8.724,    36.151 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="C",  element="6",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([42.339,     9.907,    35.831 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O",  element="8",  charge=0,  transformation_id="1"),
        struc.Atom(np.array([58.656483, 34.763695, 36.104 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="N",  element="7",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.212917, 35.263927, 34.948 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CN", element="6",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([60.296505, 34.87109 , 34.487 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="O1", element="8",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.271206, 33.732964, 36.897 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CA", element="6",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([58.49736 , 33.451305, 38.2   ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CB", element="6",  charge=0,  transformation_id="2"),
        struc.Atom(np.array([59.42145 , 33.22273 , 39.368 ]), chain_id="A", res_id=1, ins_code="", res_name="FME", hetero=False, atom_name="CG", element="6",  charge=0,  transformation_id="2"),
    ])
    # fmt: on

    assert len(get_residue_starts(atom_array)) == 2


@pytest.mark.parametrize(
    "slice_args,expected_res_ids",
    [
        ((0, 2), [1, 1, 2, 2]),  # First two residues
        ((-2, None), [2, 2, 3, 3]),  # Last two residues
        ((1, 2), [2, 2]),  # Middle residue
        ((), [1, 1, 2, 2, 3, 3]),  # Full selection
    ],
)
def test_residx_slice(basic_atom_array: struc.AtomArray, slice_args: tuple, expected_res_ids: list[int]) -> None:
    """
    Test ResIdxSlice with various slicing parameters.

    Args:
        - basic_atom_array: Fixture providing test atom array
        - slice_args: Tuple of (start, stop) indices for slicing
        - expected_res_ids: Expected residue IDs after slicing
    """
    sliced = basic_atom_array[ResIdxSlice(*slice_args)]
    assert list(sliced.res_id) == expected_res_ids


@pytest.mark.parametrize(
    "slice_args,expected_chain_ids",
    [
        ((0, 1), ["A", "A", "A", "A"]),  # First chain
        ((0, 2), ["A", "A", "A", "A", "B", "B"]),  # First chain
        ((1, None), ["B", "B"]),  # Last chain
        ((0, 1), ["A", "A", "A", "A"]),  # Single chain
        ((), ["A", "A", "A", "A", "B", "B"]),  # Full selection
        ((None, -1), ["A", "A", "A", "A"]),  # All but last chain
    ],
)
def test_chainidx_slice(basic_atom_array: struc.AtomArray, slice_args: tuple, expected_chain_ids: list[str]) -> None:
    """
    Test ChainIdxSlice with various slicing parameters.

    Args:
        - basic_atom_array: Fixture providing test atom array
        - slice_args: Tuple of (start, stop) indices for slicing
        - expected_chain_ids: Expected chain IDs after slicing
    """
    sliced = basic_atom_array[ChainIdxSlice(*slice_args)]
    assert list(sliced.chain_id) == expected_chain_ids


def test_slice_behavior(basic_atom_array: struc.AtomArray) -> None:
    """Test slice behavior with out-of-bounds indices."""
    # Out of bounds slices should return empty arrays, not raise errors
    assert len(basic_atom_array[ResIdxSlice(10, 20)]) == 0
    assert len(basic_atom_array[ChainIdxSlice(10, 20)]) == 0

    # Negative indices should work as expected
    assert list(basic_atom_array[ResIdxSlice(-1, None)].res_id) == [3, 3]
    assert list(basic_atom_array[ChainIdxSlice(-1, None)].chain_id) == ["B", "B"]


def test_edge_cases() -> None:
    """Test edge cases with empty and single-atom arrays."""
    # Empty array
    empty_array = struc.AtomArray(0)
    assert len(empty_array[ResIdxSlice(0, 1)]) == 0
    assert len(empty_array[ChainIdxSlice(0, 1)]) == 0

    # Single atom array
    single_atom = struc.array([struc.Atom(np.array([1, 1, 1]), chain_id="A", res_id=1, res_name="ALA", atom_name="N")])
    assert len(single_atom[ResIdxSlice(0, 1)]) == 1
    assert len(single_atom[ChainIdxSlice(0, 1)]) == 1
    assert list(single_atom[ResIdxSlice(0, 1)].res_id) == [1]
    assert list(single_atom[ChainIdxSlice(0, 1)].chain_id) == ["A"]
