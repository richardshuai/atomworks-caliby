"""Utility functions helpful for writings tests for AtomArray objects."""

__all__ = ["assert_same_atom_array"]


import biotite.structure as struc
import numpy as np
from biotite.structure.atoms import AtomArray, AtomArrayStack


def _get_atom_array_stats(arr: AtomArray) -> str:
    msg = f"AtomArray: {len(arr)} atoms, {struc.get_residue_count(arr)} residues, {struc.get_chain_count(arr)} chains\n"
    msg += f"\t... unique chain ids: {np.unique(arr.chain_id)}\n"
    msg += f"\t... unique residue ids: {np.unique(arr.res_id)}\n"
    msg += f"\t... unique atom types: {np.unique(arr.atom_name)}\n"
    msg += f"\t... unique elements: {np.unique(arr.element)}\n"
    return msg


def assert_same_atom_array(
    arr1: AtomArray | AtomArrayStack,
    arr2: AtomArray | AtomArrayStack,
    compare_coords: bool = True,
    compare_bonds: bool = True,
    annotations_to_compare: list[str] | None = None,
    _n_mismatches_to_show: int = 20,
) -> None:
    """Asserts that two AtomArray objects are equal.

    Args:
        arr1 (AtomArray): The first AtomArray to compare.
        arr2 (AtomArray): The second AtomArray to compare.
        compare_coords (bool, optional): Whether to compare coordinates. Defaults to True.
        compare_bonds (bool, optional): Whether to compare bonds. Defaults to True.
        annotations_to_compare (list[str] | None, optional): List of annotation categories to compare.
            Defaults to None, in which case all annotations are compared.
        _n_mismatches_to_show (int, optional): Number of mismatches to show. Defaults to 20.

    WARNING: If AtomArrayStack objects are passed, only the first array is compared.

    Raises:
        AssertionError: If the AtomArray objects are not equal.
    """
    assert isinstance(
        arr1, AtomArray | AtomArrayStack
    ), f"arr1 is not an AtomArray or AtomArrayStack but has type {type(arr1)}"
    assert isinstance(
        arr2, AtomArray | AtomArrayStack
    ), f"arr2 is not an AtomArray or AtomArrayStack but has type {type(arr2)}"

    # If the input is a stack, only compare the first array
    if isinstance(arr1, AtomArrayStack):
        arr1 = arr1[0]
    if isinstance(arr2, AtomArrayStack):
        arr2 = arr2[0]

    # Compare lengths, down to the residue-level if necessary
    if len(arr1) != len(arr2):
        msg = "AtomArrays are not the same length!\n"

        # Find the chains that are different lengths
        for chain_id in np.unique(arr1.chain_id):
            arr1_chain_aa = arr1[arr1.chain_id == chain_id]
            arr2_chain_aa = arr2[arr2.chain_id == chain_id]

            if len(arr1_chain_aa) != len(arr2_chain_aa):
                msg += f"+--------- Mismatches for chain: {chain_id} -----------+\n"
                # Find the residues that are different lengths
                for res_id in np.unique(arr1_chain_aa.res_id):
                    arr1_res_aa = arr1_chain_aa[arr1_chain_aa.res_id == res_id]
                    arr2_res_aa = arr2_chain_aa[arr2_chain_aa.res_id == res_id]

                    # Give an informative error message
                    if len(arr1_res_aa) != len(arr2_res_aa):
                        msg += f"Mismatch at residue {res_id}:\n"
                        msg += f"\tarr1: {_get_atom_array_stats(arr1_res_aa)}\n"
                        msg += f"\tarr2: {_get_atom_array_stats(arr2_res_aa)}\n"

        raise ValueError(msg)

    if compare_coords:
        assert (
            arr1.coord.shape == arr2.coord.shape
        ), f"Coord shapes do not match: {arr1.coord.shape} != {arr2.coord.shape}"
        if not np.allclose(arr1.coord, arr2.coord, equal_nan=True, atol=1e-3, rtol=1e-3):
            mismatch_idxs = np.where(arr1.coord != arr2.coord)[0]
            msg = f"Coords do not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
            for idx in mismatch_idxs[:_n_mismatches_to_show]:
                msg += f"\t{idx}: {arr1.coord[idx]} != {arr2.coord[idx]}\n"
            raise AssertionError(msg)

    if compare_bonds:
        assert arr1.bonds is not None, "arr1.bonds is None"
        assert arr2.bonds is not None, "arr2.bonds is None"
        if not np.array_equal(arr1.bonds.as_array(), arr2.bonds.as_array()):
            mismatch_idxs = np.where(arr1.bonds.as_array() != arr2.bonds.as_array())[0]
            msg = f"Bonds do not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
            for idx in mismatch_idxs[:_n_mismatches_to_show]:
                msg += f"\t{idx}: {arr1.bonds.as_array()[idx]} != {arr2.bonds.as_array()[idx]}\n"
            raise AssertionError(msg)

    if annotations_to_compare is None:
        arr1_annotation_keys = arr1.get_annotation_categories()
        arr2_annotation_keys = arr2.get_annotation_categories()
        missing_in_arr1 = set(arr2_annotation_keys) - set(arr1_annotation_keys)
        missing_in_arr2 = set(arr1_annotation_keys) - set(arr2_annotation_keys)
        assert len(missing_in_arr1) == 0, f"Annotations missing in arr1: {missing_in_arr1}"
        assert len(missing_in_arr2) == 0, f"Annotations missing in arr2: {missing_in_arr2}"
        annotations_to_compare = arr1_annotation_keys

    for annotation in annotations_to_compare:
        if annotation not in arr1.get_annotation_categories():
            raise AssertionError(f"Annotation {annotation} not in arr1.")
        if annotation not in arr2.get_annotation_categories():
            raise AssertionError(f"Annotation {annotation} not in arr2.")

        # Check if the arrays contain floating-point numbers (in which case, we allow NaN == NaN)
        if np.issubdtype(arr1.get_annotation(annotation).dtype, np.floating) and np.issubdtype(
            arr2.get_annotation(annotation).dtype, np.floating
        ):
            arrays_equal = np.array_equal(
                arr1.get_annotation(annotation), arr2.get_annotation(annotation), equal_nan=True
            )
        else:
            arrays_equal = np.array_equal(
                arr1.get_annotation(annotation), arr2.get_annotation(annotation), equal_nan=False
            )

        if not arrays_equal:
            mismatch_idxs = np.where(arr1.get_annotation(annotation) != arr2.get_annotation(annotation))[0]
            msg = (
                f"Annotation {annotation} does not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
            )
            for idx in mismatch_idxs[:_n_mismatches_to_show]:
                msg += f"\t{idx}: {arr1.get_annotation(annotation)[idx]} != {arr2.get_annotation(annotation)[idx]}\n"
            raise AssertionError(msg)
