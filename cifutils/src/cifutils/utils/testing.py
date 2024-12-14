"""
Utility functions to support proper matching and resolution of residue atoms in a structure.
"""

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
    annotations_to_compare: list[str] = ["chain_id", "res_name", "res_id", "atom_name", "element"],
    max_print_length: int = 10,
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

    # Compare annotations
    for annotation in annotations_to_compare:
        annot1 = arr1.get_annotation(annotation)
        annot2 = arr2.get_annotation(annotation)
        mismatch_mask = annot1 != annot2
        if np.any(mismatch_mask):
            msg = f"AtomArrays are not equivalent in `{annotation}`\n"
            arr1_mismatch = arr1[mismatch_mask][:max_print_length]  # max len to reduce length of print output
            arr2_mismatch = arr2[mismatch_mask][:max_print_length]
            msg += f"\tarr1: \n{arr1_mismatch}\n"
            msg += f"\tarr2: \n{arr2_mismatch}\n"
            raise ValueError(msg)

    # Compare coordinates
    if not np.allclose(arr1.coord, arr2.coord, equal_nan=True):
        msg = "AtomArrays are not equivalent in coordinates\n"
        mismatch_mask = np.any(~np.isclose(arr1.coord, arr2.coord, equal_nan=True), axis=1)
        arr1_mismatch = arr1[mismatch_mask][:max_print_length]
        arr2_mismatch = arr2[mismatch_mask][:max_print_length]
        msg = "AtomArrays are not equivalent in coordinates\n"
        msg += f"\tarr1: \n{arr1_mismatch}\n"
        msg += f"\tarr2: \n{arr2_mismatch}\n"
        raise ValueError(msg)
