"""
Utility functions to support proper matching and resolution of residue atoms in a structure.
"""

__all__ = ["get_std_alt_atom_id_conversion"]

from functools import cache

import numpy as np

import biotite.structure as struc
from biotite.structure.atoms import AtomArray, AtomArrayStack
from cifutils.common import exists


@cache
def get_std_alt_atom_id_conversion(res_name: str) -> dict:
    """
    Get a mapping from standard atom IDs to alternative atom IDs for a given residue name.

    NOTE:
     - This is a cached function, so the biotite CCD database is only queried once per residue name.
     - This function requires the IPD's 'in-house' version of biotite, since the `alt_atom_id` field
       is not available in the standard biotite package (as of v0.41.0, date 2024-06-30)
     - This function is used for backwards compatibility with older encodings (RF2AA_ATOM36_ENCODING),
       where the alternative atom names were used instead of the standard atom names for hydrogens.
       It should not be used for new code.

    Args:
        res_name (str): The 3-letter residue name (must be in the biotite CCD database).

    Returns:
        dict: A dictionary with the following keys:
            - `std_to_alt`: A mapping from standard atom IDs to alternative atom IDs.
            - `alt_to_std`: A mapping from alternative atom IDs to standard atom IDs.
    """
    std_atom_ids = struc.info.ccd.get_from_ccd("chem_comp_atom", res_name, "atom_id")
    alt_atom_ids = struc.info.ccd.get_from_ccd("chem_comp_atom", res_name, "alt_atom_id")

    assert exists(std_atom_ids) and (
        len(std_atom_ids) > 0
    ), f"{res_name} info does not exist in biotite's CCD. Try to update it to fix this assertion."
    assert len(std_atom_ids) == len(
        alt_atom_ids
    ), f"{res_name} has {len(std_atom_ids)} standard atom ids and {len(alt_atom_ids)} alternative atom ids"

    mapping = {"std_to_alt": dict(zip(std_atom_ids, alt_atom_ids)), "alt_to_std": dict(zip(alt_atom_ids, std_atom_ids))}

    return mapping


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
):
    # If the input is a stack, only compare the first array
    if isinstance(arr1, AtomArrayStack):
        arr1 = arr1[0]
    if isinstance(arr2, AtomArrayStack):
        arr2 = arr2[0]

    # Compare lengths, down to the residue-level if necessary
    if not len(arr1) == len(arr2):
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
                    if not len(arr1_res_aa) == len(arr2_res_aa):
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
