"""
Utility functions to support proper matching and resolution of residue atoms in a structure.
"""

__all__ = ["get_std_alt_atom_id_conversion"]

from functools import cache

import biotite.structure as struc
import numpy as np
from biotite.structure.atoms import AtomArray

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
    arr1: AtomArray,
    arr2: AtomArray,
    annotations_to_compare: list[str] = ["chain_id", "res_name", "res_id", "atom_name", "element"],
    max_print_length: int = 10,
):
    if not len(arr1) == len(arr2):
        msg = "AtomArrays are not the same length.\n"
        msg += f"\tarr1: {_get_atom_array_stats(arr1)}\n"
        msg += f"\tarr2: {_get_atom_array_stats(arr2)}\n"
        raise ValueError(msg)

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

    if not np.allclose(arr1.coord, arr2.coord, equal_nan=True):
        mismatch_mask = np.any(~np.isclose(arr1.coord, arr2.coord, equal_nan=True), axis=1)
        arr1_mismatch = arr1[mismatch_mask][:max_print_length]
        arr2_mismatch = arr2[mismatch_mask][:max_print_length]
        msg = "AtomArrays are not equivalent in coordinates\n"
        msg += f"\tarr1: \n{arr1_mismatch}\n"
        msg += f"\tarr2: \n{arr2_mismatch}\n"
        raise ValueError(msg)
