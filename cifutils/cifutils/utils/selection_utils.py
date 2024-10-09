"""Utility functions for selecting segments of an AtomArray"""

__all__ = ["annot_start_stop_idxs"]

from biotite.structure import AtomArray
import numpy as np


def annot_start_stop_idxs(atom_array: AtomArray, annots: str | list[str], add_exclusive_stop: bool = False):
    """
    Computes the start and stop indices for segments in an AtomArray where any of the specified annotation(s) change.

    Args:
        - atom_array (AtomArray): The AtomArray to process.
        - annots (str | list[str]): The annotation(s) to consider for determining segment boundaries.
        - add_exclusive_stop (bool): If True, an exclusive stop index (the length of the AtomArray) is added to the result.

    Returns:
        - np.ndarray: An array of start and stop indices for segments where the annotations change.

    Example:
        >>> atom_array = AtomArray(...)
        >>> start_stop_idxs = annot_start_stop_idxs(
        ...     atom_array, annots="chain_id", add_exclusive_stop=True
        ... )
        >>> print(start_stop_idxs)
        [0, 5, 10, 15]
    """

    if len(atom_array) == 0:
        return np.array([], dtype=int)

    if isinstance(annots, str):
        annots = [annots]

    annots_differ = np.zeros(len(atom_array) - 1, dtype=bool)
    for annot in annots:
        annot_array = atom_array.get_annotation(annot)
        annots_differ |= annot_array[1:] != annot_array[:-1]

    start_stop_idxs = np.where(annots_differ)[0] + 1

    if add_exclusive_stop:
        return np.concatenate(([0], start_stop_idxs, [len(atom_array)]))
    return np.concatenate(([0], start_stop_idxs))


def get_residue_starts(atom_array: AtomArray, add_exclusive_stop: bool = False):
    """
    More robust version of `biotite.structure.residues.get_residue_starts` that also
    differentiates between residues resulting from different transformation ids.

    Backwards compatible with `biotite.structure.residues.get_residue_starts` if the
    `transformation_id` annotation is not present.

    References:
        - https://github.com/biotite-dev/biotite/blob/231eefed334e1d3509c1b7cb3f2bfd71d4b0eeb0/src/biotite/structure/residues.py#L35
    """
    _annots_to_check = ["chain_id", "res_name", "res_id", "ins_code", "transformation_id"]
    existing_annots = atom_array.get_annotation_categories()
    annots_to_check = [annot for annot in _annots_to_check if annot in existing_annots]
    return annot_start_stop_idxs(atom_array, annots=annots_to_check, add_exclusive_stop=add_exclusive_stop)
