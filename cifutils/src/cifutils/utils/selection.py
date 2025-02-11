"""Utility functions for selecting segments of an AtomArray"""

__all__ = ["annot_start_stop_idxs", "get_annotation", "get_residue_starts"]

from abc import ABC, abstractmethod
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, AtomArrayStack


def annot_start_stop_idxs(
    atom_array: AtomArray | AtomArrayStack, annots: str | list[str], add_exclusive_stop: bool = False
) -> np.ndarray:
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
    if atom_array.array_length() == 0:
        return np.array([], dtype=int)

    if isinstance(annots, str):
        annots = [annots]

    annots_differ = np.zeros(atom_array.array_length() - 1, dtype=bool)
    for annot in annots:
        annot_array = atom_array.get_annotation(annot)
        annots_differ |= annot_array[1:] != annot_array[:-1]

    start_stop_idxs = np.where(annots_differ)[0] + 1

    if add_exclusive_stop:
        return np.concatenate(([0], start_stop_idxs, [atom_array.array_length()]))
    return np.concatenate(([0], start_stop_idxs))


def get_residue_starts(atom_array: AtomArray | AtomArrayStack, add_exclusive_stop: bool = False) -> np.ndarray:
    """Get the start (and optionally stop) indices of residues in an AtomArray.

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


def get_annotation(atom_array: AtomArray | AtomArrayStack, annot: str, default: Any = None) -> np.ndarray:
    """Get the annotation for an AtomArray or AtomArrayStack if it exists, otherwise return the default value."""
    if annot in atom_array.get_annotation_categories():
        return atom_array.get_annotation(annot)
    return default


class SegmentSlice(ABC):
    """
    Abstract base class for slicing segments of an AtomArray or AtomArrayStack.

    Provides functionality analogous to Python's built-in slice object but operates on structural segments
    (e.g., residues or chains indices) rather than individual atom indices. To subclass, implement the
    `_get_segment_bounds` method to return the start and stop indices of the segments.

    For example:
        - to slice residues 0-2: `atom_array[ResIdxSlice(0, 2)]`
        - to slice chains 0-1: `atom_array[ChainIdxSlice(0, 2)]`
        - to slice to the last two residues: `atom_array[ResIdxSlice(-2, None)]`

    Args:
        - start (int | None): The starting segment index. If None, starts from the beginning.
        - stop (int | None): The ending segment index (exclusive). If None, continues to the end.
    """

    def __init__(self, start: int | None = None, stop: int | None = None):
        self.start = start
        self.stop = stop

    @abstractmethod
    def _get_segment_bounds(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        pass

    def __call__(self, atom_array: AtomArray | AtomArrayStack) -> slice:
        """
        Creates a slice object for the specified segment range in the atom array.

        Args:
            - atom_array (AtomArray | AtomArrayStack): The structure to slice.

        Returns:
            - slice: A slice object that can be used to index the atom array.
        """
        seg_bounds = self._get_segment_bounds(atom_array)
        n_segments = len(seg_bounds) - 1
        if n_segments < 0:
            # edge case: empty array
            return slice(0, 0)

        seg_slice = slice(self.start, self.stop)
        start, stop, _ = seg_slice.indices(n_segments)

        return slice(seg_bounds[start], seg_bounds[stop])


class ResIdxSlice(SegmentSlice):
    """
    Slice atoms by residue indices.

    Allows for selecting ranges of residues using Python slice-like syntax. Each residue is considered
    as a segment, defined by changes in chain_id, res_name, res_id, ins_code, or transformation_id.

    Example:
        >>> atom_array = AtomArray(...)
        >>> res_slice = ResIdxSlice(0, 2)
        >>> sliced_atom_array = atom_array[
        ...     res_slice
        ... ]  # <-- returns a new AtomArray with the first two residues
    """

    def _get_segment_bounds(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        return get_residue_starts(atom_array, add_exclusive_stop=True)


class ChainIdxSlice(SegmentSlice):
    """
    Slice atoms by chain indices.

    Allows for selecting ranges of chains using Python slice-like syntax. Each chain is considered
    as a segment, defined by changes in the chain_id annotation.

    Example:
        >>> atom_array = AtomArray(...)
        >>> chain_slice = ChainIdxSlice(0, 1)
        >>> sliced_atom_array = atom_array[
        ...     chain_slice
        ... ]  # <-- returns a new AtomArray with the first chain
    """

    def _get_segment_bounds(self, atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
        return struc.get_chain_starts(atom_array, add_exclusive_stop=True)
