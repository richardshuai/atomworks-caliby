"""Concrete implementations of common molecular design conditions."""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

import numpy as np
from biotite.structure import AtomArray
from sympy.sets.sets import true

from atomworks.constants import UNKNOWN_AA
from atomworks.io.utils import scatter
from atomworks.io.utils.atom_array import apply_and_spread
from atomworks.io.utils.atom_array_plus import AnnotationList2D
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions.annotator import ensure_annotations
from atomworks.ml.conditions.base import ConditionBase
from atomworks.ml.utils.token import get_token_starts


class Level(StrEnum):
    """
    Level-hierarchy for describing information in a molecular structure.

    Used for example to specify the level at which a `Condition` applies.
    """

    ATOM = "atom"
    TOKEN = "token"
    RESIDUE = "residue"
    CHAIN = "chain"
    MOLECULE = "molecule"
    SYSTEM = "system"

    def _get_segment_or_group(self, atom_array: "AtomArray") -> tuple[str, np.ndarray]:
        if self == Level.ATOM:
            raise ValueError("Apply and spread does not make sense for atom level.")
        if self == Level.TOKEN:
            return "segment", get_token_starts(atom_array, add_exclusive_stop=True)
        elif self == Level.RESIDUE:
            return "segment", get_residue_starts(atom_array, add_exclusive_stop=True)
        elif self == Level.CHAIN:
            chain_key = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
            return "group", atom_array.get_annotation(chain_key)
        elif self == Level.MOLECULE:
            return "group", atom_array.get_annotation("molecule_id")
        elif self == Level.SYSTEM:
            return "group", np.ones(atom_array.array_length(), dtype=np.int32)
        else:
            raise ValueError(f"Invalid level: {self}")

    def apply(self, atom_array: "AtomArray", data: np.ndarray, func: "Callable[[AtomArray], Any]") -> "Any":
        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.apply_segment_wise(grouping, data, func)
        elif strategy == "group":
            return scatter.apply_group_wise(grouping, data, func)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def spread(self, atom_array: "AtomArray", data: np.ndarray) -> np.ndarray:
        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.spread_segment_wise(grouping, data)
        elif strategy == "group":
            return scatter.spread_group_wise(grouping, data)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def apply_and_spread(
        self,
        atom_array: "AtomArray",
        data: np.ndarray,
        func: "Callable[[np.ndarray], Any]",
    ) -> np.ndarray:
        """
        Apply a function to the data and spread the result back to the original positions.
        """

        strategy, grouping = self._get_segment_or_group(atom_array)
        if strategy == "segment":
            return scatter.apply_and_spread_segment_wise(grouping, data, func)
        elif strategy == "group":
            return scatter.apply_and_spread_group_wise(grouping, data, func)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")


class Sequence(ConditionBase):
    name = "sequence"
    n_body = 1
    level = Level.RESIDUE
    dtype = str

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            # ... if annotation exists, derive the mask from it
            return cls.annotation(atom_array, default="raise") != UNKNOWN_AA
        else:
            # ... otherwise, default to an empty mask
            return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        seq = np.full(atom_array.array_length(), fill_value=UNKNOWN_AA)
        if cls.mask_name in atom_array.get_annotation_categories():
            # ... if mask exists, use it to get the sequence
            seq_mask = cls.mask(atom_array, default="raise")
            seq[seq_mask] = atom_array.res_name[seq_mask]
        return seq


class Coordinate(ConditionBase):
    name = "coordinate"
    n_body = 1
    level = Level.ATOM
    dtype = float

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        if cls.full_name in atom_array.get_annotation_categories():
            # ... if annotation exists, derive the mask from it
            return np.isfinite(cls.annotation(atom_array, default="raise")).any(axis=1)
        else:
            # ... otherwise, default to an empty mask
            return np.zeros(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        coords = np.full((atom_array.array_length(), 3), fill_value=np.nan)
        if cls.mask_name in atom_array.get_annotation_categories():
            # ... if mask exists, use it to get the coordinates
            mask = cls.mask(atom_array, default="raise")
            coords[mask] = atom_array.coord[mask]
        return coords


class Index(ConditionBase):
    name = "index"
    n_body = 1
    level = Level.RESIDUE
    dtype = int

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        return np.ones(atom_array.array_length(), dtype=bool)

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> np.ndarray:
        ensure_annotations(atom_array, "within_chain_res_idx")
        return atom_array.get_annotation("within_chain_res_idx")


class Distance(ConditionBase):
    name = "distance"
    n_body = 2
    level = Level.ATOM
    is_symmetric = True
    dtype = float

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> AnnotationList2D:
        annotation = cls.annotation(atom_array, default="generate")
        return AnnotationList2D(
            n_atoms=atom_array.array_length(),
            pairs=annotation.pairs,
            values=annotation.values > 0,
        )

    @classmethod
    def default_annotation(cls, atom_array: AtomArray) -> AnnotationList2D:
        return AnnotationList2D(
            n_atoms=atom_array.array_length(),
            pairs=np.array([], dtype=int),
            values=np.array([], dtype=cls.dtype),
        )


class NTerminus(ConditionBase):
    name = "n-terminus"
    n_body = 1
    level = Level.RESIDUE
    is_mask = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        ensure_annotations(atom_array, "is_polymer", "is_chain_start", "is_res_start", "within_chain_res_idx")

        # ... get indicator atoms for N-terminus atoms
        is_n_terminus = atom_array.is_polymer & (atom_array.within_chain_res_idx == 0) & atom_array.is_chain_start

        # ... spread to full residue
        residue_segments = np.concatenate([np.where(atom_array.is_res_start)[0], [atom_array.array_length()]])
        is_n_terminus = apply_and_spread(residue_segments, is_n_terminus, np.any)

        return is_n_terminus


class CTerminus(ConditionBase):
    name = "c-terminus"
    n_body = 1
    level = Level.RESIDUE
    is_mask = true
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> np.ndarray:
        ensure_annotations(atom_array, "is_polymer", "is_chain_start", "is_res_start", "within_chain_res_idx")

        # ... get indicator atoms for C-terminus atoms
        annotations = atom_array.get_annotation_categories()
        chain_ids = atom_array.chain_iid if "chain_iid" in annotations else atom_array.chain_id
        # ... find max within_chain_res_idx for each chain
        is_max_within_chain_res_idx = np.zeros(atom_array.array_length(), dtype=bool)
        for chain_id in np.unique(chain_ids):
            is_this_chain = chain_ids == chain_id
            max_chain_idx = np.max(atom_array.get_annotation("within_chain_res_idx")[is_this_chain])
            is_max_idx = atom_array.get_annotation("within_chain_res_idx") == max_chain_idx
            is_max_within_chain_res_idx |= is_this_chain & is_max_idx

        # ... spread to full residue
        is_c_terminus = is_max_within_chain_res_idx & atom_array.is_polymer
        residue_segments = np.concatenate([np.where(atom_array.is_res_start)[0], [atom_array.array_length()]])
        is_c_terminus = apply_and_spread(residue_segments, is_c_terminus, np.any)

        return is_c_terminus


class Chain(ConditionBase):
    name = "chain"
    n_body = 2
    level = Level.CHAIN
    is_symmetric = True
    dtype = bool

    @classmethod
    def default_mask(cls, atom_array: AtomArray) -> AnnotationList2D:
        annotations = atom_array.get_annotation_categories()
        chain_iid_annotation = "chain_iid" if "chain_iid" in annotations else "chain_id"
        chain_instance = atom_array.get_annotation(chain_iid_annotation)
        is_same_chain = chain_instance[None, :] == chain_instance[:, None]
        pairs = np.stack(np.where(is_same_chain), axis=0).T
        values = np.ones(pairs.shape[0], dtype=bool)
        return AnnotationList2D(atom_array.array_length(), pairs=pairs, values=values)
