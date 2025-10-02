"""Enums used in atomworks.io."""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

import numpy as np
from biotite.structure import AtomArray

from atomworks.io.utils import scatter
from atomworks.io.utils.selection import get_residue_starts
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
