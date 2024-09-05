"""Transforms on atom arrays."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable, Iterator, Literal

import biotite.structure as struc
import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from cifutils.enums import ChainType
from cifutils.utils import get_1_from_3_letter_code, get_3_from_1_letter_code

from datahub.preprocessing.constants import SUPPORTED_CHAIN_TYPES
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.base import Transform
from datahub.utils.numpy import not_isin
from datahub.utils.token import (
    get_token_count,
    get_token_starts,
    spread_token_wise,
)

logger = logging.getLogger(__name__)


# Convenience utils
def get_chain_instance_starts(array: AtomArray, add_exclusive_stop: bool = False) -> np.ndarray:
    """
    Get indices for an atom array, each indicating the beginning of a new chain instance (chain_iid).

    Inspired by `biotite.strucutre.get_chain_starts`.

    Args:
    - atom_array (AtomArray): The atom array to get the chain_iid starts from.
    - add_exclusive_stop (bool, optional): If True, add an exclusive stop to the chain_iid starts for the last chain instance. Defaults to False.

    Returns:
    - np.ndarray: An array of indices indicating the beginning of each chain instance.
    """
    # This mask is 'true' at indices where the chain_iid changes
    chain_iid_changes = array.chain_iid[1:] != array.chain_iid[:-1]

    # Convert mask to indices
    # Add 1, to shift the indices from the end of a residue
    # to the start of a new chain instance
    chain_iid_starts = np.where(chain_iid_changes)[0] + 1

    # The first chain instance is not included yet -> Insert '[0]'
    if add_exclusive_stop:
        return np.concatenate(([0], chain_iid_starts, [array.array_length()]))
    else:
        return np.concatenate(([0], chain_iid_starts))


def chain_instance_iter(array: AtomArray) -> Iterator[AtomArray]:
    """
    Returns an iterator over the chain instances (chain_iid) in the atom array.
    This will match `biotite.structure.chain_iter` in the case where there are no transformations.
    """
    # The exclusive stop is appended to the residue starts
    starts = get_chain_instance_starts(array, add_exclusive_stop=True)
    return struc.segments.segment_iter(array, starts)


def atom_id_to_atom_idx(atom_array: AtomArray, atom_id: int) -> int:
    """Convert an atom ID to an atom index in the given array."""
    atom_idx = np.where(atom_array.atom_id == atom_id)[0]
    assert len(atom_idx) == 1, f"Expected 1 index for atom_id {atom_id}, got {atom_idx}"
    return atom_idx[0]


def atom_id_to_token_idx(atom_array: AtomArray, atom_id: int) -> int:
    """Convert an atom ID to a token index in the given array."""
    atom_idx = atom_id_to_atom_idx(atom_array, atom_id)

    # get the sorted token start idxs
    token_start_idxs = get_token_starts(atom_array)

    # the atom's token_idx is the matching or next lower token
    token_idx = np.searchsorted(token_start_idxs, atom_idx, side="right") - 1

    return token_idx


def apply_and_spread_residue_wise(
    atom_array: AtomArray, data: np.ndarray, function: Callable[[np.ndarray], np.ScalarType], axis: int | None = None
) -> np.ndarray:
    """Apply a function residue wise and then spread the result to the atoms."""
    return struc.spread_residue_wise(atom_array, struc.apply_residue_wise(atom_array, data, function, axis))


def apply_and_spread_chain_wise(
    atom_array: AtomArray, data: np.ndarray, function: Callable[[np.ndarray], np.ScalarType], axis: int | None = None
) -> np.ndarray:
    """Apply a function chain wise and then spread the result to the atoms."""
    return struc.spread_chain_wise(atom_array, struc.apply_chain_wise(atom_array, data, function, axis))


class AddMoleculeSymmetricIdAnnotation(Transform):
    """
    Adds the `molecule_symmetric_id` annotation to the AtomArray.
    For a molecule, the symmetric_id is a unique integer within the set of molecules that share the same molecule_entity.

    Example:
    - If molecule_entity 0 has 3 molecules, they will have symmetric_ids 0, 1, 2.
    """

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["molecule_entity", "molecule_iid"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Add the molecule_symmetric_id annotation to the AtomArray
        atom_array.add_annotation("molecule_symmetric_id", dtype=np.uint16)

        molecule_iids = np.unique(atom_array.molecule_iid)
        molecule_entity_counts = {}
        # Loop through every molecule
        for molecule_iid in molecule_iids:
            mask = atom_array.molecule_iid == molecule_iid

            # Get the molecule_entity (same for all atoms in the molecule)
            molecule_entity = atom_array.molecule_entity[mask][0]

            # Check whether the molecule_entity has been seen before
            if molecule_entity in molecule_entity_counts:
                molecule_entity_counts[molecule_entity] += 1
            else:
                molecule_entity_counts[molecule_entity] = 0

            # Assign a 0-indexed symmetric_id to the molecule
            symmetric_id = molecule_entity_counts[molecule_entity]
            atom_array.molecule_symmetric_id[mask] = symmetric_id

        data["atom_array"] = atom_array
        return data


class RenumberNonPolymerResidueIdx(Transform):
    """
    Re-numbers non-polymer residue indices to be one-indexed, similar to polymer residues.

    This transformation ensures that non-polymer residue indices start from 1, providing a consistent
    indexing scheme across both polymer and non-polymer residues. It addresses the issue where non-polymer
    residue indices may start at "101", which can lead to non-deterministic behavior.

    Note:
        The renumbering is applied to each non-polymer chain independently, ensuring that the indices
        are continuous and start from 1 for each chain.

    Returns:
        - data (dict): The updated data dictionary containing the modified atom_array with renumbered
            non-polymer residue indices.
    """

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_iid", "res_id", "is_polymer"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Get the non-polymer chain full IDs
        non_polymer_mask = ~atom_array.is_polymer
        non_polymer_chain_iids = np.unique(atom_array.chain_iid[non_polymer_mask])

        # Loop through every non-polymer chain, renumbering the residues
        for chain_iid in non_polymer_chain_iids:
            chain_mask = atom_array.chain_iid == chain_iid
            num_residues = struc.get_residue_count(atom_array[chain_mask])
            renumbered_res_ids = np.arange(1, num_residues + 1)  # 1-indexed
            atom_array.res_id[chain_mask] = struc.spread_residue_wise(atom_array[chain_mask], renumbered_res_ids)

        data["atom_array"] = atom_array
        return data


def get_within_poly_res_idx(atom_array: AtomArray) -> np.ndarray:
    # Add annotation, where we default to -1 for residues that are not within a polymer
    within_poly_res_idx = np.full(len(atom_array), -1, dtype=np.int16)

    # Filter to polymers
    polymer_atom_array = atom_array[atom_array.is_polymer]  # NOTE: This creates a COPY of the atom array! Danger!

    # Loop through ever unique chain_iid (which for polymers, is the same as the pn_unit_iid)
    for chain_iid in np.unique(polymer_atom_array.chain_iid):
        chain_mask = atom_array.chain_iid == chain_iid

        # Spread residue-wise
        new_res_idx = struc.spread_residue_wise(atom_array[chain_mask], np.arange(0, np.sum(chain_mask)))

        # Update the atom_array with the generated res_ids, indexing into the full atom array
        within_poly_res_idx[chain_mask] = new_res_idx

    return within_poly_res_idx


def get_within_group_res_idx(atom_array: AtomArray, group_by: str) -> np.ndarray:
    """
    Get the within-group residue index for the atom array.
        - Groups do not need to be contiguous.
        - Groups are defined by the unique values of the `group_by` annotation.
    """
    # Add annotation, where we default to -1 for residues that are not within a group
    within_group_res_idx = np.empty(len(atom_array), dtype=np.int32)

    group_annotation = atom_array.get_annotation(group_by)

    for group_id in np.unique(group_annotation):
        group_mask = group_annotation == group_id
        in_group_res_idx = struc.spread_residue_wise(atom_array[group_mask], np.arange(0, np.sum(group_mask)))
        within_group_res_idx[group_mask] = in_group_res_idx

    return within_group_res_idx


def get_within_group_atom_idx(atom_array: AtomArray, group_by: str) -> np.ndarray:
    """
    Get the within-group atom index for the atom array.
        - Groups do not need to be contiguous.
        - Groups are defined by the unique values of the `group_by` annotation.
    """
    within_group_atom_idx = np.empty(len(atom_array), dtype=np.int32)

    group_annotation = atom_array.get_annotation(group_by)

    for group_id in np.unique(group_annotation):
        group_mask = group_annotation == group_id
        in_group_atom_idx = np.arange(0, np.sum(group_mask))
        within_group_atom_idx[group_mask] = in_group_atom_idx

    return within_group_atom_idx


def get_within_entity_idx(
    atom_array: AtomArray, level: Literal["chain", "pn_unit", "molecule"]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Get the within-entity instance index for the atom array.
        - Allowed levels are "chain", "pn_unit", or "molecule".
        - Entities do not need to be contiguous.
        - Entities are defined by the unique values of the `{level}_entity` annotation.

    Args:
        - atom_array (AtomArray): The atom array to process.
        - level (Literal["chain", "pn_unit", "molecule"]): The level at which to calculate the within-entity index.

    Returns:
        - np.ndarray: An array of within-entity instance indices for each atom in the atom array.

    Example:
        >>> import biotite.structure as struc
        >>> atom_array = struc.AtomArray(7)
        >>> atom_array.set_annotation("chain_iid", ["A", "A", "B", "C", "D", "D", "E"])
        >>> atom_array.set_annotation("chain_entity", ["1", "1", "1", "1", "2", "2", "2"])
        >>> iids, within_entity_idx = get_within_entity_idx(atom_array, level="chain")
        >>> print(within_entity_idx)
        [0 0 1 2 0 0 1]
        >>> print(iids)
        ['A' 'B' 'C'] ['D' 'E']
    """
    within_entity_idx = np.empty(len(atom_array), dtype=np.int32)

    entity_annotation = atom_array.get_annotation(f"{level}_entity")
    instance_annotation = atom_array.get_annotation(f"{level}_iid")

    iids = []
    for entity_id in np.unique(entity_annotation):
        entity_mask = entity_annotation == entity_id

        in_entity_iids, in_entity_instance_idx = np.unique(instance_annotation[entity_mask], return_inverse=True)
        iids.append(in_entity_iids)
        within_entity_idx[entity_mask] = in_entity_instance_idx

    return iids, within_entity_idx


class AddWithinPolyResIdxAnnotation(Transform):
    """
    Adds the `within_poly_res_idx` (within polymer residue index) annotation to the AtomArray.

    For polymers, the `within_poly_res_idx` is a zero-indexed, continuous residue index within the chain.
    For non-polymers, the `within_poly_res_idx` is set to -1. This annotation is later used to index into the
    MSA, as it remains consistent with MSA indices even after cropping the AtomArray.

    Note:
        The `within_poly_res_idx` is zero-indexed, since it is used as an index into the MSA. In contrast,
        the `res_id` annotation (derived from the mmCIF file) is one-indexed. We generate `within_poly_res_idx`
        from scratch rather than inferring from `res_id` to avoid any mmCIF annotation errors.

    """

    incompatible_previous_transforms = [
        "CropContiguousLikeAF3",
        "CropSpatialLikeAF3",
    ]  # cropping changes the residue indices

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_iid"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        within_poly_res_idx = get_within_poly_res_idx(atom_array, group_by="chain_iid")
        within_poly_res_idx[~atom_array.is_polymer] = -1
        atom_array.set_annotation("within_poly_res_idx", within_poly_res_idx)

        data["atom_array"] = atom_array
        return data


class RemoveUnresolvedPNUnits(Transform):
    """
    Filters PN units that have all unresolved atoms (i.e., atoms with occupancy 0) from the AtomArray.
    Can be applied before or after croppping, since cropping may lead to PN units with all unresolved atoms that were previously not entirely unresolved.
    At training time, these unresolved PN units provide minimal value and can lead to errors in the model.
    """

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_iid", "occupancy"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Get the PN units with resolved atoms
        pn_units_with_resolved_atoms = np.unique(atom_array.pn_unit_iid[atom_array.occupancy != 0])

        # Restrict the AtomArray to only include PN units with at least one resolved atom
        atom_array = atom_array[np.isin(atom_array.pn_unit_iid, pn_units_with_resolved_atoms)]

        data["atom_array"] = atom_array
        return data


class RemoveUnsupportedChainTypes(Transform):
    """
    Filter out chains with unsupported chain types from the AtomArray.
    Additionally, asserts that none of the query pn_units are of an unsupported chain type (in which case they should have been filtered out upstream, otherwise our example is not valid).
    """

    requires_previous_transforms = []

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array", "query_pn_unit_iids", "pdb_id"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_type", "pn_unit_iid"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        # We first assert that none of the query pn_units are of an unsupported chain type, which means the example should have been filtered out upstream
        query_pn_unit_chain_types = np.unique(
            atom_array.chain_type[np.isin(atom_array.pn_unit_iid, data["query_pn_unit_iids"])]
        )
        assert np.all(
            np.isin(query_pn_unit_chain_types, SUPPORTED_CHAIN_TYPES)
        ), f"{data['pdb_id']}: Query PN unit has an unsupported chain type: {query_pn_unit_chain_types}"

        # Then, we filter out chains with unsupported chain types
        data["atom_array"] = atom_array[np.isin(atom_array.chain_type, SUPPORTED_CHAIN_TYPES)]
        return data


class RemoveHydrogens(Transform):
    """
    Remove hydrogens from the atom array.
    """

    def __init__(self, hydrogen_names: tuple | list = ("1", "H", "D", "T")):
        self.hydrogen_names = hydrogen_names

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        is_heavy = atom_array.element != 1
        is_heavy &= not_isin(atom_array.element, self.hydrogen_names)
        data["atom_array"] = atom_array[is_heavy]
        return data


class ApplyFunctionToAtomArray(Transform):
    """
    Apply a function to the atom array.
    """

    def __init__(self, func: Callable[[AtomArray], AtomArray]):
        self.func = func

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

    def forward(self, data: dict) -> dict:
        data["atom_array"] = self.func(data["atom_array"])
        return data


class RemoveTerminalOxygen(ApplyFunctionToAtomArray):
    """
    Remove terminal oxygen atoms (`OXT`) from the atom array.
    """

    def __init__(self):
        super().__init__(func=lambda arr: arr[arr.atom_name != "OXT"])


class FilterToProteins(ApplyFunctionToAtomArray):
    """
    Filter atom array to only include protein residues.
    """

    def __init__(self, min_size: int = 5):
        super().__init__(func=lambda arr: arr[struc.filter_polymer(arr, pol_type="peptide", min_size=min_size)])


class AddProteinTerminiAnnotation(Transform):
    """
    Annotate protein termini (i.e. N- and C-terminus) for protein chains in the atom array.
    """

    incompatible_previous_transforms = ["CropContiguousLikeAF3", "CropSpatialLikeAF3"]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_id", "chain_id", "chain_type"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        is_linear_protein = np.isin(
            atom_array.chain_type, [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L]
        )  # We can't use PROTEINS from data_constants.py, since that includes CYCLIC_PSEUDO_PEPTIDE

        # Annotate N-termini
        is_first_in_chain = atom_array.res_id == 1
        atom_array.set_annotation("is_N_terminus", is_first_in_chain & is_linear_protein)

        # Annotate C-termini
        last_res_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)[1:] - 1
        is_last_in_chain = np.zeros(len(atom_array), dtype=bool)
        is_last_in_chain[last_res_idxs] = True
        atom_array.set_annotation("is_C_terminus", is_last_in_chain & is_linear_protein)

        data["atom_array"] = atom_array
        return data


def add_global_atom_id_annotation(atom_array: AtomArray) -> AtomArray:
    """
    Adds a global atom ID annotation `atom_id` to the atom array.
    This annotation is useful for tracking atoms after operations such as cropping,
    slicing, or shuffling. The `atom_id` is generated as a sequence of integers
    corresponding to the number of atoms in the atom array.

    Args:
        atom_array (AtomArray): The AtomArray to which the atom ID annotation will be added.

    Returns:
        AtomArray: The AtomArray with the added `atom_id` annotation.
    """
    atom_array.set_annotation("atom_id", np.arange(len(atom_array), dtype=np.uint32))
    return atom_array


class AddGlobalAtomIdAnnotation(Transform):
    """
    Adds a global atom ID annotation to the atom array.
    Useful for keeping track of atoms after cropping, slicing or shuffling operations.
    """

    incompatible_previous_transforms = ["AddGlobalAtomIdAnnotation"]

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, required=[], forbidden=["atom_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        data["atom_array"] = add_global_atom_id_annotation(atom_array)
        return data


def add_global_token_id_annotation(atom_array: AtomArray) -> AtomArray:
    """
    Adds a global token ID annotation `token_id` to the atom array.
    This annotation is useful for tracking tokens after operations such as cropping,
    slicing, or shuffling. The `token_id` is generated as a sequence of integers
    corresponding to the number of tokens in the atom array, and is spread across
    the atom array to maintain the association with each atom.

    Args:
        atom_array (AtomArray): The AtomArray to which the token ID annotation will be added.

    Returns:
        AtomArray: The AtomArray with the added `token_id` annotation.
    """
    token_id = np.arange(get_token_count(atom_array), dtype=np.uint32)  # [n_tokens]
    atom_array.set_annotation("token_id", spread_token_wise(atom_array, token_id))
    return atom_array


class AddGlobalTokenIdAnnotation(Transform):
    """
    Adds a global token ID annotation `token_id` to the atom array.
    Useful for keeping track of tokens after cropping, slicing or shuffling operations.
    """

    incompatible_previous_transforms = ["AddGlobalTokenIdAnnotation"]

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, required=[], forbidden=["token_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # add the global token id annotation
        data["atom_array"] = add_global_token_id_annotation(atom_array)
        return data


def sort_poly_then_non_poly(atom_array: AtomArray, treat_atomized_as_non_poly: bool = True) -> AtomArray:
    """
    Sort the atom array such that polymer chains are first, followed by non-polymer chains.
    The order within the `poly` and `non_poly` chains is preserved.

    This function is useful for ensuring that models like `RF2AA`, which expect the input to be
    formatted as [polys, non-polys], receive the correctly ordered atom array.

    Args:
        - atom_array (AtomArray): The AtomArray to be sorted.
        - treat_atomized_as_non_poly (bool): If True, atomized structures are treated as non-polymer.
            Defaults to True.

    Returns:
        AtomArray: The sorted AtomArray with polymer chains first, followed by non-polymer chains.
    """
    is_atomized = np.zeros(len(atom_array), dtype=bool)
    if treat_atomized_as_non_poly and "atomize" in atom_array.get_annotation_categories():
        is_atomized = atom_array.atomize

    # Find indices of polymer and non-polymer atoms
    is_poly = atom_array.is_polymer & ~is_atomized
    is_non_poly = is_atomized | ~atom_array.is_polymer

    # Sort by indexing (instead of masking/slicing), since this leads to correctly
    # tracking and updating the inter-poly-non-poly bonds
    poly_idxs = np.where(is_poly)[0]
    non_poly_idxs = np.where(is_non_poly)[0]
    sort_poly_then_non_poly = np.concatenate([poly_idxs, non_poly_idxs])

    return atom_array[sort_poly_then_non_poly]


def sort_like_rf2aa(atom_array: AtomArray) -> AtomArray:
    """
    Sort the atom array such that non-polymer chains are sorted by their covalent bonds and PN unit IIDs.
    """
    is_atomized = np.zeros(len(atom_array), dtype=bool)
    if "atomize" in atom_array.get_annotation_categories():
        is_atomized = atom_array.atomize

    # Find indices of polymer and non-polymer atoms
    is_poly = atom_array.is_polymer & (~is_atomized)
    is_non_poly = is_atomized | (~atom_array.is_polymer)
    is_bonded_non_poly = np.zeros(len(atom_array), dtype=bool)
    for pn_unit_iid in np.unique(atom_array.pn_unit_iid):
        pn_unit_mask = atom_array.pn_unit_iid == pn_unit_iid
        is_bonded_non_poly[pn_unit_mask] = np.any(is_poly[pn_unit_mask]) & is_non_poly[pn_unit_mask]
    is_free_non_poly = is_non_poly & (~is_bonded_non_poly)
    assert np.sum(is_poly) + np.sum(is_bonded_non_poly) + np.sum(is_free_non_poly) == len(
        atom_array
    ), "overlapping groups"

    # Sort by indexing according to
    #  0: by poly / bonded non-poly / free non-poly
    #  1: within groups by moelcule_iid
    #  2: within molecules by pn_unit_iid
    #  3: within pn_units by chain_iid
    _sort_table = pd.DataFrame(
        {
            "atom_idx": np.arange(len(atom_array)),
            "group": is_poly.astype(np.int8)
            + 2 * is_bonded_non_poly.astype(np.int8)
            + 3 * is_free_non_poly.astype(np.int8),
            "molecule_entity": atom_array.molecule_entity,
            "molecule_iid": atom_array.molecule_iid,
            "pn_unit_iid": atom_array.pn_unit_iid,
            "chain_entity": atom_array.chain_entity,
            "chain_iid": atom_array.chain_iid,
        }
    )
    to_sorted = _sort_table.sort_values(
        by=["group", "molecule_entity", "molecule_iid", "pn_unit_iid", "chain_entity", "chain_iid", "atom_idx"]
    )["atom_idx"].values

    # ... ensure all indices occur exactly once
    assert np.all(np.sort(to_sorted) == np.arange(len(atom_array))), "indices must occur exactly once"

    return atom_array[to_sorted]


class SortLikeRF2AA(Transform):
    """
    Sort the atom array in 3 groups (in this order). Within each group the atoms are ordered by
    their pn_unit_iid (and within a pn_unit their order is preserved).
        - (1) polymer atoms
        - (2) non-poly atoms of a pn-unit bonded to a polymer (covalent modifications)
        - (3) non-poly atoms of a free-floating pn-unit (free-floating ligands)
    """

    requires_previous_transforms = ["AtomizeResidues"]
    incompatible_previous_transforms = ["EncodeAtomArray", "CropSpatialLikeAF3", "CropContiguousLikeAF3"]

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(
            data,
            ["is_polymer", "pn_unit_iid", "molecule_iid", "molecule_entity", "chain_entity", "chain_iid", "atomize"],
        )

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # perform the sorting
        data["atom_array"] = sort_like_rf2aa(atom_array)

        return data


class SortPolyThenNonPoly(Transform):
    """
    Sort the atom array such that polymer chains are first, followed by non-polymer chains.
    The order within the `poly` and `non_poly` chains is preserved.

    This transformation is useful for models like `RF2AA`, which expect the input to be formatted
    as [polys, non-polys].

    Args:
        - treat_atomized_as_non_poly (bool): If True, atomized structures are treated as non-polymer.
            Defaults to True.
    """

    incompatible_previous_transforms = ["EncodeAtomArray", "CropSpatialLikeAF3", "CropContiguousLikeAF3"]

    def __init__(self, treat_atomized_as_non_poly: bool = True):
        self.treat_atomized_as_non_poly = treat_atomized_as_non_poly

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["is_polymer"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # perform the sorting
        data["atom_array"] = sort_poly_then_non_poly(atom_array, self.treat_atomized_as_non_poly)

        return data


class RemoveUnresolvedLigandAtomsIfTooMany(Transform):
    """
    If the number of unresolved (zero-occupancy) ligand atoms exceeds a specified threshold, remove all masked (zero-occupancy) ligand atoms from the atom array.

    This Transform is needed to avoid a significant proportion of the crop window from being filled with unresolved ligand atoms. Most commonly, this occurs with poorly resolved liposomes.

    Parameters:
        - unresolved_ligand_atom_limit(int): The maximum number of unresolved ligand atoms allowed in the atom array.

    Example: See PDB ID `6CLZ`, which contains a liposome with many unresolved atoms.
    """

    def __init__(self, unresolved_ligand_atom_limit: int):
        self.unresolved_ligand_atom_limit = unresolved_ligand_atom_limit

    def check_input(self, data):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data):
        atom_array = data["atom_array"]

        # Create a mask for unresolved ligand atoms
        is_ligand_atom = ~atom_array.is_polymer
        is_unresolved_atom = atom_array.occupancy == 0
        is_unresolved_ligand_atom = is_ligand_atom & is_unresolved_atom

        # If the number of unresolved ligand atoms exceeds the limit, remove all unresolved ligand atoms in the example
        num_unresolved_ligand_atoms = np.sum(is_unresolved_ligand_atom)
        if num_unresolved_ligand_atoms > self.unresolved_ligand_atom_limit:
            logger.info(
                f"Removing {num_unresolved_ligand_atoms} unresolved ligand atoms from the atom array, exceeding the limit of {self.unresolved_ligand_atom_limit}"
            )
            data["atom_array"] = atom_array[~is_unresolved_ligand_atom]

        return data

        # TODO: Write tests; find an example to check
        # TODO: Trial-and-error a couple approaches to this challenge (e.g., best way to avoid liposomes)


class HandleUndesiredResTokens(Transform):
    """
    Remove undesired residue tokens from the AtomArray.

    For undesired residue names `res_name`, the following actions are taken:
        - For undesired residues in non-polymer residues:
            - Remove the entire non-polymer (pn_unit_iid)
        - For undesired residues in polymer residues:
            - Map to the closest canonical residue name (if possible)
            - Else, map to an unknown residue name (if possible, i.e if backbone atoms are present)
            - Else, atomize
    """

    def __init__(self, undesired_res_tokens: list | tuple):
        """
        HandleUndesiredResTokens is a Transform that removes undesired residue tokens from an AtomArray.

        This class processes an AtomArray to identify and handle undesired residue names. The actions taken
        depend on whether the residues are part of a polymer or non-polymer. For non-polymer residues, the
        entire non-polymer unit is removed. For polymer residues, the undesired residue is mapped to the
        closest canonical residue name.

        Args:
            - undesired_res_tokens (list | tuple): A list or tuple of undesired residue names to be removed
              or mapped.

        Example:
            >>> transform = HandleUndesiredResTokens(undesired_res_tokens=["PTR", "SO4"])
        """
        self.undesired_res_tokens = undesired_res_tokens

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name", "is_polymer", "pn_unit_iid", "chain_type"])

    def _get_closest_canonical_residue(self, res_name: str, chain_type: int, force_unknown: bool = False) -> str:
        """Map a residue name to the closest canonical residue name."""
        one_letter_canonical = get_1_from_3_letter_code(
            res_name, chain_type=ChainType(chain_type), use_closest_canonical=not force_unknown
        )
        three_letter_canonical = get_3_from_1_letter_code(one_letter_canonical, chain_type=ChainType(chain_type))
        return three_letter_canonical

    @lru_cache(maxsize=1000)
    def _map_to_closest_canonical_residue(
        self, res_name: str, chain_type: int, has_hydrogens: bool, atom_name: tuple[str]
    ) -> tuple[np.ndarray, str]:
        """Map a residue name to the closest canonical residue name."""

        for force_unknown in (False, True):
            canonical_res_name = self._get_closest_canonical_residue(res_name, chain_type, force_unknown)
            canonical_res = struc.info.residue(canonical_res_name)

            # Remove hydrogens if non-canonical residue doesn't have hydrogens
            if not has_hydrogens:
                canonical_res = canonical_res[canonical_res.element != "H"]

            # If canonical residue is a strict subset of the original residue,
            #  keep all matching atom names and delete the rest
            if np.all(np.isin(canonical_res.atom_name, atom_name)):
                to_keep = np.isin(atom_name, canonical_res.atom_name)
                # ... if we match without `force_unknown` break loop early
                return to_keep, canonical_res_name

        # If we could not find a canonical residue, or map to unknown, atomize the residue
        return np.ones(len(atom_name), dtype=bool), res_name

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        if "atomize" not in atom_array.get_annotation_categories():
            atom_array.set_annotation("atomize", np.zeros(len(atom_array), dtype=bool))

        # Mark undesired residues
        to_remove = np.isin(atom_array.res_name, self.undesired_res_tokens)

        # Case 1: Undesired residue is part of non-polymer:
        #  - Remove the entire non-polymer (pn_unit_iid)
        is_undesired_non_poly = to_remove & (~atom_array.is_polymer)
        if np.any(is_undesired_non_poly):
            pn_unit_iids_to_remove = np.unique(atom_array.pn_unit_iid[is_undesired_non_poly])
            to_remove |= np.isin(atom_array.pn_unit_iid, pn_unit_iids_to_remove)

        # Case 2: Undesired residue is part of polymer:
        #  - Map to closest canonical residue
        is_undesired_poly = to_remove & atom_array.is_polymer
        if np.any(is_undesired_poly):
            # Iterate over all undesired residues
            _token_start_stop_idx = get_token_starts(atom_array, add_exclusive_stop=True)
            _token_starts = _token_start_stop_idx[:-1]
            _token_stops = _token_start_stop_idx[1:]
            undesired_poly_token_idxs = np.where(is_undesired_poly[_token_starts])[0]

            for token_idx in undesired_poly_token_idxs:
                token_start, token_stop = _token_starts[token_idx], _token_stops[token_idx]

                old_res_name = atom_array.res_name[token_start]
                to_keep, new_res_name = self._map_to_closest_canonical_residue(
                    res_name=old_res_name,
                    chain_type=atom_array.chain_type[token_start],
                    has_hydrogens="1" in atom_array.element[token_start:token_stop],
                    atom_name=tuple(atom_array.atom_name[token_start:token_stop]),  # tuple for hashability
                )

                # if new_res_name is the same as the original res_name (i.e. we didn't map to a canonical residue),
                # we atomize the residue
                if new_res_name == old_res_name:
                    atom_array.atomize[token_start:token_stop] = True

                # ... override the `to_remove` flag as `False` for the atoms that we want to keep
                to_remove[token_start:token_stop] = ~to_keep

                # ... override the old res name
                atom_array.res_name[token_start:token_stop] = new_res_name

        # Drop undesired residues
        atom_array = atom_array[~to_remove]
        data["atom_array"] = atom_array
        return data


class RaiseIfTooManyAtoms(Transform):
    def __init__(self, max_atoms: int):
        self.max_atoms = max_atoms

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array", "example_id"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        num_atoms = len(data["atom_array"])
        if num_atoms > self.max_atoms:
            example_id = data["example_id"]
            raise ValueError(f"{example_id} exceeds max allowed number of atoms! ({num_atoms:,} > {self.max_atoms:,}).")
        return data
