"""
Transforms to handle featurization of edge cases with unresolved residues.
NOTE: Transforms that "filter" based on unresolved residues will be found in the "filters" file, not here.
"""

from typing import Any

import numpy as np
from biotite.structure import AtomArray
from cifutils.enums import ChainTypeInfo

from datahub.transforms._checks import check_atom_array_annotation, check_contains_keys, check_is_instance
from datahub.transforms.atom_array import apply_and_spread_residue_wise
from datahub.transforms.base import Transform
from datahub.utils.numpy import get_nearest_true_index_for_each_false
from datahub.utils.token import (
    apply_token_wise,
    get_af3_token_center_masks,
    get_af3_token_representative_masks,
    spread_token_wise,
)


def mask_residues_with_unresolved_backbone_atoms(atom_array: AtomArray) -> AtomArray:
    """If a polymer residue has an unresolved backbone atom (occupancy == 0), set the occupancy of the entire residue to zero."""
    backbone_atom_names = ["N", "CA", "C"]

    # ... subset to backbone atoms within polymers with unresolved coordinates
    # (We treat partially occupied atoms as occupied; e.g., those resolved from "altlocs")
    protein_mask = (
        np.isin(atom_array.chain_type, ChainTypeInfo.PROTEINS) if "chain_type" in atom_array else atom_array.is_polymer
    )
    unresolved_polymer_backbone_mask = (
        protein_mask & np.isin(atom_array.atom_name, backbone_atom_names) & (atom_array.occupancy == 0)
    )

    # ... get the residue-wise mask for unresolved backbone atoms
    unresolved_backbone_res_mask = apply_and_spread_residue_wise(
        atom_array, unresolved_polymer_backbone_mask, function=np.any
    )

    # ... mask the occupancy of the entire residue
    atom_array.occupancy[unresolved_backbone_res_mask] = 0

    return atom_array


class MaskResiduesWithUnresolvedBackboneAtoms(Transform):
    """For residues with at least one unresolved backbone atom, mask (set to occupancy zero) the entire residue.

    If we don't have backbone atoms, then:
        - We cannot build backbone frames
        - The local structure quality is likely poor

    As an example, see PDB ID `6Z3R`, which has unresolved C and CA atoms.

    NOTE: This transform must be applied before other transform that rely on the `occupancy` annotation.
    """

    incompatible_previous_transforms = [
        "EncodeAtomArray",
        "CropContiguousLikeAF3",
        "CropSpatialLikeAF3",
        "PlaceUnresolvedTokenOnClosestResolvedTokenInSequence",
    ]

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = mask_residues_with_unresolved_backbone_atoms(data["atom_array"])
        return data


def place_unresolved_token_on_closest_resolved_token_in_sequence(
    atom_array: AtomArray, annotation_to_update: str = "coord_to_be_noised", annotation_to_copy: str = "coord"
) -> AtomArray:
    """Place all atoms within fully-unresolved residues on the closest resolved neighbor in sequence space.

    NOTE: For non-polymers, each atom is considered a token, so this transform will place unresolved
    atoms on the closest resolved token in sequence space (i.e., the previous or next atom).

    NOTE: We only perform the operation WITHIN chains, such that we don't resolve across chain boundaries.

    Args:
        atom_array (AtomArray): The atom array to modify.
        annotation_to_update (str): The annotation to update with the new coordinates. E.g., "coord" (if we want to modify the ground-truth),
            or "coord_to_be_noised" (if we want to modify only the coordinates that will be noised).
        annotation_to_copy (str): The annotation to copy from the resolved atom to the unresolved atom. E.g., "coord" (if we want to copy the ground-truth),
            or "coord_to_be_noised" (if we want to copy the coordinates that will be noised, which may have been modified by previous transforms).
            In the AF-3 pipeline, we want to copy "coord_to_be_noised", to correctly resolve residues after applying PlaceUnresolvedTokenAtomsOnRepresentativeAtom.

    Returns:
        AtomArray: The modified atom array.
    """

    # ... loop through chains with unresolved atoms, such that we don't resolve across chain boundaries
    # (NOTE: We only iterate through chain instances containing any unresolved atoms for efficiency)
    for chain_iid in np.unique(atom_array.chain_iid[atom_array.occupancy == 0]):
        chain_mask = atom_array.chain_iid == chain_iid
        chain_atom_array = atom_array[chain_mask]

        # ... map each unresolved token to the nearest resolved token
        # (Here, we consider a token resolve if any atom within the token is resolved)
        is_token_resolved_token_level = apply_token_wise(
            chain_atom_array, chain_atom_array.occupancy, np.any
        )  # (n_tokens)
        is_token_resolved_atom_level = spread_token_wise(chain_atom_array, is_token_resolved_token_level)  # (n_atoms)

        # (Early exit if no tokens are resolved, or all tokens are resolved, as we can't then place unresolved tokens on closest resolved token)
        # NOTE: Such cases should not occur if the `RemoveUnresolvedPNUnits` Transform is applied first
        if np.all(is_token_resolved_atom_level) or np.all(~is_token_resolved_token_level):
            continue

        # ... get the nearest resolved token indices for each unresolved token
        nearest_resolved_token_indices_token_wise = get_nearest_true_index_for_each_false(
            is_token_resolved_token_level
        )  # (n_tokens)
        nearest_resolved_token_indices_atom_wise = spread_token_wise(
            chain_atom_array[~is_token_resolved_atom_level], nearest_resolved_token_indices_token_wise
        )  # (n_atoms)

        # Where the entire token is unresolved, set the atom coordinates to the nearest resolved token representative atom coordinates

        # ... get coordinates of representative atoms from the specified annotation to copy
        if annotation_to_copy == "coord":
            source_coordinates = chain_atom_array.coord
        else:
            source_coordinates = chain_atom_array.get_annotation(annotation_to_copy)

        representative_atom_coordinates_atom_level = source_coordinates[
            get_af3_token_representative_masks(chain_atom_array)
        ]  # (n_atoms, 3)

        assert len(representative_atom_coordinates_atom_level) == len(is_token_resolved_token_level)

        # ... update the coordinates for the specified annotation (e.g., "coord" or "coord_to_be_noised")
        if annotation_to_update == "coord":
            # (We must handle "coord" explicitly, as it is treated differently than other annotations)
            chain_atom_array.coord[~is_token_resolved_atom_level] = representative_atom_coordinates_atom_level[
                nearest_resolved_token_indices_atom_wise
            ]
            atom_array.coord[chain_mask] = chain_atom_array.coord
        else:
            chain_atom_array.get_annotation(annotation_to_update)[~is_token_resolved_atom_level] = (
                representative_atom_coordinates_atom_level[nearest_resolved_token_indices_atom_wise]
            )
            atom_array.get_annotation(annotation_to_update)[chain_mask] = chain_atom_array.get_annotation(
                annotation_to_update
            )

    return atom_array


class PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(Transform):
    """Place fully unresolved tokens on their closest resolved neighbor in sequence space, breaking ties by choosing the "leftmost" neighbor.

    This heuristic is helpful to avoid noising unresolved residue coordinates from the origin during diffusion training.

    Args:
        annotation_to_update (str): The annotation to update with the new coordinates. E.g., "coord" (if we want to modify the ground-truth),
            or "coord_to_be_noised" (if we want to modify only the coordinates that will be noised).
            NOTE: Must match the annotation used for `PlaceUnresolvedTokenAtomsOnRepresentativeAtom`.
        annotation_to_copy (str): The annotation to copy from the resolved atom to the unresolved atom.
    """

    requires_previous_transforms = [
        "AtomizeByCCDName",
        "PlaceUnresolvedTokenAtomsOnRepresentativeAtom",
    ]

    def __init__(
        self, annotation_to_update: str = "coord_to_be_noised", annotation_to_copy="coord_to_be_noised"
    ) -> None:
        self.annotation_to_update = annotation_to_update
        self.annotation_to_copy = annotation_to_copy

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

        annotations_to_check = {"occupancy"}
        if self.annotation_to_update != "coord":
            # "coord" is a special annotation, and technically not in `atom_array.get_annotation_categories()`
            annotations_to_check.add(self.annotation_to_update)
        if self.annotation_to_copy != "coord":
            annotations_to_check.add(self.annotation_to_copy)

        check_atom_array_annotation(data, list(annotations_to_check))

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = place_unresolved_token_on_closest_resolved_token_in_sequence(
            data["atom_array"],
            annotation_to_update=self.annotation_to_update,
            annotation_to_copy=self.annotation_to_copy,
        )
        return data


def place_unresolved_token_atoms_on_token_representative_atom(
    atom_array: AtomArray, annotation_to_update: str = "coord_to_be_noised"
) -> AtomArray:
    """Place unresolved token atoms (e.g., side chain atoms) on the representative atom of the corresponding residue (token).

    In cases where the representative atom is unresolved, we also try the token center atom (e.g., if the CB is unresolved but the CA is resolved, like in 8E83 for chain A, residue 194).
    Helpful for diffusive models to avoid noising unresolved side-chain atoms from the origin.

    NOTE: For non-polymers, all atoms are considered tokens (and are atomized); in such cases this Transform will have no effect.

    Args:
        atom_array (AtomArray): The atom array to modify.
        annotation_to_update (str): The annotation to update with the new coordinates. E.g., "coord" (if we want to modify the ground-truth),
            or "coord_to_be_noised" (if we want to modify only the coordinates that will be noised).

    Returns:
        AtomArray: The modified atom array.
    """
    # ... get a mask of all unresolved atoms
    unresolved_atom_mask = atom_array.occupancy == 0

    # ... get the unique chain IIDs of polymers with unresolved atoms (as this transform only applies to polymers; e.g., chains without full atomization)
    chain_iids_with_unresolved_atoms = np.unique(atom_array.chain_iid[(unresolved_atom_mask) & (~atom_array.atomize)])

    # ... prepare a mask of representative atoms for each residue
    representative_atom_mask = get_af3_token_representative_masks(atom_array)

    # (For cases where the representative atom is unresolved, we also try the token center atom)
    center_atom_mask = get_af3_token_center_masks(atom_array)

    for chain_iid in chain_iids_with_unresolved_atoms:
        # NOTE: We cannot rely on the `is_polymer` annotation, as in some instances (like acyl groups) we may have non-polymer tokens within a polymer chain (see: 7RCU)
        residues_with_unresolved_atoms = np.unique(
            atom_array.res_id[(atom_array.chain_iid == chain_iid) & unresolved_atom_mask & (~atom_array.atomize)]
        )
        for res_id in residues_with_unresolved_atoms:
            # ... create a mask for the unresolved atoms in the residue
            residue_mask = (atom_array.chain_iid == chain_iid) & (atom_array.res_id == res_id)
            unresolved_atoms_in_residue_mask = residue_mask & unresolved_atom_mask

            # ... get a mask indicating where to place unresolved atoms
            placement_atom_mask = representative_atom_mask & residue_mask

            # ... if the chosen atom (by default, the representative atom) is unresolved, try the center atom
            if not np.any(placement_atom_mask & ~unresolved_atom_mask):
                placement_atom_mask = center_atom_mask & residue_mask

            # ... get the index of the representative atom (there should be exactly one instance of the chain)
            assert np.sum(placement_atom_mask) == 1
            placement_atom_idx = np.where(placement_atom_mask)[0]

            # ... set the unresolved atom coordinates to the placment atom coordinates
            if annotation_to_update == "coord":
                # (We must handle "coord" explicitly, as it is treated differently than other annotations)
                atom_array.coord[unresolved_atoms_in_residue_mask] = atom_array.coord[placement_atom_idx]
            else:
                atom_array.get_annotation(annotation_to_update)[unresolved_atoms_in_residue_mask] = (
                    atom_array.get_annotation(annotation_to_update)[placement_atom_idx]
                )

    return atom_array


class PlaceUnresolvedTokenAtomsOnRepresentativeAtom(Transform):
    """Place unresolved token atoms (e.g., side chain atoms) on the representative atom of the residue (token).

    Note that this Transform has no impact on non-polymers, as all atoms are considered tokens.

    Args:
        annotation_to_update (str): The annotation to update with the new coordinates. E.g., "coord" (if we want to modify the ground-truth),
        or "coord_to_be_noised" (if we want to modify only the coordinates that will be noised).
    """

    requires_previous_transforms = ["MaskResiduesWithUnresolvedBackboneAtoms", "AtomizeByCCDName"]

    def __init__(self, annotation_to_update: str = "coord_to_be_noised") -> None:
        self.annotation_to_update = annotation_to_update

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)

        annotations_to_check = ["occupancy"]
        if self.annotation_to_update != "coord":
            # "coord" is a special annotation, and technically not in `atom_array.get_annotation_categories()`
            annotations_to_check += [self.annotation_to_update]
        check_atom_array_annotation(data, annotations_to_check)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        data["atom_array"] = place_unresolved_token_atoms_on_token_representative_atom(
            data["atom_array"], annotation_to_update=self.annotation_to_update
        )
        return data
