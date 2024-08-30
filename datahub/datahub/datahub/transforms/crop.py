import itertools

import numpy as np
from assertpy import assert_that
from biotite.structure import AtomArray
from scipy.spatial import KDTree

from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.atom_array import atom_id_to_atom_idx, atom_id_to_token_idx
from datahub.transforms.base import Transform
from datahub.utils.token import (
    get_af3_token_representative_coords,
    get_token_count,
    get_token_starts,
    spread_token_wise,
)


def crop_contiguous_af2_multimer(iids: list[int | str], instance_lens: list[int], crop_size: int) -> dict:
    """
    Crop contiguous tokens from the given instances to reach the given crop size probabilistically.

    Implements the `crop_contiguous` (algorithm 1 in section 7.2.1) of AF2 Multimer and section 2.7.2 of AF3.

    Args:
        iids (list[int | str]): List of instance identifiers.
        instance_lens (list[int]): List of lengths corresponding to each instance.
        crop_size (int): Desired number of tokens to crop. Must be greater than 0.

    Returns:
        keep_tokens (dict[int | str, np.ndarray[bool]]): Dictionary mapping instance identifiers
            (iids) to crop masks (i.e. boolean arrays) indicating which tokens to keep.

    References:
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf

    Example:
        >>> iids = [1, 2, 3]
        >>> instance_lens = [3, 4, 2]
        >>> crop_size = 5
        >>> result = crop_contiguous_af2_multimer(iids, instance_lens, crop_size)
        >>> print(result)
        # Output might look like (probabilistic!):
        # {
        #     3: array([False, True]),
        #     2: array([False, True, True, False]),
        #     1: array([True, True, False])
        # }
    """
    iids = np.asarray(iids)
    instance_lens = np.asarray(instance_lens)

    assert_that(crop_size).is_greater_than(0)
    assert_that(len(iids)).is_equal_to(len(instance_lens)).is_equal_to(np.unique(iids).size)

    # randomly permute the order of the instances to avoid cropping bias
    permutation = np.random.permutation(len(iids))
    iids = iids[permutation]
    instance_lens = instance_lens[permutation]

    # init variables to keep track of remaining budget
    n_budget = crop_size  # ... number of tokens that can still be added to the crop
    n_available = np.sum(instance_lens)  # ... number of tokens that are still available in the remaining instances

    keep_tokens = {}
    for iid, instance_len in zip(iids, instance_lens):
        if n_budget == 0:
            # ... early stop if budget is already exhausted
            break

        # ... after cropping the current instance, n_remaining tokens are still available
        n_available -= instance_len

        # Determine the min/max crop sizes
        # ... maximally take at most
        #   (1) the remaining budget or
        #   (2) all tokens of the current instance,
        # whichever is smaller
        crop_size_max = min(n_budget, instance_len)

        # ... take at least
        #  (1) 0 if there is still more than enough tokens available to reach the budget
        #  (2) how much would be needed to reach the budget if we took all remaining available tokens
        #  (3) or take all tokens of the current instance if we cannot reach the budget even if
        #   we take all remaining available tokens
        crop_size_min = min(instance_len, max(0, n_budget - n_available))

        # ... sample a crop size for this instance and update the budget
        n_crop_in_instance = np.random.randint(crop_size_min, crop_size_max + 1)
        n_budget -= n_crop_in_instance
        assert n_budget >= 0, "The budget cannot be negative!"

        # ... sample a crop start position for this instance
        crop_start = np.random.randint(0, instance_len - n_crop_in_instance + 1)

        keep_token = np.zeros(instance_len, dtype=bool)
        keep_token[crop_start : crop_start + n_crop_in_instance] = True

        # ... add the crop to the keep dictionary
        keep_tokens[iid] = keep_token

    return keep_tokens  # dict[int | str, np.ndarray[bool]]


def get_spatial_crop_center(atom_array: AtomArray, query_pn_unit_iids: list[str], cutoff_distance: float = 15.0):
    """
    Sample a crop center from a spatial region of the atom array.

    Implements the selection of a crop center as described in AF3.
    ```
        In this procedure, polymer residues and ligand atoms are selected that
        are within close spatial distance of an interface atom. The interface
        atom is selected at random from the set of token centre atoms (defined
        in subsection 2.6) with a distance under 15 Å to another chain's token
        centre atom. For examples coming out of the Weighted PDB or Disordered
        protein PDB complex datasets, where a preferred chain or interface is
        provided (subsection 2.5), the reference atom is selected at random
        from interfacial token centre atoms that exist within this chain or
        interface.
    ```

    Args:
        atom_array (AtomArray): The array containing atom information.
        query_pn_unit_iids (list[str]): List of PN unit instance IDs to query.
        cutoff_distance (float, optional): The distance cutoff to consider for spatial proximity. Defaults to 15.0.

    Returns:
        np.ndarray: A boolean mask indicating the crop center.
    """
    # ... get mask for query polymer/non-polymer unit
    is_query_pn_unit = np.isin(atom_array.pn_unit_iid, query_pn_unit_iids)

    # ... get mask for occupied atoms
    is_occupied = atom_array.occupancy > 0

    if len(query_pn_unit_iids) == 1:
        # If there's only one query unit, we don't need to check for spatial proximity,
        # so we can just return the mask for the query unit.
        can_be_crop_center = is_query_pn_unit & is_occupied
        assert np.any(
            can_be_crop_center
        ), f"No crop center found! It appears `query_pn_unit_iid` {query_pn_unit_iids} is not in the atom array or unresolved."

        return can_be_crop_center

    # ... get mask for ligands of interest
    is_at_interface = np.zeros_like(is_query_pn_unit, dtype=bool)
    for pn_unit_1_iid, pn_unit_2_iid in itertools.combinations(query_pn_unit_iids, 2):
        # ... get mask, indices, and kdtree for pn_unit_1
        pn_unit_1_mask = (atom_array.pn_unit_iid == pn_unit_1_iid) & is_occupied
        pn_unit_1_indices = np.where(pn_unit_1_mask)[0]
        _tree1 = KDTree(atom_array.coord[pn_unit_1_mask])

        # ... get mask, indices, and kdtree for pn_unit_2
        pn_unit_2_mask = (atom_array.pn_unit_iid == pn_unit_2_iid) & is_occupied
        pn_unit_2_indices = np.where(pn_unit_2_mask)[0]
        _tree2 = KDTree(atom_array.coord[pn_unit_2_mask])

        dists = _tree1.sparse_distance_matrix(_tree2, max_distance=cutoff_distance, output_type="coo_matrix")

        # ... update the interface mask (by converting the local idxs to the global idxs)
        is_at_interface[pn_unit_1_indices[np.unique(dists.row)]] = True
        is_at_interface[pn_unit_2_indices[np.unique(dists.col)]] = True

    # ... assemble final crop mask
    can_be_crop_center = is_query_pn_unit & is_at_interface & is_occupied

    assert np.any(can_be_crop_center), "No crop center found!"
    return can_be_crop_center


def crop_spatial_af2_multimer(
    coord: np.ndarray, crop_center_idx: int, crop_size: int, jitter_scale: float = 1e-3
) -> np.ndarray:
    """
    Crop spatial tokens around a given `crop_center` by keeping the `crop_size` nearest neighbors (with jitter).

    Implements the `crop_spatial` (algorithm 2 in section 7.2.1) of AF2 Multimer and AF3

    Args:
        coord (np.ndarray): A 2D numpy array of shape (N, 3) representing the 3D token-level coordinates.
            Coordinates are expected to be in Angstroms.
        crop_center_idx (int): The index of the token to be used as the center of the crop.
        crop_size (int): The number of nearest neighbors to include in the crop.
        jitter_scale (float): The scale of the jitter to add to the coordinates.

    Returns:
        crop_mask (np.ndarray): A boolean mask of shape (N,) where True indicates that the token is within the crop.

    References:
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf

    Example:
        >>> coord = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        >>> crop_center_idx = 1
        >>> crop_size = 2
        >>> crop_mask = crop_spatial_af2_multimer(coord, crop_center_idx, crop_size)
        >>> print(crop_mask)
        [ True  True False False]
    """
    assert_that(coord.ndim).is_equal_to(2)
    assert_that(coord.shape[1]).is_equal_to(3)
    assert_that(crop_center_idx).is_less_than(coord.shape[0])
    assert_that(crop_size).is_greater_than(0)
    assert_that(jitter_scale).is_greater_than_or_equal_to(0)

    # Add small jitter to coordinates to break ties
    if jitter_scale > 0:
        coord = coord + np.random.normal(scale=jitter_scale, size=coord.shape)

    # ... get query center
    query_center = coord[crop_center_idx]

    # ... extract a mask for valid coordiantes (i.e. no `nan`'s, which indicate unknown token centers)
    #     including including unoccupied tokens in the crop
    is_valid = np.isfinite(coord).all(axis=1)

    # ... build a KDTree for efficient querying, excluding invalid coordinates
    tree = KDTree(coord[is_valid])

    # ... query the `crop_size` nearest neighbors of the crop center
    _, nearest_neighbor_idxs = tree.query(query_center, k=crop_size, p=2)
    # ... filter out missing neighbours (index equal to `tree.n`)
    nearest_neighbor_idxs = nearest_neighbor_idxs[nearest_neighbor_idxs < tree.n]

    # ... crop mask is True for the `crop_size` nearest neighbors of the crop center
    crop_mask = np.zeros(coord.shape[0], dtype=bool)
    is_valid_and_in_crop_idxs = np.where(is_valid)[0][nearest_neighbor_idxs]
    crop_mask[is_valid_and_in_crop_idxs] = True

    return crop_mask


class CropContiguousLikeAF3(Transform):
    """
    A transform that performs contiguous cropping similar to AF3.

    This class implements the contiguous cropping procedure as described in AF3. It selects a crop center
    from a contiguous region of the atom array and samples a crop around this center.

    WARNING: This transform is probabilistic if the atom array is larger than the crop size!

    References:
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf

    Attributes:
        crop_size (int): The maximum number of tokens to crop.
        keep_uncropped_atom_array (bool): Whether to keep the uncropped atom array in the data.
            If `True`, the uncropped atom array will be stored in the `crop_info` dictionary
            under the key `"atom_array"`. Defaults to `False`.
    """

    requires_previous_transforms = ["AtomizeResidues"]
    incompatible_previous_transforms = ["EncodeAtomArray", "CropSpatialLikeAF3", "CropContiguousLikeAF3"]

    def __init__(self, crop_size: int, keep_uncropped_atom_array: bool = False):
        self.crop_size = crop_size
        self.keep_uncropped_atom_array = keep_uncropped_atom_array

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["chain_iid", "atomize"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        requires_crop = get_token_count(atom_array) > self.crop_size
        if requires_crop:
            # Extract chain data
            chain_iids = np.unique(atom_array.chain_iid)
            chain_n_tokens = [
                get_token_count(atom_array[atom_array.chain_iid == chain_iid]) for chain_iid in chain_iids
            ]

            # Sample crop as in AF2 multimer
            keep_token_dict = crop_contiguous_af2_multimer(chain_iids, chain_n_tokens, self.crop_size)

            # Turn crop information into atom-level mask
            is_token_start_idxs = get_token_starts(atom_array)
            is_token_in_crop = np.zeros_like(is_token_start_idxs, dtype=bool)
            for chain_iid, keep_token_idxs in keep_token_dict.items():
                chain_idxs = np.where(atom_array[is_token_start_idxs].chain_iid == chain_iid)[0]
                is_token_in_crop[chain_idxs[keep_token_idxs]] = True
            is_atom_in_crop = spread_token_wise(atom_array, is_token_in_crop)
        else:
            # ... no need to crop since the atom array is already small enough
            is_atom_in_crop = np.ones(len(atom_array), dtype=bool)
            is_token_in_crop = np.ones(get_token_count(atom_array), dtype=bool)

        # Update data
        data["crop_info"] = {
            "type": self.__class__.__name__,
            "requires_crop": requires_crop,
            "crop_token_idxs": np.where(is_token_in_crop)[0],
            "crop_atom_idxs": np.where(is_atom_in_crop)[0],
        }
        if self.keep_uncropped_atom_array:
            data["crop_info"]["atom_array"] = atom_array
        data["atom_array"] = atom_array[is_atom_in_crop]

        return data


class CropSpatialLikeAF3(Transform):
    """
    A transform that performs spatial cropping similar to AF3 and AF2 Multimer.

    This class implements the spatial cropping procedure as described in AF3. It selects a crop center
    from a spatial region of the atom array and samples a crop around this center.

    WARNING: This transform is probabilistic if the atom array is larger than the crop size!

    References:
        - AF3 https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
        - AF2 Multimer https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf

    Attributes:
        crop_size (int): The maximum number of tokens to crop. Must be greater than 0.
        jitter_scale (float): The scale of the jitter to apply to the crop center. This is to break
            ties between atoms with the same spatial distance. Defaults to 1e-3.
        crop_center_cutoff_distance (float): The cutoff distance to consider for selecting crop
            centers. Measured in Angstroms. Defaults to 15.0.
        keep_uncropped_atom_array (bool): Whether to keep the uncropped atom array in the data.
            If `True`, the uncropped atom array will be stored in the `crop_info` dictionary
            under the key `"atom_array"`. Defaults to `False`.
    """

    requires_previous_transforms = ["AddGlobalAtomIdAnnotation", "AtomizeResidues"]
    incompatible_previous_transforms = ["EncodeAtomArray", "CropContiguousLikeAF3", "CropSpatialLikeAF3"]

    def __init__(
        self,
        crop_size: int,
        jitter_scale: float = 1e-3,
        crop_center_cutoff_distance: float = 15.0,
        keep_uncropped_atom_array: bool = False,
    ):
        """
        Initialize the CropSpatialLikeAF3 transform.

        Args:
            crop_size (int): The maximum number of tokens to crop. Must be greater than 0.
            jitter_scale (float, optional): The scale of the jitter to apply to the crop center.
                This is to break ties between atoms with the same spatial distance. Defaults to 1e-3.
            crop_center_cutoff_distance (float, optional): The cutoff distance to consider for
                selecting crop centers. Measured in Angstroms. Defaults to 15.0.
            keep_uncropped_atom_array (bool, optional): Whether to keep the uncropped atom array in the data.
                If `True`, the uncropped atom array will be stored in the `crop_info` dictionary
                under the key `"atom_array"`. Defaults to `False`.
        """
        self.crop_size = crop_size
        self.jitter_scale = jitter_scale
        self.crop_center_cutoff_distance = crop_center_cutoff_distance
        self.keep_uncropped_atom_array = keep_uncropped_atom_array

    def check_input(self, data: dict):
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["pn_unit_iid", "atomize", "atom_id"])

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        if "query_pn_unit_iids" in data:
            query_pn_units = data["query_pn_unit_iids"]
        else:
            query_pn_units = np.unique(atom_array.pn_unit_iid)

        requires_crop = get_token_count(atom_array) > self.crop_size
        if requires_crop:
            # Get possible crop centers
            can_be_crop_center = get_spatial_crop_center(atom_array, query_pn_units, self.crop_center_cutoff_distance)

            # Sample crop atom
            crop_atom_id = np.random.choice(atom_array[can_be_crop_center].atom_id)
            # ... sample crop
            token_coords = get_af3_token_representative_coords(atom_array)  # < SLOW!
            token_crop_idx = atom_id_to_token_idx(atom_array, crop_atom_id)
            is_token_in_crop = crop_spatial_af2_multimer(
                token_coords, token_crop_idx, crop_size=self.crop_size, jitter_scale=self.jitter_scale
            )
            # ... spread token-level crop mask to atom-level
            is_atom_in_crop = spread_token_wise(atom_array, is_token_in_crop)
        else:
            # ... no need to crop since the atom array is already small enough
            crop_atom_id = np.nan
            token_crop_idx = np.nan
            is_atom_in_crop = np.ones(len(atom_array), dtype=bool)
            is_token_in_crop = np.ones(get_token_count(atom_array), dtype=bool)

        # Update crop center atom for history
        data["crop_info"] = {
            "type": self.__class__.__name__,
            "requires_crop": requires_crop,
            "crop_center_atom_id": crop_atom_id,
            "crop_center_atom_idx": atom_id_to_atom_idx(atom_array, crop_atom_id) if requires_crop else np.nan,
            "crop_center_token_idx": token_crop_idx,
            "crop_token_idxs": np.where(is_token_in_crop)[0],
            "crop_atom_idxs": np.where(is_atom_in_crop)[0],
        }
        if self.keep_uncropped_atom_array:
            data["crop_info"]["atom_array"] = atom_array

        # Update data with cropped atom array
        data["atom_array"] = atom_array[is_atom_in_crop]  # note: this is a copy

        return data
