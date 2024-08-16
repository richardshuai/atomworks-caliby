"""
Transforms operating predominantly on Biotite's `AtomArray` objects.
These operations should take as input, and return, `AtomArray` objects.
"""

from biotite.structure import AtomArray, AtomArrayStack
import numpy as np
import pandas as pd
import biotite.structure as struc
from collections import Counter
import logging
from biotite.structure.io.pdbx import CIFBlock
from cifutils.cifutils_biotite.utils.bond_utils import add_bonds_from_struct_conn, get_inter_and_intra_residue_bonds
from cifutils.cifutils_biotite.common import exists
from cifutils.cifutils_biotite.utils.cifutils_biotite_utils import (
    deduplicate_iterator,
)

logger = logging.getLogger(__name__)


def remove_atoms_by_residue_names(atom_array: AtomArray, residues_to_remove: list) -> AtomArray:
    """
    Remove atoms from the AtomArray that have residue names in the residues_to_remove list.

    Parameters:
        atom_array (AtomArray): The array of atoms.
        residues_to_remove (list): A list of residue names to be removed from the atom array.

    Returns:
        AtomArray: The filtered atom array.
    """
    return atom_array[~np.isin(atom_array.res_name, residues_to_remove)]


def resolve_arginine_naming_ambiguity(atom_array: AtomArray) -> AtomArray:
    """
    Arginine naming ambiguities are fixed (ensuring NH1 is always closer to CD than NH2)
    """
    arg_mask = atom_array.res_name == "ARG"

    arg_nh1_mask = (atom_array.atom_name == "NH1") & arg_mask
    arg_nh2_mask = (atom_array.atom_name == "NH2") & arg_mask
    arg_cd_mask = (atom_array.atom_name == "CD") & arg_mask

    cd_nh1_dist = np.linalg.norm(atom_array.coord[arg_cd_mask] - atom_array.coord[arg_nh1_mask], axis=1)
    cd_nh2_dist = np.linalg.norm(atom_array.coord[arg_cd_mask] - atom_array.coord[arg_nh2_mask], axis=1)

    # Check if there are any name swamps required
    _to_swap = cd_nh1_dist > cd_nh2_dist  # local mask
    # turn local mask into global mask
    to_swap = np.zeros(len(atom_array), dtype=bool)
    to_swap[arg_nh1_mask] = _to_swap
    to_swap[arg_nh2_mask] = _to_swap

    # Swap NH1 and NH2 names if NH1 is further from CD than NH2
    if np.any(to_swap):
        logger.debug(f"Resolving {np.sum(_to_swap)} arginine naming ambiguities.")
        atom_array.atom_name[arg_nh1_mask & to_swap] = "NH2"
        atom_array.atom_name[arg_nh2_mask & to_swap] = "NH1"

        # apply reorder to ensure standardized order
        atom_array[arg_mask] = atom_array[arg_mask][struc.info.standardize_order(atom_array[arg_mask])]
    return atom_array


def mse_to_met(atom_array: AtomArray) -> AtomArray:
    """
    Convert MSE to MET for arginine residues.
    """
    mse_mask = atom_array.res_name == "MSE"
    if np.any(mse_mask):
        se_mask = (atom_array.atom_name == "SE") & mse_mask
        logger.debug(f"Converting {np.sum(se_mask)} MSE residues to MET.")

        # Update residue name, hetero flag, and element
        atom_array.res_name[mse_mask] = "MET"
        atom_array.hetero[mse_mask] = False
        atom_array.atom_name[se_mask] = "SD"

        # ... handle cases for integer or string representatiosn of element
        _elt_prev = atom_array.element[se_mask][0]
        if _elt_prev == "SE":
            atom_array.element[se_mask] = "S"
        elif _elt_prev == 34:
            atom_array.element[se_mask] = 16
        elif _elt_prev == "34":
            atom_array.element[se_mask] = "16"

        # Reorder atoms for canonical MET ordering
        atom_array[mse_mask] = atom_array[mse_mask][struc.info.standardize_order(atom_array[mse_mask])]

    return atom_array


def keep_last_residue(atom_array: AtomArray) -> AtomArray:
    """
    Removes duplicate residues in the atom array, keeping only the last occurrence.

    Args:
        atom_array (AtomArray): The atom array containing the chain information.

    Returns:
        AtomArray: The atom array with duplicate residues removed.
    """
    atom_df = pd.DataFrame(
        {
            "chain_id": atom_array.chain_id,
            "res_id": atom_array.res_id,
            "res_name": atom_array.res_name,
        }
    )

    # Get the mask of duplicates based on the combination of chain_id, res_id, and res_name
    collapsed_df = atom_df.drop_duplicates(subset=["chain_id", "res_id", "res_name"])

    # Get duplicates based on res_id, keeping the last
    duplicate_mask = collapsed_df.duplicated(subset=["chain_id", "res_id"], keep="last")
    duplicates_df = collapsed_df[duplicate_mask]

    # Perform a left merge to find rows in atom_df that are also in duplicates_df
    merged_df = atom_df.merge(duplicates_df, on=["chain_id", "res_id", "res_name"], how="left", indicator=True)

    # Create a mask where True indicates the row is not in duplicates_df
    mask = merged_df["_merge"] == "left_only"

    # Remove rows from atom_array with the deletion mask
    return atom_array[mask]


def maybe_patch_non_polymer_at_symmetry_center(
    atom_array_stack: AtomArrayStack, clash_distance: float = 1.0, clash_ratio: float = 0.5
) -> AtomArrayStack:
    """
    In some PDB entries, non-polymer molecules are placed at the symmetry center and clash with themselves when
    transformed via symmetry operations. We should remove the duplicates in these cases, keeping the identity copy.

    We consider a non-polymer to be clashing with itself if at least `clash_ratio` of its atoms clash with the symmetric copy.

    Examples:
    — PDB ID `7mub` has a potassium ion at the symmetry center that when reflected with the symmetry operation clashes with itself.
    — PDB ID `1xan` has a ligand at a symmetry center that similarly when refelcted clashes with itself.

    Args:
        atom_array (AtomArray): The atom array to be patched.
        clash_distance (float): The distance threshold for two atoms to be considered clashing.
        clash_ratio (float): The percentage of atoms that must clash for the molecule to be considered clashing.

    Returns:
        AtomArray: The patched atom array.
    """
    # Select one model AtomArray to simplify computations
    atom_array = atom_array_stack[0]

    # Filter to only atoms with coordinates to avoid non-physical clashes at the origin
    resolved_atom_array = atom_array[atom_array.occupancy > 0]

    if not np.any(~resolved_atom_array.is_polymer):
        return atom_array_stack  # Early exit
    else:
        non_polymers = resolved_atom_array[~resolved_atom_array.is_polymer]  # [n]

        # Build cell list for rapid distance computations
        cell_list = struc.CellList(non_polymers, cell_size=3.0)

        # Quick check to see whether any non-polymer is closer than 0.05A to any other.
        clash_matrix = cell_list.get_atoms(non_polymers.coord, clash_distance, as_mask=True)  # [n, n]
        identity_matrix = np.identity(len(non_polymers), dtype=bool)
        if np.array_equal(clash_matrix, identity_matrix):
            return atom_array_stack  # Early exit
        else:
            # Remove identity matrix so we don't count self-clashes
            clash_matrix = clash_matrix & ~identity_matrix
        logger.debug("Found clashing non-polymer at a symmetry center, resolving.")

        # Get list of chain_ids with clashing atoms (for computational efficiency)
        clashing_atom_mask = np.sum(clash_matrix, axis=1) > 0
        clashing_chain_ids = np.unique(non_polymers.chain_id[clashing_atom_mask])

        # For each clashing chain, we check whether any non-polymer is clashing with a symmetric copy of itself
        # We count the clashes with each symmetric copy of itself and remove those that have a clash ratio above the threshold
        # We keep the identity transformation, or the lowest transformation ID in the case of multiple symmetric copies
        chain_iids_to_remove = []
        for chain_id in clashing_chain_ids:
            chain_mask = non_polymers.chain_id == chain_id
            mask = chain_mask & clashing_atom_mask  # Mask for clashing atoms in the current chain
            chain_clash_matrix = clash_matrix[mask][:, mask]

            # Loop through possible transformation ID's
            transformation_ids_to_check = sorted(np.unique(non_polymers.transformation_id[mask].astype(str)).tolist())
            while transformation_ids_to_check:
                transformation_id = str(transformation_ids_to_check.pop(0))
                transformation_mask = non_polymers.transformation_id == str(transformation_id)
                # Create matrix where the rows correspond to the atoms of the current transformation and the columns corresponded to the other transformations
                chain_clash_matrix = clash_matrix[mask & transformation_mask][
                    :, mask & ~transformation_mask
                ]  # [current transformation clashing atoms, other transformations clashing atoms]
                # We can then count clashes by transformation ID
                transformation_id_matrix = np.tile(
                    non_polymers.transformation_id[mask & ~transformation_mask], (chain_clash_matrix.shape[0], 1)
                )

                # Apply chain_clash_matrix to transformation_id_matrix so we can count clashes by transformation ID
                clashing_transformation_ids = np.where(chain_clash_matrix, transformation_id_matrix, None).flatten()
                clash_count_by_transformation_id = Counter(
                    clashing_transformation_ids[clashing_transformation_ids != np.array(None)]
                )
                threshold = clash_ratio * np.sum(chain_mask & transformation_mask)

                # For each transformation ID with a clash ratio above the threshold, note the chain_iid to remove, and remove from the list to check
                transformation_ids_to_remove = [
                    trans_id for trans_id, count in clash_count_by_transformation_id.items() if count > threshold
                ]
                chain_iids_to_remove.extend([f"{chain_id}_{trans_id}" for trans_id in transformation_ids_to_remove])
                transformation_ids_to_check = [
                    id_ for id_ in transformation_ids_to_check if str(id_) not in transformation_ids_to_remove
                ]

        # Filter and return
        keep_mask = ~np.isin(atom_array.chain_iid, chain_iids_to_remove)
        atom_array_stack = atom_array_stack[:, keep_mask]
        return atom_array_stack


def add_polymer_annotation(atom_array: AtomArray, chain_info_dict: dict) -> AtomArray:
    """
    Adds an annotation to the atom array to indicate whether a chain is a polymer.

    Args:
        atom_array (AtomArray): The atom array containing the chain information.
        chain_info_dict (dict): Dictionary containing the sequence details of each chain.

    Returns:
        AtomArray: The updated atom array with the polymer annotation added.
    """
    chain_ids = atom_array.get_annotation("chain_id")
    is_polymer = np.array([chain_info_dict[chain_id]["is_polymer"] for chain_id in chain_ids])
    atom_array.set_annotation("is_polymer", is_polymer)
    return atom_array


def update_nonpoly_seq_ids(atom_array: AtomArray, chain_info_dict: dict) -> AtomArray:
    """
    Updates the sequence IDs of non-polymeric chains in the atom array to the author sequence IDs.

    Args:
        atom_array (AtomArray): The atom array containing the chain information.
        chain_info_dict (dict): Dictionary containing the sequence details of each chain.

    Returns:
        AtomArray: The updated atom array with the sequence IDs updated for non-polymeric chains.
    """
    # For non-polymeric chains, we use the author sequence ids
    author_seq_ids = atom_array.get_annotation("auth_seq_id")
    chain_ids = atom_array.get_annotation("chain_id")

    # Create mask based on the is_polymer column
    non_polymer_mask = ~np.array([chain_info_dict[chain_id]["is_polymer"] for chain_id in chain_ids])

    # Update the atom_array_label with the (1-indexed) author sequence ids
    atom_array.res_id[non_polymer_mask] = author_seq_ids[non_polymer_mask]

    return atom_array


def add_bonds_to_bondlist(
    cif_block: CIFBlock,
    atom_array: AtomArray,
    chain_info_dict: dict,
    keep_hydrogens: bool,
    known_residues: list[str],
    get_intra_residue_bonds: callable,
    converted_res: dict = {},
    ignored_res: list = [],
) -> AtomArray:
    """
    Add bonds to the atom array using precomputed CCD data and the mmCIF `struct_conn` field.

    Args:
        cif_block (CIFBlock): The CIF file block containing the structure data.
        atom_array (AtomArray): The atom array to which the bonds will be added.
        chain_info_dict (dict): A dictionary containing information about the chains in the structure.
        keep_hydrogens (bool): Whether to add hydrogens to the atom array.
        known_residues (list): A list of known residues.
        get_intra_residue_bonds (callable): A function that returns the intra-residue bonds for a given residue.
        converted_res (dict): A dictionary containing the residue conversions.
        ignored_res (list): A list of residues to ignore when adding bonds.

    Returns:
        AtomArray: The updated atom array with bonds added.
    """
    # Step 0: Add index to atom_array for ease of access
    atom_array.set_annotation("index", np.arange(len(atom_array)))

    # Step 1: Add inter-residue and inter-chain bonds from the `struct_conn` category in the CIF file
    leaving_atom_indices = []
    struct_conn_bonds, struct_conn_leaving_atom_indices = add_bonds_from_struct_conn(
        cif_block, chain_info_dict, atom_array, converted_res, ignored_res
    )

    if exists(struct_conn_leaving_atom_indices) and len(struct_conn_leaving_atom_indices) > 0:
        leaving_atom_indices.append(np.concatenate(struct_conn_leaving_atom_indices))

    # Step 2: Add inter-residue and intra-residue bonds
    inter_and_intra_residue_bonds = []
    for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
        chain_bonds, chain_leaving_atom_indices = get_inter_and_intra_residue_bonds(
            atom_array=atom_array,
            chain_id=chain_id,
            chain_type=chain_info_dict[chain_id]["type"],
            known_residues=known_residues,
            get_intra_residue_bonds=get_intra_residue_bonds,
            keep_hydrogens=keep_hydrogens,
        )
        if exists(chain_bonds):
            inter_and_intra_residue_bonds.append(chain_bonds)
        if exists(chain_leaving_atom_indices) and len(chain_leaving_atom_indices) > 0:
            leaving_atom_indices.append(chain_leaving_atom_indices)

    if len(struct_conn_bonds) == 0:
        combined_bonds = np.vstack(inter_and_intra_residue_bonds)
    else:
        combined_bonds = np.vstack((np.vstack(inter_and_intra_residue_bonds), struct_conn_bonds))

    # Step 3: Add the bonds to the atom array
    bond_list = struc.BondList(len(atom_array), combined_bonds)
    atom_array.bonds = bond_list

    # Delete leaving atoms and bonds to leaving atoms
    leaving_atoms = np.unique(np.concatenate(leaving_atom_indices))
    all_atom_indices = atom_array.index
    return atom_array[np.setdiff1d(all_atom_indices, leaving_atoms, True)]
