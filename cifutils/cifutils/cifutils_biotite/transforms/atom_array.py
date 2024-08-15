"""
Transforms operating on Biotite's `AtomArray` objects.
"""

from biotite.structure import AtomArray 
import numpy as np
from cifutils.cifutils_biotite.cifutils_biotite_utils import logger
import biotite.structure as struc

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