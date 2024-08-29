"""
Utility functions to handle creation and manipulation of residues.
"""

__all__ = [
    "cached_residue_utils_factory",
    "add_missing_atoms_as_unresolved",
    "get_chem_comp_type",
]

import numpy as np
from biotite.structure import AtomArray, Atom
import biotite.structure as struc
import logging
from cifutils.common import (
    deduplicate_iterator,
    exists,
)
from functools import lru_cache
from cifutils.utils.atom_matching_utils import standardize_heavy_atom_ids
from cifutils.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER

logger = logging.getLogger("cifutils")


def cached_residue_utils_factory(known_residues: list[str], data_by_residue: callable) -> tuple[callable, callable]:
    """
    Factory function to build cached helper functions for building residue atoms.
    We must invoke closure since dictionaries are not hashable and cannot be used as keys in lru_cache.

    Args:
        known_residues (list): A list of valid residue names.
        data_by_residue (callable): A function that returns CCD data for a given residue name.
    """
    known_residues_set = set(known_residues)

    @lru_cache(maxsize=None)
    def build_residue_atoms(residue_name: str, keep_hydrogens: bool) -> list[Atom]:
        """
        Build a list of atoms for a given residue name from CCD data.

        Args:
            residue_name (str): The name of the residue.
            keep_hydrogens (bool): Whether to add hydrogens to the residue.

        Returns:
            list[Atom]: A list of Atom objects initialized with zero coordinates.
        """
        if residue_name not in known_residues_set:
            raise ValueError(f"Residue {residue_name} not found in precompiled CCD data.")

        ccd_atoms = data_by_residue(residue_name)["atoms"]
        atom_list = [
            struc.Atom(
                [0.0, 0.0, 0.0],
                res_name=residue_name,
                atom_name=atom_name,
                element=atom_data["element"],
                charge=atom_data["charge"],
                hyb=atom_data["hyb"],
                nhyd=atom_data["nhyd"],
                hvydeg=atom_data["hvydeg"],
                leaving_atom_flag=atom_data["leaving_atom_flag"],
                leaving_group=atom_data["leaving_group"],
                is_metal=atom_data["is_metal"],
            )
            for atom_name, atom_data in ccd_atoms.items()
        ]

        # Remove hydrogens, if necessary
        if not keep_hydrogens:
            atom_list = [atom for atom in atom_list if atom.element != 1]

        return atom_list

    return build_residue_atoms


def add_missing_atoms_as_unresolved(
    atom_array: AtomArray, chain_info_dict: dict, keep_hydrogens: bool, build_residue_atoms: callable
) -> AtomArray:
    """
    Add missing atoms to a polymer chain based on its sequence.

    Iterates through the residues in a given chain, identifies missing atoms based on the reference residue,
    and inserts the missing atoms into the atom array. Also augments atom data with Open Babel data.

    Args:
        atom_array (AtomArray): An array of atom objects representing the current state of the polymer chain.
        chain_info_dict (dict): A dictionary containing chain information, including whether the chain is a polymer and the sequence of residues.
        keep_hydrogens (bool): Whether to add hydrogens when building the residue atoms. Most, if not all, hydrogens will be unresolved.

    Returns:
        AtomArray: An updated array of atom objects with the missing atoms added.

    TODO: Break into two functions, one that adds missing atoms, and another that adds the OpenBabel annotations (e.g., leaving group) that we need to make bonds.
    """
    full_atom_list = []
    residue_ids = []
    chain_ids = []
    # NOTE: We need deduplicate_iterator() since biotite considers a decrease in sequence_id as a new chain (see `5xnl`)
    for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
        # Iterate through the sequence and create all atoms with zero coordinates
        residue_name_list = chain_info_dict[chain_id]["residue_name_list"]
        if chain_info_dict[chain_id]["is_polymer"]:
            # For polymer chains, we can just assign the residue ID based on the sequential index, and don't need to worry about UNL (unknown ligands)
            # NOTE: In the future, we may want to allow more flexibility for polymers (e.g., custom NCAA); for now, they will raise an error in `build_residue_atoms`
            # We could achieve this through adding a "NCR"  (non-canonical residue) name, for instance, which would behavior similarly to "UNL"
            for residue_index_sequential, residue_name in enumerate(residue_name_list, start=1):
                residue_atom_list = build_residue_atoms(residue_name=residue_name, keep_hydrogens=keep_hydrogens)
                # We assign the residue ID as the sequential index for polymers, consistent with the PDB label ids (but not author ids)
                residue_ids.append(np.full(len(residue_atom_list), residue_index_sequential))

                chain_ids.append(np.full(len(residue_atom_list), chain_id))
                full_atom_list.append(residue_atom_list)
        else:
            # For non-polymer chains, we need to assign the residue ID based on the residue ID list, and handle UNL (unknown ligands)
            for residue_index_sequential, residue_name in enumerate(residue_name_list, start=1):
                residue_id_original = chain_info_dict[chain_id]["residue_id_list"][
                    residue_index_sequential - 1
                ]  # Recall that residue_index_sequential is 1-indexed
                if residue_name == "UNL":
                    # Directly add the atoms for the unknown ligand to the full_atom_list:
                    # ...get the unknown ligand atoms from the AtomArray
                    unknown_residue_atom_array = atom_array[
                        (atom_array.chain_id == chain_id)
                        & (atom_array.res_name == residue_name)
                        & (atom_array.res_id == residue_id_original)
                    ]

                    # ...limit the annotations to only those in the new AtomArray
                    new_annotations = [
                        "chain_id",
                        "res_id",
                        "ins_code",
                        "res_name",
                        "hetero",
                        "atom_name",
                        "element",
                        "charge",
                        "nhyd",
                        "hyb",
                        "hvydeg",
                        "leaving_atom_flag",
                        "leaving_group",
                        "is_metal",
                    ]
                    for annotation in unknown_residue_atom_array.get_annotation_categories():
                        if annotation not in new_annotations:
                            unknown_residue_atom_array.del_annotation(annotation)

                    # ...add the necessary annotations that are not present in the AtomArray, and default them
                    empty_lists_array = np.empty(len(unknown_residue_atom_array), dtype=object)
                    empty_lists_array.fill(
                        []
                    )  # See: https://stackoverflow.com/questions/43483663/how-do-i-make-a-grid-of-empty-lists-in-numpy-that-accepts-appending
                    unknown_residue_atom_array.set_annotation("leaving_group", empty_lists_array)
                    unknown_residue_atom_array.add_annotation("leaving_atom_flag", dtype=bool)
                    unknown_residue_atom_array.add_annotation("is_metal", dtype=bool)
                    unknown_residue_atom_array.add_annotation(
                        "charge", dtype=int
                    )  # NOTE: Will be removed in the future
                    unknown_residue_atom_array.add_annotation(
                        "hvydeg", dtype=int
                    )  # NOTE: Will be removed in the future
                    unknown_residue_atom_array.add_annotation("hyb", dtype=int)  # NOTE: Will be removed in the future
                    unknown_residue_atom_array.add_annotation("nhyd", dtype=int)  # NOTE: Will be removed in the future

                    # ...convert element to int to match the outputs from build_residue_atoms
                    unknown_residue_atom_array.element = np.array(
                        [ELEMENT_NAME_TO_ATOMIC_NUMBER[el] for el in unknown_residue_atom_array.element]
                    )

                    # ...decompose into a list of atoms
                    residue_atom_list = [atom for atom in unknown_residue_atom_array]
                else:
                    residue_atom_list = build_residue_atoms(residue_name, keep_hydrogens)
                # Preserve the residue ID for non-polymer chains (e.g., often starts at 101)
                residue_ids.append(np.full(len(residue_atom_list), residue_id_original))

                chain_ids.append(np.full(len(residue_atom_list), chain_id))
                full_atom_list.append(residue_atom_list)

    # Create atom array object and flatten residue_ids and chain_ids
    full_atom_array = struc.array(np.concatenate(full_atom_list))
    full_atom_array.chain_id = np.concatenate(chain_ids)
    full_atom_array.res_id = np.concatenate(residue_ids)
    # Shenanigans to fix the data type of element
    elements = full_atom_array.element.astype(int)
    full_atom_array.del_annotation("element")
    full_atom_array.add_annotation("element", dtype=int)
    full_atom_array.set_annotation("element", elements)

    # Standardize heavy atom naming
    atom_array.atom_name = standardize_heavy_atom_ids(atom_array)
    full_atom_array.atom_name = standardize_heavy_atom_ids(full_atom_array)

    # Compute index mapping between `full_atom_array`
    # ... get lookup table of id -> idx for full_atom_array
    id_to_idx_full_atom_array = {
        id: idx
        for idx, id in enumerate(
            zip(
                full_atom_array.chain_id,
                full_atom_array.res_id,
                full_atom_array.res_name,
                full_atom_array.atom_name,
            )
        )
    }
    assert len(id_to_idx_full_atom_array) == len(full_atom_array), "Duplicate atom ids in `full_atom_array`!"

    # ... inspect all present atoms in `atom_array` and get the matching idx in `full_atom_array`
    full_atom_array_match_idx = []
    atom_array_match_idx = []
    _failed_to_match = []
    for idx, id in enumerate(zip(atom_array.chain_id, atom_array.res_id, atom_array.res_name, atom_array.atom_name)):
        if id in id_to_idx_full_atom_array:
            full_atom_array_match_idx.append(id_to_idx_full_atom_array[id])
            atom_array_match_idx.append(idx)
        else:
            _failed_to_match.append(idx)
            logger.info(f"Atom {id} not found in `full_atom_array`!")

    # ... turn arrays into np arrays
    full_atom_array_match_idx = np.array(full_atom_array_match_idx)
    atom_array_match_idx = np.array(atom_array_match_idx)

    # ... verify that there is a 1-to-1 mapping between the two arrays
    if not len(full_atom_array_match_idx) == len(atom_array_match_idx):
        unique, counts = np.unique(full_atom_array_match_idx, return_counts=True)
        # ... find duplicates in `full_atom_array_match_idx` for error message
        duplicates = unique[counts > 1]
        duplicates_id = full_atom_array[full_atom_array_match_idx][duplicates]
        raise ValueError(
            f"Mismatch between `full_atom_array` and `atom_array`! Found {len(duplicates)} duplicates in `full_atom_array_match_idx`:\n{duplicates_id}"
        )

    # Carry over the annotations from `atom_array` to `full_atom_array` for corresponding atoms
    # ...always carry over coordinates and occupancy
    full_atom_array.coord[full_atom_array_match_idx] = atom_array[atom_array_match_idx].coord
    occupancy = np.zeros(len(full_atom_array), dtype=np.float32)
    occupancy[full_atom_array_match_idx] = atom_array[atom_array_match_idx].occupancy
    full_atom_array.set_annotation("occupancy", occupancy)
    # ...carry over b_factor, if present
    if "b_factor" in atom_array.get_annotation_categories():
        b_factor = np.zeros(len(full_atom_array), dtype=np.float32)
        b_factor[full_atom_array_match_idx] = atom_array[atom_array_match_idx].b_factor
        full_atom_array.set_annotation("b_factor", b_factor)

    # If any heavy atom in a residue cannot be matched, then mask the whole residue (i.e., set occupancy to 0)
    if len(_failed_to_match) > 0:
        failing_atoms = atom_array[np.array(_failed_to_match)]
        is_heavy = ~np.isin(failing_atoms.element, ["H", "D", "T"])
        if len(failing_atoms[is_heavy]) > 0:
            for atom in failing_atoms[is_heavy]:
                chain_id, res_id, res_name = atom.chain_id, atom.res_id, atom.res_name
                residue_mask = (
                    (full_atom_array.chain_id == chain_id)
                    & (full_atom_array.res_id == res_id)
                    & (full_atom_array.res_name == res_name)
                )
                full_atom_array.occupancy[residue_mask] = 0
            logger.info(
                f"Masked residues for {len(failing_atoms[is_heavy])} heavy atoms in `atom_array` that failed to match."
            )

    # ...ensure no hydrogens are present in the full atom array, if we're not keeping hydrogens
    if not keep_hydrogens:
        # It's possible, especially with unknown ligands, that some hydrogens snuck through
        full_atom_array = full_atom_array[full_atom_array.element != "1"]

    return full_atom_array


@lru_cache(maxsize=10000)
def get_chem_comp_type(res_name: str) -> str:
    """
    Get the chemical component type for a residue name from the Chemical Component Dictionary (CCD).
    Can be combined with CHEM_TYPES from `cifutils_biotite.constants` to determine if a residue is a protein, nucleic acid, or carbohydrate.

    Args:
        res_name (str): The residue name.

    Example:
        >>> get_chem_comp_type("ALA")
        'L-PEPTIDE LINKING'
    """
    chemcomp_type = struc.info.ccd.get_from_ccd("chem_comp", res_name, "type")
    if exists(chemcomp_type):
        return chemcomp_type[0].upper()
    else:
        logger.info(f"Chemical component type for `{res_name}` not found in CCD. Using 'other'.")
        return "other".upper()
