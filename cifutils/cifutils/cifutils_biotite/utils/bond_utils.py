"""
Utility functions for the detection, and creation, of bonds in a structure.
"""

import numpy as np
from biotite.structure import AtomArray
import biotite.structure as struc
from cifutils.cifutils_biotite.common import exists
from biotite.structure.io.pdbx import CIFBlock
import logging
from cifutils.cifutils_biotite.utils.atom_matching_utils import get_matching_atom
from cifutils.cifutils_biotite.utils.cifutils_biotite_utils import (
    get_bond_type_from_order_and_is_aromatic,
)
from cifutils.cifutils_biotite.transforms.categories import category_to_df
from functools import lru_cache

logger = logging.getLogger(__name__)


def cached_bond_utils_factory(data_by_residue: callable) -> tuple[callable, callable]:
    """
    Factory function to build cached helper functions for for constructing bonds.
    We must invoke closure since functions are not hashable and cannot be used as keys in lru_cache.

    Args:
        data_by_residue (callable): A function that returns CCD data for a given residue name.
    """

    @lru_cache(maxsize=None)
    def get_intra_residue_bonds(residue_name: dict, add_hydrogens: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve intra-residue bonds for a given residue.

        Args:
            residue_data (dict): Dictionary containing keys for the intra-residue bonds and constituent atoms, derived from OpenBabel.
            add_hydrogens (bool): Whether or not hydrogens are being added to the structure. Relevant for bond removal.

        Returns:
            tuple: Three arrays representing the atom indices and bond types within the residue frame.
        """
        residue_data = data_by_residue(residue_name)

        # If we aren't adding hydrogens, we need to remove any bonds to hydrogens, and any hydrogen atoms from the atom list
        if not add_hydrogens:
            residue_data["intra_residue_bonds"] = [
                # NOTE: We are assuming that all, and only, hydrogen atoms are named with an 'H' prefix
                bond
                for bond in residue_data["intra_residue_bonds"]
                if not bond["atom_a_id"].startswith("H") and not bond["atom_b_id"].startswith("H")
            ]
            residue_data["atoms"] = {
                atom_id: atom_data
                for atom_id, atom_data in residue_data["atoms"].items()
                if not atom_data["element"] == 1
            }

        # Create a mapping of atom IDs to indices
        atom_id_to_index = {atom_id: index for index, atom_id in enumerate(residue_data["atoms"].keys())}
        atom_a_indices = []
        atom_b_indices = []
        bond_types = []
        for bond in residue_data["intra_residue_bonds"]:
            atom_a_index = atom_id_to_index[bond["atom_a_id"]]
            atom_b_index = atom_id_to_index[bond["atom_b_id"]]
            bond_type = get_bond_type_from_order_and_is_aromatic(bond["order"], bond["is_aromatic"])
            atom_a_indices.append(atom_a_index)
            atom_b_indices.append(atom_b_index)
            bond_types.append(bond_type)
        return np.array(atom_a_indices), np.array(atom_b_indices), np.array(bond_types)

    return get_intra_residue_bonds


def add_bonds_from_struct_conn(
    cif_block: CIFBlock,
    chain_info_dict: dict,
    atom_array: AtomArray,
    converted_res: dict = {},
    ignored_res: list = [],
) -> tuple[list[list[int]], list[int]]:
    """
    Adds bonds from the 'struct_conn' category of a CIF block to an atom array. Only covalent bonds are considered.

    Args:
        cif_block (CIFBlock): The CIF block for the entry.
        chain_info_dif (Dict): A dictionary containing information about the chains.
        atom_array (AtomArray): The atom array used to get atom indices.
        converted_res (dict): A dictionary of residues that have been converted to a different residue name.
        ignored_res (list): A list of residues that should be ignored.

    Returns:
        struct_conn_bonds: A List of bonds to be added to the atom array.
        leaving_atom_indices: A List of indices of atoms that are leaving groups for bookkeeping.
    """
    if "struct_conn" not in cif_block:
        return [], []

    struct_conn_df = category_to_df(cif_block, "struct_conn")
    struct_conn_df = struct_conn_df[
        struct_conn_df["conn_type_id"].str.startswith("covale")
    ]  # Only consider covalent bonds (throw out disulfide bods, metal coordination covalent bonds, hydrogen bonds)

    struct_conn_bonds = []
    leaving_atom_indices = []

    if not struct_conn_df.empty:
        logger.debug(f"Attempting to add {len(struct_conn_df)} bonds from `struct_conn`")
        for _, row in struct_conn_df.iterrows():
            a_chain_id = row["ptnr1_label_asym_id"]
            b_chain_id = row["ptnr2_label_asym_id"]
            a_atom_id = row["ptnr1_label_atom_id"]
            b_atom_id = row["ptnr2_label_atom_id"]
            a_res_name = converted_res.get(row["ptnr1_label_comp_id"], row["ptnr1_label_comp_id"])
            b_res_name = converted_res.get(row["ptnr2_label_comp_id"], row["ptnr2_label_comp_id"])

            # Check if res_name is ignored (e.g., water, crystallization aids, ignored ligands), in which case we early exit:
            if (a_res_name in ignored_res) or (b_res_name in ignored_res):
                # skip
                continue

            # Check if the chains for each of the residues exist in the structure
            if (a_chain_id not in chain_info_dict) or (b_chain_id not in chain_info_dict):
                # skip, but warn
                logger.warning(
                    f"Found covalent bond involving chains {a_chain_id} and {b_chain_id}, but at least one "
                    "chain was removed during cleaning. This is likely because the chain is made up of a "
                    "residue that is not in the pre-compiled CCD. This should automatically"
                    f"be resolved once you update your CCD."
                )
                continue

            a_seq_id = (
                row["ptnr1_label_seq_id"] if chain_info_dict[a_chain_id]["is_polymer"] else row["ptnr1_auth_seq_id"]
            )
            b_seq_id = (
                row["ptnr2_label_seq_id"] if chain_info_dict[b_chain_id]["is_polymer"] else row["ptnr2_auth_seq_id"]
            )

            # Get the indices of the atoms and append to the list
            residue_a = atom_array[
                (atom_array.chain_id == a_chain_id)
                & (atom_array.res_id == int(a_seq_id))
                & (atom_array.res_name == a_res_name)
            ]
            residue_b = atom_array[
                (atom_array.chain_id == b_chain_id)
                & (atom_array.res_id == int(b_seq_id))
                & (atom_array.res_name == b_res_name)
            ]

            # Ensure that the we picked the correct residue (to handle sequence heterogeneity; see PDB ID `3nez` for an example)
            #  (short circuit eval to avoid indexing errors in cases where we don't have one of the residues due to seq. heterogeneity
            #   - e.g. 3nez)
            if (
                (len(residue_a) == 0)
                or (len(residue_b) == 0)
                or (a_res_name != residue_a.res_name[0])
                or (b_res_name != residue_b.res_name[0])
            ):
                # skip, but warn
                logger.warning(
                    f"Covalent bond involving residues {a_chain_id}:{a_seq_id}:{a_res_name} and "
                    f"{b_chain_id}:{b_seq_id}:{b_res_name} was found in `struct_conn`, but the "
                    f"residues are not present in the atom array. This is likely due to "
                    f"resolved sequence heterogeneity which removed one of the residues."
                )
                continue

            # Get the atoms that participate in the bond
            atom_a = get_matching_atom(residue_a, a_atom_id)
            atom_b = get_matching_atom(residue_b, b_atom_id)

            struct_conn_bonds.append([atom_a.index[0], atom_b.index[0], struc.BondType.SINGLE])

            # Leaving group bookkeeping
            leaving_atom_indices.append(residue_a.index[np.isin(residue_a.atom_name, atom_a.leaving_group[0])])
            leaving_atom_indices.append(residue_b.index[np.isin(residue_b.atom_name, atom_b.leaving_group[0])])

    return struct_conn_bonds, leaving_atom_indices


def get_inter_and_intra_residue_bonds(
    atom_array: AtomArray,
    chain_id: str,
    chain_type: str,
    known_residues: list[str],
    get_intra_residue_bonds: callable,
    add_hydrogens: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds inter-residue and intra_residue bonds to an atom array for a given chain.

    Args:
        atom_array (AtomArray): The atom array to which the bonds are added.
        chain_id (str): The ID of the chain for which bonds are added.
        chain_type (str): The type of the chain, used to determine the type of bond.
        known_residues (list): A list of valid residue names.
        get_intra_residue_bonds (str): A function that takes as input a residue name and returns tuples of intra-residue bonds.
        add_hydrogens (str): Whether we are adding hydrogens to the residues (relevant for removing leaving groups).

    Returns:
        intra_residue_bonds: An np.array of intra-residue bonds to be added to the atom array.
        leaving_atom_indices: An np.array of indices of atom indices that are leaving groups for bookkeeping.
    """
    known_residues = set(known_residues)

    # Possible types given at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
    atom_pairs = {
        "polydeoxyribonucleotide": ("O3'", "P"),  # phosphodiester bond
        "polydeoxyribonucleotide/polyribonucleotide hybrid": ("O3'", "P"),  # phosphodiester bond
        "polypeptide(D)": ("C", "N"),  # peptide bond
        "polypeptide(L)": ("C", "N"),  # peptide bond
        "polyribonucleotide": ("O3'", "P"),  # phosphodiester bond
    }

    # Append as we go along and then concatenate at the end
    inter_residue_bonds = []
    atom_a_intra_residue_indices = []
    atom_b_intra_residue_indices = []
    intra_residue_bond_types = []
    leaving_atom_indices = []

    bond_atoms = atom_pairs.get(chain_type, None)
    atom_chain_array = atom_array[atom_array.chain_id == chain_id]

    # Create iterators for the current and next residues
    residues = list(struc.residue_iter(atom_chain_array))

    for i in range(len(residues)):
        current_res = residues[i]
        next_res = residues[i + 1] if i + 1 < len(residues) else None
        # Add inter-residue bond if there is a next residue
        if next_res and exists(bond_atoms):
            atom_a = current_res[current_res.atom_name == bond_atoms[0]]
            atom_b = next_res[next_res.atom_name == bond_atoms[1]]
            if atom_a and atom_b:
                inter_residue_bonds.append([atom_a.index[0], atom_b.index[0], struc.BondType.SINGLE])

                # Leaving group bookkeeping
                leaving_atom_indices.append(current_res.index[np.isin(current_res.atom_name, atom_a.leaving_group[0])])
                leaving_atom_indices.append(next_res.index[np.isin(next_res.atom_name, atom_b.leaving_group[0])])

        # Add intra-residue bonds for the current residue
        residue_name = current_res.res_name[
            0
        ]  # current_res.res_name is a list of identical values, so we just take the first one
        if residue_name in known_residues:
            atom_a_local_indices, atom_b_local_indices, bond_types = get_intra_residue_bonds(
                residue_name, add_hydrogens
            )
            if atom_a_local_indices.size and atom_b_local_indices.size and bond_types.size:
                atom_a_intra_residue_indices.append(current_res.index[atom_a_local_indices])
                atom_b_intra_residue_indices.append(current_res.index[atom_b_local_indices])
                intra_residue_bond_types.append(bond_types)

    # At the end, we concatenate the lists to form the final arrays
    if atom_a_intra_residue_indices and atom_b_intra_residue_indices and intra_residue_bond_types:
        intra_residue_bonds = np.column_stack(
            (
                np.concatenate(atom_a_intra_residue_indices),
                np.concatenate(atom_b_intra_residue_indices),
                np.concatenate(intra_residue_bond_types),
            )
        )
    else:
        intra_residue_bonds = np.array([], dtype=np.int32).reshape(0, 3)

    leaving_atom_indices = (
        np.concatenate(leaving_atom_indices) if leaving_atom_indices else np.array([], dtype=np.int32)
    )

    if inter_residue_bonds:
        return np.vstack((np.array(inter_residue_bonds), intra_residue_bonds)), leaving_atom_indices
    else:
        return intra_residue_bonds, leaving_atom_indices
