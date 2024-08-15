"""
Utility functions for the detection, and creation, of bonds in a structure. 
"""
import numpy as np
from biotite.structure import AtomArray, Atom, AtomArrayStack
import biotite.structure as struc
from cifutils.cifutils_biotite.common import exists
from biotite.structure.io.pdbx import CIFBlock, CIFCategory
import logging
from cifutils.cifutils_biotite.utils.atom_matching_utils import get_matching_atom
from functools import lru_cache
from cifutils.cifutils_biotite.utils.cifutils_biotite_utils import (
    apply_assembly_transformation,
    deduplicate_iterator,
    fix_bonded_atom_charges,
    get_bond_type_from_order_and_is_aromatic,
    parse_transformations,
    read_cif_file,
)
from cifutils.cifutils_biotite.transforms.categories import category_to_df

logger = logging.getLogger(__name__)

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
    - cif_block (CIFBlock): The CIF block for the entry.
    - chain_info_dif (Dict): A dictionary containing information about the chains.
    - atom_array (AtomArray): The atom array used to get atom indices.
    - converted_res (dict): A dictionary of residues that have been converted to a different residue name.
    - ignored_res (list): A list of residues that should be ignored.

    Returns:
    - struct_conn_bonds: A List of bonds to be added to the atom array.
    - leaving_atom_indices: A List of indices of atoms that are leaving groups for bookkeeping.
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

            # Fix charges
            atom_a_updates = fix_bonded_atom_charges(atom_a[0])
            atom_a.charge, atom_a.hyb, atom_a.nhyd = (
                np.array([atom_a_updates["charge"]]),
                np.array([atom_a_updates["hyb"]]),
                np.array([atom_a_updates["nhyd"]]),
            )

            atom_b_updates = fix_bonded_atom_charges(atom_b[0])
            atom_b.charge, atom_b.hyb, atom_b.nhyd = (
                np.array([atom_b_updates["charge"]]),
                np.array([atom_b_updates["hyb"]]),
                np.array([atom_b_updates["nhyd"]]),
            )

    return struct_conn_bonds, leaving_atom_indices
