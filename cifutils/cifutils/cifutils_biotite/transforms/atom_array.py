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
import networkx as nx
from collections import defaultdict
from cifutils.cifutils_biotite.utils.bond_utils import (
    add_bonds_from_struct_conn,
    get_inter_and_intra_residue_bonds,
    get_coarse_graph_as_nodes_and_edges,
    get_connected_nodes,
    hash_graph,
    generate_inter_level_bond_hash,
)
from cifutils.cifutils_biotite.common import exists, deduplicate_iterator
from cifutils.cifutils_biotite.enums import ChainType
from cifutils.cifutils_biotite.utils.selection_utils import annot_start_stop_idxs
from cifutils.cifutils_biotite.common import sum_string_arrays

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
        keep_mask = ~np.isin(atom_array.chain_iid, np.array(chain_iids_to_remove, dtype=atom_array.chain_iid.dtype))
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


def add_pn_unit_id_annotation(full_atom_array: AtomArray) -> AtomArray:
    """
    Adds the polymer/non-polymer unit ID (pn_unit_id) annotation to the AtomArray.
    Two covalently bonded ligands are considered one PN unit, but a ligand bonded to a protein is considered two PN units.
    See the README glossary for more details on how we define `chains`, `pn_units`, and `molecules` within this codebase.

    Args:
        full_atom_array (AtomArray): The AtomArray to process.

    Returns:
        full_atom_array (AtomArray): The AtomArray including the `pn_unit_id` annotation.
    """
    # ...initialize the pn_unit_id to chain_id (we will later update for multi-chain non-polymer PN units)
    full_atom_array.add_annotation("pn_unit_id", dtype="<U20")
    full_atom_array.pn_unit_id = full_atom_array.chain_id

    # ...make the NetworkX graph for non-polymer chains
    non_polymer_atom_array = full_atom_array[~full_atom_array.is_polymer]
    connected_chains = get_connected_nodes(*get_coarse_graph_as_nodes_and_edges(non_polymer_atom_array, "chain_id"))

    for connected_chain in connected_chains:
        # ...set the same the pn_unit_id for each chain in the connected chain
        pn_unit_id = ",".join(sorted(connected_chain))
        assert len(pn_unit_id) < 20, f"pn_unit_id exceeds 20 characters: {pn_unit_id}"
        for chain_id in connected_chain:
            full_atom_array.pn_unit_id[full_atom_array.chain_id == chain_id] = np.array(pn_unit_id, dtype="<U20")

    return full_atom_array


def add_molecule_id_annotation(atom_array: AtomArray) -> AtomArray:
    """Adds the molecule ID (molecule_id) annotation to the AtomArray."""
    # ...initialize the pn_unit_id to chain_id (we will later update for multi-chain non-polymer PN units)
    atom_array.add_annotation("molecule_id", dtype=np.int16)

    # ...make the NetworkX graph for all pn_units
    connected_pn_units = get_connected_nodes(*get_coarse_graph_as_nodes_and_edges(atom_array, "pn_unit_id"))

    # ...iterate through connected pn_units
    for idx, connected_pn_unit in enumerate(connected_pn_units):
        # ...set the same the molecule_id for each pn_unit in the connected pn_unit
        molecule_id = idx
        for pn_unit_id in connected_pn_unit:
            atom_array.molecule_id[atom_array.pn_unit_id == pn_unit_id] = molecule_id

    return atom_array


def add_molecule_iid_annotation(atom_array_stack: AtomArrayStack) -> AtomArrayStack:
    """Adds the molecule instance ID (molecule_iid) annotation to the AtomArrayStack"""
    # ...count the number of unique molecule ID's
    num_unique_molecule_ids = len(np.unique(atom_array_stack.molecule_id))
    unique_transformation_ids = np.unique(atom_array_stack.transformation_id)
    transformation_id_int_map = {
        transformation_id: idx for idx, transformation_id in enumerate(unique_transformation_ids)
    }

    # ...set the molecule instance ID (molecule_iid) to the molecule ID plus stride times transformation ID
    offset = np.array(list(map(transformation_id_int_map.get, atom_array_stack.transformation_id)))
    molecule_iid_arr = atom_array_stack.molecule_id + num_unique_molecule_ids * offset
    atom_array_stack.set_annotation("molecule_iid", molecule_iid_arr)

    return atom_array_stack


def annotate_entities(
    atom_array: AtomArray,
    level: str,
    lower_level_id: str | list[str],
    lower_level_entity: str,
    add_inter_level_bond_hash: bool = True,
) -> tuple[AtomArray, dict]:
    """
    Annotates entities in an AtomArray at a given `id` level, based on the connectivity and annotations at the lower level.

    The intended use is, for example:
        - For the `molecule` level, `molecule_entities` are generated for each `molecule_id` based on the connectivty
            at the `pn_unit` level.
        - For the `pn_unit` level, `pn_unit_entities` are generated for each `pn_unit_id` based on the connectivty
            at the `chain` level.
        - For the `chain` level, `chain_entities` are generated for each `chain_id` based on the connectivty at the `residue`
            level.

    Args:
        - atom_array (AtomArray): The AtomArray to process.
        - level (str): The level at which to annotate entities (e.g., "chain", "pn_unit", "entity")
        - lower_level_id (str | list[str]): A list of annotations to consider for determining segment boundaries at a lower level.
            E.g. "pn_unit_id", "chain_id" or "res_id".
        - lower_level_entity (str): The annotation to use for identifying entities at the lower level.
            E.g. "pn_unit_entity", "chain_entity" or "res_name".
        - add_inter_level_bond_hash (bool): Whether to add a hash of the inter-level bonds to the entity hash.
            For some cases, this may be necessary to distinguish entities (e.g., when determining molecule-level
            entities). In others (e.g., for polymers), this may be overkill.

    Returns:
        - Tuple[AtomArray, dict]: A tuple containing:
            - atom_array (AtomArray): The updated AtomArray with the entity annotation.
            - entities_info (dict): A dictionary mapping entity IDs to lists of instance IDs.

    Example:
        >>> atom_array = AtomArray(...)
        >>> entities_at_level, entities_info = annotate_entities(
        ...     atom_array, level="chain", lower_level_id="res_id", lower_level_entity="res_name"
        ... )
        >>> print(entities_at_level)
        [0, 0, 1, 1, 2, 2]
        >>> print(entities_info)
        {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    """
    _next_available_entity_id = 0
    _hash_to_entity_id = {}

    ids_at_level = np.unique(atom_array.get_annotation(level + "_id"))

    # ... initialize annotations to fill
    entities_annotation = np.zeros(len(atom_array), dtype=int)
    entities_info = defaultdict(list)

    for instance_id in np.unique(ids_at_level):
        is_instance = atom_array.get_annotation(level + "_id") == instance_id
        instance = atom_array[is_instance]

        # ... get connectivity and node annotations for the coarse graph at the lower level
        _, edges = get_coarse_graph_as_nodes_and_edges(instance, lower_level_id)
        instance_graph = nx.Graph()
        instance_graph.add_edges_from(edges)

        # ... set node attributes to lower level entities
        lower_level_iter = struc.resutil.segment_iter(
            instance, annot_start_stop_idxs(instance, lower_level_id, add_exclusive_stop=True)
        )
        node_attrs = {
            idx: lower_level_instance.get_annotation(lower_level_entity)[0]
            for idx, lower_level_instance in enumerate(lower_level_iter)
        }
        nx.set_node_attributes(instance_graph, node_attrs, "node")

        # ... create the graph hash
        hash = hash_graph(instance_graph, node_attr="node")

        # ... add the inter-level bond hash (only consider the first lower level id; since we hash at the atom-level, this simplication is valid)
        if add_inter_level_bond_hash:
            hash += generate_inter_level_bond_hash(
                atom_array=instance,
                lower_level_id=lower_level_id[0] if isinstance(lower_level_id, list) else lower_level_id,
                lower_level_entity=lower_level_entity,
            )

        # ... check if the graph has been seen before
        if hash in _hash_to_entity_id:
            entity_id = _hash_to_entity_id[hash]
        else:
            entity_id = _next_available_entity_id
            _hash_to_entity_id[hash] = entity_id
            _next_available_entity_id += 1

        # ... assign the entity id to the instance
        entities_annotation[is_instance] = entity_id
        entities_info[entity_id].append(instance_id)

    atom_array.set_annotation(level + "_entity", entities_annotation)

    return atom_array, dict(entities_info)


def add_chain_iid_annotation(atom_array_stack: AtomArrayStack) -> AtomArrayStack:
    """Adds the chain instance ID (chain_iid) annotation to the AtomArrayStack"""
    # ...concatenate chain_id and transformation_id to create a unique chain instance ID
    chain_iid = sum_string_arrays(
        atom_array_stack.chain_id,
        "_",
        atom_array_stack.transformation_id,
    )
    atom_array_stack.set_annotation("chain_iid", chain_iid)
    return atom_array_stack


def add_pn_unit_iid_annotation(atom_array_stack: AtomArrayStack) -> AtomArrayStack:
    """Adds the polymer/non-polymer unit instance ID (pn_unit_iid) annotation to the AtomArrayStack."""
    # ...create an array that concatenates the pn_unit_id and transformation_id
    # TODO: Make string dynamically resize its length to fit the data
    _temp_pn_unit_iid = sum_string_arrays(atom_array_stack.pn_unit_id, "_", atom_array_stack.transformation_id).astype(
        "<U30"
    )

    atom_array_stack.set_annotation("pn_unit_iid", _temp_pn_unit_iid)

    # ...iterate through unique pn_unit_iids
    # (We implicitly assume that a given pn_unit_id will have the same transformation_id across all atoms in the unit)
    for pn_unit_iid in np.unique(atom_array_stack.pn_unit_iid):
        pn_unit_atom_array = atom_array_stack[:, atom_array_stack.pn_unit_iid == pn_unit_iid]
        # ...get the transformation_id and pn_unit_id (which is the same for all atoms in the unit)
        transformation_id = pn_unit_atom_array.transformation_id[0]
        pn_unit_id = pn_unit_atom_array.pn_unit_id[0].astype(str)

        # ...split apart the pn_unit_id by commas
        pn_unit_ids = pn_unit_id.split(",")

        # ...add the transformation_id to each pn_unit_id
        pn_unit_iids = [f"{unit_id}_{transformation_id}" for unit_id in pn_unit_ids]

        # ...join the instance-level identifiers back into a single string
        pn_unit_iid_formatted = ",".join(pn_unit_iids)

        # ...update the AtomArray with the instance-level identifier
        atom_array_stack.pn_unit_iid[atom_array_stack.pn_unit_iid == pn_unit_iid] = np.array(
            pn_unit_iid_formatted, dtype="<U30"
        )

    return atom_array_stack


def add_iid_annotations_to_assemblies(assemblies_dict: dict) -> dict:
    """Adds chain, PN unit, and molecule IIDs to assembly AtomArrayStacks."""
    for assembly_id, assembly in assemblies_dict.items():
        # ...add chain IIDs
        assembly = add_chain_iid_annotation(assembly)

        # ...add PN unit IIDs
        assembly = add_pn_unit_iid_annotation(assembly)

        # ...add molecule IIDs
        assembly = add_molecule_iid_annotation(assembly)

        # ...update the dictionary
        assemblies_dict[assembly_id] = assembly

    return assemblies_dict


def add_chain_type_annotation(atom_array: AtomArray, chain_info_dict: dict) -> AtomArray:
    """
    Adds a chain_type annotation to the AtomArray.

    Args:
        - atom_array (AtomArray): The full atom array.
        - chain_info_dict (dict): A dictionary mapping chain IDs to chain information,
            including the chain type (output of CIFUtils Biotite parser).

    Returns:
        - AtomArray: The AtomArray with the chain_type annotation added.
    """
    # Add annotation for chain_type as an integer
    atom_array.add_annotation("chain_type", dtype=np.int8)
    for chain_id in np.unique(atom_array.chain_id):
        chain_type = chain_info_dict[chain_id]["type"]
        chain_type_enum = ChainType.from_string(chain_type)
        atom_array.chain_type[atom_array.chain_id == chain_id] = chain_type_enum.value

    # Return the modified atom array
    return atom_array


def add_bonds_to_bondlist_and_remove_leaving_atoms(
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
    After adding the bonds, remove the leaving atoms and bonds to the leaving atoms.

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

    # Step 4: Delete leaving atoms and bonds to leaving atoms
    leaving_atoms = np.unique(np.concatenate(leaving_atom_indices))
    all_atom_indices = atom_array.index
    return atom_array[np.setdiff1d(all_atom_indices, leaving_atoms, True)]
