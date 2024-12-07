"""
Utility functions for the detection, and creation, of bonds in a structure.
"""

__all__ = [
    "get_intra_residue_bonds",
    "add_bonds_from_struct_conn",
    "get_inter_and_intra_residue_bonds",
    "get_coarse_graph_as_nodes_and_edges",
    "get_connected_nodes",
    "hash_graph",
    "generate_inter_level_bond_hash",
]

import numpy as np
from biotite.structure import AtomArray
import biotite.structure as struc
from cifutils.common import exists
from biotite.structure.io.pdbx import CIFBlock
import logging
from cifutils.utils.atom_matching_utils import get_matching_atom
from cifutils.transforms.categories import category_to_df
from cifutils.common import to_hashable
from cifutils.enums import ChainType
from functools import cache
import networkx as nx
from cifutils.constants import CCD_MIRROR_PATH, HYDROGEN_LIKE_SYMBOLS
from cifutils.utils.ccd import get_available_ccd_codes, get_ccd_component
import os
from cifutils.common import not_isin

logger = logging.getLogger("cifutils")


def _get_bond_type_from_order_and_is_aromatic(order, is_aromatic):
    """Get the biotite struc.BondType from the bond order and aromaticity."""
    aromatic_bond_types = {
        1: struc.BondType.AROMATIC_SINGLE,
        2: struc.BondType.AROMATIC_DOUBLE,
        3: struc.BondType.AROMATIC_TRIPLE,
    }

    non_aromatic_bond_types = {
        1: struc.BondType.SINGLE,
        2: struc.BondType.DOUBLE,
        3: struc.BondType.TRIPLE,
        4: struc.BondType.QUADRUPLE,
    }

    return (
        aromatic_bond_types.get(order, struc.BondType.ANY)
        if is_aromatic
        else non_aromatic_bond_types.get(order, struc.BondType.ANY)
    )


@cache
def get_intra_residue_bonds(
    ccd_code: str, keep_hydrogens: bool, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieve intra-residue bonds for a given residue.

    Args:
        ccd_code (str): The CCD code for the residue. E.g. `ALA` for alanine, `NAP` for N-acetyl-D-glucosamine.
        keep_hydrogens (bool): Whether or not hydrogens are being added to the structure. Relevant for bond removal.
        ccd_mirror_path (os.PathLike): Path to the local mirror of the Chemical Component Dictionary (recommended).
            If not provided, Biotite's built-in CCD will be used.

    Returns:
        tuple: Three arrays representing the atom indices and bond types within the residue frame.
    """
    # TODO:(smathis) Modify
    chem_comp = get_ccd_component(ccd_code, ccd_mirror_path=ccd_mirror_path, coords=None, add_properties=False)

    if not keep_hydrogens:
        chem_comp = chem_comp[not_isin(chem_comp.element, HYDROGEN_LIKE_SYMBOLS)]

    bonds = chem_comp.bonds.as_array()
    return bonds[:, 0], bonds[:, 1], bonds[:, 2]


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
                logger.info(
                    f"Found covalent bond involving chains {a_chain_id} and {b_chain_id}, but at least one "
                    "chain was removed during cleaning. This is likely because the chain is made up of a "
                    "residue that is not in the pre-compiled CCD. This should automatically"
                    f"be resolved once you update your CCD."
                )
                continue

            # For non-polymers, we use the auth_seq_id, otherwise we use the label_seq_id
            # NOTE: For non-polymers within PDB files, we may not have the auth_seq_id; we use the label_seq_id in such cases
            a_seq_id = (
                row["ptnr1_label_seq_id"]
                if chain_info_dict[a_chain_id]["is_polymer"] or "ptnr1_auth_seq_id" not in row
                else row["ptnr1_auth_seq_id"]
            )
            b_seq_id = (
                row["ptnr2_label_seq_id"]
                if chain_info_dict[b_chain_id]["is_polymer"] or "ptnr2_auth_seq_id" not in row
                else row["ptnr2_auth_seq_id"]
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
                logger.info(
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


def get_inter_and_intra_residue_bonds(
    atom_array: AtomArray,
    chain_id: str,
    chain_type: str,
    keep_hydrogens: bool,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds inter-residue and intra_residue bonds to an atom array for a given chain.

    Args:
        atom_array (AtomArray): The atom array to which the bonds are added.
        chain_id (str): The ID of the chain for which bonds are added.
        chain_type (str): The type of the chain, used to determine the type of bond.
        known_residues (list): A list of valid residue names.
        keep_hydrogens (bool): Whether we are adding hydrogens to the residues (relevant for removing leaving groups).
        processed_ccd_path (os.PathLike): The path to the processed CCD data from which
            reference bond information will be read.

    Returns:
        intra_residue_bonds: An np.array of intra-residue bonds to be added to the atom array.
        leaving_atom_indices: An np.array of indices of atom indices that are leaving groups for bookkeeping.
    """
    # Possible types given at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
    # TODO: This should really be a property of the residues, to deal with peptide-nucleic acid hybrids
    atom_pairs = {
        ChainType.DNA: ("O3'", "P"),  # phosphodiester bond
        ChainType.DNA_RNA_HYBRID: ("O3'", "P"),  # phosphodiester bond
        ChainType.POLYPEPTIDE_D: ("C", "N"),  # peptide bond
        ChainType.POLYPEPTIDE_L: ("C", "N"),  # peptide bond
        ChainType.CYCLIC_PSEUDO_PEPTIDE: ("C", "N"),  # peptide bond
        ChainType.RNA: ("O3'", "P"),  # phosphodiester bond
    }

    # Append as we go along and then concatenate at the end
    inter_residue_bonds = []
    atom_a_intra_residue_indices = []
    atom_b_intra_residue_indices = []
    intra_residue_bond_types = []
    leaving_atom_indices = []

    chain_type = ChainType.as_enum(chain_type)
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

        # Add intra-residue bonds for the current residue
        ccd_code = current_res.res_name[
            0
        ]  # current_res.res_name is a list of identical values, so we just take the first one
        if ccd_code in get_available_ccd_codes(ccd_mirror_path=ccd_mirror_path):
            atom_a_local_indices, atom_b_local_indices, bond_types = get_intra_residue_bonds(
                ccd_code, keep_hydrogens, ccd_mirror_path=ccd_mirror_path
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


def get_coarse_graph_as_nodes_and_edges(atom_array: AtomArray, annotations: str | tuple[str]):
    """
    Returns the coarse-grained nodes and edges at the given annotation level based on the atom array's bond connectivity.

    Args:
        - atom_array (AtomArray): The atom array containing atomic information and bonds.
        - annotations (str | tuple[str]): A single annotation or a tuple of annotations to be used for node
            identification.

    Returns:
        - nodes (np.ndarray): An array of unique nodes, each represented by a combination of annotations.
        - edges (np.ndarray): An array of edges, where each edge is a tuple of node indices representing a bond
            between two nodes.

    Example:
        >>> atom_array = cached_parse("5ocm")["atom_array"]
        >>> nodes, edges = get_coarse_graph(atom_array, ["chain_id", "transformation_id"])
        >>> print(nodes)
        array([('A', '1'), ('F', '1'), ('G', '1'), ('H', '1'), ('I', '1'),
               ('W', '1'), ('X', '1'), ('Y', '1')],
              dtype=[('chain_id', '<U4'), ('transformation_id', '<U1')])
        >>> print(edges)
        array([[0, 0],
               [1, 1],
               [2, 2],
               [3, 3],
               [5, 5],
               [6, 6]])
    """
    annotations = [annotations] if isinstance(annotations, str) else annotations

    atom1, atom2, _ = atom_array.bonds.as_array().T

    if len(annotations) > 1:
        _annots = np.zeros(
            len(atom_array), dtype=[(annot, atom_array.get_annotation(annot).dtype) for annot in annotations]
        )
        for annot in annotations:
            _annots[annot] = atom_array.get_annotation(annot)  # [n_atoms, n_annotations]
    else:
        _annots = atom_array.get_annotation(annotations[0])  # [n_atoms]

    annot1 = _annots[atom1]  # [n_bonds, n_annotations]
    annot2 = _annots[atom2]  # [n_bonds, n_annotations]

    nodes = np.unique(_annots, axis=0)  # [n_nodes, n_annotations]
    self_edges = np.vstack([nodes, nodes]).T  # [n_nodes, 2]
    edges = np.unique(np.vstack([self_edges, np.vstack([annot1, annot2]).T]), axis=0)  # [n_edges, 2]

    # Map nodes to integers
    node_to_idx = {to_hashable(node): i for i, node in enumerate(nodes)}
    if len(edges) > 0:
        edges = np.apply_along_axis(
            lambda x: (node_to_idx[to_hashable(x[0])], node_to_idx[to_hashable(x[1])]), 1, edges
        )

    return nodes, edges


def get_connected_nodes(nodes: np.ndarray, edges: np.ndarray):
    """Returns connected nodes as a mapped list given corresponding arrays of nodes and edges."""
    # ...make the graph
    graph = nx.Graph()
    graph.add_edges_from(edges)

    # ...return lists of connected chains
    return list(map(lambda xs: [nodes[x] for x in xs], nx.connected_components(graph)))


def hash_graph(
    graph: nx.Graph,
    node_attr: str | None = None,
    edge_attr: str | None = None,
    iterations: int = 3,
    digest_size: int = 16,
) -> str:
    """
    Computes a hash for a given graph using the Weisfeiler-Lehman (WL) graph hashing algorithm and additionally
    adds a node and edge attribute hash, if specified, to deal with common edge cases where WL fails (e.g.
     disconnected graphs).

    Args:
        - graph (networkx.Graph): The input graph to be hashed.
        - node_attr (str | None): The node attribute to be used for hashing. If None, node attributes are ignored.
        - edge_attr (str | None): The edge attribute to be used for hashing. If None, edge attributes are ignored.
        - iterations (int): The number of iterations for the WL algorithm. Default is 3.
        - digest_size (int): The size of the hash digest for WL. Default is 16.

    Returns:
        - str: The computed hash of the graph.

    Example:
        >>> import networkx as nx
        >>> G = nx.gnm_random_graph(10, 15)
        >>> hash_graph(G)
        '504894f49dd84b17c391b163af69624b'
    """
    # ... compute WL-hash
    hash = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
        graph, node_attr=node_attr, edge_attr=edge_attr, iterations=iterations, digest_size=digest_size
    )

    if node_attr is not None:
        # ... add number of unique nodes to hash
        hash += f"_{len(graph.nodes)}"
        # ... add number of unique node attributes with counts to hash
        node_attr_dict = nx.get_node_attributes(graph, node_attr)
        hash += "_" + ",".join(
            [f"{elt}:{count}" for elt, count in zip(*np.unique(list(node_attr_dict.values()), return_counts=True))]
        )
    if edge_attr is not None:
        # ... add number of unique edges to hash
        hash += f"_{len(graph.edges)}"
    return hash


def generate_inter_level_bond_hash(
    atom_array: AtomArray, lower_level_id: str, lower_level_entity: str | None = None
) -> str:
    """
    Generates a hash string representing the inter-level bonds within an AtomArray.
    When computing entities IDs, we must consider inter-level bonds at the atom- and residue-level to avoid ambiguity.

    Args:
        atom_array (AtomArray): The array of atoms containing bond and annotation information.
        lower_level_id (str): The level which to find, and hash, the inter-level bonds. For example, when computing molecule entities, we'd consider the inter-PN Unit bonds.
        lower_level_entity (str } None): An additional entity annotation to use when computing the hash. Optional; if None, then only residue ID, residue name, and atom name are used.

    Returns:
        str: A hash string representing the inter-level bonds.
    """

    # ...find the inter-level bonds
    bond_a = atom_array.get_annotation(lower_level_id)[atom_array.bonds.as_array()[:, 0]]
    bond_b = atom_array.get_annotation(lower_level_id)[atom_array.bonds.as_array()[:, 1]]
    inter_level_bonds = atom_array.bonds.as_array()[bond_a != bond_b]

    if inter_level_bonds.size:
        # ...loop over the bonds and create a (sorted) list of tuples with the relevant information
        bond_tuples = []
        for atom_idx in range(inter_level_bonds.shape[0]):
            atom_a = atom_array[inter_level_bonds[atom_idx, 0]]
            atom_b = atom_array[inter_level_bonds[atom_idx, 1]]
            bond_tuples.append(
                tuple(
                    sorted(
                        [
                            (
                                getattr(atom_a, lower_level_entity) if lower_level_entity else None,
                                atom_a.res_id,
                                atom_a.res_name,
                                atom_a.atom_name,
                            ),
                            (
                                getattr(atom_b, lower_level_entity) if lower_level_entity else None,
                                atom_b.res_id,
                                atom_b.res_name,
                                atom_b.atom_name,
                            ),
                        ]
                    )
                )
            )

        # ...sort the list of tuples, and hash
        return str(hash(tuple(sorted(bond_tuples))))
    else:
        return ""


def fix_bonded_atom_charges(atom):
    """
    Fix charges and hydrogen counts for cases when
    charged a atom is connected by an inter-residue bond.

    Args:
        atom (Atom): The atom object to be modified.

    Returns:
        dict: A dictionary with updated 'charge', 'hyb', and 'nhyd' values.
    """
    # TODO(smathis): Obsolete.
    # ...convert to int for comparison
    element = atom.element.astype(int)
    charge = atom.charge.astype(int)
    nhyd = atom.nhyd.astype(int)
    hyb = atom.hyb.astype(int)
    hvydeg = atom.hvydeg.astype(int)

    # ...manually fix charges and hydrogen counts for certain cases
    if element == 7 and charge == 1 and hyb == 3 and nhyd == 2 and hvydeg == 2:  # -(NH2+)-
        return {"charge": 0, "hyb": 2, "nhyd": 0}
    elif element == 7 and charge == 1 and hyb == 3 and nhyd == 3 and hvydeg == 0:  # free NH3+ group
        return {"charge": 0, "hyb": 2, "nhyd": 2}
    elif element == 8 and charge == -1 and hyb == 3 and nhyd == 0:
        return {"charge": 0, "hyb": hyb, "nhyd": nhyd}
    elif element == 8 and charge == -1 and hyb == 2 and nhyd == 0:  # O-linked connections
        return {"charge": 0, "hyb": hyb, "nhyd": nhyd}

    # ...default (no change)
    return {"charge": charge, "hyb": hyb, "nhyd": nhyd}


