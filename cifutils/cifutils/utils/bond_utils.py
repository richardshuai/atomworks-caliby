"""
Utility functions for the detection, and creation, of bonds in a structure.
"""

__all__ = [
    "get_struct_conn_bonds",
    "get_inferred_polymer_bonds",
    "get_coarse_graph_as_nodes_and_edges",
    "get_connected_nodes",
    "hash_graph",
    "generate_inter_level_bond_hash",
]

import numpy as np
from cifutils.utils.selection_utils import get_residue_starts
from biotite.structure import AtomArray
import biotite.structure as struc
import logging
from cifutils.common import to_hashable
from cifutils.enums import ChainType, ChainTypeInfo
from cifutils.constants import (
    CHEM_TYPE_POLYMERIZATION_ATOMS,
    AA_LIKE_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
    DNA_LIKE_CHEM_TYPES,
    STRUCT_CONN_BOND_TYPES,
)
import networkx as nx
from cifutils.utils.residue_utils import get_chem_comp_type
from cifutils.utils.ccd import get_chem_comp_leaving_atom_names
import pandas as pd

logger = logging.getLogger("cifutils")

POLYMER_CHEM_COMP_TYPES = AA_LIKE_CHEM_TYPES | RNA_LIKE_CHEM_TYPES | DNA_LIKE_CHEM_TYPES


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


def _get_leaving_atom_idxs_for(atom_name: str, res_name: str, atom_names: np.ndarray, offset: int = 0) -> np.ndarray:
    """
    Get the indices of the leaving atoms for a given residue and atom.
    """
    leaving_atoms = get_chem_comp_leaving_atom_names(res_name).get(atom_name, ())
    return offset + np.where(np.isin(atom_names, leaving_atoms))[0]


def get_inferred_polymer_bonds(atom_array: AtomArray) -> tuple[list[tuple[int, int, struc.BondType]], np.ndarray]:
    """
    Infers and returns polymer bonds between consecutive residues in an atom array based on chemical component types
    and chain types.

    The function identifies bonds by looking at consecutive residues within the same chain and determining the
    appropriate bonding atoms based on either the chain type (as a fallback) or more detailed chemical component
    types. It also tracks leaving atoms that are displaced during bond formation. Leaving groups are inferred from
    the CCD entries for the chemical components. If a CCD code is missing from your local CCD mirror,
    leaving groups will not be inferred.

    Args:
        - atom_array (AtomArray): The atom array containing the structure information. Must include annotations for
            chain_id, res_id, res_name, and atom_name. Optionally includes chain_type annotation.

    Returns:
        - polymer_bonds (List[Tuple[int, int, struc.BondType]]): List of tuples containing (atom1_idx, atom2_idx,
            bond_type) for each inferred polymer bond.
        - leaving_atom_idxs (np.ndarray): Array of atom indices that represent leaving groups displaced during bond
            formation.

    Example:
        >>> # Create an atom array with two consecutive peptide residues
        >>> atom_array = AtomArray(length=10)
        >>> atom_array.chain_id = ["A"] * 10
        >>> atom_array.res_id = [1] * 5 + [2] * 5
        >>> atom_array.res_name = ["ALA"] * 5 + ["GLY"] * 5
        >>> atom_array.atom_name = ["N", "CA", "C", "OXT", "CB"] + ["N", "CA", "C", "O", "H2"]
        >>> # Get the polymer bonds
        >>> bonds, leaving = get_inferred_polymer_bonds(atom_array)
        >>> print(bonds)  # Shows C-N peptide bond between residues
        [(2, 5, <BondType.SINGLE>)]  # C of ALA to N of GLY
        >>> print(
        ...     leaving
        ... )  # Shows leaving OXT from C and H2 from N (other hydrogen atom names not shown for simplicity)
        [array([3]), array([9])]
    """

    # ... initialize return values
    bonds: list[tuple[int, int, struc.BondType]] = []
    leaving: list[np.ndarray] = []

    # ... get annotations we need to work with
    chain_ids = atom_array.chain_id
    res_ids = atom_array.res_id
    res_names = atom_array.res_name
    atom_names = atom_array.atom_name
    chain_types = atom_array.chain_type if "chain_type" in atom_array.get_annotation_categories() else None

    # ... get iterators over the residues
    residue_starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    this_res_starts = residue_starts[:-2]
    next_res_starts = residue_starts[1:-1]
    next_res_stops = residue_starts[2:]

    # ... loop over the residues and add the bonds
    for this_res_start, next_res_start, next_res_stop in zip(this_res_starts, next_res_starts, next_res_stops):
        # ... skip if residues are not on the same chain
        if chain_ids[this_res_start] != chain_ids[next_res_start]:
            continue

        # ... and skip if residues don't have consecutive res_id's
        #     (NOTE: same res_id is allowed, if ins_code is different)
        if res_ids[next_res_start] - res_ids[this_res_start] > 1:
            continue

        # ... get fallback default bonding atoms based on chain type
        bonding_atoms = None
        if chain_types is not None:
            chain_type = ChainType.as_enum(chain_types[this_res_start])
            bonding_atoms = ChainTypeInfo.ATOMS_AT_POLYMER_BOND.get(chain_type, None)

        # ... get (more detailed) bonding atoms based on chem-comp types
        this_link = get_chem_comp_type(res_names[this_res_start], strict=False)
        next_link = get_chem_comp_type(res_names[next_res_start], strict=False)

        # ... decide which bonds to form:
        if this_link in CHEM_TYPE_POLYMERIZATION_ATOMS and next_link in POLYMER_CHEM_COMP_TYPES:
            bonding_atoms = CHEM_TYPE_POLYMERIZATION_ATOMS[this_link]

        # ... add the bonds if we have bonding atoms
        if bonding_atoms is not None:
            # bonding_atoms: tuple[str, str] = (atom1_name, atom2_name)
            atom1_name, atom2_name = bonding_atoms

            # ... get the atoms names within the current residues
            this_res_atom_names = atom_names[this_res_start:next_res_start]
            next_res_atom_names = atom_names[next_res_start:next_res_stop]

            # ... find the indices of the bonding atoms based on the atoms names
            atom1_idx = np.where(this_res_atom_names == atom1_name)[0]
            atom2_idx = np.where(next_res_atom_names == atom2_name)[0]

            if len(atom1_idx) == 0 or len(atom2_idx) == 0:
                # ... bonding atoms are not found in the adjacent residues
                # ... -> skip this bond
                logger.info(
                    f"Bonding atoms {atom1_name} and {atom2_name} not found "
                    f"in the adjacent residues {this_res_start} and {next_res_start}!"
                )
                continue

            # ... add the bond
            bonds.append(
                (
                    this_res_start + atom1_idx[0],  # ... add global atom idx offset
                    next_res_start + atom2_idx[0],  # ... add global atom idx offset
                    struc.BondType.SINGLE,
                )
            )

            # ... compute the leaving atoms
            leaving_this_res = _get_leaving_atom_idxs_for(
                atom_name=atom1_name,
                res_name=res_names[this_res_start],
                atom_names=this_res_atom_names,
                offset=this_res_start,
            )
            leaving_next_res = _get_leaving_atom_idxs_for(
                atom_name=atom2_name,
                res_name=res_names[next_res_start],
                atom_names=next_res_atom_names,
                offset=next_res_start,
            )
            leaving.append(leaving_this_res)
            leaving.append(leaving_next_res)

    return bonds, np.concatenate(leaving)


def get_struct_conn_bonds(
    struct_conn_dict: dict[str, np.ndarray],
    chain_info_dict: dict,
    atom_array: AtomArray,
    ignore_ccd: list[str] = [],
    keep_bonds: list[str] = ["covale"],
) -> tuple[list[list[int]], list[int]]:
    """
    Adds bonds from the 'struct_conn' category of a CIF block to an atom array. Only covalent bonds are considered.

    Args:
        struct_conn_dict (dict): The struct_conn category of a CIF block as a dictionary. E.g.
            ```
                {'id': array(['disulf1',...]),
                'conn_type_id': array(['disulf', ...]),
                'pdbx_leaving_atom_flag': array(['?', ...]),
                ...
                'ptnr1_label_asym_id': array(['A', ...]),
                'ptnr1_label_comp_id': array(['CYS', ...]),
                'ptnr1_label_seq_id': array(['6', ...]),
                'ptnr1_label_atom_id': array(['SG', ...]),
                'ptnr1_symmetry': array(['1_555', ...]),
                'ptnr2_label_asym_id': array(['A', ...]),
                'ptnr2_label_comp_id': array(['CYS', ...]),
                'ptnr2_label_seq_id': array(['127', ...]),
                'ptnr2_label_atom_id': array(['SG', ...]),
                ...
            ```
        chain_info_dict (Dict): A dictionary containing information about the chains.
        atom_array (AtomArray): The atom array used to get atom indices.
        ignore_ccd (list): A list of CCD codes that should be ignored.
        keep_bonds (list): A list of bond types that should be kept. Valid bond types
            are: ["covale", "disulf", "metalc", "hydrog"]. Defaults to ["covale"], which is
            the use-case in structure-prediction, where we would a-priori know covalent bonds
            (except for disulfides).

            Reference: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_struct_conn.conn_type_id.html

    Returns:
        struct_conn_bonds: A List of bonds to be added to the atom array.
        leaving_atom_indices: A List of indices of atoms that are leaving groups for bookkeeping.
    """
    # ... validate input
    invalid_bond_types = set(keep_bonds) - STRUCT_CONN_BOND_TYPES
    if len(invalid_bond_types) > 0:
        raise ValueError(
            f"Invalid bond type(s) provided: {invalid_bond_types}! Valid bond types are: {STRUCT_CONN_BOND_TYPES}"
        )

    # ... initialize return values
    bonds: list[tuple[int, int, struc.BondType]] = []
    leaving: list[np.ndarray] = []

    # ... get the standard-to-alternative atom name mapping
    # ... convert struct_conn_dict to a DataFrame and filter for the bond types we want to keep
    struct_conn_df = pd.DataFrame(struct_conn_dict)
    struct_conn_df = struct_conn_df[struct_conn_df["conn_type_id"].isin(keep_bonds)]

    if struct_conn_df.empty:
        # ... skip if no bonds to add
        return bonds, leaving
    logger.debug(f"Attempting to add {len(struct_conn_df)} bonds from `struct_conn`")

    # ... extract relevant annotations
    chain_ids = atom_array.chain_id
    res_names = atom_array.res_name
    res_ids = atom_array.res_id
    atom_names = atom_array.atom_name
    global_atom_idx = np.arange(atom_array.array_length())
    alt_atom_ids = atom_array.alt_atom_id if "alt_atom_id" in atom_array.get_annotation_categories() else atom_names
    uses_alt_atom_id = (
        atom_array.uses_alt_atom_id
        if "uses_alt_atom_id" in atom_array.get_annotation_categories()
        else np.zeros(len(atom_array), dtype=bool)
    )

    for _, row in struct_conn_df.iterrows():
        res_name1 = row["ptnr1_label_comp_id"]
        res_name2 = row["ptnr2_label_comp_id"]
        if (res_name1 in ignore_ccd) or (res_name2 in ignore_ccd):
            # ... skip if the residues are ignored
            continue

        chain_id1 = row["ptnr1_label_asym_id"]
        chain_id2 = row["ptnr2_label_asym_id"]
        if (chain_id1 not in chain_info_dict) or (chain_id2 not in chain_info_dict):
            # ... skip, but warn if the chains are not present in the structure
            logger.info(
                f"Found covalent bond involving chains {chain_id1} and {chain_id2}, but at least one "
                "chain was removed during cleaning. This is likely because the chain is made up of a "
                "residue that is not in the local CCD. This should automatically be resolved once you "
                "update your CCD, unless you are working with an outdated structure file."
            )
            continue

        # For non-polymers, we use the auth_seq_id, otherwise we use the label_seq_id
        # NOTE: For non-polymers within PDB files, we may not have the auth_seq_id; we use the label_seq_id in such cases
        res_id1 = int(
            row["ptnr1_label_seq_id"]
            if chain_info_dict[chain_id1]["is_polymer"] or "ptnr1_auth_seq_id" not in row
            else row["ptnr1_auth_seq_id"]
        )
        res_id2 = int(
            row["ptnr2_label_seq_id"]
            if chain_info_dict[chain_id2]["is_polymer"] or "ptnr2_auth_seq_id" not in row
            else row["ptnr2_auth_seq_id"]
        )
        # ... get masks for the residues to which atoms 1 & 2 belong
        in_res1 = (chain_ids == chain_id1) & (res_ids == res_id1) & (res_names == res_name1)
        in_res2 = (chain_ids == chain_id2) & (res_ids == res_id2) & (res_names == res_name2)
        in_res1_start = global_atom_idx[in_res1][0]
        in_res2_start = global_atom_idx[in_res2][0]

        # Ensure that the we picked the correct residue (to handle sequence heterogeneity; see PDB ID `3nez` for an example)
        #  (short circuit eval to avoid indexing errors in cases where we don't have one of the residues due to seq. heterogeneity
        #   - e.g. 3nez)
        if (
            (in_res1.sum() == 0)
            or (in_res2.sum() == 0)
            or (res_name1 != res_names[in_res1_start])
            or (res_name2 != res_names[in_res2_start])
        ):
            # skip, but warn
            logger.info(
                f"Covalent bond involving residues {chain_id1}:{res_id1}:{res_name1} and "
                f"{chain_id2}:{res_id2}:{res_name2} was found in `struct_conn`, but the "
                f"residues are not present in the atom array. This is likely due to "
                f"resolved sequence heterogeneity which removed one of the residues."
            )
            continue

        # If all residues are present, we can proceed with identifying the global indices of the
        # atoms in the bond and add the bond
        # ... get the indices of the atoms and append to the list
        atom_name1 = row["ptnr1_label_atom_id"]
        atom_name2 = row["ptnr2_label_atom_id"]

        if uses_alt_atom_id[in_res1_start]:
            atom1_local_idx = np.where(alt_atom_ids[in_res1] == atom_name1)[0]
        else:
            atom1_local_idx = np.where(atom_names[in_res1] == atom_name1)[0]

        if uses_alt_atom_id[in_res2_start]:
            atom_2_local_idx = np.where(alt_atom_ids[in_res2] == atom_name2)[0]
        else:
            atom_2_local_idx = np.where(atom_names[in_res2] == atom_name2)[0]

        # ... convert local atom indices to global indices
        atom1_global_idx = in_res1_start + atom1_local_idx
        atom2_global_idx = in_res2_start + atom_2_local_idx

        # ... add the bond
        bonds.append([atom1_global_idx, atom2_global_idx, struc.BondType.SINGLE])

        # ... and identify the leaving atoms
        leaving_res1 = _get_leaving_atom_idxs_for(
            atom_name=atom_names[atom1_global_idx],
            res_name=res_name1,
            atom_names=atom_names[in_res1],
            offset=in_res1_start,
        )
        leaving_res2 = _get_leaving_atom_idxs_for(
            atom_name=atom_names[atom2_global_idx],
            res_name=res_name2,
            atom_names=atom_names[in_res2],
            offset=in_res2_start,
        )
        leaving.append(leaving_res1)
        leaving.append(leaving_res2)

        # Fix charges (TODO: Implement)

    return bonds, leaving


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
    #    for nitrogen
    if element == 7 and charge == 1 and hyb == 3 and nhyd == 2 and hvydeg == 2:  # -(NH2+)-
        return {"charge": 0, "hyb": 2, "nhyd": 0}
    elif element == 7 and charge == 1 and hyb == 3 and nhyd == 3 and hvydeg == 0:  # free NH3+ group
        return {"charge": 0, "hyb": 2, "nhyd": 2}
    #    for oxygen
    elif element == 8 and charge == -1 and hyb == 3 and nhyd == 0:
        return {"charge": 0, "hyb": hyb, "nhyd": nhyd}
    elif element == 8 and charge == -1 and hyb == 2 and nhyd == 0:  # O-linked connections
        return {"charge": 0, "hyb": hyb, "nhyd": nhyd}

    # ...default (no change)
    return {"charge": charge, "hyb": hyb, "nhyd": nhyd}
