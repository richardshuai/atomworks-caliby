"""Transforms to handle the assignment of RF2AA's atom frames"""

from typing import Any

import networkx as nx
import numpy as np
import torch
from biotite.structure import AtomArray

from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.atomize import AtomizeByCCDName
from datahub.transforms.base import Transform
from datahub.transforms.encoding import EncodeAtomArray
from datahub.utils.token import get_token_starts

# Constants copied from `chemdata` to decouple the RF2AA repository from the datahub pipeline
NUM2AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
    "MAS",
    " DA",
    " DC",
    " DG",
    " DT",
    " DX",
    " RA",
    " RC",
    " RG",
    " RU",
    " RX",
    "HIS_D",  # only used for cart_bonded
    "Al",
    "As",
    "Au",
    "B",
    "Be",
    "Br",
    "C",
    "Ca",
    "Cl",
    "Co",
    "Cr",
    "Cu",
    "F",
    "Fe",
    "Hg",
    "I",
    "Ir",
    "K",
    "Li",
    "Mg",
    "Mn",
    "Mo",
    "N",
    "Ni",
    "O",
    "Os",
    "P",
    "Pb",
    "Pd",
    "Pr",
    "Pt",
    "Re",
    "Rh",
    "Ru",
    "S",
    "Sb",
    "Se",
    "Si",
    "Sn",
    "Tb",
    "Te",
    "U",
    "W",
    "V",
    "Y",
    "Zn",
    "ATM",
]

FRAME_PRIORITY_TO_ATOM = [
    "F",
    "Cl",
    "Br",
    "I",
    "O",
    "S",
    "Se",
    "Te",
    "N",
    "P",
    "As",
    "Sb",
    "C",
    "Si",
    "Sn",
    "Pb",
    "B",
    "Al",
    "Zn",
    "Hg",
    "Cu",
    "Au",
    "Ni",
    "Pd",
    "Pt",
    "Co",
    "Rh",
    "Ir",
    "Pr",
    "Fe",
    "Ru",
    "Os",
    "Mn",
    "Re",
    "Cr",
    "Mo",
    "W",
    "V",
    "U",
    "Tb",
    "Y",
    "Be",
    "Mg",
    "Ca",
    "Li",
    "K",
    "ATM",
]
ATOM_TO_FRAME_PRIORITY = {x: i for i, x in enumerate(FRAME_PRIORITY_TO_ATOM)}


def find_all_paths_of_length_n(G: nx.Graph, n: int, order_independent_atom_frame_prioritization: bool = True) -> list:
    """
    Find all paths of a given length n in a NetworkX graph.

    Parameters:
    G (nx.Graph): The input graph.
    n (int): The length of the paths to find.
    order_independent_frame_prioritization (bool, optional):
        If True, considers paths with the same nodes but in different orders as equivalent.
        Defaults to True.

    Returns:
    np.ndarray: A tensor containing all unique paths of length n.

    Reference:
        https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph'''
    """

    def findPaths(G: nx.Graph, u: Any, n: int):
        """Find all paths of length n starting from node u in graph G."""
        if n == 0:
            return [[u]]
        paths = [[u] + path for neighbor in G.neighbors(u) for path in findPaths(G, neighbor, n - 1) if u not in path]
        return paths

    # All paths of length n
    if order_independent_atom_frame_prioritization:
        # Reverse paths if the first node is greater than the last node (which we later deduplicate with a set)
        allpaths = [tuple(p) if p[0] < p[-1] else tuple(reversed(p)) for node in G for p in findPaths(G, node, n)]
    else:
        # If order_independent_frame_prioritization is False, do not reverse paths
        allpaths = [tuple(p) for node in G for p in findPaths(G, node, n)]

    # Ensure paths are unique
    allpaths = list(set(allpaths))

    return allpaths


def get_rf2aa_atom_frames(
    encoded_query_pn_unit: np.ndarray, G: nx.Graph, order_independent_atom_frame_prioritization: bool = True
):
    """
    Choose a frame of 3 bonded atoms for each atom in the molecule,
    using a rule-based system that prioritizes frames based on atom types.

    Parameters:
        encoded_query_pn_unit (torch.Tensor): Sequence of the pn_unit that we want to build frames for,
            encoded using the RF2AA TokenEncoding.
        G (nx.Graph): The input graph representing the non-polymer molecule.
        order_independent_frame_prioritization (bool, optional):
            If True, sorts atom types within frames to consider them order-independent.
            Defaults to True.

    Returns:
        torch.Tensor: A tensor containing the selected frames for each atom.
    """

    frames = find_all_paths_of_length_n(G, 2, order_independent_atom_frame_prioritization)
    selected_frames = []

    for n in range(encoded_query_pn_unit.shape[0]):
        frames_with_n = [frame for frame in frames if n == frame[1]]

        # Some chemical groups don't have two bonded heavy atoms; so, choose a frame with an atom two bonds away
        if not frames_with_n:
            frames_with_n = [frame for frame in frames if n in frame]

        # If the atom isn't in a three-atom frame, it should be ignored in loss calculation; set all the atoms to n
        if not frames_with_n:
            selected_frames.append([(0, 1), (0, 1), (0, 1)])
            continue

        frame_priorities = []
        for frame in frames_with_n:
            # HACK: Uses the "query_seq" to convert index of the atom into an "atom type", and converts that into a priority
            indices = [index for index in frame if index != n]
            aas = [NUM2AA[int(encoded_query_pn_unit[index])] for index in indices]

            #
            if order_independent_atom_frame_prioritization:
                frame_priorities.append(sorted([ATOM_TO_FRAME_PRIORITY[aa] for aa in aas]))
            else:
                frame_priorities.append([ATOM_TO_FRAME_PRIORITY[aa] for aa in aas])

        # NOTE: np.argsort doesn't sort tuples correctly so just sort a list of indices using a key
        sorted_indices = sorted(range(len(frame_priorities)), key=lambda i: frame_priorities[i])

        # Calculate residue offset for frame
        frame = [(frame - n, 1) for frame in frames_with_n[sorted_indices[0]]]
        selected_frames.append(frame)

    assert encoded_query_pn_unit.shape[0] == len(selected_frames)
    return torch.tensor(selected_frames).long()


class AddAtomFrames(Transform):
    """
    Add atom frames to the data dictionary. See the RF2AA supplement for more details.

    NOTE: We do not assume that all atomized residues are at the end of the AtomArray to allow for more flexibility in the future.

    Parameters:
        order_independent_atom_frame_prioritization (bool, optional):
            If True, sorts atom types within frames to consider them order-independent.
            Defaults to True.
    """

    requires_previous_transforms = [AtomizeByCCDName, EncodeAtomArray]

    def __init__(self, order_independent_atom_frame_prioritization: bool = True):
        self.order_independent_atom_frame_prioritization = order_independent_atom_frame_prioritization

    def check_input(self, data):
        check_contains_keys(data, ["encoded", "atom_array"])
        check_is_instance(data, "atom_array", AtomArray)  # TODO: Add other checks
        check_atom_array_annotation(data, ["pn_unit_iid"])

    def forward(self, data):
        atom_array = data["atom_array"]
        token_starts = get_token_starts(atom_array)
        token_wise_atom_array = atom_array[token_starts]

        # Initialize the atom frames
        seq = data["encoded"]["seq"]
        atom_frames = torch.zeros((seq.shape[0], 3, 2), dtype=torch.int64)  # [n_tokens_across_chains, 3, 2] (int)

        # Loop through all atomized pn_units_iids
        pn_unit_iids = np.unique(atom_array.pn_unit_iid[atom_array.atomize])
        for pn_unit_iid in pn_unit_iids:
            token_level_pn_unit_mask = (token_wise_atom_array.pn_unit_iid == pn_unit_iid) & (
                token_wise_atom_array.atomize
            )

            # Generate the networkx graph for the pn_unit
            pn_unit_instance_bonds = token_wise_atom_array.bonds[token_level_pn_unit_mask]
            G = pn_unit_instance_bonds.as_graph()

            # Get the frames
            pn_unit_instance_atom_frames = get_rf2aa_atom_frames(
                seq[token_level_pn_unit_mask], G, self.order_independent_atom_frame_prioritization
            )

            # Fill in the atom frames
            atom_frames[token_level_pn_unit_mask] = pn_unit_instance_atom_frames

        data["rf2aa_atom_frames"] = atom_frames
        return data

    # TODO: Tests for `AddAtomFrames`
