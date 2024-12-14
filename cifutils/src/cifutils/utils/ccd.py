import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from functools import cache
from typing import Literal

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import networkx as nx
import numpy as np
import toolz

from cifutils.common import exists, immutable_lru_cache
from cifutils.constants import (
    AA_LIKE_CHEM_TYPES,
    CCD_MIRROR_PATH,
    DNA_LIKE_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_LIGAND,
    UNKNOWN_RNA,
)
from cifutils.enums import ChainType, ChainTypeInfo

logger = logging.getLogger(__name__)


@cache
def get_available_ccd_codes_in_mirror(ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> frozenset[str]:
    """Returns a frozenset of all CCD codes available in the local mirror."""
    cif = pdbx.CIFFile.read(os.path.join(ccd_mirror_path, "components.cif"))
    return frozenset(cif.keys())


@cache
def get_available_ccd_codes_in_biotite() -> frozenset[str]:
    """Returns a frozenset of all CCD codes available in Biotite's built-in Chemical Component Dictionary."""
    return frozenset(struc.info.ccd.get_ccd()["chem_comp"]["id"].as_array())


def get_available_ccd_codes(ccd_mirror_path: os.PathLike | None = CCD_MIRROR_PATH) -> frozenset[str]:
    """Returns a frozenset of all CCD codes available.

    If a mirror path is provided, it will be used to check the local mirror first.
    Otherwise, Biotite's built-in CCD will be used.
    """
    return (
        get_available_ccd_codes_in_mirror(ccd_mirror_path) if ccd_mirror_path else get_available_ccd_codes_in_biotite()
    )


def get_ccd_component_from_biotite(ccd_code: str) -> struc.AtomArray:
    """
    Retrieves a component from the Chemical Component Dictionary using Biotite's built-in functionality.

    Args:
        - ccd_code (str): The three-letter code of the chemical component to retrieve.

    Returns:
        - AtomArray: The atomic structure of the requested component.
    """
    try:
        atom_array = struc.info.residue(ccd_code)
        return atom_array
    except KeyError:
        raise ValueError(f"No atom information found for residue '{ccd_code}' in Biotite's CCD") from None


def check_ccd_codes_are_available(
    ccd_codes: Iterable[str], ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, mode: Literal["warn", "raise"] = "warn"
) -> bool:
    """Checks if the provided CCD codes are available in the local mirror."""
    available_ccds = get_available_ccd_codes(ccd_mirror_path)
    invalid_ccds = set(ccd_codes) - available_ccds
    if invalid_ccds:
        which_mirror = "Biotite's built-in CCD" if ccd_mirror_path is None else f"the local mirror at {ccd_mirror_path}"
        if mode == "warn":
            logger.warning(f"The following CCD codes were not found in {which_mirror}: {invalid_ccds}")
        elif mode == "raise":
            raise ValueError(f"The following CCD codes were not found in {which_mirror}: {invalid_ccds}")
    return not bool(invalid_ccds)


def _get_ccd_path(ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> os.PathLike:
    """
    Constructs the file path for a Chemical Component Dictionary entry in the local mirror.

    Args:
        - ccd_code (str): The three-letter code of the chemical component.
        - ccd_mirror_path (os.PathLike): Path to the root of the CCD mirror directory.

    Returns:
        - os.PathLike: Full path to the component's CIF file.
    """
    return os.path.join(ccd_mirror_path, ccd_code[0], ccd_code, ccd_code + ".cif")


def parse_ccd_cif(
    cif: pdbx.CIFFile,
    coords: Literal["model", "ideal_pdbx", "ideal_rdkit"] | None = "ideal_pdbx",
    add_properties: bool = True,
    add_mapping: bool = False,
) -> struc.AtomArray:
    """
    Parses a Chemical Component Dictionary CIF file into a Biotite AtomArray structure.

    Args:
        - cif (CIFFile): The CIF file containing the component data.
        - coords (Literal["model", "ideal_pdbx", "ideal_rdkit"] | None):
            Type of coordinates to use. Defaults to "ideal_pdbx".
                - "model": Use the coordinates that are found in a random (but fixed) pdb file.
                - "ideal_pdbx": Use the idealized coordinates computed by the RCSB PDB (sometimes not available).
                - "ideal_rdkit": Use the idealized coordinates computed by RDKit (sometimes unrealistic).
        - add_properties (bool): Whether to include RDKit-computed properties. Defaults to True.
            Properties are available under the `properties` attribute of the returned `AtomArray`.
        - add_mapping (bool): Whether to include external resource mappings, such as e.g. the ChEMBL ID.
            Defaults to False.
            Mappings are available under the `mapping` attribute of the returned `AtomArray`.

    Returns:
        - AtomArray: The parsed atomic structure with requested annotations and properties.

    Example:
        >>> cif = pdbx.CIFFile.read("path/to/ALA.cif")
        >>> atom_array = parse_ccd_cif(cif, coords="ideal_pdbx")
    """
    if coords not in ("model", "ideal_pdbx", "ideal_rdkit", None):
        raise ValueError(
            f"Invalid coordinate type: {coords}. Must be one of 'model', 'ideal_pdbx', 'ideal_rdkit' or `None`."
        )

    block = pdbx.convert._get_block(cif, None)

    # Extract metadata
    metadata = block.get("chem_comp")
    ccd_code = metadata["id"].as_item()

    if (metadata["pdbx_ideal_coordinates_missing_flag"].as_item() == "Y") and (coords == "ideal_pdbx"):
        logger.warning(f"No ideal coordinates found for `{ccd_code}`. Coordinates will be `nan`.")
        coords = None
    elif (metadata["pdbx_model_coordinates_missing_flag"].as_item() == "Y") and (coords == "model"):
        logger.warning(f"No model coordinates found for `{ccd_code}`. Coordinates will be `nan`.")
        coords = None

    # Extract atom specific information
    atom_data = block.get("chem_comp_atom")

    # Initialize the empty array:
    atoms = struc.AtomArray(atom_data.row_count)

    # Fill standard annotations
    as_bool = lambda x: np.where(x.as_array(str) == "Y", True, False)  # noqa: E731
    atoms.set_annotation("res_name", atom_data.get("comp_id").as_array(str))
    atoms.set_annotation("atom_name", atom_data.get("atom_id").as_array(str))
    atoms.set_annotation("alt_atom_id", atom_data.get("alt_atom_id").as_array(str))
    atoms.set_annotation("element", atom_data.get("type_symbol").as_array(str))
    atoms.set_annotation("charge", atom_data.get("charge").as_array(np.int8))
    atoms.set_annotation("stereo", atom_data.get("pdbx_stereo_config").as_array(str))
    atoms.set_annotation("is_aromatic", as_bool(atom_data.get("pdbx_aromatic_flag")))
    atoms.set_annotation("is_leaving_atom", as_bool(atom_data.get("pdbx_leaving_atom_flag")))
    atoms.set_annotation("is_backbone_atom", as_bool(atom_data.get("pdbx_backbone_atom_flag")))
    atoms.set_annotation("is_n_terminal_atom", as_bool(atom_data.get("pdbx_n_terminal_atom_flag")))
    atoms.set_annotation("is_c_terminal_atom", as_bool(atom_data.get("pdbx_c_terminal_atom_flag")))
    atoms.set_annotation("res_id", np.full(len(atoms), 1))  # We 1-index residue IDs to be consistent with RCSB

    # Try setting hetero flag
    hetero = True
    with suppress(ValueError):
        hetero = get_ccd_component_from_biotite(ccd_code).hetero[0]
    atoms.set_annotation("hetero", [hetero] * len(atoms))

    # Fill in the coordinates
    try:
        if coords is None:
            pass
        elif coords == "model":
            for i, name in enumerate(["model_Cartn_x", "model_Cartn_y", "model_Cartn_z"]):
                atoms.coord[:, i] = atom_data[name].as_array(np.float32)
        elif coords == "ideal_pdbx":
            for i, name in enumerate(
                ["pdbx_model_Cartn_x_ideal", "pdbx_model_Cartn_y_ideal", "pdbx_model_Cartn_z_ideal"]
            ):
                atoms.coord[:, i] = atom_data[name].as_array(np.float32)
        elif coords == "ideal_rdkit":
            rdkit_data = block.get("pdbe_chem_comp_rdkit_conformer")
            assert np.all(rdkit_data["atom_id"].as_array(str) == atoms.get_annotation("atom_name"))
            for i, name in enumerate(["Cartn_x_rdkit", "Cartn_y_rdkit", "Cartn_z_rdkit"]):
                atoms.coord[:, i] = rdkit_data[name].as_array(np.float32)
    except KeyError:
        raise ValueError(f"No coordinate data found for `{coords}` coordinates. Coords will be `nan`.") from None

    # Extract bond data
    try:
        bond_data = block.get("chem_comp_bond")
        if bond_data is not None:
            bond_dict = pdbx.convert._parse_intra_residue_bonds(bond_data)
            atoms.bonds = struc.connect_via_residue_names(atoms, custom_bond_dict=bond_dict)
    except KeyError:
        atoms.bonds = None
        logger.warning(f"No bond data found for `{ccd_code}`. Bonds will be `None`.")

    # Set general annotations:
    if add_properties:
        try:
            atoms.properties = toolz.valmap(lambda x: x.as_item(), dict(block["pdbe_chem_comp_rdkit_properties"]))
        except KeyError:
            logger.warning(f"No properties data found for `{ccd_code}`. Properties will be `None`.")
            atoms.properties = None

    if add_mapping:
        try:
            mapping = block.get("pdbe_chem_comp_external_mappings")
            atoms.mapping = dict(zip(mapping["resource"], mapping["resource_id"], strict=True))
        except KeyError:
            atoms.mapping = None
            logger.warning(f"No mapping data found for `{ccd_code}`. Mapping will be `None`.")

    return atoms


@immutable_lru_cache(maxsize=2000)
def get_ccd_component_from_mirror(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, **parse_ccd_cif_kwargs
) -> struc.AtomArray:
    """
    Retrieves and parses a component from a local mirror of the Chemical Component Dictionary.

    Args:
        - ccd_code (str): The three-letter code of the chemical component.
        - ccd_mirror_path (os.PathLike): Path to the root of the CCD mirror directory.
        - **parse_ccd_cif_kwargs: Additional keyword arguments passed to parse_ccd_cif():
            - coords (Literal["model", "ideal_pdbx", "ideal_rdkit"] | None):
                Type of coordinates to use. Defaults to "ideal_pdbx".
            - add_properties (bool): Whether to include RDKit-computed properties. Defaults to True.
            - add_mapping (bool): Whether to include external resource mappings, such as e.g. the ChEMBL ID.
                Defaults to False.

    Returns:
        - AtomArray: The parsed atomic structure of the requested component.

    Example:
        >>> atom_array = get_ccd_component_from_mirror("ALA", coords="ideal_pdbx")
    """
    cif = pdbx.CIFFile.read(_get_ccd_path(ccd_code, ccd_mirror_path))
    atom_array = parse_ccd_cif(cif, **parse_ccd_cif_kwargs)
    return atom_array


def atom_array_from_ccd_code(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, **parse_ccd_cif_kwargs
) -> struc.AtomArray:
    """Retrieves and parses a component from a local mirror of the Chemical Component Dictionary.

    If a mirror path is provided, it will be used to check the local mirror first.
    Otherwise, Biotite's built-in CCD will be used.

    Wrapper around `get_ccd_component_from_mirror()` and `get_ccd_component_from_biotite()`.
    """
    if ccd_mirror_path and ccd_code in get_available_ccd_codes_in_mirror(ccd_mirror_path):
        return get_ccd_component_from_mirror(ccd_code, ccd_mirror_path, **parse_ccd_cif_kwargs)
    else:
        return get_ccd_component_from_biotite(ccd_code)


def _find_connected_components_after_removal(graph: nx.Graph, node_to_remove: int) -> list[list[int]]:
    """
    Identifies connected components that would form after removing a node from a graph.

    Args:
        - graph (nx.Graph): The input graph.
        - node_to_remove (int): The node to hypothetically remove.

    Returns:
        - list[list[int]]: List of lists containing node indices in each new component.
    """
    # Get the neighbors before removal
    neighbors = set(graph.neighbors(node_to_remove))
    if not neighbors:
        return []

    # Create subgraph without the node
    subgraph = graph.subgraph(set(graph.nodes) - {node_to_remove})

    # Use BFS from any neighbor to find components
    components = []
    unvisited = neighbors.copy()

    while unvisited:
        start = unvisited.pop()
        if start not in subgraph:
            continue

        # Find all nodes reachable from this neighbor
        component = list(nx.bfs_tree(subgraph, start))
        components.append(component)
        unvisited -= set(component)

    return components


@immutable_lru_cache(maxsize=2000)
def get_chem_comp_leaving_atom_names(
    ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH, mode: Literal["warn", "raise"] = "warn"
) -> dict[str, tuple[str, ...]]:
    """
    Computes the canonical leaving groups for a given CCD entry based on the PDBs annotation
    of leaving atoms.

    The returned dictionary maps the name of the atom to the names of the atoms that would
    become disconnected if the atom were removed.

    Example:
        >>> get_chem_comp_leaving_atom_names("ALA")
        {'N': ('H2',), 'C': ('OXT', 'HXT'), 'OXT': ('HXT',)}
    """
    chem_comp = atom_array_from_ccd_code(ccd_code, ccd_mirror_path)

    if "is_leaving_atom" not in chem_comp.get_annotation_categories():
        if mode == "warn":
            logger.warning(
                f"No 'is_leaving_atom' annotation found for `{ccd_code}`. "
                "Cannot compute leaving groups, returning empty dictionary. "
                "Check if your CCD mirror is up to date."
            )
        elif mode == "raise":
            raise ValueError(
                f"No 'is_leaving_atom' annotation found for `{ccd_code}`. "
                "Cannot compute leaving groups. Check if your CCD mirror is up to date."
            )
        return {}

    # ... initialize output
    leaving_atom_names = defaultdict(list)

    # ... get relevant annotations
    is_leaving_atom = chem_comp.get_annotation("is_leaving_atom")
    atom_name = chem_comp.get_annotation("atom_name")
    element = chem_comp.get_annotation("element")

    # ... skip if no atoms are annotated as leaving atoms (majority of CCD entries)
    if not any(is_leaving_atom):
        return {}

    # ... compute the leaving groups based on the bond graph and annotation
    bond_graph = chem_comp.bonds.as_graph()
    for atom_idx in range(chem_comp.array_length()):
        # ... find the connected groups of atoms if the current atom were removed
        connected_groups = _find_connected_components_after_removal(bond_graph, atom_idx)

        # ... check if all atoms in the connected group are flagged as leaving atoms
        #     by the CCD entry
        for connected_group in connected_groups:
            heavy_atoms = list(filter(lambda x: element[x] != "H", connected_group))
            if all(is_leaving_atom[heavy_atoms]):
                leaving_atom_names[atom_name[atom_idx]] += [atom_name[idx] for idx in connected_group]

    # ... turn leaving_atom_names into a dictionary of tuples
    leaving_atom_names = {k: tuple(v) for k, v in leaving_atom_names.items()}

    return leaving_atom_names


@cache
def _chem_comp_type_dict() -> dict[str, str]:
    """Get a dictionary of all residue names and their corresponding chemical component types."""
    ccd = struc.info.ccd.get_ccd()  # NOTE: biotite caches this internally
    chem_comp_ids = np.char.upper(ccd["chem_comp"]["id"].as_array())
    chem_comp_types = np.char.upper(ccd["chem_comp"]["type"].as_array())
    return dict(zip(chem_comp_ids, chem_comp_types, strict=False))


def get_chem_comp_type(ccd_code: str, mode: Literal["warn", "raise"] = "warn") -> str:
    """Get the chemical component type for a CCD code from the Chemical Component Dictionary (CCD).

    Can be combined with CHEM_TYPES from `cifutils_biotite.constants` to determine if a component is a
    protein, nucleic acid, or carbohydrate.

    Args:
        ccd_code (str): The CCD code for the component. E.g. `ALA` for alanine, `NAP` for N-acetyl-D-glucosamine.
        mode (Literal["warn", "raise"]): How to handle unknown chemical component types.

    Example:
        >>> get_chem_comp_type("ALA")
        'L-PEPTIDE LINKING'
    """
    chem_comp_type = _chem_comp_type_dict().get(ccd_code, None)

    # ... handle unknown chemical component types
    if not exists(chem_comp_type):
        if mode == "raise":
            # ... raise an error if we want to fail loudly
            raise ValueError(f"Chemical component type for `{ccd_code=}` not found in CCD.")
        elif mode == "warn":
            # ... otherwise set chemical component type to "other" - the equivalent of unknown.
            logger.info(f"Chemical component type for `{ccd_code=}` not found in CCD. Using 'other'.")
            chem_comp_type = "OTHER"

    return chem_comp_type


def get_chain_type_from_chem_comp_type(chem_comp_type: str) -> ChainType:
    """Get the ChainType enum corresponding to a chemical component type."""
    return ChainTypeInfo.CHEM_COMP_TYPE_TO_ENUM.get(chem_comp_type, ChainType.OTHER_POLYMER)


def get_chain_type_from_ccd_code(ccd_code: str) -> ChainType:
    """Get the ChainType enum corresponding to a CCD code."""
    return get_chain_type_from_chem_comp_type(get_chem_comp_type(ccd_code))


def get_unknown_ccd_code_for_chem_comp_type(chem_comp_type: str) -> str:
    """Get the CCD code for an unknown chemical component type."""
    if chem_comp_type in AA_LIKE_CHEM_TYPES:
        return UNKNOWN_AA
    elif chem_comp_type in DNA_LIKE_CHEM_TYPES:
        return UNKNOWN_DNA
    elif chem_comp_type in RNA_LIKE_CHEM_TYPES:
        return UNKNOWN_RNA
    else:
        return UNKNOWN_LIGAND


def get_std_to_alt_atom_name_map(ccd_code: str, ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH) -> dict[str, str]:
    """Get a map from standard atom names to alternative atom names."""
    chem_comp = atom_array_from_ccd_code(ccd_code, ccd_mirror_path)
    return dict(zip(chem_comp.atom_name, chem_comp.alt_atom_id, strict=True))
