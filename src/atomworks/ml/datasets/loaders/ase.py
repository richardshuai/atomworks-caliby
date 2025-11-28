"""Loader for ASE datasets (e.g., from XYZ files) into AtomWorks format."""

import functools
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from atomworks.io.parser import parse_atom_array
from atomworks.io.tools.rdkit import atom_array_from_rdkit, atom_array_to_rdkit
from atomworks.io.transforms.atom_array import (
    get_coarse_graph_as_nodes_and_edges,
    get_connected_nodes,
)
from atomworks.io.utils.ase_conversions import ase_to_atom_array
from atomworks.io.utils.chain import create_chain_id_generator

logger = logging.getLogger(__name__)


def _ase_loader_function(
    raw_data: tuple,
    per_atom_properties: list[str],
    global_properties: list[str],
    add_missing_atoms: bool,
) -> dict[str, Any]:
    """ASE loader function (picklable when used with functools.partial)."""
    # Unpack the tuple
    atoms_row, example_id, global_idx = raw_data

    # Convert ASE row to atoms object
    atoms = atoms_row.toatoms()

    # Update atoms.info with any data from the row
    if isinstance(atoms_row.data, dict):
        atoms.info.update(atoms_row.data)

    # Convert to Biotite AtomArray
    atom_array = ase_to_atom_array(atoms)

    # Add bonds, hybridization, chirality, etc. via RDKit
    mol = atom_array_to_rdkit(
        atom_array,
        infer_bonds=True,
        timeout_seconds=2,
        hydrogen_policy="keep",
        system_charge=atoms.info.get("charge", 0),
    )
    atom_array = atom_array_from_rdkit(mol, remove_hydrogens=False)

    # Create unique atom IDs and assign chain IDs and residue IDs based on connectivity
    atom_array.set_annotation("atom_id", np.arange(atom_array.array_length()))
    connected_atoms = get_connected_nodes(*get_coarse_graph_as_nodes_and_edges(atom_array, "atom_id"))

    # Assign chain IDs based on connected components
    chain_id_gen = create_chain_id_generator()
    for connected_atom in connected_atoms:
        chain_letter = next(chain_id_gen)
        res_number = 1
        element_counts = {}

        for atom_id in connected_atom:
            atom_array.chain_id[atom_array.atom_id == atom_id] = chain_letter
            atom_array.res_id[atom_array.atom_id == atom_id] = res_number
            atom_array.res_name[atom_array.atom_id == atom_id] = f"{chain_letter}:{res_number}"

            element = atom_array.element[atom_array.atom_id == atom_id][0]
            element_counts[element] = element_counts.get(element, 0) + 1
            atom_name = f"{element}{element_counts[element]}"
            atom_array.atom_name[atom_array.atom_id == atom_id] = atom_name

    # Extract per-atom properties BEFORE parse_atom_array
    for prop in per_atom_properties:
        if prop in ("numbers", "positions"):
            continue

        if prop in atoms.arrays:
            if not hasattr(atom_array, prop):
                atom_array.set_annotation(prop, atoms.arrays[prop])
        elif prop in atoms.info:
            prop_data = atoms.info[prop]
            if not hasattr(atom_array, prop):
                if hasattr(prop_data, "__len__") and len(prop_data) == atom_array.array_length():
                    atom_array.set_annotation(prop, np.array(prop_data))
                else:
                    logger.warning(
                        f"Property '{prop}' found in atoms.info but not compatible as per-atom "
                        f"(length {len(prop_data) if hasattr(prop_data, '__len__') else 'scalar'} vs {atom_array.array_length()} atoms)"
                    )
        else:
            logger.warning(
                f"Requested per-atom property '{prop}' not found for example {global_idx}. "
                f"Available in atoms.arrays: {list(atoms.arrays.keys())}, "
                f"available in atoms.info: {list(atoms.info.keys())}"
            )

    # Parse atom array
    data = parse_atom_array(
        atom_array,
        add_missing_atoms=add_missing_atoms,
        remove_waters=False,
        remove_ccds=None,
        build_assembly="_spoof",
        hydrogen_policy="keep",
        fix_formal_charges=False,
        fix_bond_types=False,
    )

    # Extract the processed AtomArray
    atom_array = data["assemblies"]["1"][0]
    data["atom_array"] = atom_array

    # Extract global properties
    for prop in global_properties:
        if prop in atoms.info:
            data["extra_info"][prop] = atoms.info[prop]
        elif prop in atoms_row:
            data["extra_info"][prop] = atoms_row[prop]
        else:
            logger.warning(
                f"Requested global property '{prop}' not found in atoms.info for example {global_idx}. "
                f"Available properties: {list(atoms.info.keys())}"
            )

    data["example_id"] = example_id
    return data


def create_ase_loader(
    per_atom_properties: list[str] | None = None,
    global_properties: list[str] | None = None,
    add_missing_atoms: bool = False,
) -> Callable:
    """Factory function that creates a picklable loader for ASE LMDB datasets.

    The loader processes ASE atoms_row objects into AtomWorks-compatible dictionaries
    with atom_array, extra_info, and example_id fields.

    Args:
        per_atom_properties: List of per-atom properties to extract from atoms.arrays
            as AtomArray annotations (e.g., forces, charges, spins)
        global_properties: List of global properties to extract from atoms.info
            and/or atoms_row into extra_info dict (e.g., energy, charge, spin)
        add_missing_atoms: Whether to add missing atoms during parse_atom_array

    Returns:
        A picklable loader function (via functools.partial) for multiprocessing.
    """
    return functools.partial(
        _ase_loader_function,
        per_atom_properties=per_atom_properties or [],
        global_properties=global_properties or [],
        add_missing_atoms=add_missing_atoms,
    )
