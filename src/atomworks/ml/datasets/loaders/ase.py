"""Loader for ASE datasets (e.g., from XYZ files) into AtomWorks format."""

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

logger = logging.getLogger(__name__)


def create_ase_loader(
    per_atom_properties: list[str] | None = None,
    global_properties: list[str] | None = None,
    add_missing_atoms: bool = False,
) -> Callable:
    """Factory function that creates a loader for ASE LMDB datasets.

    The loader processes ASE atoms_row objects into AtomWorks-compatible dictionaries
    with atom_array, extra_info, and example_id fields.

    Args:
        per_atom_properties: List of per-atom properties to extract from atoms.arrays
            as AtomArray annotations (e.g., forces, charges, spins)
        global_properties: List of global properties to extract from atoms.info
            and/or atoms_row into extra_info dict (e.g., energy, charge, spin)
        add_missing_atoms: Whether to add missing atoms during parse_atom_array

    Returns:
        A loader function that takes a tuple (atoms_row, example_id, global_idx) and returns a dict

    Example:
        >>> loader = create_ase_loader(
        ...     per_atom_properties=["numbers", "positions"],
        ...     global_properties=["source", "charge", "spin", "num_atoms"],
        ... )
        >>> # Use with AseDBDataset
        >>> dataset = AseDBDataset(lmdb_path="...", loader=loader, transform=pipeline)
    """
    per_atom_properties = per_atom_properties or []
    global_properties = global_properties or []

    def loader_function(raw_data: tuple) -> dict[str, Any]:
        """Process ASE atoms_row into AtomWorks format.

        Args:
            raw_data: Tuple of (atoms_row, example_id, global_idx)

        Returns:
            Dictionary with atom_array, extra_info, and example_id
        """
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
        # (We infer bonds from coordinates and elements - not 100% accurate; fails or times out on many cases containing transition metals)
        mol = atom_array_to_rdkit(
            atom_array,
            infer_bonds=True,
            timeout_seconds=1,
            hydrogen_policy="keep",
            system_charge=atoms.info.get("charge", 0),
        )
        atom_array = atom_array_from_rdkit(mol, remove_hydrogens=False)

        # Create unique atom IDs and assign chain IDs and residue IDs based on connectivity
        atom_array.set_annotation("atom_id", np.arange(atom_array.array_length()))
        connected_atoms = get_connected_nodes(*get_coarse_graph_as_nodes_and_edges(atom_array, "atom_id"))

        # Assign chain IDs based on connected components
        for chain_idx, connected_atom in enumerate(connected_atoms):
            # Convert to letter for chain_id (A, B, C, ...) and 1-indexed for res_id
            chain_letter = chr(ord("A") + chain_idx)
            res_number = chain_idx + 1

            # Track element counts within this residue for atom naming
            element_counts = {}

            for atom_id in connected_atom:
                atom_array.chain_id[atom_array.atom_id == atom_id] = chain_letter
                atom_array.res_id[atom_array.atom_id == atom_id] = res_number
                atom_array.res_name[atom_array.atom_id == atom_id] = f"L:{chain_letter}:{res_number}"

                # Get element for this atom and assign numbered atom name
                element = atom_array.element[atom_array.atom_id == atom_id][0]
                element_counts[element] = element_counts.get(element, 0) + 1
                atom_name = f"{element}{element_counts[element]}"
                atom_array.atom_name[atom_array.atom_id == atom_id] = atom_name

        # Extract per-atom properties BEFORE parse_atom_array
        # This ensures properties are filtered correctly when hydrogens are removed
        for prop in per_atom_properties:
            # Skip properties already handled
            if prop in ("numbers", "positions"):
                continue

            # Check both atoms.arrays and atoms.info (OMol25 stores per-atom props in info)
            if prop in atoms.arrays:
                if not hasattr(atom_array, prop):
                    atom_array.set_annotation(prop, atoms.arrays[prop])
            elif prop in atoms.info:
                # Property is in atoms.info (common for OMol25)
                prop_data = atoms.info[prop]
                if not hasattr(atom_array, prop):
                    # Verify it's actually per-atom (array-like with correct length)
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

        # Parse atom array (removes hydrogens, etc.)
        # Note: parse_atom_array returns a dict with "asym_unit" containing the processed AtomArray(Stack)
        # Per-atom properties added above will be automatically filtered along with atom removal
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

        # Extract the processed AtomArray from the parsed dict
        # The "asym_unit" contains an AtomArrayStack (even for single structures)
        atom_array = data["assemblies"]["1"][0]

        # Update data dict to have the extracted atom_array
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

        # Set the example ID
        data["example_id"] = example_id
        return data

    return loader_function
