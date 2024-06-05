from __future__ import annotations
import gzip
import itertools
from collections import OrderedDict
from pathlib import Path
from os import PathLike
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
import toolz
from biotite.structure.atoms import repeat
from biotite.structure.io.pdbx import CIFBlock, CIFFile, BinaryCIFFile
from cifutils.cifutils_biotite.common import exists
from functools import cache
from biotite.structure.atoms import AtomArray
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def category_to_dict(cif_block: CIFBlock, category: str) -> dict[str, np.ndarray]:
    """Convert a CIF block to a dictionary."""
    if exists(cif_block.get(category)):
        return toolz.valmap(lambda x: x.as_array(), dict(cif_block[category]))
    else:
        return {}


def category_to_df(cif_block: CIFBlock, category: str) -> pd.DataFrame | None:
    """Convert a CIF block to a pandas DataFrame."""
    return pd.DataFrame(category_to_dict(cif_block, category)) if category in cif_block.keys() else None


def deduplicate_iterator(iterator):
    """Deduplicate an iterator while preserving order."""
    return iter(OrderedDict.fromkeys(iterator))


def get_bond_type_from_order_and_is_aromatic(order, is_aromatic):
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


def read_cif_file(filename: PathLike) -> CIFFile | BinaryCIFFile:
    """Reads a CIF, BCIF, or gzipped CIF/BCIF file and returns its contents."""
    if not isinstance(filename, Path):
        filename = Path(filename)

    file_ext = filename.suffix

    if file_ext == ".gz":
        with gzip.open(filename, "rt") as f:
            # Handle gzipped CIF files
            if filename.name.endswith(".cif.gz"):
                cif_file = pdbx.CIFFile.read(f)
            elif filename.name.endswith(".bcif.gz"):
                with gzip.open(filename, "rb") as bf:
                    cif_file = pdbx.BinaryCIFFile.read(bf)
            else:
                raise ValueError("Unsupported file format for gzip compressed file")
    elif file_ext == ".bcif":
        # Handle BinaryCIF files
        cif_file = pdbx.BinaryCIFFile.read(filename)
    elif file_ext == ".cif":
        # Handle plain CIF files
        cif_file = pdbx.CIFFile.read(filename)
    else:
        raise ValueError(f"Unsupported file format {file_ext} in {filename}")

    return cif_file


def parse_transformations(struct_oper):
    """
    Get transformation operation in terms of rotation matrix and
    translation for each operation ID in ``pdbx_struct_oper_list``.

    Copied from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398
    """
    transformation_dict = {}
    for index, id in enumerate(struct_oper["id"].as_array(str)):
        rotation_matrix = np.array(
            [[struct_oper[f"matrix[{i}][{j}]"].as_array(float)[index] for j in (1, 2, 3)] for i in (1, 2, 3)]
        )
        translation_vector = np.array([struct_oper[f"vector[{i}]"].as_array(float)[index] for i in (1, 2, 3)])
        transformation_dict[id] = (rotation_matrix, translation_vector)
    return transformation_dict


def parse_operation_expression(expression):
    """
    Get successive operation steps (IDs) for the given
    ``oper_expression``.
    Form the cartesian product, if necessary.

    Copied from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398
    """
    # Split groups by parentheses:
    # use the opening parenthesis as delimiter
    # and just remove the closing parenthesis
    expressions_per_step = expression.replace(")", "").split("(")
    expressions_per_step = [e for e in expressions_per_step if len(e) > 0]
    # Important: Operations are applied from right to left
    expressions_per_step.reverse()

    operations = []
    for expr in expressions_per_step:
        if "-" in expr:
            # Range of operation IDs, they must be integers
            first, last = expr.split("-")
            operations.append([str(id) for id in range(int(first), int(last) + 1)])
        elif "," in expr:
            # List of operation IDs
            operations.append(expr.split(","))
        else:
            # Single operation ID
            operations.append([expr])

    # Cartesian product of operations
    return list(itertools.product(*operations))


def apply_transformations(structure, transformation_dict, operations):
    """
    Get subassembly by applying the given operations to the input
    structure containing affected asym IDs.

    Copied from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398
    """
    # Additional first dimesion for 'structure.repeat()'
    assembly_coord = np.zeros((len(operations),) + structure.coord.shape)

    # Apply corresponding transformation for each copy in the assembly
    for i, operation in enumerate(operations):
        coord = structure.coord
        # Execute for each transformation step
        # in the operation expression
        for op_step in operation:
            rotation_matrix, translation_vector = transformation_dict[op_step]
            # Rotate
            coord = matrix_rotate(coord, rotation_matrix)
            # Translate
            coord += translation_vector
        assembly_coord[i] = coord

    return repeat(structure, assembly_coord)


def matrix_rotate(v, matrix):
    """
    Perform a rotation using a rotation matrix.

    Parameters
    ----------
    v : ndarray
        The coordinates to rotate.
    matrix : ndarray
        The rotation matrix.

    Returns
    -------
    rotated : ndarray
        The rotated coordinates.
    """
    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = v.ndim
    if orig_ndim > 2:
        orig_shape = v.shape
        v = v.reshape(-1, 3)
    # Apply rotation
    v = np.dot(matrix, v.T).T
    # Reshape back into original shape
    if orig_ndim > 2:
        v = v.reshape(*orig_shape)
    return v


def fix_bonded_atom_charges(atom):
    """
    Fix charges and hydrogen counts for cases when
    charged a atom is connected by an inter-residue bond.

    Args:
        atom (Atom): The atom object to be modified.

    Returns:
        dict: A dictionary with updated 'charge', 'hyb', and 'nhyd' values.
    """
    if atom.element == 7 and atom.charge == 1 and atom.hyb == 3 and atom.nhyd == 2 and atom.hvydeg == 2:  # -(NH2+)-
        return {"charge": 0, "hyb": 2, "nhyd": 0}
    elif (
        atom.element == 7 and atom.charge == 1 and atom.hyb == 3 and atom.nhyd == 3 and atom.hvydeg == 0
    ):  # free NH3+ group
        return {"charge": 0, "hyb": 2, "nhyd": 2}
    elif atom.element == 8 and atom.charge == -1 and atom.hyb == 3 and atom.nhyd == 0:
        return {"charge": 0, "hyb": atom.hyb, "nhyd": atom.nhyd}
    elif atom.element == 8 and atom.charge == -1 and atom.hyb == 2 and atom.nhyd == 0:  # O-linked connections
        return {"charge": 0, "hyb": atom.hyb, "nhyd": atom.nhyd}
    elif atom.charge != 0:
        # Additional logic for other cases if needed
        pass
    return {"charge": atom.charge, "hyb": atom.hyb, "nhyd": atom.nhyd}


def build_modified_residues_dict(cif_block, chain_info_dict):
    """
    Build a dictionary mapping modified residue names to their canonical names.
    Note that one modified residue could be derived from multiple canonical residues (e.g., NRQ, circularized tri-peptide chromophore).
    In such cases, we store the modified residue name as the key and a list of canonical residue names as the value.

    Args:
    - cif_block (CIFBlock): The CIF block for the entry.
    - chain_info_dict (Dict): A dictionary containing information about the chains.

    Returns:
    - dict: A dictionary mapping modified residue names to their canonical names.
    """
    modified_residues_category = cif_block.get("pdbx_struct_mod_residue")
    if not modified_residues_category:
        return {}

    # Extract relevant fields
    auth_asym_ids = modified_residues_category["auth_asym_id"].as_array(str)
    label_seq_ids = modified_residues_category["label_seq_id"].as_array(str)
    auth_seq_ids = modified_residues_category["auth_seq_id"].as_array(str)
    label_comp_ids = modified_residues_category["auth_comp_id"].as_array(str)
    parent_comp_ids = modified_residues_category["parent_comp_id"].as_array(str)

    # Build the dictionary
    modified_residues_dict = {}
    for chain_id, label_seq_id, auth_seq_id, mod_res_name, canon_res_name in zip(
        auth_asym_ids, label_seq_ids, auth_seq_ids, label_comp_ids, parent_comp_ids
    ):
        # Check if we removed the chain (e.g., for unknown ligands / UNL residues)
        if chain_id not in chain_info_dict:
            continue
        res_id = label_seq_id if chain_info_dict[chain_id]["is_polymer"] else auth_seq_id
        key = (chain_id, res_id, mod_res_name)
        if mod_res_name != canon_res_name:
            modified_residues_dict.setdefault(key, []).append(canon_res_name)

    return modified_residues_dict


@cache
def get_std_alt_atom_id_conversion(res_name: str) -> dict:
    std_atom_ids = struc.info.ccd.get_from_ccd("chem_comp_atom", res_name, "atom_id")
    alt_atom_ids = struc.info.ccd.get_from_ccd("chem_comp_atom", res_name, "alt_atom_id")

    assert len(std_atom_ids) > 0, f"{res_name} info does not exist."
    assert len(std_atom_ids) == len(
        alt_atom_ids
    ), f"{res_name} has {len(std_atom_ids)} standard atom ids and {len(alt_atom_ids)} alternative atom ids"

    mapping = {"std_to_alt": dict(zip(std_atom_ids, alt_atom_ids)), "alt_to_std": dict(zip(alt_atom_ids, std_atom_ids))}

    return mapping


def standardize_atom_ids(atom_array: AtomArray) -> np.ndarray:
    _found_alt_atom_ids = 0
    atom_name_all = []
    for res in struc.residue_iter(atom_array):
        res_name = res.res_name
        atom_name = res.atom_name

        # Check if an atom array uses standard atom ids
        mapping = get_std_alt_atom_id_conversion(res_name[0])
        std_atoms = np.array(list(mapping["std_to_alt"].keys()))
        if not np.all(np.isin(atom_name, std_atoms)):
            _found_alt_atom_ids += 1
            # Convert to standard atom ids
            atom_name = np.array([mapping["alt_to_std"][atom_id] for atom_id in atom_name])

        atom_name_all.append(atom_name)

    if _found_alt_atom_ids > 0:
        logger.info(f"Found {_found_alt_atom_ids} alternative atom ids.")

    return np.concatenate(atom_name_all)
