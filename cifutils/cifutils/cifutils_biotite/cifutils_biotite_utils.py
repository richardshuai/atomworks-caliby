from __future__ import annotations
import gzip
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
from Bio.Data.PDBData import (
    protein_letters_3to1_extended,
    nucleic_letters_3to1_extended,
    protein_letters_3to1,
    protein_letters_1to3,
    nucleic_letters_3to1,
)

logger = logging.getLogger(__name__)

# Manually encode mapping from 1 to 3 for nuclelic acids (Bio.Data.PDBData is insufficient)
rna_letters_1to3 = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
}

dna_letters_1to3 = {
    "A": "DA",
    "C": "DC",
    "G": "DG",
    "T": "DT",
}


def get_1_from_3_letter_code(
    res_name: str,
    chain_type: str,
    use_closest_canonical: bool = False,
    gap_three_letter: str = "<GAP>",
    gap_one_letter: str = "-",
) -> str:
    """
    Converts a 3-letter residue name to its 1-letter code based on the chain type.
    Optionally, the closest canonical mapping can be used.

    Parameters:
    - res_name (str): The 3-letter residue name.
    - chain_type (str): The type of chain, e.g., "polypeptide(D)", "polypeptide(L)", "polydeoxyribonucleotide", or "polyribonucleotide".
    - use_closest_canonical (bool): Whether to use the closest canonical mapping (from BioPython). Defaults to False.
    - gap_three_letter (str): The three-letter code for a gap. Defaults to "<GAP>".
    - gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).

    Returns:
    - str: The corresponding 1-letter code. Returns "X" if the residue name or chain type is not supported.
    """
    # Convert gaps ("<GAP>") to "-", or whatever is specified
    if res_name == gap_three_letter:
        return gap_one_letter

    chain_type = chain_type.lower()
    if chain_type == "polypeptide(d)" or chain_type == "polypeptide(l)":
        # Proteins
        if use_closest_canonical:
            return protein_letters_3to1_extended.get(res_name, "X")
        else:
            return protein_letters_3to1.get(res_name, "X")
    elif chain_type == "polydeoxyribonucleotide" or chain_type == "polyribonucleotide":
        # Nucleic acids

        # Pad the residue name to 3 characters
        res_name = res_name.ljust(3)

        if use_closest_canonical:
            return nucleic_letters_3to1_extended.get(res_name, "X")
        else:
            return nucleic_letters_3to1.get(res_name, "X")
    else:
        logger.warning(f"Unsupported chain type: {chain_type}")
        return "X"


def get_3_from_1_letter_code(
    letter: str,
    chain_type: str,
    gap_one_letter: str = "-",
    gap_three_letter: str = "<GAP>",
    unknown_protein_three_letter: str = "UNK",
    unknown_rna_three_letter: str = "RX",
    unknown_dna_three_letter: str = "DX",
) -> str:
    """
    Converts a 1-letter residue name to its 3-letter code based on the chain type.
    NOTE: Converting from a three-letter, to a one-letter, back to a three-letter code is not invertible (i.e., 1:1) and may result in a different three-letter sequence.

    Parameters:
    - letter (str): The 1-letter residue name.
    - chain_type (str): The type of chain, e.g., "polypeptide(D)", "polypeptide(L)", "polydeoxyribonucleotide", or "polyribonucleotide".
    - gap_one_letter (str): The one-letter code for a gap. Defaults to "-" (as is standard within MSAs).
    - gap_three_letter (str): The three-letter code for a gap. Defaults to "<GAP>".
    - unknown_protein_three_letter (str): The three-letter code for an unknown protein residue. Defaults to "UNK_PROT".
    - unknown_rna_three_letter (str): The three-letter code for an unknown RNA residue. Defaults to "UNK_RNA".
    - unknown_dna_three_letter (str): The three-letter code for an unknown DNA residue. Defaults to "UNK_DNA".

    Returns:
    - str: The corresponding 3-letter code. Returns "UNK" if the letter or chain type is not supported.
    """
    chain_type = chain_type.lower()

    # Convert gaps (-) to "<GAP>", or whatever is specified
    if letter == gap_one_letter:
        return gap_three_letter

    if chain_type == "polypeptide(d)" or chain_type == "polypeptide(l)":
        # Proteins
        return protein_letters_1to3.get(letter, unknown_protein_three_letter)
    elif chain_type == "polydeoxyribonucleotide":
        # DNA
        return dna_letters_1to3.get(letter, unknown_dna_three_letter)
    elif chain_type == "polyribonucleotide":
        # RNA
        return rna_letters_1to3.get(letter, unknown_rna_three_letter)
    else:
        logger.warning(f"Unsupported chain type: {chain_type}")
        return unknown_protein_three_letter


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


def apply_transformation(structure, transformation_dict, operation):
    """
    Get subassembly by applying the given operation to the input
    structure containing affected asym IDs.

    Modified from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398
    """
    coord = structure.coord
    # Execute for each transformation step
    # in the operation expression
    for op_step in operation:
        rotation_matrix, translation_vector = transformation_dict[op_step]
        # Rotate
        coord = matrix_rotate(coord, rotation_matrix)
        # Translate
        coord += translation_vector

    # Add a dimension to coord to match expected shape or `repeat` (first dimension is # repeats)
    coord = coord[np.newaxis, ...]

    return repeat(structure, coord)


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

    assert exists(std_atom_ids) and (
        len(std_atom_ids) > 0
    ), f"{res_name} info does not exist in biotite's CCD. Try to update it to fix this assertion."
    assert len(std_atom_ids) == len(
        alt_atom_ids
    ), f"{res_name} has {len(std_atom_ids)} standard atom ids and {len(alt_atom_ids)} alternative atom ids"

    mapping = {"std_to_alt": dict(zip(std_atom_ids, alt_atom_ids)), "alt_to_std": dict(zip(alt_atom_ids, std_atom_ids))}

    return mapping


def standardize_heavy_atom_ids(atom_array: AtomArray) -> np.ndarray:
    _found_alt_atom_ids = 0
    atom_name_all = []
    for res in struc.residue_iter(atom_array):
        res_name = res.res_name

        # NOTE: We do not rename any H atoms, as we only care about
        #  covalent bonds in the struct_conn category later and so
        #  we will never have to match up H's.
        is_heavy = res.element != 1  # 1 is hydrogen, deuterium, tritium here
        is_heavy &= ~np.isin(res.element, ["H", "D", "H2", "T", "1"])

        atom_name = res.atom_name

        # Check if an atom array uses standard atom ids
        try:
            mapping = get_std_alt_atom_id_conversion(res_name[0])
        except AssertionError as e:
            # deal with residues which do not yet exist in biotite's CCD
            # skip, but warn
            logger.warning(
                f"{e.__class__.__name__}: {e}. Trying to continue processing, but consider updating biotite's CCD."
            )
            atom_name_all.append(atom_name)
            continue

        std_atoms = np.array(list(mapping["std_to_alt"].keys()))
        if not np.all(np.isin(atom_name[is_heavy], std_atoms)):
            _found_alt_atom_ids += 1
            # Convert to standard atom ids
            atom_name_renamed = np.array(
                [mapping["alt_to_std"].get(atom_id, atom_id) for atom_id in atom_name[is_heavy]]
            )

            # Ensure that renaming created no dupliates
            if len(np.unique(atom_name_renamed)) != len(atom_name_renamed):
                # if updates resulted in non-unique atom names, warn the user and
                # proceed with old atom names
                logger.error(
                    "Duplicate atom names found after renaming. This is likely because a mix of "
                    "standard and alternative atom ids was used in the input residue. Trying to "
                    "proceed without renaming."
                )
            else:
                # if updates are unique, rename and proceed
                atom_name[is_heavy] = atom_name_renamed

        atom_name_all.append(atom_name)

    if _found_alt_atom_ids > 0:
        logger.debug(f"Found {_found_alt_atom_ids} alternative atom ids.")

    return np.concatenate(atom_name_all)


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
