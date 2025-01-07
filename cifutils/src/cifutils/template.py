import logging
import os
from typing import Any, Final

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, BondList

import cifutils.transforms.atom_array as ta
from cifutils.common import exists
from cifutils.constants import CCD_MIRROR_PATH, UNKNOWN_LIGAND, WATER_LIKE_CCDS
from cifutils.utils.bonds import (
    correct_formal_charges_for_specified_atoms,
    get_inferred_polymer_bonds,
    get_struct_conn_bonds,
)
from cifutils.utils.ccd import atom_array_from_ccd_code, check_ccd_codes_are_available
from cifutils.utils.selection import get_annotation

logger = logging.getLogger(__file__)


DO_NOT_MATCH_CCD: Final[frozenset[str]] = frozenset((*WATER_LIKE_CCDS, UNKNOWN_LIGAND))
"""
CCDs that should not be matched to a template for the
purpose of adding missing atoms.
"""


def get_empty_ccd_template(
    ccd_code: str,
    *,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    remove_hydrogens: bool = True,
    **res_wise_annotations: int | float | str,
) -> AtomArray:
    """
    Creates an empty template AtomArray from a Chemical Component Dictionary (CCD) entry with optional residue-wise
    annotations.

    Args:
        - ccd_code (str): The three-letter code of the chemical component to create a template for.
        - ccd_mirror_path (os.PathLike, optional): Path to the local CCD mirror directory. Defaults to CCD_MIRROR_PATH.
        - remove_hydrogens (bool, optional): Whether to remove hydrogen atoms from the template. Defaults to True.
        - **res_wise_annotations: Additional residue-wise annotations to add to the template. Values can be int, float,
            or str and will be broadcast to all atoms in the template.

    Returns:
        - AtomArray: An empty template structure with nan coordinates but with bonds and annotations from the CCD entry,
            plus any additional specified annotations.

    Example:
        >>> template = get_empty_ccd_template("ALA", chain_id="A", res_id=1, occupancy=1.0)
    """
    template_cc = atom_array_from_ccd_code(ccd_code, ccd_mirror_path, coords=None)

    if remove_hydrogens:
        template_cc = ta.remove_hydrogens(template_cc)

    # set default residue-wise annotations
    for annot, value in res_wise_annotations.items():
        if value is not None:
            template_cc.set_annotation(annot, np.full(len(template_cc), value))

    return template_cc


def match_residue_to_template(
    template: AtomArray,
    real: AtomArray,
    use_ccd_charges: bool,
) -> AtomArray:
    """
    Matches atoms from a real structure to a template structure, copying over coordinates and annotations while preserving
    the template's topology.

    The function attempts to match atoms first by standard atom names, then by alternative atom IDs if available and if
    they provide better matching. Coordinates and annotations from matched atoms in the real structure are copied to the
    template. Unmatched atoms in the real structure are dropped with a warning.

    Args:
        - template (AtomArray): Template structure containing the reference topology and complete set of atoms.
        - real (AtomArray): Real structure containing the atoms to be matched to the template.
        - use_ccd_charges (bool): Whether to keep template charges (True) or copy charges from real structure (False).

    Returns:
        - AtomArray: Template structure with coordinates and annotations copied from matched atoms in the real structure.

    Raises:
        - ValueError: If multiple atoms in the real structure have the same name.

    Notes:
        - Atoms in real structure not found in template are dropped (with warning)
        - If multiple template atoms match a real atom, only first match is used (with warning)
        - Records whether alternative atom IDs were used for matching in 'uses_alt_atom_id' annotation
    """
    # ... get information about the residue
    ccd_code = real.res_name[0]
    annotations = set(real.get_annotation_categories())

    # ... fail if there are multiple atoms with the same name in the `real` array
    if len(np.unique(real.atom_name)) != len(real):
        raise ValueError(f"CCD {ccd_code}: Multiple atoms with the same name in \n{real}")

    # ... determine whether to use the standard or alternative atom naming
    match_by = "atom_name"
    if "alt_atom_id" in template.get_annotation_categories():
        n_matches_std = np.sum(np.isin(real.atom_name, template.atom_name))
        n_matches_alt = np.sum(np.isin(real.atom_name, template.alt_atom_id))
        match_alt = n_matches_alt > n_matches_std
        match_by = "alt_atom_id" if match_alt else "atom_name"
        if match_alt:
            logger.warning(f"CCD {ccd_code}: Having to use alternative atom IDs for matching.")
    # ... and record what we used to match
    template.set_annotation("uses_alt_atom_id", [(match_by == "alt_atom_id")] * len(template))

    # ... match the atoms that exist in the chemical component
    for atom in real:
        # match_by: 'atom_name' or 'alt_atom_id'
        match = np.where(template.get_annotation(match_by) == atom.atom_name)[0]

        if len(match) == 0:
            # ... drop atoms that are not in the template (!This should not happen, except for UNK/X/DX!)
            logger.warning(f"{ccd_code}: Atom {atom} not found in template {template}. Will be dropped.")
            continue
        elif len(match) > 1:
            # ... drop atoms that are duplicated in the template (!This should not happen!)
            logger.warning(
                f"{ccd_code}: Atom {atom} found multiple times in template {template}. Only first will be matched."
            )
            continue

        # ... copy over the annotations
        template.coord[match] = atom.coord
        template.occupancy[match] = atom._annot.get("occupancy", 1.0)
        template.ins_code[match] = atom._annot.get("ins_code", template.ins_code[match])
        if "b_factor" in annotations:
            template.b_factor[match] = atom._annot.get("b_factor", np.nan)
        if not use_ccd_charges:
            template.charge[match] = atom._annot.get("charge", template.charge[match])

    # ... copy over general residue annotations
    template.chain_id = [atom.chain_id] * len(template)
    template.res_id = [atom.res_id] * len(template)
    template.ins_code = [atom.ins_code] * len(template)

    # ... return matched array
    return template


def build_template_atom_array(
    chain_info_dict: dict[str, dict[str, Any]],
    atom_array: AtomArray | None = None,
    remove_hydrogens: bool = True,
    use_ccd_charges: bool = True,
    ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
) -> AtomArray:
    """
    Builds a template AtomArray by matching residues to CCD templates and copying coordinates/annotations from an existing
    structure.

    For each residue in chain_info_dict, creates a template from the Chemical Component Dictionary (CCD) and either:
    1. Copies coordinates and annotations from matching atoms in atom_array, or
    2. Leaves coordinates as NaN if no matching atoms exist, or
    3. Copies atoms verbatim from atom_array if no CCD template exists (e.g., for UNL) or for CCD codes that are not available
        we want to ignore for matching (e.g., for water molecules)

    Args:
        - chain_info_dict (dict): Dictionary mapping chain IDs to dicts containing residue information with keys:
            - 'res_id': List of residue IDs
            - 'res_name': List of residue names (CCD codes)
            - 'is_polymer': Boolean indicating if chain is polymeric
            - 'chain_type': Chain type enum value
        - atom_array (AtomArray, optional): Structure containing coordinates and annotations to copy. Defaults to None.
        - remove_hydrogens (bool, optional): Whether to remove hydrogens from CCD templates. Defaults to True.
        - use_ccd_charges (bool, optional): Whether to use charges from CCD (True) or atom_array (False). Defaults to True.
        - ccd_mirror_path (os.PathLike, optional): Path to local CCD mirror. Defaults to CCD_MIRROR_PATH.

    Returns:
        AtomArray: Template structure with coordinates and annotations copied from atom_array where available.

    Raises:
        ValueError: If chains in atom_array don't match chains in chain_info_dict.
    """
    # ... check if the chain_to_sequence_map is consistent with the atom_array
    if exists(atom_array) and (not set(struc.get_chains(atom_array)) == set(chain_info_dict)):
        raise ValueError(
            "Mismatch between `atom_array` and `chain_to_sequence`! "
            f"Atom array contains chains {struc.get_chains(atom_array)} but chain_to_sequence "
            f"contains chains {chain_info_dict.keys()}."
        )

    # ... extract the relevant entries from the chain_info_dict for readability
    chain_id_to_res_ids = {chain_id: chain_info_dict[chain_id]["res_id"] for chain_id in chain_info_dict}
    chain_id_to_res_names = {chain_id: chain_info_dict[chain_id]["res_name"] for chain_id in chain_info_dict}
    chain_id_to_is_polymer = {chain_id: chain_info_dict[chain_id]["is_polymer"] for chain_id in chain_info_dict}
    chain_id_to_types = {chain_id: chain_info_dict[chain_id]["chain_type"] for chain_id in chain_info_dict}
    annotations = set(atom_array.get_annotation_categories()) if exists(atom_array) else set()

    # ... extract the relevant entries from the atom_array if it exists
    all_false = lambda n: np.zeros(n, dtype=bool)  # noqa: E731, convenience function
    chain_ids = atom_array.chain_id if exists(atom_array) else all_false(0)
    res_ids = atom_array.res_id if exists(atom_array) else all_false(0)
    res_names = atom_array.res_name if exists(atom_array) else all_false(0)

    # ... create a list of atoms based on the reference CCD entries
    template_residues = []
    for chain_id in chain_info_dict:
        chain_res_ids = chain_id_to_res_ids[chain_id]
        chain_res_names = chain_id_to_res_names[chain_id]
        chain_is_polymer = chain_id_to_is_polymer[chain_id]
        chain_type = chain_id_to_types[chain_id].value  # chain_type is an IntEnum; we want the value

        assert len(chain_res_ids) == len(chain_res_names), "Lenght mismatch between chain_res_ids, chain_res_names!"

        for res_id, (res_id_original, ccd_code) in enumerate(
            zip(chain_res_ids, chain_res_names, strict=False), start=1
        ):
            res_id_original = int(res_id_original)
            # ... and corresponding mask
            res_mask = (chain_ids == chain_id) & (res_names == ccd_code) & (res_ids == res_id_original)

            # ... if we cannot get a template from the CCD, we copy over the atoms from the atom_array verbatim
            if (ccd_code in DO_NOT_MATCH_CCD) or not check_ccd_codes_are_available(
                [ccd_code], ccd_mirror_path, mode="warn"
            ):
                if not res_mask.any():
                    # ... skip if we cannot find the residue in the reference atom_array
                    logger.warning(
                        f"No atoms found for residue {ccd_code} in chain {chain_id} with ID {res_id_original}!"
                    )
                    continue

                # ... otherwise, we copy over the atoms from the atom_array
                real = atom_array[res_mask]
                n_atoms = real.array_length()
                real.set_annotation("stereo", np.full(n_atoms, fill_value="N", dtype="<U1"))
                real.set_annotation("is_leaving_atom", all_false(n_atoms))
                real.set_annotation("is_backbone_atom", all_false(n_atoms))
                real.set_annotation("is_n_terminal_atom", all_false(n_atoms))
                real.set_annotation("is_c_terminal_atom", all_false(n_atoms))
                real.set_annotation("uses_alt_atom_id", all_false(n_atoms))
                real.set_annotation("chain_type", [chain_type] * n_atoms)
                real.set_annotation("charge", get_annotation(real, "charge", default=np.zeros(n_atoms)))
                real.set_annotation("is_polymer", [chain_is_polymer] * n_atoms)
                template_residues.append(real)
                continue

            # ... get empty template (no occupation, nan coordinates)
            tmpl = get_empty_ccd_template(
                ccd_code,
                ccd_mirror_path=ccd_mirror_path,
                remove_hydrogens=remove_hydrogens,
                # ... add required residue-wise annotations
                chain_id=chain_id,
                res_id=res_id_original if chain_is_polymer else res_id,
                occupancy=0.0,
                # ... add custom residue-wise annotations if they exist
                is_polymer=chain_is_polymer,
                b_factor=np.nan if "b_factor" in annotations else None,
                chain_type=chain_type,
            )

            # ... copy over the annotations & coordinates from the atom_array if the residue exists
            if res_mask.any():
                real = atom_array[res_mask]
                tmpl = match_residue_to_template(template=tmpl, real=real, use_ccd_charges=use_ccd_charges)

            template_residues.append(tmpl)

    # ... concatenate all template residues into a single AtomArray
    template_array = struc.concatenate(template_residues)

    return template_array


def add_missing_atoms(
    atom_array: AtomArray,
    chain_info_dict: dict[str, dict[str, Any]],
    struct_conn_dict: dict = {},
    add_bond_types_from_struct_conn: list[str] = ["covale"],
    remove_hydrogens: bool = True,
    use_ccd_charges: bool = True,
    fix_formal_charges: bool = True,
) -> AtomArray:
    """
    Adds missing atoms to an AtomArray by matching residues to CCD templates and handling inter-residue bonds.

    This function performs several steps:
    1. Matches residues to CCD templates to add missing atoms and intra-residue bonds
    2. Infers and adds polymer bonds between residues
    3. Adds additional inter-residue bonds from struct_conn records
    4. Removes leaving atoms from bond formation
    5. Fixes formal charges on atoms involved in inter-residue bonds

    Args:
        atom_array (AtomArray): Input structure containing atoms to be completed.
        chain_info_dict (dict): Dictionary mapping chain IDs to dicts containing 'res_id', 'res_name', 'is_polymer', and
            'chain_type' info.
        struct_conn_dict (dict, optional): Dictionary containing structural connectivity information. Defaults to {}.
        add_bond_types_from_struct_conn (list[str], optional): Types of bonds to add from struct_conn. Defaults to
            ["covale"].
        remove_hydrogens (bool, optional): Whether to remove hydrogen atoms from templates. Defaults to True.
        use_ccd_charges (bool, optional): Whether to use charges from CCD or input structure. Defaults to True.
        fix_formal_charges (bool, optional): Whether to fix formal charges on atoms involved in inter-residue bonds.

    Returns:
        AtomArray: Completed structure with missing atoms added and proper bonding.

    Raises:
        ValueError: If chain_info_dict is inconsistent with atom_array chains.
    """
    # ... match all residues to a CCD template
    #     (unless no CCD template esits, in which case we copy over)
    #     this also creates the intra-residue bonds from the CCD
    atoms = build_template_atom_array(
        chain_info_dict=chain_info_dict,
        atom_array=atom_array,
        use_ccd_charges=use_ccd_charges,
        remove_hydrogens=False,  # we keep hydrogens here, to allow fixing formal charges
    )

    # ... infer inter-residue polymer bonds
    polymer_bonds, polymer_bond_leaving_atom_idxs = get_inferred_polymer_bonds(atoms)

    # ... create any remaining inter-residue bonds that
    #     are specified in struct_conn
    struct_conn_bonds, struct_conn_leaving_atom_idxs = get_struct_conn_bonds(
        atoms,
        struct_conn_dict=struct_conn_dict,
        add_bond_types=add_bond_types_from_struct_conn,
    )

    # ... merge all inter-residue bonds
    inter_bonds = BondList(
        atom_count=atoms.array_length(),
        bonds=np.concatenate((polymer_bonds, struct_conn_bonds)),
    )
    #    and add them to the atom array
    atoms.bonds = atoms.bonds.merge(inter_bonds)
    #    and record which atoms make inter-residue bonds
    atoms_with_inter_bonds = np.unique(inter_bonds.as_array()[:, :2])
    makes_inter_bond = np.zeros(atoms.array_length(), dtype=bool)
    makes_inter_bond[atoms_with_inter_bonds] = True

    # ... merge all leaving group indices
    is_leaving = np.zeros(len(atoms), dtype=bool)
    is_leaving[np.concatenate((polymer_bond_leaving_atom_idxs, struct_conn_leaving_atom_idxs), axis=0)] = True
    #     and remove them from the atom array
    atoms = atoms[~is_leaving]
    makes_inter_bond = makes_inter_bond[~is_leaving]

    # ... fix charges of newly bonded atoms, where needed
    if fix_formal_charges:
        atoms = correct_formal_charges_for_specified_atoms(atoms, to_update=makes_inter_bond)

    # ... remove hydrogens
    if remove_hydrogens:
        atoms = ta.remove_hydrogens(atoms)

    return atoms
