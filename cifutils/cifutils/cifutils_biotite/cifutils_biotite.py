"""
Implementation of original `citufils` functions using `biotite` library.
Retains all functionality of orginal library, with increased performance and additional features.
"""

from __future__ import annotations
import logging
import pickle
import time
from functools import lru_cache
from typing import Sequence, Literal

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
from os import PathLike
from collections import Counter

from cifutils.cifutils_biotite.cifutils_biotite_utils import (
    apply_transformations,
    category_to_df,
    category_to_dict,
    deduplicate_iterator,
    fix_bonded_atom_charges,
    get_bond_type_from_order_and_is_aromatic,
    parse_operation_expression,
    parse_transformations,
    read_cif_file,
    build_modified_residues_dict,
)
from cifutils.cifutils_biotite.common import exists

from biotite.structure.io.pdbx import CIFBlock
from biotite.structure import AtomArray, Atom
from cifutils.cifutils_biotite.constants import CRYSTALLIZATION_AIDS

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

__all__ = ["CIFParser"]


class CIFParser:
    def __init__(
        self,
        by_residue_pickle="/projects/ml/RF2_allatom/cifutils_extended/ligands_by_residue.pkl",
        by_atom_pickle="/projects/ml/RF2_allatom/cifutils_extended/ligands_by_atom.pkl",
        residues_to_skip: Sequence[str] = None,
        add_missing_atoms=True,
        add_bonds=True,
        convert_residues_dict=None,
        remove_waters=False,
        exclude_crystallization_aid=False,
        patch_symmetry_centers=True,
    ):
        """
        Initialize a CIFParser object.

        Args:
        - by_residue_pickle (str, optional): Path to the pre-compiled residue-level CCD and OB data built from `make_residue_library_from_ccd.py`.
        - by_atom_pickle (str, optional): Path to the pre-compiled atom-level CCD and OB data built from `make_residue_library_from_ccd.py`.
        - residues_to_skip (list, optional): List of residue names to skip.
        - add_missing_atoms (bool, optional): Whether to add missing atoms to the structure, including relevant OpenBabel atom-level data.
        - add_bonds (bool, optional): Whether to add bonds to the structure. Cannot be True if `add_missing_atoms` is False.
        - convert_residues_dict (dict, optional): Dictionary of residue name conversions. Keys are the original residue names, and values are the new residue names.
        - remove_waters (bool, optional): Whether to remove water molecules from the structure.
        - exclude_crystallization_aid (bool, optional): Whether to exclude crystallization aids and ions from the structure. Uses the AF-3 exclusion list.
        - patch_symmetry_centers (bool, optional): Whether to patch non-polymer residues at symmetry centers that clash with themselves when transformed.
        """

        # Step 0: Set and validate arguments
        self.add_missing_atoms = add_missing_atoms
        self.add_bonds = add_bonds
        self.convert_residues_dict = convert_residues_dict
        self.remove_waters = remove_waters
        self.exclude_crystallization_aid = exclude_crystallization_aid
        self.patch_symmetry_cetner = patch_symmetry_centers
        self._validate_arguments()

        # Step 1: Parse pre-compiled library (from the CCD, augmented with OpenBabel) of all residues observed in the PDB
        logger.info(f"Loading residue-level CCD and OB data from {by_residue_pickle}...")
        start_time = time.time()
        with open(by_residue_pickle, "rb") as file:
            self.data_by_residue = pickle.load(file)
        end_time = time.time()
        loading_time = end_time - start_time
        logger.info(f"Precompiled CCD data loaded successfully in {round(loading_time)} seconds.")

        # Step 2: Preparse atom-centric transformation of the precompiled library
        logger.info(f"Loading atom-level CCD and OB data from {by_atom_pickle}...")
        start_time = time.time()
        with open(by_atom_pickle, "rb") as file:
            self.data_by_atom = pickle.load(file)
        end_time = time.time()
        loading_time = end_time - start_time
        logger.info(f"Built atom-level dataframe in {round(loading_time)} seconds.")

        # Residues to be ignored during parsing are deleted from the precomputed library
        if exists(residues_to_skip):
            self.data_by_residue = self.data_by_residue[~self.data_by_residue["name"].isin(residues_to_skip)]
            self.data_by_atom = self.data_by_atom[~self.data_by_residue["name"].isin(residues_to_skip)]

        # Set indices
        self.data_by_residue.set_index("name", inplace=True)
        self.extra_info = {}  # For backwards compatability

    def _validate_arguments(self):
        """Validate the arguments passed to the CIFParser object."""
        if not self.add_missing_atoms and self.add_bonds:
            raise ValueError("add_bonds cannot be True if add_missing_atoms is False")
        if not isinstance(self.convert_residues_dict, dict) and exists(self.convert_residues_dict):
            raise ValueError("convert_residues_dict must be a dictionary.")
        if exists(self.convert_residues_dict):
            raise NotImplementedError("convert_residues_dict is not yet implemented.")

    def parse(
        self,
        filename: PathLike,
        build_assembly: Literal["first", "all"] | list[str] | None = None,
    ) -> dict:
        """
        Parse the CIF file and return chain information, residue information, atom array, metadata, and legacy data.

        Args:
        - filename (str): Path to the CIF file. May be any format of CIF file (e.g., gz, bcif, etc.).
        - build_assembly (str, optional): Which assembly to build, if any. Options are None (e.g., asymmetric unit), "first", "all", or a list of assembly IDs. Defaults to None.

        Returns:
        - dict: A dictionary containing the following keys:
        'chain_info': A dictionary mapping chain ID to sequence, type, entity ID, and other information.
        'residue_info': A dictionary mapping residue name to reference structure (atoms, bonds, automorphisms, etc.).
        'ligand_info': A dictionary containing ligand of interest information.
        'atom_array': An AtomArrayStack instance representing the asymmetric unit.
        'assemblies': A dictionary mapping assembly IDs to AtomArray instances.
        'metadata': A dictionary containing metadata about the structure.
        'modified_residues': A dictionary mapping modified residue names to their canonical name(s).
        'extra_info': A dictionary with legacy information for cross-compatibility; should not typically be used.
        """
        cif_file = read_cif_file(filename)
        cif_block = cif_file.block

        # Load metadata
        metadata = self._get_metadata(cif_block)

        # Load structure using the RCSB labels for sequence ids, and later update for non-polymers
        atom_array = pdbx.get_structure(
            cif_block,
            extra_fields=[
                "label_entity_id",
                "auth_seq_id",  # for non-polymer residue indexing
                "atom_id",
                "b_factor",
                "occupancy",
            ],
            use_author_fields=False,
            altloc="occupancy",
            model=1,
        )

        # Load chain information (uses atom_array to build chain list)
        chain_info_dict = self._get_chain_info(cif_block, atom_array)

        # Remove waters
        if self.remove_waters:
            atom_array = atom_array[atom_array.res_name != "HOH"]

        # Remove crystallization aids and ions from the atom array
        if self.exclude_crystallization_aid:
            atom_array = self._remove_crystallization_aids_and_ions(atom_array)

        # Replace non-polymeric chain sequence ids with author sequence ids
        atom_array = self._update_nonpoly_seq_ids(atom_array, chain_info_dict)

        # Remove unmatched residues from the atom array
        atom_array = self._remove_atoms_with_unmatched_residues(atom_array)

        # Load monomer sequence information into chain_info_dict and residue details (e.g., automorphisms) into residue_info_dict
        chain_info_dict, residue_info_dict = self._load_monomer_sequence_information(
            cif_block, chain_info_dict, atom_array
        )

        # Handle sequence heterogeneity by selecting the residue that appears last
        atom_array = self._keep_last_residue(atom_array)

        # Order atoms within each residue according to standard CCD ordering
        atom_array = atom_array[struc.info.standardize.standardize_order(atom_array)]

        # Create a larger atom array that includes missing atoms (e.g., hydrogens), then populate with atoms details loaded from structure
        if self.add_missing_atoms:
            atom_array = self._add_missing_atoms(atom_array, chain_info_dict)

        # Get bonds from the preprocessed CCD and OpenBabel data
        if self.add_bonds:
            atom_array = self._add_bonds(cif_block, atom_array, chain_info_dict)

        # Build the assembly and add the transformation_id annotation (defaults to identity)
        if exists(build_assembly):
            assemblies = self._build_assembly(cif_block, atom_array, assembly_ids=build_assembly)
        else:
            assemblies = {}

        # Get ligand of interest information
        loi_info = self._get_ligand_of_interest_info(cif_block)

        # Modified residue information
        modified_residues_dict = build_modified_residues_dict(cif_block, chain_info_dict)

        return {
            "chain_info": chain_info_dict,
            "residue_info": residue_info_dict,
            "ligand_info": loi_info,
            "atom_array": atom_array,
            "assemblies": assemblies,
            "metadata": metadata,
            "modified_residues": modified_residues_dict,
            "extra_info": {**self.extra_info},  # modified residues, struct_conn bonds
        }

    def _remove_atoms_with_unmatched_residues(self, atom_array: AtomArray) -> AtomArray:
        """Remove atoms from the atom array that do not have a corresponding residue in the precompiled CCD data."""
        unknown_residues = np.setdiff1d(np.unique(atom_array.res_name), self.data_by_residue.index.to_numpy())

        for unknown_residue in unknown_residues:
            mask = atom_array.res_name != unknown_residue
            atom_array = atom_array[mask]

        return atom_array

    def _remove_crystallization_aids_and_ions(self, atom_array: AtomArray) -> AtomArray:
        """Remove crystallization aids and ions from the atom array."""

        # Remove atoms from the atom array that are crystallization aids
        atom_array = atom_array[~np.isin(atom_array.res_name, CRYSTALLIZATION_AIDS)]

        return atom_array

    def _build_assembly(
        self, cif_block: CIFBlock, atom_array: AtomArray, assembly_ids: Literal["all", "first"] | list[str] = "first"
    ) -> AtomArray:
        """
        Build the first biological assembly found within the mmCIF file and update the `transformation_id` annotation.

        Code modified from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1318

        Args:
        - cif_block (CIFBlock): The CIF block containing the structure data.
        - atom_array (AtomArray): The atom array to which the transformations will be applied.
        - assembly_id (int, optional): The ID of the assembly to build. Defaults to None, which means the first assembly will be built.

        Returns:
        - AtomArray: The atom array with the biological assembly built and transformation_id annotations updated.
        """

        if "pdbx_struct_assembly" not in cif_block.keys():
            return atom_array

        # Parse CIF blocks and select assembly (either by passed assembly_id or the first assembly)
        assembly_gen_category = cif_block["pdbx_struct_assembly_gen"]
        struct_oper_category = cif_block["pdbx_struct_oper_list"]
        available_assembly_ids = assembly_gen_category["assembly_id"].as_array(str)

        # parse `assembly_ids` option
        if assembly_ids == "first":
            to_build = [available_assembly_ids[0]]
        elif assembly_ids == "all":
            to_build = available_assembly_ids
        else:
            to_build = assembly_ids

        # ensure instructions for each requested assembly id exist
        if not all(_id in available_assembly_ids for _id in to_build):
            raise ValueError(
                f"Invalid assembly ID(s) provided: {to_build}. Available assembly IDs: {available_assembly_ids}"
            )

        # get the transformations and apply to affected asym IDs
        transformations = parse_transformations(struct_oper_category)  # {id: rotation, translation}
        assemblies = {}
        for _id, op_expr, asym_id_expr in zip(
            assembly_gen_category["assembly_id"].as_array(str),
            assembly_gen_category["oper_expression"].as_array(str),
            assembly_gen_category["asym_id_list"].as_array(str),
        ):
            # Find the operation expressions for given assembly ID
            if _id in to_build:
                operations = parse_operation_expression(op_expr)
                asym_ids = asym_id_expr.split(",")
                # Filter affected asym IDs
                sub_structure = atom_array[..., np.isin(atom_array.chain_id, asym_ids)]
                for operation in operations:
                    sub_assembly = apply_transformations(sub_structure, transformations, operation)
                    # Add transformation ID annotation (e.g., 1 for identity operation)
                    sub_assembly.set_annotation("transformation_id", np.full(len(sub_assembly), operation))
                    # Merge the chains with asym IDs for this operation with chains from other operations
                    assemblies[_id] = assemblies[_id] + sub_assembly if _id in assemblies else sub_assembly

                # Create a composite chain_id, transformation_id annotation for ease of access
                chain_full_id = np.char.add(
                    np.char.add(assemblies[_id].chain_id, "_"), assemblies[_id].transformation_id.astype(str)
                )
                assemblies[_id].set_annotation("chain_full_id", chain_full_id)

                # For molecules with multiple transformations, we need to check for non-polymers at symmetry centers
                if len(operations) > 1 and self.patch_symmetry_centers:
                    assemblies[_id] = self._maybe_patch_non_polymer_at_symmetry_center(assemblies[_id])
        return assemblies

    @lru_cache(maxsize=None)
    def _build_residue_atoms(self, residue_name: str) -> list[Atom]:
        """
        Build a list of atoms for a given residue name from CCD data.

        Args:
        - residue_name (str): The name of the residue.

        Returns:
        - AtomList: An AtomList object initialized with zero coordinates.
        """
        if residue_name not in self.data_by_residue.index:
            raise ValueError(f"Residue {residue_name} not found in precompiled CCD data.")

        ccd_atoms = self.data_by_residue.loc[residue_name]["atoms"]
        atom_list = [
            struc.Atom(
                [0.0, 0.0, 0.0],
                res_name=residue_name,
                atom_name=atom_name,
                element=atom_data["element"],
                charge=atom_data["charge"],
                leaving_atom_flag=atom_data["leaving_atom_flag"],
                leaving_group=atom_data["leaving_group"],
                is_metal=atom_data["is_metal"],
                hyb=atom_data["hyb"],
                nhyd=atom_data["nhyd"],
                hvydeg=atom_data["hvydeg"],
                align=atom_data["align"],
            )
            for atom_name, atom_data in ccd_atoms.items()
        ]
        return atom_list

    def _add_missing_atoms(self, atom_array: AtomArray, chain_info_dict: dict) -> AtomArray:
        """
        Add missing atoms to a polymer chain based on its sequence.

        Iterates through the residues in a given chain, identifies missing atoms based on the reference residue,
        and inserts the missing atoms into the atom array. Also augments atom data with Open Babel data.

        Args:
        - atom_array (AtomArray): An array of atom objects representing the current state of the polymer chain.
        - chain_info_dict (dict): A dictionary containing chain information, including whether the chain is a polymer and the sequence of residues.

        Returns:
        - AtomArray: An updated array of atom objects with the missing atoms added.
        """
        full_atom_list = []
        residue_ids = []
        chain_ids = []
        for chain_id in deduplicate_iterator(
            struc.get_chains(atom_array)
        ):  # NOTE: We need set() since biotite considers a decrease in sequence_id as a new chain (see `5xnl`)
            # Iterate through the sequence and create all atoms with zero coordinates
            residue_name_list = chain_info_dict[chain_id]["residue_name_list"]
            for residue_index_sequential, residue_name in enumerate(residue_name_list, start=1):
                residue_atom_list = self._build_residue_atoms(residue_name)
                if chain_info_dict[chain_id]["is_polymer"]:
                    # We assign the residue ID as the sequential index for polymers, consistent with the PDB label ids (but not author ids)
                    residue_ids.append(np.full(len(residue_atom_list), residue_index_sequential))
                else:
                    residue_id_nonpoly = chain_info_dict[chain_id]["residue_id_list"][
                        residue_index_sequential - 1
                    ]  # Recall that residue_index_sequential is 1-indexed
                    residue_ids.append(np.full(len(residue_atom_list), residue_id_nonpoly))
                chain_ids.append(np.full(len(residue_atom_list), chain_id))
                full_atom_list.append(residue_atom_list)

        # Create atom array object and flatten residue_ids and chain_ids
        full_atom_array = struc.array(np.concatenate(full_atom_list))
        full_atom_array.chain_id = np.concatenate(chain_ids)
        full_atom_array.res_id = np.concatenate(residue_ids)
        # Shenanigans to fix the data type of element
        elements = full_atom_array.element.astype(int)
        full_atom_array.del_annotation("element")
        full_atom_array.add_annotation("element", dtype=int)
        full_atom_array.set_annotation("element", elements)

        # Find overlap between populated atoms from atom_site and the full atom list derived from the sequences
        def create_structured_array(atom_array):
            """Create a structured array from an AtomArray object. Used for efficient element comparison."""
            dtype = np.dtype(
                [("chain_id", "U3"), ("res_id", "i4"), ("atom_name", "U4")]
            )  # TODO: Why not also compare `res_name`? Are res_id's guaranteed to match?
            structured_array = np.zeros(len(atom_array), dtype=dtype)
            structured_array["chain_id"] = atom_array.chain_id
            structured_array["res_id"] = atom_array.res_id
            structured_array["atom_name"] = atom_array.atom_name
            return structured_array

        full_atom_array_structured = create_structured_array(full_atom_array)
        present_atom_array_structured = create_structured_array(atom_array)
        full_atom_array_match_mask = np.isin(full_atom_array_structured, present_atom_array_structured)
        present_atom_array_match_mask = np.isin(present_atom_array_structured, full_atom_array_structured)

        if not np.array_equal(
            full_atom_array[full_atom_array_match_mask].atom_name,
            atom_array[present_atom_array_match_mask].atom_name,
        ):
            raise ValueError("Order of atom names in full_atom_array and atom_array do not match.")

        # Prepare np arrays to add b_factor and occupancy annotations to the AtomArray
        b_factor = np.zeros(len(full_atom_array), dtype=np.float32)
        b_factor[full_atom_array_match_mask] = atom_array[present_atom_array_match_mask].b_factor
        occupancy = np.zeros(len(full_atom_array), dtype=np.float32)
        occupancy[full_atom_array_match_mask] = atom_array[present_atom_array_match_mask].occupancy

        # Replace annotations
        full_atom_array.coord[full_atom_array_match_mask] = atom_array[present_atom_array_match_mask].coord
        full_atom_array.set_annotation("b_factor", b_factor)
        full_atom_array.set_annotation("occupancy", occupancy)

        # Create polymer annotation
        full_atom_array.add_annotation("is_polymer", dtype=bool)
        polymer_chain_ids = [chain_id for chain_id, chain_info in chain_info_dict.items() if chain_info["is_polymer"]]
        polymer_mask = np.isin(full_atom_array.chain_id, polymer_chain_ids)
        full_atom_array.set_annotation("is_polymer", polymer_mask)

        # If any heavy atom in a residue cannot be matched, then mask the whole residue
        unmatched_heavy_atoms_mask = ~present_atom_array_match_mask & (
            (atom_array.element != "H")
            & (atom_array.element != "D")  # Note that in atom_array the elements are still strings
        )
        unmatched_heavy_atoms = atom_array[unmatched_heavy_atoms_mask]
        for i in range(len(unmatched_heavy_atoms)):
            atom = unmatched_heavy_atoms[i]
            residue_mask = (full_atom_array.chain_id == atom.chain_id) & (full_atom_array.res_id == atom.res_id)
            full_atom_array.occupancy[residue_mask] = np.zeros(np.sum(residue_mask), dtype=np.half)

        return full_atom_array

    @lru_cache(maxsize=None)
    def _get_intra_residue_bonds(self, residue_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve intra-residue bonds for a given residue.

        Args:
        - residue_name (str): The name of the residue.

        Returns:
        - tuple: Three arrays representing the atom indices and bond types within the residue frame.
        """
        residue_data = self.data_by_residue.loc[residue_name]
        # Create a mapping of atom IDs to indices
        atom_id_to_index = {atom_id: index for index, atom_id in enumerate(residue_data["atoms"].keys())}
        atom_a_indices = []
        atom_b_indices = []
        bond_types = []
        for bond in residue_data["intra_residue_bonds"]:
            atom_a_index = atom_id_to_index[bond["atom_a_id"]]
            atom_b_index = atom_id_to_index[bond["atom_b_id"]]
            bond_type = get_bond_type_from_order_and_is_aromatic(bond["order"], bond["is_aromatic"])
            atom_a_indices.append(atom_a_index)
            atom_b_indices.append(atom_b_index)
            bond_types.append(bond_type)
        return np.array(atom_a_indices), np.array(atom_b_indices), np.array(bond_types)

    def _get_inter_and_intra_residue_bonds(
        self, atom_array: AtomArray, chain_id: str, chain_type: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Adds inter-residue and intra_residue bonds to an atom array for a given chain.

        Args:
        - atom_array (AtomArray): The atom array to which the bonds are added.
        - chain_id (str): The ID of the chain for which bonds are added.
        - chain_type (str): The type of the chain, used to determine the type of bond.

        Returns:
        - intra_residue_bonds: An np.array of intra-residue bonds to be added to the atom array.
        - leaving_atom_indices: An np.array of indices of atom indices that are leaving groups for bookkeeping.
        """
        # Possible types given at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
        atom_pairs = {
            "polydeoxyribonucleotide": ("O3'", "P"),  # phosphodiester bond
            "polydeoxyribonucleotide/polyribonucleotide hybrid": ("O3'", "P"),  # phosphodiester bond
            "polypeptide(D)": ("C", "N"),  # peptide bond
            "polypeptide(L)": ("C", "N"),  # peptide bond
            "polyribonucleotide": ("O3'", "P"),  # phosphodiester bond
        }

        # Append as we go along and then concatenate at the end
        inter_residue_bonds = []
        atom_a_intra_residue_indices = []
        atom_b_intra_residue_indices = []
        intra_residue_bond_types = []
        leaving_atom_indices = []

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
                    leaving_atom_indices.append(
                        current_res.index[np.isin(current_res.atom_name, atom_a.leaving_group[0])]
                    )
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
            residue_name = current_res.res_name[
                0
            ]  # current_res.res_name is a list of identical values, so we just take the first one
            if (
                residue_name in self.data_by_residue.index
                and len(self.data_by_residue.loc[residue_name]["intra_residue_bonds"]) > 0
            ):
                atom_a_local_indices, atom_b_local_indices, bond_types = self._get_intra_residue_bonds(residue_name)
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

    def _get_ligand_of_interest_info(self, cif_block: CIFBlock) -> dict:
        """Extract ligand of interest information from a CIF block.

        Reference:
            - https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/small-molecule-ligands
        """

        # Extract binary flag for whether the ligand of interest is specified
        # NOTE: This is being used in addition to the below as it has slightly higher coverage across the PDB
        # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_pdbx_entry_details.has_ligand_of_interest.html
        has_loi = (
            category_to_dict(cif_block, "pdbx_entry_details").get("has_ligand_of_interest", np.array(["N"]))[0] == "Y"
        )

        # Extract which ligand is of interest if specified
        # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_pdbx_entity_instance_feature.feature_type.html
        entity_instance_feature = category_to_dict(cif_block, "pdbx_entity_instance_feature")
        comp_id_names = entity_instance_feature.get("comp_id", np.array([], dtype="<U3"))
        comp_id_mask = entity_instance_feature.get("feature_type", np.array([])) == "SUBJECT OF INVESTIGATION"

        return {
            "ligand_of_interest": comp_id_names[comp_id_mask],
            "has_ligand_of_interest": has_loi | (len(comp_id_names) > 0),
        }

    def _add_bonds_from_struct_conn(
        self, cif_block: CIFBlock, chain_info_dict: dict, atom_array: AtomArray
    ) -> tuple[list[list[int]], list[int]]:
        """
        Adds bonds from the 'struct_conn' category of a CIF block to an atom array. Only covalent bonds are considered.

        Args:
        - cif_block (CIFBlock): The CIF block for the entry.
        - chain_info_dif (Dict): A dictionary containing information about the chains.
        - atom_array (AtomArray): The atom array used to get atom indices.

        Returns:
        - struct_conn_bonds: A List of bonds to be added to the atom array.
        - leaving_atom_indices: A List of indices of atoms that are leaving groups for bookkeeping.
        """
        if "struct_conn" not in cif_block:
            return [], []

        struct_conn_df = category_to_df(cif_block, "struct_conn")
        struct_conn_df = struct_conn_df[
            struct_conn_df["conn_type_id"] == "covale"
        ]  # Only consider covalent bonds (throw out disulfide bods, metal coordination covalent bonds, hydrogen bonds)
        struct_conn_bonds = []
        leaving_atom_indices = []

        if not struct_conn_df.empty:
            for _, row in struct_conn_df.iterrows():
                a_chain_id = row["ptnr1_label_asym_id"]
                b_chain_id = row["ptnr2_label_asym_id"]
                a_seq_id = (
                    row["ptnr1_label_seq_id"] if chain_info_dict[a_chain_id]["is_polymer"] else row["ptnr1_auth_seq_id"]
                )
                b_seq_id = (
                    row["ptnr2_label_seq_id"] if chain_info_dict[b_chain_id]["is_polymer"] else row["ptnr2_auth_seq_id"]
                )
                a_atom_id = row["ptnr1_label_atom_id"]
                b_atom_id = row["ptnr2_label_atom_id"]
                a_res_name = row["ptnr1_label_comp_id"]
                b_res_name = row["ptnr2_label_comp_id"]

                # Get the indices of the atoms and append to the list
                residue_a = atom_array[(atom_array.chain_id == a_chain_id) & (atom_array.res_id == int(a_seq_id))]
                residue_b = atom_array[(atom_array.chain_id == b_chain_id) & (atom_array.res_id == int(b_seq_id))]

                # Ensure that the we picked the correct residue (to handle sequence heterogeneity; see PDB ID `3nez` for an example)
                if a_res_name != residue_a.res_name[0] or b_res_name != residue_b.res_name[0]:
                    continue

                atom_a = residue_a[residue_a.atom_name == a_atom_id]
                atom_b = residue_b[residue_b.atom_name == b_atom_id]
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

        # Add to legacy
        self.extra_info["struct_conn_bonds"] = struct_conn_bonds

        return struct_conn_bonds, leaving_atom_indices

    def _add_bonds(self, cif_block: CIFBlock, atom_array: AtomArray, chain_info_dict: dict) -> AtomArray:
        """
        Add bonds to the atom array stack using precomputed CCD data and the mmCIF `struct_conn` field.

        Args:
        - cif_block (CIFBlock): The CIF file block containing the structure data.
        - atom_array (AtomArray): The atom array to which the bonds will be added.
        - chain_info_dict (dict): A dictionary containing information about the chains in the structure.

        Returns:
        - AtomArray: The updated atom array with bonds added.
        """
        # Step 0: Add index to atom_array for ease of access
        atom_array.set_annotation("index", np.arange(len(atom_array)))

        # Step 1: Add inter-residue and inter-chain bonds from the `struct_conn` category in the CIF file
        leaving_atom_indices = []
        struct_conn_bonds, struct_conn_leaving_atom_indices = self._add_bonds_from_struct_conn(
            cif_block, chain_info_dict, atom_array
        )
        if exists(struct_conn_leaving_atom_indices) and len(struct_conn_leaving_atom_indices) > 0:
            leaving_atom_indices.append(np.concatenate(struct_conn_leaving_atom_indices))

        # Step 2: Add inter-residue and intra-residue bonds
        inter_and_intra_residue_bonds = []
        for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
            chain_bonds, chain_leaving_atom_indices = self._get_inter_and_intra_residue_bonds(
                atom_array, chain_id, chain_info_dict[chain_id]["type"]
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

        # Delete leaving atoms and bonds to leaving atoms
        leaving_atoms = np.unique(np.concatenate(leaving_atom_indices))
        all_atom_indices = atom_array.index
        return atom_array[np.setdiff1d(all_atom_indices, leaving_atoms, True)]

    def _keep_last_residue(self, atom_array: AtomArray) -> AtomArray:
        """
        Removes duplicate residues in the atom array, keeping only the last occurrence.

        Args:
        - atom_array (AtomArray): The atom array stack containing the chain information.

        Returns:
        - AtomArray: The atom array with duplicate residues removed.
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

    def _update_nonpoly_seq_ids(self, atom_array: AtomArray, chain_info_dict: dict) -> None:
        """
        Updates the sequence IDs of non-polymeric chains in the atom array stack.
        Additionally, adds an annotation to the atom array stack to indicate whether a chain is a polymer.

        Args:
        - atom_array (AtomArray): The atom array stack containing the chain information.
        - chain_info_dict (dict): Dictionary containing the sequence details of each chain.

        Returns:
        - AtomArray: The updated atom array stack with the sequence IDs updated for non-polymeric chains.
        """
        # For non-polymeric chains, we use the author sequence ids
        author_seq_ids = atom_array.get_annotation("auth_seq_id")
        chain_ids = atom_array.get_annotation("chain_id")

        # Create mask based on the is_polymer column
        non_polymer_mask = ~np.array([chain_info_dict[chain_id]["is_polymer"] for chain_id in chain_ids])

        # Update the atom_array_label with the (1-indexed) author sequence ids
        atom_array.res_id[non_polymer_mask] = author_seq_ids[non_polymer_mask]

        return atom_array

    def _load_monomer_sequence_information(
        self, cif_block: CIFBlock, chain_info_dict: dict, atom_array: AtomArray
    ) -> tuple[dict, dict]:
        """
        Load monomer sequence information into a chain_info_dict.

        For polymers, uses the 'entity_poly_seq' category in the CIF block to get the sequence.
        For non-polymers, uses the atom array to get the sequence.

        Args:
            cif_block (CIFBlock): The CIF block containing the monomer sequence information.
            chain_info_dict (dict): The dictionary where the monomer sequence information will be stored.
            atom_array (AtomArray): The atom array used to get the sequence for non-polymers.

        Returns:
            tuple: The updated chain_info_dict with monomer sequence information and a dictionary with residue information.
        """
        # Handle polymers by using `entity_poly_seq`
        polymer_seq_df = category_to_df(cif_block, "entity_poly_seq")
        polymer_seq_df = polymer_seq_df.loc[:, ["entity_id", "num", "mon_id"]].rename(
            columns={"num": "residue_id", "mon_id": "residue_name"}
        )

        # Keep only the last occurrence of each residue
        polymer_seq_df.drop_duplicates(subset=["entity_id", "residue_id"], keep="last", inplace=True)

        # Filter out residues that are not in the precompiled CCD data
        polymer_seq_df = polymer_seq_df[polymer_seq_df["residue_name"].isin(self.data_by_residue.index)]

        # Map entity_id to lists of residue names and residue IDs
        polymer_seq_df["entity_id"] = polymer_seq_df["entity_id"].astype(float)
        polymer_entity_id_to_residue_names_and_ids = {
            entity_id: {"residue_names": group["residue_name"].tolist(), "residue_ids": None}
            for entity_id, group in polymer_seq_df.groupby("entity_id")
        }

        # Build up the chain_info_dict with the sequence information
        unique_residues = {
            residue
            for residues in polymer_entity_id_to_residue_names_and_ids.values()
            for residue in residues["residue_names"]
        }
        for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
            entity_id = int(chain_info_dict[chain_id]["entity_id"])
            if entity_id in polymer_entity_id_to_residue_names_and_ids:
                # For polymers, we use the stored entity residue list
                if polymer_entity_id_to_residue_names_and_ids[entity_id]["residue_names"]:
                    chain_info_dict[chain_id]["residue_name_list"] = polymer_entity_id_to_residue_names_and_ids[
                        entity_id
                    ]["residue_names"]
                    chain_info_dict[chain_id]["residue_id_list"] = polymer_entity_id_to_residue_names_and_ids[
                        entity_id
                    ]["residue_ids"]
            else:
                # For non-polymers, we must re-compute every time, since entities are not guaranteed to have the same monomer sequence (e.g., for H2O chains)
                chain_atom_array = atom_array[atom_array.chain_id == chain_id]
                residue_id_list, residue_name_list = struc.get_residues(chain_atom_array)
                # We don't need to filter out unmatched residues for non-polymers here, since we handled that by filtering the AtomArray earlier
                chain_info_dict[chain_id]["residue_name_list"] = residue_name_list
                chain_info_dict[chain_id]["residue_id_list"] = residue_id_list
                unique_residues.update(residue_name_list)

        # Remove entries from chain_info_dict that have no residues
        chain_info_dict = {
            chain_id: chain_info
            for chain_id, chain_info in chain_info_dict.items()
            if "residue_name_list" in chain_info
        }

        # Store information about each present residue in a dictionary
        residue_info_dict = {
            residue_name: {
                "planars": self.data_by_residue.loc[residue_name]["planars"],
                "chirals": self.data_by_residue.loc[residue_name]["chirals"],
                "automorphisms": self.data_by_residue.loc[residue_name]["automorphisms"],
            }
            for residue_name in unique_residues
        }

        return chain_info_dict, residue_info_dict

    def _get_chain_info(self, cif_block: CIFBlock, atom_array: AtomArray) -> dict:
        """
        Extracts chain information from the CIF block.

        Args:
        - cif_block (CIFBlock): Parsed CIF block.
        - atom_array (AtomArray): Atom array stack containing the chain information.

        Returns:
        - dict: Dictionary containing the sequence details of each chain.
        """
        chain_info_dict = {}

        # Step 1: Build a mapping of chain id to entity id from the `atom_site`
        chain_ids = atom_array.get_annotation("chain_id")
        entity_ids = atom_array.get_annotation("label_entity_id").astype(str)
        unique_chain_entity_map = {chain_id: entity_id for chain_id, entity_id in zip(chain_ids, entity_ids)}

        # Step 2: Load additional chain information
        entity_df = category_to_df(cif_block, "entity")
        entity_df["id"] = entity_df["id"].astype(str)
        entity_df.rename(columns={"type": "entity_type"}, inplace=True)
        entity_dict = entity_df.set_index("id").to_dict(orient="index")

        # From `entity_poly`
        polymer_df = category_to_df(cif_block, "entity_poly")
        polymer_df = polymer_df[
            ["entity_id", "type", "pdbx_seq_one_letter_code", "pdbx_seq_one_letter_code_can", "pdbx_strand_id"]
        ]
        polymer_df.rename(
            columns={
                "type": "polymer_type",
                "pdbx_seq_one_letter_code": "non_canonical_sequence",
                "pdbx_seq_one_letter_code_can": "canonical_sequence",
            },
            inplace=True,
        )
        polymer_df["entity_id"] = polymer_df["entity_id"].astype(str)
        polymer_dict = polymer_df.set_index("entity_id").to_dict(orient="index")

        # Step 3: Merge additional information into the dictionary
        for chain_id, entity_id in unique_chain_entity_map.items():
            chain_info = entity_dict.get(entity_id, {})
            polymer_info = polymer_dict.get(entity_id, {})

            chain_info_dict[chain_id] = {
                "entity_id": entity_id,
                "type": polymer_info.get("polymer_type", chain_info.get("entity_type", "")),
                "canonical_sequence": polymer_info.get("canonical_sequence", ""),
                "non_canonical_sequence": polymer_info.get("non_canonical_sequence", ""),
                "is_polymer": chain_info.get("entity_type") == "polymer",
            }

        return chain_info_dict

    def _get_metadata(self, cif_block: CIFBlock) -> dict:
        """Extract metadata from the CIF block."""
        metadata = {}
        metadata["pdb_id"] = cif_block["entry"]["id"].as_item().lower()
        exptl = cif_block["exptl"] if "exptl" in cif_block.keys() else None
        status = cif_block["pdbx_database_status"] if "pdbx_database_status" in cif_block.keys() else None
        refine = cif_block["refine"] if "refine" in cif_block.keys() else None
        em_reconstruction = cif_block["em_3d_reconstruction"] if "em_3d_reconstruction" in cif_block.keys() else None

        # Method
        metadata["method"] = exptl["method"].as_item().replace(" ", "_") if exptl else None
        # Initial deposition date (date)
        metadata["date"] = status["recvd_initial_deposition_date"].as_item() if status else None
        # Resolution
        metadata["resolution"] = None
        if refine:
            try:
                metadata["resolution"] = float(refine["ls_d_res_high"].as_item())
            except (KeyError, ValueError):
                pass

        if metadata["resolution"] is None and em_reconstruction:
            try:
                metadata["resolution"] = float(em_reconstruction["resolution"].as_item())
            except (KeyError, ValueError):
                pass

        return metadata

    @staticmethod
    def _maybe_patch_non_polymer_at_symmetry_center(
        atom_array: AtomArray, clash_distance=1.0, clash_ratio=0.5
    ) -> AtomArray:
        """
        In some PDB entries, non-polymer molecules are placed at the symmetry center and clash with themselves when
        transformed via symmetry operations. We should remove the duplicates in these cases, keeping the identity copy.

        We consider a non-polymer to be clashing with itself if at least `clash_ratio` of its atoms clash with the symmetric copy.

        Examples:
        — PDB ID `7mub` has a potassium ion at the symmetry center that when reflected with the symmetry operation clashes with itself.
        — PDB ID `1xan` has a ligand at a symmetry center that similarly when refelcted clashes with itself.

        Arguments:
        — atom_array (AtomArray): The atom array to be patched.
        — clash_distance (float): The distance threshold for two atoms to be considered clashing.
        - clash_ratio (float): The percentage of atoms that must clash for the molecule to be considered clashing.

        Returns:
        — AtomArray: The patched atom array.
        """
        # Filter to only atoms with coordinates to avoid non-physical clashes at the origin
        resolved_atom_array = atom_array[atom_array.occupancy > 0]

        if not np.any(~resolved_atom_array.is_polymer):
            return atom_array  # Early exit
        else:
            non_polymers = resolved_atom_array[~resolved_atom_array.is_polymer]  # [n]

            # Build cell list for rapid distance computations
            cell_list = struc.CellList(non_polymers, cell_size=3.0)

            # Quick check to see whether any non-polymer is closer than 0.05A to any other.
            clash_matrix = cell_list.get_atoms(non_polymers.coord, clash_distance, as_mask=True)  # [n, n]
            identity_matrix = np.identity(len(non_polymers), dtype=bool)
            if np.array_equal(clash_matrix, identity_matrix):
                return atom_array  # Early exit
            else:
                # Remove identity matrix so we don't count self-clashes
                clash_matrix = clash_matrix & ~identity_matrix

            # Get list of chain_ids with clashing atoms (for computational efficiency)
            clashing_atom_mask = np.sum(clash_matrix, axis=1) > 0
            clashing_chain_ids = np.unique(non_polymers.chain_id[clashing_atom_mask])

            # For each clashing chain, we check whether any non-polymer is clashing with a symmetric copy of itself
            # We count the clashes with each symmetric copy of itself and remove those that have a clash ratio above the threshold
            # We keep the identity transformation, or the lowest transformation ID in the case of multiple symmetric copies
            chain_full_ids_to_remove = []
            for chain_id in clashing_chain_ids:
                chain_mask = non_polymers.chain_id == chain_id
                mask = chain_mask & clashing_atom_mask  # Mask for clashing atoms in the current chain
                chain_clash_matrix = clash_matrix[mask][:, mask]

                # Loop through possible transformation ID's
                transformation_ids_to_check = sorted(
                    np.unique(non_polymers.transformation_id[mask].astype(int)).tolist()
                )
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

                    # For each transformation ID with a clash ratio above the threshold, note the chain_full_id to remove, and remove from the list to check
                    transformation_ids_to_remove = [
                        trans_id for trans_id, count in clash_count_by_transformation_id.items() if count > threshold
                    ]
                    chain_full_ids_to_remove.extend(
                        [f"{chain_id}_{trans_id}" for trans_id in transformation_ids_to_remove]
                    )
                    transformation_ids_to_check = [
                        id_ for id_ in transformation_ids_to_check if str(id_) not in transformation_ids_to_remove
                    ]

            # Filter and return
            keep_mask = ~np.isin(atom_array.chain_full_id, chain_full_ids_to_remove)
            atom_array = atom_array[keep_mask]
            return atom_array
