"""
Implementation of original `citufils` functions using `biotite` library.
Retains all functionality of orginal library, with increased performance and additional features.
"""

from __future__ import annotations
import logging
from functools import lru_cache
from typing import Literal
from cifutils.cifutils_biotite.constants import CRYSTALLIZATION_AIDS
from cifutils.cifutils_biotite.utils.atom_matching_utils import standardize_heavy_atom_ids

import numpy as np
import pandas as pd
from os import PathLike
from pathlib import Path

from cifutils.cifutils_biotite.utils.cifutils_biotite_utils import (
    deduplicate_iterator,
    read_cif_file,
)
from cifutils.cifutils_biotite.common import exists
from cifutils.cifutils_biotite.transforms.categories import get_chain_info, get_metadata, load_monomer_sequence_information_from_category, get_ligand_of_interest_info
from cifutils.cifutils_biotite.transforms.atom_array import mse_to_met, remove_atoms_by_residue_names, resolve_arginine_naming_ambiguity, keep_last_residue, update_nonpoly_seq_ids, add_bonds_to_bondlist
from cifutils.cifutils_biotite.utils.bond_utils import add_bonds_from_struct_conn, cached_bond_utils_factory, get_inter_and_intra_residue_bonds
from cifutils.cifutils_biotite.utils.assembly_utils import build_bioassembly_from_asym_unit
from cifutils.cifutils_biotite.utils.residue_utils import cached_residue_utils_factory

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx import CIFBlock
from biotite.structure import AtomArray, Atom, AtomArrayStack
from biotite.file import InvalidFileError

logger = logging.getLogger(__name__)

__all__ = ["CIFParser"]

class CIFParser:
    def __init__(
        self,
        by_residue_ligand_dir: str = "/projects/ml/RF2_allatom/cifutils_biotite/ccd_library",
    ):
        """
        Initialize a CIFParser object.

        Args:
            by_residue_ligand_dir (str, optional): Directory path to the pre-compiled residue-level CCD and OB data built from `make_residue_library_from_ccd.py`.
        """
        self.by_residue_ligand_dir = by_residue_ligand_dir

        # For backwards compatability
        self.extra_info = {}

    def _validate_arguments(self, **kwargs):
        """Validate the arguments passed to the CIFParser object."""
        add_missing_atoms = kwargs.get('add_missing_atoms')
        add_bonds = kwargs.get('add_bonds')
        load_from_cache = kwargs.get('load_from_cache')
        save_to_cache = kwargs.get('save_to_cache')
        cache_dir = kwargs.get('cache_dir')

        if not add_missing_atoms and add_bonds:
            raise ValueError("add_bonds cannot be True if add_missing_atoms is False")

        if not add_missing_atoms and add_bonds:
            raise ValueError("add_bonds cannot be True if add_missing_atoms is False")
        
        if load_from_cache and not cache_dir:
            raise ValueError("Must provide a cache directory to load from cache")
        
        if save_to_cache and not cache_dir:
            raise ValueError("Must provide a cache directory to save to cache")

    @lru_cache(maxsize=1000)
    def _data_by_residue(self, residue_name: str) -> dict:
        """Loads the data for a given residue from the precompiled library."""
        # Find the residue in the precompiled library
        path = Path(self.by_residue_ligand_dir) / f"{residue_name}.pkl"
        assert path.exists(), f"Residue {residue_name} not found in precompiled library."

        # Load and return the residue data
        return pd.read_pickle(path)

    @lru_cache(maxsize=1000)
    def _residue_file_exists(self, residue_name: str) -> bool:
        """Check if a residue file exists in the precompiled library."""
        path = Path(self.by_residue_ligand_dir) / f"{residue_name}.pkl"
        return path.exists()

    @lru_cache(maxsize=None)
    def _all_precompiled_residues(self) -> list[str]:
        """Return a list of all supported residues in the precompiled library."""
        # Get all residues in the precompiled library
        all_residues = [path.stem.upper() for path in Path(self.by_residue_ligand_dir).glob("*.pkl")]

        # Remove unsupported residues
        supported_residues = [residue for residue in all_residues if residue not in self._residues_to_remove]
        
        # Sanity check -- ensure that we removed at least one residue, if we provided a list of residues to remove
        if len(self._residues_to_remove) > 0:
            assert len(supported_residues) < len(all_residues), "No residues were removed from the precompiled library with the provided residues_to_remove list."

        return supported_residues
    
    def parse(self, load_from_cache: bool = False, save_to_cache: bool = False, cache_dir: PathLike = None, **kwargs):
        """
        Entrypoint for CIF parsing, which can either:
            - Directly parse from CIF, using the specified keyword arguments; or,
            - Load the CIF from a cached directory, re-building bioassemblies on-the-fly

        In addition to the arguments in `parse_cif_from_rcsb`, this function can also include the following arguments:
            - load_from_cache (bool, optional): Whether to load pre-compiled results from cache. Defaults to False.
            - save_to_cache (bool, optional): Whether to save pre-compiled results to cache. Defaults to False.
            - cache_dir (PathLike, optional): Directory path to save pre-compiled results. Defaults to None.
        """
        self._validate_arguments(**kwargs)

        if cache_dir:
            cache_dir = Path(cache_dir)
            # Make the cache directory if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)

        if load_from_cache and cache_dir:
            # Load from cache
            cache_file_path = cache_dir / f"{Path(kwargs['filename']).stem}.pkl"
            if cache_file_path.exists():
                # Load the result from the cache
                result =  pd.read_pickle(cache_file_path)

                # Build assemblies
                atom_array_stack = result["atom_array_stack"]

                if "assembly_gen_category" in result["extra_info"]:
                    assembly_gen_category = result["extra_info"]["assembly_gen_category"]
                    struct_oper_category = result["extra_info"]["struct_oper_category"]
                    assemblies = build_bioassembly_from_asym_unit(
                        assembly_gen_category=assembly_gen_category,
                        struct_oper_category=struct_oper_category,
                        atom_array_stack=atom_array_stack,
                        assembly_ids=kwargs.get("build_assembly", "all"),
                        patch_symmetry_centers=kwargs.get("patch_symmetry_centers", True),
                    )
                else:
                    assemblies = atom_array_stack

                # Return updated result
                result["assemblies"] = assemblies
                return result

        # Parse from CIF
        result = self.parse_from_cif(save_to_cache=save_to_cache, **kwargs)
        if save_to_cache and cache_dir:
            # We want our cache to include:
            #   (1) All keys in `result` excep the assemblies and 
            #   (2) The information needed to rebuild the assemblies, which is stored in `result["extra_info"]`
            cache_file_path = cache_dir / f"{Path(kwargs['filename']).stem}.pkl"
            if not cache_file_path.exists():
                # Save the result to the cache, excluding the assemblies
                result_to_cache = {k: v for k, v in result.items() if k != "assemblies"}
                pd.to_pickle(result_to_cache, cache_file_path)
            
        return result

    def parse_from_cif(
        self,
        filename: PathLike,
        save_to_cache: bool,
        assume_residues_all_resolved: bool = False, # TODO: Implement
        add_missing_atoms: bool = True,
        add_bonds: bool = True,
        remove_waters: bool = True,
        residues_to_remove: list[str] = CRYSTALLIZATION_AIDS,
        patch_symmetry_centers: bool = True,
        build_assembly: Literal["first", "all"] | list[str] | None = "all",
        fix_arginines: bool = True,
        convert_mse_to_met: bool = False,
        add_hydrogens: bool = True,
        model: int | None = None,
    ) -> dict:
        """
        Parse the CIF file (must contain information from the PDB) and return 
        chain information, residue information, atom array, metadata, and 
        legacy data.

        Args:
            save_to_cache (bool): Whether to save the results to cache.
            filename (str): Path to the CIF file. May be any format of CIF file 
                (e.g., gz, bcif, etc.).
            assume_residues_all_resolved (bool): Whether we can assume when 
                parsing that all residues are represented, and all atoms are 
                present. Required for distillation examples that do not have 
                all RCSB fields. Defaults to False.
            add_missing_atoms (bool, optional): Whether to add missing atoms to 
                the structure, including relevant OpenBabel atom-level data.
            add_bonds (bool, optional): Whether to add bonds to the structure. 
                Cannot be True if `add_missing_atoms` is False.
            remove_waters (bool, optional): Whether to remove water molecules 
                from the structure.
            residues_to_remove (list, optional): A list of residue names to 
                remove from the structure. Defaults to crystallization aids. 
                NOTE: Exclusion of polymer residues and common multi-chain 
                ligands must be done with care to avoid sequence gaps.
            patch_symmetry_centers (bool, optional): Whether to patch 
                non-polymer residues at symmetry centers that clash with 
                themselves when transformed.
            build_assembly (str, optional): Which assembly to build, if any. 
                Options are None (e.g., asymmetric unit), "first", "all", or a 
                list of assembly IDs.
            fix_arginines (bool, optional): Whether to fix arginine naming 
                ambiguity, see the AF-3 supplement for details.
            convert_mse_to_met (bool, optional): Whether to convert 
                selenomethionine (MSE) residues to methionine (MET) residues.
            add_hydrogens (bool, optional): Whether to add hydrogens to the 
                structure. Defaults to True.
            model (int, optional): The model number to parse from the CIF file 
                for NMR entries. Defaults to all models (None).

        Returns:
            dict: A dictionary containing the following keys:
                'chain_info': A dictionary mapping chain ID to sequence, type, 
                    entity ID, EC number, and other information.
                'ligand_info': A dictionary containing ligand of interest 
                    information.
                'atom_array_stack': An AtomArrayStack instance representing the 
                    asymmetric unit.
                'assemblies': A dictionary mapping assembly IDs to 
                    AtomArrayStack instances.
                'metadata': A dictionary containing metadata about the 
                    structure (e.g., resolution, deposition date, etc.).
                'extra_info': A dictionary with information for 
                    cross-compatibility and caching. Should typically not be 
                    used directly.
        """
        # Set class variables for universal access
        self._add_hydrogens = add_hydrogens
        self._residues_to_remove = [residue.upper() for residue in residues_to_remove]

        # Initialize internal processing variables, which we will later populate
        _converted_res = {}
        _ignored_res = []

        cif_file = read_cif_file(filename)
        cif_block = cif_file.block

        # Load metadata (either from RCSB standard fields, or from the custom `extra_metadata` field)
        fallback_filename = Path(filename).stem
        metadata = get_metadata(cif_block, fallback_id=fallback_filename)

        # Load structure using the RCSB labels for sequence ids, and later update for non-polymers

        common_extra_fields = [
            "label_entity_id",
            "auth_seq_id",  # for non-polymer residue indexing
            "atom_id",
            "b_factor",
            "occupancy",
        ]
        try:
            atom_array_stack = pdbx.get_structure(
                cif_block,
                extra_fields=common_extra_fields,
                use_author_fields=False,
                altloc="occupancy",
                model=model,
            )
        except InvalidFileError:
            logger.warning(f"Invalid file error encountered for {filename}; loading with only one model")
            # Try again, choosing only one model
            atom_array_stack = pdbx.get_structure(
                cif_block,
                extra_fields=common_extra_fields,
                use_author_fields=True,
                altloc="occupancy",
                model=1,
            )
        
        # Ensure we have an atom array stack (e.g., if we selected a specific model, we may get an AtomArray)
        if not isinstance(atom_array_stack, AtomArrayStack):
            atom_array_stack = struc.stack([atom_array_stack])

        # Load chain information from the first model (uses atom_array to build chain list)
        # NOTE: If not loading from RCSB (e.g., distillation), then the chain_info_dict will be empty
        chain_info_dict = get_chain_info(cif_block, atom_array_stack[0])

        # Loop through models
        models = []
        for model_idx in range(atom_array_stack.stack_depth()):
            atom_array = atom_array_stack[model_idx]

            # Remove hydrogens (most examples will not have any hydrogens; only NMR studies and small molecules)
            if not self._add_hydrogens:
                atom_array = atom_array[~np.isin(atom_array.element, ["H", "D"])]

            # Remove waters
            if remove_waters:
                atom_array = atom_array[atom_array.res_name != "HOH"]
                _ignored_res.append("HOH")

            # Replace non-polymeric chain sequence ids with author sequence ids
            atom_array = update_nonpoly_seq_ids(atom_array, chain_info_dict)

            # Remove unmatched residues from the atom array
            excluded_residues = set(atom_array.res_name) - set(self._all_precompiled_residues())
            atom_array = remove_atoms_by_residue_names(atom_array, list(excluded_residues))

            # Load monomer sequence information into chain_info_dict
            # NOTE: We MAY NOT delete atoms from the AtomArray after this step, as it may modify the sequence
            chain_info_dict = load_monomer_sequence_information_from_category(
                cif_block = cif_block, 
                chain_info_dict = chain_info_dict, 
                atom_array = atom_array, 
                known_residues = self._all_precompiled_residues()
            )

            # Handle sequence heterogeneity by selecting the residue that appears last
            atom_array = keep_last_residue(atom_array)

            # Create a larger atom array that includes missing atoms (e.g., hydrogens), then populate with atoms details loaded from structure
            if add_missing_atoms:
                # Create cached function to build residues
                self._build_residue_atoms = cached_residue_utils_factory(
                    known_residues=self._all_precompiled_residues(), 
                    data_by_residue=self._data_by_residue, 
                )
                # Create cached function to get intra-residue bonds
                self._get_intra_residue_bonds = cached_bond_utils_factory(
                    data_by_residue=self._data_by_residue, 
                )
                # Add missing atoms to the atom array, using the reference residue as a template
                atom_array = self._add_missing_atoms(atom_array, chain_info_dict)
            
            # Remove any excluded residues (e.g., crystallization solvents)
            # NOTE: If the excluded residues are part of a polymer chain, or part of a multi-chain ligand, this may create sequence gaps!
            if self._residues_to_remove:
                atom_array = remove_atoms_by_residue_names(atom_array, self._residues_to_remove)
                _ignored_res.extend(self._residues_to_remove)

            # Resolve arginine naming ambiguity
            if fix_arginines:
                atom_array = resolve_arginine_naming_ambiguity(atom_array)

            # Convert MSE to MET
            if convert_mse_to_met:
                atom_array = mse_to_met(atom_array)
                _converted_res["MSE"] = "MET"

            # Get bonds from the preprocessed CCD and OpenBabel data
            if add_bonds:
                atom_array = add_bonds_to_bondlist(
                    cif_block=cif_block, 
                    atom_array=atom_array, 
                    chain_info_dict=chain_info_dict, 
                    add_hydrogens=self._add_hydrogens,
                    known_residues=self._all_precompiled_residues(),
                    get_intra_residue_bonds=self._get_intra_residue_bonds,
                    converted_res=_converted_res,
                    ignored_res=_ignored_res,
                )

            models.append(atom_array)

        # Create an AtomArrayStack from the models list
        processed_atom_array_stack = struc.stack(models)

        # Build the assembly and add the transformation_id annotation (defaults to identity)
        if exists(build_assembly):
            if "pdbx_struct_assembly" not in cif_block.keys():
                assemblies = atom_array_stack
            else:
                assembly_gen_category = cif_block["pdbx_struct_assembly_gen"]
                struct_oper_category = cif_block["pdbx_struct_oper_list"]
                assemblies = build_bioassembly_from_asym_unit(
                    assembly_gen_category=assembly_gen_category,
                    struct_oper_category=struct_oper_category,
                    atom_array_stack=processed_atom_array_stack,
                    assembly_ids=build_assembly,
                    patch_symmetry_centers=patch_symmetry_centers,
                )

                # If we're caching, we need to store the assembly information in extra_info
                if save_to_cache:
                    self.extra_info["assembly_gen_category"] = assembly_gen_category
                    self.extra_info["struct_oper_category"] = struct_oper_category
        else:
            assemblies = {}

        # Get ligand of interest information
        loi_info = get_ligand_of_interest_info(cif_block)

        # Add final sequence information to chain_info_dict
        return {
            "chain_info": chain_info_dict,
            "ligand_info": loi_info,
            "atom_array_stack": processed_atom_array_stack,
            "assemblies": assemblies,
            "metadata": metadata,
            "extra_info": {**self.extra_info}, 
        }


    def _add_missing_atoms(self, atom_array: AtomArray, chain_info_dict: dict) -> AtomArray:
        """
        Add missing atoms to a polymer chain based on its sequence.

        Iterates through the residues in a given chain, identifies missing atoms based on the reference residue,
        and inserts the missing atoms into the atom array. Also augments atom data with Open Babel data.

        Args:
            atom_array (AtomArray): An array of atom objects representing the current state of the polymer chain.
            chain_info_dict (dict): A dictionary containing chain information, including whether the chain is a polymer and the sequence of residues.

        Returns:
            AtomArray: An updated array of atom objects with the missing atoms added.
        """
        full_atom_list = []
        residue_ids = []
        chain_ids = []
        # NOTE: We need deduplicate_iterator() since biotite considers a decrease in sequence_id as a new chain (see `5xnl`)
        for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
            # Iterate through the sequence and create all atoms with zero coordinates
            residue_name_list = chain_info_dict[chain_id]["residue_name_list"]
            for residue_index_sequential, residue_name in enumerate(residue_name_list, start=1):
                residue_atom_list = self._build_residue_atoms(residue_name, self._add_hydrogens)
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

        # Standardize heavy atom naming
        atom_array.atom_name = standardize_heavy_atom_ids(atom_array)
        full_atom_array.atom_name = standardize_heavy_atom_ids(full_atom_array)

        # Compute index mapping between `full_atom_array`
        # ... get lookup table of id -> idx for full_atom_array
        id_to_idx_full_atom_array = {
            id: idx
            for idx, id in enumerate(
                zip(
                    full_atom_array.chain_id,
                    full_atom_array.res_id,
                    full_atom_array.res_name,
                    full_atom_array.atom_name,
                )
            )
        }
        assert len(id_to_idx_full_atom_array) == len(full_atom_array), "Duplicate atom ids in `full_atom_array`!"

        # ... inspect all present atoms in `atom_array` and get the matching idx in `full_atom_array`
        full_atom_array_match_idx = []
        atom_array_match_idx = []
        _failed_to_match = []
        for idx, id in enumerate(
            zip(atom_array.chain_id, atom_array.res_id, atom_array.res_name, atom_array.atom_name)
        ):
            if id in id_to_idx_full_atom_array:
                full_atom_array_match_idx.append(id_to_idx_full_atom_array[id])
                atom_array_match_idx.append(idx)
            else:
                _failed_to_match.append(idx)
                logger.warning(f"Atom {id} not found in `full_atom_array`!")

        # ... turn arrays into np arrays
        full_atom_array_match_idx = np.array(full_atom_array_match_idx)
        atom_array_match_idx = np.array(atom_array_match_idx)

        # ... verify that there is a 1-to-1 mapping between the two arrays
        if not len(full_atom_array_match_idx) == len(atom_array_match_idx):
            unique, counts = np.unique(full_atom_array_match_idx, return_counts=True)
            # ... find duplicates in `full_atom_array_match_idx` for error message
            duplicates = unique[counts > 1]
            duplicates_id = full_atom_array[full_atom_array_match_idx][duplicates]
            raise ValueError(
                f"Mismatch between `full_atom_array` and `atom_array`! Found {len(duplicates)} duplicates in `full_atom_array_match_idx`:\n{duplicates_id}"
            )

        # Carry over the annotations from `atom_array` to `full_atom_array` for corresponding atoms
        # ... initialize
        b_factor = np.zeros(len(full_atom_array), dtype=np.float32)
        occupancy = np.zeros(len(full_atom_array), dtype=np.float32)
        # ... carry over annotations
        full_atom_array.coord[full_atom_array_match_idx] = atom_array[atom_array_match_idx].coord
        b_factor[full_atom_array_match_idx] = atom_array[atom_array_match_idx].b_factor
        occupancy[full_atom_array_match_idx] = atom_array[atom_array_match_idx].occupancy
        full_atom_array.set_annotation("b_factor", b_factor)
        full_atom_array.set_annotation("occupancy", occupancy)
        # ... polymer annotation
        full_atom_array.add_annotation("is_polymer", dtype=bool)
        polymer_chain_ids = [chain_id for chain_id, chain_info in chain_info_dict.items() if chain_info["is_polymer"]]
        polymer_mask = np.isin(full_atom_array.chain_id, polymer_chain_ids)
        full_atom_array.set_annotation("is_polymer", polymer_mask)

        # If any heavy atom in a residue cannot be matched, then mask the whole residue
        if len(_failed_to_match) > 0:
            failing_atoms = atom_array[np.array(_failed_to_match)]
            is_heavy = ~np.isin(failing_atoms.element, ["H", "D", "T"])
            if len(failing_atoms[is_heavy]) > 0:
                for atom in failing_atoms[is_heavy]:
                    chain_id, res_id, res_name = atom.chain_id, atom.res_id, atom.res_name
                    residue_mask = (
                        (full_atom_array.chain_id == chain_id)
                        & (full_atom_array.res_id == res_id)
                        & (full_atom_array.res_name == res_name)
                    )
                    full_atom_array.occupancy[residue_mask] = 0
                logger.warning(
                    f"Masked residues for {len(failing_atoms[is_heavy])} heavy atoms in `atom_array` that failed to match."
                )

        return full_atom_array
