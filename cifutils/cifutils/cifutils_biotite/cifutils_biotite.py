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
    apply_assembly_transformation,
    deduplicate_iterator,
    fix_bonded_atom_charges,
    get_bond_type_from_order_and_is_aromatic,
    parse_transformations,
    read_cif_file,
)
from cifutils.cifutils_biotite.common import exists
from cifutils.cifutils_biotite.transforms.categories import get_chain_info, get_metadata, load_monomer_sequence_information_from_category, get_ligand_of_interest_info
from cifutils.cifutils_biotite.transforms.atom_array import mse_to_met, remove_atoms_by_residue_names, resolve_arginine_naming_ambiguity, keep_last_residue, update_nonpoly_seq_ids, maybe_patch_non_polymer_at_symmetry_center
from cifutils.cifutils_biotite.utils.bond_utils import add_bonds_from_struct_conn

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx import CIFBlock, CIFCategory
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
        - by_residue_ligand_dir (str, optional): Directory path to the pre-compiled residue-level CCD and OB data built from `make_residue_library_from_ccd.py`.
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
        supported_residues = [residue for residue in all_residues if residue not in self.residues_to_remove]
        
        # Sanity check -- ensure that we removed at least one residue, if we provided a list of residues to remove
        if len(self.residues_to_remove) > 0:
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
                    assemblies = self._build_assembly(
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
        self.add_hydrogens = add_hydrogens
        self.residues_to_remove = [residue.upper() for residue in residues_to_remove]

        # Initialize internal processing variables, which we will later populate
        converted_res = {}
        ignored_res = []

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
            if not self.add_hydrogens:
                atom_array = atom_array[~np.isin(atom_array.element, ["H", "D"])]

            # Remove waters
            if remove_waters:
                atom_array = atom_array[atom_array.res_name != "HOH"]
                ignored_res.append("HOH")

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
                # Add missing atoms to the atom array, using the reference residue as a template
                atom_array = self._add_missing_atoms(atom_array, chain_info_dict)
            
            # Remove any excluded residues (e.g., crystallization solvents, AF-3 excluded ligands)
            # NOTE: We must remove AFTER adding missing atoms, otherwise we add them back as unresolved residues if they are still part of the polymer sequence
            if residues_to_remove:
                atom_array = remove_atoms_by_residue_names(atom_array, residues_to_remove)
                ignored_res.extend(residues_to_remove)

            # Resolve arginine naming ambiguity
            if fix_arginines:
                atom_array = resolve_arginine_naming_ambiguity(atom_array)

            # Convert MSE to MET
            if convert_mse_to_met:
                atom_array = mse_to_met(atom_array)
                converted_res["MSE"] = "MET"

            # Get bonds from the preprocessed CCD and OpenBabel data
            if add_bonds:
                atom_array = self._add_bonds(cif_block, atom_array, chain_info_dict, converted_res, ignored_res)

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
                assemblies = self._build_assembly(
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

    def _build_assembly(
        self,
        assembly_gen_category: CIFCategory,
        struct_oper_category: CIFCategory,
        atom_array_stack: AtomArrayStack,
        assembly_ids: Literal["all", "first"] | list[str] = "first",
        patch_symmetry_centers: bool = True,
    ) -> AtomArrayStack:
        """
        Build the first biological assembly found within the mmCIF file and update the `transformation_id` annotation.

        Code modified from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1318

        Args:
        - cif_block (CIFBlock): The CIF block containing the structure data.
        - atom_array_stack (AtomArrayStack): The atom array stack to which the transformations will be applied.
        - assembly_id (int, optional): The ID of the assembly to build. Defaults to None, which means the first assembly will be built.

        Returns:
        - AtomArray: The atom array with the biological assembly built and transformation_id annotations updated.
        """

        # Parse CIF blocks and select assembly (either by passed assembly_id or the first assembly)
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
                operations = pdbx.convert._parse_operation_expression(op_expr)
                asym_ids = asym_id_expr.split(",")
                # Filter affected asym IDs
                sub_structure = atom_array_stack[..., np.isin(atom_array_stack.chain_id, asym_ids)]
                for operation in operations:
                    sub_assembly = apply_assembly_transformation(sub_structure, transformations, operation)
                    # Add transformation ID annotation (e.g., 1 for identity operation)
                    if len(operation) > 1:
                        # Rarely, operation expressions will have multiple elements defining their name
                        # (e.g. ('1', 'X0') for `2fs3`), in this case we combine them into a single string
                        # for referencing the operation later on
                        operation = "".join(operation)
                    sub_assembly.set_annotation("transformation_id", np.full(sub_assembly.array_length(), operation))
                    # Merge the chains with asym IDs for this operation with chains from other operations
                    assemblies[_id] = assemblies[_id] + sub_assembly if _id in assemblies else sub_assembly

                # Create a composite chain_id, transformation_id annotation for ease of access (named chain instance ID, e.g., chain_iid)
                chain_iid = np.char.add(
                    np.char.add(assemblies[_id].chain_id.astype("<U20"), "_"),
                    assemblies[_id].transformation_id.astype(str),
                )
                assemblies[_id].set_annotation("chain_iid", chain_iid)

                # For molecules with multiple transformations, we need to check for non-polymers at symmetry centers
                if len(operations) > 1 and patch_symmetry_centers:
                    assemblies[_id] = maybe_patch_non_polymer_at_symmetry_center(assemblies[_id])
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
        if not self._residue_file_exists(residue_name):
            raise ValueError(f"Residue {residue_name} not found in precompiled CCD data.")

        ccd_atoms = self._data_by_residue(residue_name)["atoms"]
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

        # Remove hydrogens, if necessary
        if not self.add_hydrogens:
            atom_list = [atom for atom in atom_list if atom.element != 1]

        return atom_list

    def _add_bonds(
        self,
        cif_block: CIFBlock,
        atom_array: AtomArray,
        chain_info_dict: dict,
        converted_res: dict = {},
        ignored_res: list = [],
    ) -> AtomArray:
        """
        Add bonds to the atom array using precomputed CCD data and the mmCIF `struct_conn` field.

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
        struct_conn_bonds, struct_conn_leaving_atom_indices = add_bonds_from_struct_conn(
            cif_block, chain_info_dict, atom_array, converted_res, ignored_res
        )
        self.extra_info["struct_conn_bonds"] = struct_conn_bonds

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
                self._residue_file_exists(residue_name)
                and len(self._data_by_residue(residue_name)["intra_residue_bonds"]) > 0
            ):
                atom_a_local_indices, atom_b_local_indices, bond_types = self.get_intra_residue_bonds(residue_name, self.add_hydrogens)
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

    @lru_cache(maxsize=None)
    def get_intra_residue_bonds(self, residue_name: dict, add_hydrogens: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve intra-residue bonds for a given residue.

        Args:
            residue_data (dict): Dictionary containing keys for the intra-residue bonds and constituent atoms, derived from OpenBabel.
            add_hydrogens (bool): Whether or not hydrogens are being added to the structure. Relevant for bond removal.

        Returns:
            tuple: Three arrays representing the atom indices and bond types within the residue frame.
        """
        residue_data = self._data_by_residue(residue_name)

        # If we aren't adding hydrogens, we need to remove any bonds to hydrogens, and any hydrogen atoms from the atom list
        if not add_hydrogens:
            residue_data["intra_residue_bonds"] = [
                # NOTE: We are assuming that all, and only, hydrogen atoms are named with an 'H' prefix
                bond for bond in residue_data["intra_residue_bonds"] if not bond["atom_a_id"].startswith("H") and not bond["atom_b_id"].startswith("H")
            ]
            residue_data["atoms"] = {
                atom_id: atom_data
                for atom_id, atom_data in residue_data["atoms"].items()
                if not atom_data["element"] == 1
            }
        
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
        # NOTE: We need deduplicate_iterator() since biotite considers a decrease in sequence_id as a new chain (see `5xnl`)
        for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
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
