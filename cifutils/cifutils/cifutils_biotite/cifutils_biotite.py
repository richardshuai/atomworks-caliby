"""
Full-featured CIF parsing library that can:
* Parse CIF files from the RCSB PDB, including metadata, chain information, ligand information, and atom arrays
* Parse distillation CIF files, which may not contain all fields from the RCSB PDB, but can be assumed to be complete
* Add missing atoms and residues to the atom array using precompiled residue-level data (a pre-requisite for structure prediction)
* Add bonds to the atom array using precompiled residue-level data, and standard inter-residue bonds (a pre-requisite for structure prediction)
* Resolve naming ambiguities, such as arginine naming ambiguity
* Resolve clashing residues at symmetry centers
* ...and more!

Written by Nate Corley and Simon Mathis in Summer of 2024.
"""

from __future__ import annotations
import logging
from functools import lru_cache
from typing import Literal
from cifutils.cifutils_biotite.constants import CRYSTALLIZATION_AIDS

import numpy as np
import pandas as pd
from os import PathLike
from pathlib import Path

from cifutils.cifutils_biotite.utils.io_utils import read_cif_file, load_structure
from cifutils.cifutils_biotite.common import exists
from cifutils.cifutils_biotite.transforms.categories import (
    get_chain_info_from_category,
    get_metadata_from_category,
    load_monomer_sequence_information_from_category,
    get_ligand_of_interest_info,
)
from cifutils.cifutils_biotite.transforms.atom_array import (
    mse_to_met,
    remove_atoms_by_residue_names,
    resolve_arginine_naming_ambiguity,
    keep_last_residue,
    update_nonpoly_seq_ids,
    add_bonds_to_bondlist_and_remove_leaving_atoms,
    add_polymer_annotation,
    add_pn_unit_id_annotation,
    add_molecule_id_annotation,
    add_chain_type_annotation,
    annotate_entities,
)
from cifutils.cifutils_biotite.utils.non_rcsb_utils import (
    load_monomer_sequence_information_from_atom_array,
    infer_chain_info_from_atom_array,
)
from cifutils.cifutils_biotite.utils.bond_utils import cached_bond_utils_factory
from cifutils.cifutils_biotite.utils.assembly_utils import process_assemblies
from cifutils.cifutils_biotite.utils.residue_utils import cached_residue_utils_factory, add_missing_atoms_as_unresolved

import biotite.structure as struc
from biotite.structure import AtomArrayStack
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
        add_missing_atoms = kwargs.get("add_missing_atoms")
        add_bonds = kwargs.get("add_bonds")
        load_from_cache = kwargs.get("load_from_cache")
        save_to_cache = kwargs.get("save_to_cache")
        cache_dir = kwargs.get("cache_dir")

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
    def _all_precompiled_residues(self, residues_to_remove: tuple) -> list[str]:
        """Return a list of all supported residues in the precompiled library."""
        # Get all residues in the precompiled library
        all_residues = [path.stem.upper() for path in Path(self.by_residue_ligand_dir).glob("*.pkl")]

        # Remove unsupported residues
        supported_residues = [residue for residue in all_residues if residue not in residues_to_remove]

        # Sanity check -- ensure that we removed at least one residue, if we provided a list of residues to remove
        if len(self._residues_to_remove) > 0:
            assert len(supported_residues) < len(
                all_residues
            ), "No residues were removed from the precompiled library with the provided residues_to_remove list."

        return supported_residues

    def parse(
        self, *, load_from_cache: bool = False, save_to_cache: bool = False, cache_dir: PathLike = None, **kwargs
    ):
        """
        Entrypoint for CIF parsing, which can either:
            - Directly parse from CIF, using the specified keyword arguments; or,
            - Load the CIF from a cached directory, re-building bioassemblies on-the-fly

        In addition to the arguments in `parse_cif_from_rcsb`, this function can also include the following arguments:
            load_from_cache (bool, optional): Whether to load pre-compiled results from cache. Defaults to False.
            save_to_cache (bool, optional): Whether to save pre-compiled results to cache. Defaults to False.
            cache_dir (PathLike, optional): Directory path to save pre-compiled results. Defaults to None.
        """
        self._validate_arguments(**kwargs)

        if cache_dir:
            cache_dir = Path(cache_dir)
            # Make the cache directory if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file_path = (
                cache_dir / f"{Path(kwargs['filename']).stem}_assembly_{kwargs.get('build_assembly', 'all')}.pkl"
            )

        if load_from_cache and cache_dir:
            # Load from cache
            if cache_file_path.exists():
                # Load the result from the cache
                result = pd.read_pickle(cache_file_path)

                # Build assemblies
                atom_array_stack = result["atom_array_stack"]
                if "assembly_gen_category" in result["extra_info"]:
                    assemblies = process_assemblies(
                        assembly_gen_category=result["extra_info"]["assembly_gen_category"],
                        struct_oper_category=result["extra_info"]["struct_oper_category"],
                        atom_array_stack=atom_array_stack,
                        build_assembly=kwargs.get("build_assembly", "all"),
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
            #   (2) The information needed to rebuild the assembly(s), which is stored in `result["extra_info"]`
            if not cache_file_path.exists():
                # Save the result to the cache, excluding the assemblies
                result_to_cache = {k: v for k, v in result.items() if k != "assemblies"}
                pd.to_pickle(result_to_cache, cache_file_path)

        return result

    def parse_from_cif(
        self,
        *,
        filename: PathLike,
        save_to_cache: bool,
        assume_residues_all_resolved: bool = False,
        add_missing_atoms: bool = True,
        add_bonds: bool = True,
        remove_waters: bool = True,
        residues_to_remove: list[str] = CRYSTALLIZATION_AIDS,
        patch_symmetry_centers: bool = True,
        build_assembly: Literal["first", "all"] | list[str] | None = "all",
        fix_arginines: bool = True,
        convert_mse_to_met: bool = False,
        keep_hydrogens: bool = True,
        model: int | None = None,
    ) -> dict:
        """
        Parse the CIF file (must contain information from the PDB) and return chain
        information, residue information, atom array, metadata, and legacy data.

        Args:
            save_to_cache (bool): Whether to save the results to cache (see `parse`).
            filename (str): Path to the CIF file. May be any format of CIF file
                (e.g., gz, bcif, etc.).
            assume_residues_all_resolved (bool): Whether we can assume when parsing
                that all residues are represented, and all atoms are present. Required
                for distillation examples that do not have all RCSB fields. Defaults to False.
            add_missing_atoms (bool, optional): Whether to add missing atoms to the
                structure (from entirely or partially unresolved residues), including
                relevant OpenBabel atom-level data. Defaults to True.
            add_bonds (bool, optional): Whether to add bonds to the structure. Leverages
                (a) precompiled intra-residue bond data, (b) standard inter-residue bonds,
                and (c) the `struct_conn` category to determine connectivity. Cannot be True
                if `add_missing_atoms` is False. Defaults to True.
            remove_waters (bool, optional): Whether to remove water molecules from the
                structure. Defaults to True.
            residues_to_remove (list, optional): A list of residue names to remove from
                the structure. Defaults to crystallization aids. NOTE: Exclusion of polymer
                residues and common multi-chain ligands must be done with care to avoid sequence gaps.
            patch_symmetry_centers (bool, optional): Whether to patch non-polymer residues
                at symmetry centers that clash with themselves when transformed. Defaults to True.
            build_assembly (str, optional): Which assembly to build, if any. Options are None
                (e.g., asymmetric unit), "first", "all", or a list of assembly IDs. Defaults to "all".
            fix_arginines (bool, optional): Whether to fix arginine naming ambiguity, see the
                AF-3 supplement for details. Defaults to True.
            convert_mse_to_met (bool, optional): Whether to convert selenomethionine (MSE)
                residues to methionine (MET) residues. Defaults to False.
            keep_hydrogens (bool, optional): Whether to add hydrogens to the structure
                (e.g., when adding missing atoms). Defaults to True.
            model (int, optional): The model number to parse from the CIF file for NMR entries.
                Defaults to all models (None).

        Returns:
            dict: A dictionary containing the following keys:
                'chain_info': A dictionary mapping chain ID to sequence, type, RCSB entity,
                    EC number, and other information.
                'ligand_info': A dictionary containing ligand of interest information.
                'atom_array_stack': An AtomArrayStack instance representing the asymmetric unit.
                'assemblies': A dictionary mapping assembly IDs to AtomArrayStack instances.
                'metadata': A dictionary containing metadata about the structure
                    (e.g., resolution, deposition date, etc.).
                'extra_info': A dictionary with information for cross-compatibility and caching.
                    Should typically not be used directly.
        """
        # ...initializations
        self._residues_to_remove = [residue.upper() for residue in residues_to_remove]
        _converted_res = {}
        _ignored_res = []

        # ...default running dictionary, which we will populate through a series of Transforms
        data_dict = {}
        data_dict["extra_info"] = {}

        # ...read the CIF file into the dictionary (we will clean up the dictionary before returning)
        cif_file = read_cif_file(filename)
        data_dict["cif_block"] = cif_file.block

        # ...load metadata into "metadata" key (either from RCSB standard fields, or from the custom `extra_metadata` field)
        fallback_filename = Path(filename).stem
        data_dict["metadata"] = get_metadata_from_category(data_dict["cif_block"], fallback_id=fallback_filename)

        # ...load structure into the "atom_array_stack" key using the RCSB labels for sequence ids, and later update for non-polymers
        common_extra_fields = [
            "label_entity_id",
            "auth_seq_id",  # for non-polymer residue indexing
            "atom_id",
        ]
        if not assume_residues_all_resolved:
            # If we're not assuming residues are all resolved, we need to load the b_factor and occupancy fields
            common_extra_fields += ["b_factor", "occupancy"]

        try:
            atom_array_stack = load_structure(
                data_dict["cif_block"],
                common_extra_fields,
                assume_residues_all_resolved,
                model,
            )
        except InvalidFileError:
            logger.warning(f"Invalid file error encountered for {filename}; loading with only one model")
            # Try again, choosing only the first model
            atom_array_stack = load_structure(
                data_dict["cif_block"],
                common_extra_fields,
                assume_residues_all_resolved,
                model=1,
            )

        # ...ensure we have an atom array stack (e.g., if we selected a specific model, we may get an AtomArray)
        if not isinstance(atom_array_stack, AtomArrayStack):
            atom_array_stack = struc.stack([atom_array_stack])
        data_dict["atom_array_stack"] = atom_array_stack

        # ...load chain information from the first model (uses atom_array to build chain list)
        if "entity" and "entity_poly" in data_dict["cif_block"].keys():
            # We can get the chain information directly from the CIF file
            data_dict["chain_info_dict"] = get_chain_info_from_category(
                data_dict["cif_block"], data_dict["atom_array_stack"][0]
            )
        else:
            # We must infer the chain information from the AtomArray residue names (not bulletproof)
            data_dict["chain_info_dict"] = infer_chain_info_from_atom_array(data_dict["atom_array_stack"][0])

        # ...loop through models
        models = []
        for model_idx in range(atom_array_stack.stack_depth()):
            atom_array = data_dict["atom_array_stack"][model_idx]

            # ...optionally, remove hydrogens (most examples will not have any hydrogens; only NMR studies and small molecules)
            if not keep_hydrogens:
                atom_array = atom_array[~np.isin(atom_array.element, ["H", "D"])]

            # ...optionally, remove waters
            if remove_waters:
                atom_array = atom_array[atom_array.res_name != "HOH"]
                _ignored_res.append("HOH")  # We keep track of removed residues for bond resolution downstream

            # ...replace non-polymeric chain sequence ids with author sequence ids (since the non-polymer sequence ID's are not informative)
            atom_array = update_nonpoly_seq_ids(atom_array, data_dict["chain_info_dict"])

            # ...remove any explicitly excluded residues (e.g., crystallization solvents)
            # NOTE: If the excluded residues are part of a polymer chain, or part of a multi-chain ligand, this may create sequence gaps!
            if self._residues_to_remove:
                atom_array = remove_atoms_by_residue_names(atom_array, self._residues_to_remove)
                _ignored_res.extend(self._residues_to_remove)

            # ...load monomer sequence information into chain_info_dict
            # NOTE: We MAY NOT delete polymer atoms from the AtomArray after this step, as the sequences won't be updated
            if not assume_residues_all_resolved:
                # Use the `entity_poly_seq` category as ground-truth for polymers, and the AtomArray as ground-truth for non-polymers
                data_dict["chain_info_dict"] = load_monomer_sequence_information_from_category(
                    cif_block=data_dict["cif_block"],
                    chain_info_dict=data_dict["chain_info_dict"],
                    atom_array=atom_array,
                    known_residues=self._all_precompiled_residues(tuple(self._residues_to_remove)),
                )
            else:
                # Use the AtomArray as ground-truth for all residues (e.g., distillation sets)
                data_dict["chain_info_dict"] = load_monomer_sequence_information_from_atom_array(
                    chain_info_dict=data_dict["chain_info_dict"],
                    atom_array=atom_array,
                )

            # ...handle sequence heterogeneity by selecting the residue that appears last
            atom_array = keep_last_residue(atom_array)

            # ...optionally, create a larger atom array that includes missing atoms (e.g., hydrogens), then populate with atoms details loaded from structure
            # NOTE: If adding bonds, we must add missing atoms to ensure we can later remove leaving atoms
            if add_missing_atoms:
                # Create cached function to build residues
                self._build_residue_atoms = cached_residue_utils_factory(
                    known_residues=self._all_precompiled_residues(tuple(self._residues_to_remove)),
                    data_by_residue=self._data_by_residue,
                )
                # Add missing atoms to the atom array, using the reference residue as a template
                atom_array = add_missing_atoms_as_unresolved(
                    atom_array, data_dict["chain_info_dict"], keep_hydrogens, self._build_residue_atoms
                )

            # ...add the is_polymer annotation to the AtomArray
            atom_array = add_polymer_annotation(atom_array, data_dict["chain_info_dict"])

            # ...add the ChainType annotation to the AtomArray
            atom_array = add_chain_type_annotation(atom_array, data_dict["chain_info_dict"])

            # ...optionally, resolve arginine naming ambiguity (AF3-style)
            if fix_arginines:
                atom_array = resolve_arginine_naming_ambiguity(atom_array)

            # ...optionally, convert MSE to MET (AF3-style)
            if convert_mse_to_met:
                atom_array = mse_to_met(atom_array)
                _converted_res["MSE"] = "MET"

            # ...optionally, generate and add bonds to the atom array bond list
            if add_bonds:
                # ...create cached function to get intra-residue bonds
                self._get_intra_residue_bonds = cached_bond_utils_factory(
                    data_by_residue=self._data_by_residue,
                )

                # ...update the AtomArray bondlist
                atom_array = add_bonds_to_bondlist_and_remove_leaving_atoms(
                    cif_block=data_dict["cif_block"],
                    atom_array=atom_array,
                    chain_info_dict=data_dict["chain_info_dict"],
                    keep_hydrogens=keep_hydrogens,  # needed for leaving group resolution
                    known_residues=self._all_precompiled_residues(tuple(self._residues_to_remove)),
                    get_intra_residue_bonds=self._get_intra_residue_bonds,
                    converted_res=_converted_res,  # needed for leaving group resolution
                    ignored_res=_ignored_res,  # needed for leaving group resolution
                )

                # ...annotate PN units
                atom_array = add_pn_unit_id_annotation(atom_array)

                # ...annotate molecules
                atom_array = add_molecule_id_annotation(atom_array)

                levels = ["chain", "pn_unit", "molecule"]
                lower_level_ids = ["res_id", "chain_id", "pn_unit_id"]
                lower_level_entities = ["res_name", "chain_entity", "pn_unit_entity"]

                for level, lower_level_id, lower_level_entity in zip(levels, lower_level_ids, lower_level_entities):
                    # ...annotate entities at appropriate level
                    atom_array, _ = annotate_entities(
                        atom_array=atom_array,
                        level=level,
                        lower_level_id=lower_level_id,
                        lower_level_entity=lower_level_entity,
                    )

            models.append(atom_array)

        # ...create an AtomArrayStack from the list of AtomArrays
        data_dict["atom_array_stack"] = struc.stack(models)

        # ...optionally, build assemblies and add assembly-specifc annotation (instance IDs)
        if exists(build_assembly):
            if "pdbx_struct_assembly" not in data_dict["cif_block"].keys():
                # ...if there are no assemblies, return the atom array stack as the only assembly
                data_dict["assemblies"] = {"1": atom_array_stack}
            else:
                # ...otherwise, build the assemblies from the CIF file, adding the `iid` annotations as we do so
                assembly_gen_category = data_dict["cif_block"]["pdbx_struct_assembly_gen"]
                struct_oper_category = data_dict["cif_block"]["pdbx_struct_oper_list"]
                data_dict["assemblies"] = process_assemblies(
                    assembly_gen_category=assembly_gen_category,
                    struct_oper_category=struct_oper_category,
                    atom_array_stack=data_dict["atom_array_stack"],
                    build_assembly=build_assembly,
                    patch_symmetry_centers=patch_symmetry_centers,
                )

                # If we're caching, we need to store the assembly information in extra_info
                if save_to_cache:
                    data_dict["extra_info"]["assembly_gen_category"] = assembly_gen_category
                    data_dict["extra_info"]["struct_oper_category"] = struct_oper_category
        else:
            data_dict["assemblies"] = {}

        # ...get ligand of interest information
        data_dict["loi_info"] = get_ligand_of_interest_info(data_dict["cif_block"])

        # ...remove annotations that are no longer needed to save memory
        unneeded_annotations = [
            "ins_code",
            "hetero",
            "leaving_atom_flag",
            "leaving_group",
            "index",
        ]
        for annotation in unneeded_annotations:
            if annotation in data_dict["atom_array_stack"].get_annotation_categories():
                data_dict["atom_array_stack"].del_annotation(annotation)
            if "assemblies" in data_dict:
                for assembly in data_dict["assemblies"].values():
                    if annotation in assembly.get_annotation_categories():
                        assembly.del_annotation(annotation)

        # ...and subset to only the keys we want to return, verbosely for clarity
        return {
            "chain_info": data_dict["chain_info_dict"],
            "ligand_info": data_dict["loi_info"],
            "atom_array_stack": data_dict["atom_array_stack"],
            "assemblies": data_dict["assemblies"],
            "metadata": data_dict["metadata"],
            "extra_info": data_dict["extra_info"],
        }
