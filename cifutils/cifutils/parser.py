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
from typing import Literal
from cifutils.constants import CRYSTALLIZATION_AIDS
import numpy as np
import pandas as pd
import os
from pathlib import Path
import io
from cifutils.utils.io_utils import read_any, get_structure, to_cif_buffer
from cifutils.utils.chain_utils import create_chain_id_generator
from cifutils.common import exists
from cifutils.transforms.categories import (
    get_chain_info_from_category,
    get_metadata_from_category,
    load_monomer_sequence_information_from_category,
    get_ligand_of_interest_info,
)
import cifutils.transforms.atom_array as ta
from cifutils.utils.non_rcsb_utils import (
    load_monomer_sequence_information_from_atom_array,
    infer_chain_info_from_atom_array,
    infer_processed_entity_sequences_from_atom_array,
)
from cifutils.utils.assembly_utils import process_assemblies
from cifutils.utils.residue_utils import add_missing_atoms_as_unresolved
from cifutils.utils.non_rcsb_utils import get_identity_assembly_gen_category, get_identity_op_expr_category
import biotite.structure as struc
from biotite.structure import AtomArrayStack
from biotite.file import InvalidFileError
from cifutils.constants import CCD_MIRROR_PATH, WATER_LIKE_CCDS
from cifutils.utils.ccd import check_ccd_codes_are_available
from toolz import keyfilter

logger = logging.getLogger("cifutils")

__all__ = ["CIFParser"]


class CIFParser:
    def __init__(
        self,
        ccd_mirror_path: os.PathLike = CCD_MIRROR_PATH,
    ):
        """
        Initialize a CIFParser object.

        Args:
            ccd_mirror_path (str, optional): Path to the local mirror of the Chemical Component Dictionary (recommended).
                If not provided, Biotite's built-in CCD will be used.
        """
        if exists(ccd_mirror_path) and not os.path.exists(ccd_mirror_path):
            logger.warning(
                f"Local mirror of the Chemical Component Dictionary does not exist: {ccd_mirror_path}. Falling back to Biotite's built-in CCD."
            )
            ccd_mirror_path = None
        self.ccd_mirror_path = ccd_mirror_path

    def _validate_arguments(self, **kwargs):
        """Validate the arguments passed to the CIFParser object."""

        # Transform related arguments
        add_missing_atoms = kwargs.get("add_missing_atoms")
        add_bonds = kwargs.get("add_bonds")
        add_entity_annotations = kwargs.get("add_entity_annotations")
        if (not add_missing_atoms) and add_bonds:
            raise ValueError("`add_bonds` requires `add_missing_atoms` to be True")

        if (not add_bonds) and add_entity_annotations:
            raise ValueError("`add_entity_annotations` requires `add_bonds` to be True")

        residues_to_remove = kwargs.get("residues_to_remove")
        check_ccd_codes_are_available(residues_to_remove, ccd_mirror_path=self.ccd_mirror_path, mode="warn")

        # Caching related arguments
        save_to_cache = kwargs.get("save_to_cache")
        cache_dir = kwargs.get("cache_dir")
        load_from_cache = kwargs.get("load_from_cache")
        if load_from_cache and not cache_dir:
            raise ValueError("Must provide a cache directory to load from cache")

        if save_to_cache and not cache_dir:
            raise ValueError("Must provide a cache directory to save to cache")

    def parse(
        self,
        filename: os.PathLike,
        *,
        load_from_cache: bool = False,
        save_to_cache: bool = False,
        cache_dir: os.PathLike = None,
        **kwargs,
    ):
        """
        Entrypoint for CIF parsing, which can either:
            - Directly parse from CIF, using the specified keyword arguments; or,
            - Load the CIF from a cached directory, re-building bioassemblies on-the-fly

        In addition to the arguments in `parse_cif_from_rcsb`, this function can also include the following arguments:

        Args:
            - filename (PathLike | io.StringIO | io.BytesIO): Path to the CIF file. May be any format of CIF file
                (e.g. .cif, .cif.gz, .pdb), Although .cif files are *strongly* recommended.
            - load_from_cache (bool, optional): Whether to load pre-compiled results from cache. Defaults to False.
            - save_to_cache (bool, optional): Whether to save pre-compiled results to cache. Defaults to False.
            - cache_dir (PathLike, optional): Directory path to save pre-compiled results. Defaults to None.
            - `parse_from_cif` arguments: Any arguments supported by `parse_from_cif`
        """
        self._validate_arguments(**kwargs)

        if cache_dir:
            cache_dir = Path(cache_dir)
            # Make the cache directory if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Build the cache file path
            assembly_info = kwargs.get("build_assembly", "all")
            if isinstance(assembly_info, (list, tuple)):
                assembly_info = ",".join(assembly_info)

            cache_file_path = cache_dir / f"{Path(filename).stem}_assembly_{assembly_info}.pkl.gz"

        if load_from_cache and cache_dir:
            try:
                # Try to load the result from the cache
                if cache_file_path.exists():
                    # Load the result from the cache
                    result = pd.read_pickle(cache_file_path)

                    # Build assemblies
                    asym_unit = result["asym_unit"]
                    extra_info = result["extra_info"]
                    if "assembly_gen_category" in extra_info:
                        assemblies = process_assemblies(
                            assembly_gen_category=extra_info["assembly_gen_category"],
                            struct_oper_category=extra_info["struct_oper_category"],
                            asym_unit_atom_array_stack=asym_unit,
                            build_assembly=kwargs.get("build_assembly", "all"),
                            patch_symmetry_centers=kwargs.get("patch_symmetry_centers", True),
                        )
                    else:
                        assemblies = asym_unit

                    # Return updated result
                    result["assemblies"] = assemblies
                    return result
            except Exception as e:
                # Log an error, and continue to parse from CIF
                logger.error(f"Error loading from cache: {e}")

        filename = Path(filename)
        # Parse from PDB
        if str(filename).endswith((".pdb", ".pdb.gz")):
            result = self.parse_from_pdb(
                filename=filename,
                save_to_cache=save_to_cache,
                cache_dir=cache_dir,
                load_from_cache=load_from_cache,
                **kwargs,
            )
        # Parse from CIF
        elif str(filename).endswith((".cif", ".cif.gz", ".bcif", ".bcif.gz")):
            result = self.parse_from_cif(
                filename=filename,
                save_to_cache=save_to_cache,
                cache_dir=cache_dir,
                load_from_cache=load_from_cache,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported file type: {filename.suffix}. Please use a .cif, .cif.gz, or .pdb file.")

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
        filename: os.PathLike | io.StringIO | io.BytesIO,
        *,
        save_to_cache: bool = False,
        assume_residues_all_resolved: bool = False,
        add_missing_atoms: bool = True,
        add_bonds: bool = True,
        add_entity_annotations: bool = True,
        residues_to_remove: list[str] = CRYSTALLIZATION_AIDS,
        patch_symmetry_centers: bool = True,
        build_assembly: Literal["first", "all"] | list[str] | tuple[str] | None = "all",
        fix_arginines: bool = True,
        convert_mse_to_met: bool = False,
        keep_hydrogens: bool = True,
        remove_waters: bool = True,
        model: int | None = None,
        **kwargs,
    ) -> dict:
        """
        Parse the CIF file (must contain information from the PDB) and return chain
        information, residue information, atom array, metadata, and legacy data.

        Args:
            save_to_cache (bool): Whether to save the results to cache (see `parse`).
            filename (str): Path to the CIF file. May be any format of CIF file
                (e.g., gz, bcif, etc.). This can be a path to a file or a buffer.
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
            add_entity_annotations (bool, optional): Whether to add entity annotations to the structure.
                Defaults to True.
            remove_waters (bool, optional): Whether to remove water molecules from the
                structure. Defaults to True.
            residues_to_remove (list, optional): A list of residue names to remove from
                the structure. Defaults to crystallization aids. NOTE: Exclusion of polymer
                residues and common multi-chain ligands must be done with care to avoid sequence gaps.
            patch_symmetry_centers (bool, optional): Whether to patch non-polymer residues
                at symmetry centers that clash with themselves when transformed. Defaults to True.
            build_assembly (string, list, or tuple, optional): Specifies which assembly to build, if any. Options are None
                (e.g., asymmetric unit), "first", "all", or a list or tuple of assembly IDs. Defaults to "all".
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
                'asym_unit': An AtomArrayStack instance representing the asymmetric unit.
                'assemblies': A dictionary mapping assembly IDs to AtomArrayStack instances.
                'metadata': A dictionary containing metadata about the structure
                    (e.g., resolution, deposition date, etc.).
                'extra_info': A dictionary with information for cross-compatibility and caching.
                    Should typically not be used directly.
        """
        # ...initializations

        _converted_res = {}
        _ignored_res = []

        # ...default running dictionary, which we will populate through a series of Transforms
        data_dict = {}
        data_dict["extra_info"] = {}

        # ...read the CIF file into the dictionary (we will clean up the dictionary before returning)
        cif_file = read_any(filename)
        data_dict["cif_block"] = cif_file.block

        # ...load metadata into "metadata" key (either from RCSB standard fields, or from the custom `extra_metadata` field)
        if isinstance(filename, (io.StringIO, io.BytesIO)):
            fallback_filename = "unknown_id"
        else:
            fallback_filename = Path(filename).stem
        data_dict["metadata"] = get_metadata_from_category(data_dict["cif_block"], fallback_id=fallback_filename)

        # ...load structure into the "asym_unit" key using the RCSB labels for sequence ids, and later update for non-polymers
        common_extra_fields = [
            "label_entity_id",
            "auth_seq_id",  # for non-polymer residue indexing
            "atom_id",
            "b_factor",
            "occupancy",
            "charge",
        ]

        try:
            asym_unit_stack = get_structure(
                cif_file,
                extra_fields=common_extra_fields,
                assume_residues_all_resolved=assume_residues_all_resolved,
                model=model,
            )
        except InvalidFileError:
            logger.info("Invalid file error encountered; loading with only one model")
            # Try again, choosing only the first model
            asym_unit_stack = get_structure(
                cif_file,
                extra_fields=common_extra_fields,
                assume_residues_all_resolved=assume_residues_all_resolved,
                model=1,
            )

        # ...ensure we have an atom array stack (e.g., if we selected a specific model, we may get an AtomArray)
        if not isinstance(asym_unit_stack, AtomArrayStack):
            asym_unit_stack = struc.stack([asym_unit_stack])
        data_dict["asym_unit"] = asym_unit_stack

        # ...load chain information from the first model (uses atom_array to build chain list)
        if "entity" and "entity_poly" in data_dict["cif_block"].keys():
            # We can get the chain information directly from the CIF file
            data_dict["chain_info"] = get_chain_info_from_category(data_dict["cif_block"], data_dict["asym_unit"][0])
        else:
            # We must infer the chain information from the AtomArray residue names (not bulletproof)
            data_dict["chain_info"] = infer_chain_info_from_atom_array(data_dict["asym_unit"][0])

        if not keep_hydrogens:
            # ...most examples, except NMR studies and small molecules, will not have any hydrogens
            asym_unit_stack = ta.remove_hydrogens(asym_unit_stack)

        # ...remove any explicitly excluded residues (e.g., crystallization solvents, waters)
        if residues_to_remove or remove_waters:
            # NOTE: If the excluded residues are part of a polymer chain, or part of a
            #  multi-chain ligand, this may create sequence gaps!
            # ... remove the residues we don't want to keep
            residues_to_remove = set(map(str.upper, residues_to_remove))
            if remove_waters:
                residues_to_remove.update(WATER_LIKE_CCDS)

            asym_unit_stack = ta.remove_ccd_components(asym_unit_stack, residues_to_remove)
            _ignored_res.extend(residues_to_remove)

        # ...replace non-polymeric chain sequence ids with author sequence ids (since the non-polymer sequence ID's are not informative)
        asym_unit_stack = ta.update_nonpoly_seq_ids(asym_unit_stack, data_dict["chain_info"])

        # ...loop through models
        models = []
        for model_idx in range(asym_unit_stack.stack_depth()):
            atom_array = data_dict["asym_unit"][model_idx]

            # ...load monomer sequence information into chain_info_dict
            # NOTE: We MAY NOT delete polymer atoms from the AtomArray after this step, as the sequences won't be updated
            if not assume_residues_all_resolved:
                # Use the `entity_poly_seq` category as ground-truth for polymers, and the AtomArray as ground-truth for non-polymers
                data_dict["chain_info"] = load_monomer_sequence_information_from_category(
                    cif_block=data_dict["cif_block"],
                    chain_info_dict=data_dict["chain_info"],
                    atom_array=atom_array,
                    ccd_mirror_path=self.ccd_mirror_path,
                )
            else:
                # Use the AtomArray as ground-truth for all residues (e.g., distillation sets)
                data_dict["chain_info"] = load_monomer_sequence_information_from_atom_array(
                    chain_info_dict=data_dict["chain_info"],
                    atom_array=atom_array,
                )

            # ...handle sequence heterogeneity by selecting the residue that appears last
            atom_array = ta.keep_last_residue(atom_array)

            # ...optionally, create a larger atom array that includes missing atoms (e.g., hydrogens), then populate with atoms details loaded from structure
            # NOTE: If adding bonds, we must add missing atoms to ensure we can later remove leaving atoms
            if add_missing_atoms:
                # Add missing atoms to the atom array, using the reference residue as a template
                atom_array = add_missing_atoms_as_unresolved(
                    atom_array,
                    chain_info_dict=data_dict["chain_info"],
                    keep_hydrogens=keep_hydrogens,
                    ccd_mirror_path=self.ccd_mirror_path,
                )

            # ...add the is_polymer annotation to the AtomArray
            atom_array = ta.add_polymer_annotation(atom_array, data_dict["chain_info"])

            # ...add the ChainType annotation to the AtomArray
            atom_array = ta.add_chain_type_annotation(atom_array, data_dict["chain_info"])

            if assume_residues_all_resolved:
                data_dict["chain_info"] = infer_processed_entity_sequences_from_atom_array(
                    data_dict["chain_info"], atom_array
                )

            # ...resolve arginine naming ambiguity
            if fix_arginines:
                atom_array = ta.resolve_arginine_naming_ambiguity(atom_array)

            # ...convert MSE to MET
            if convert_mse_to_met:
                atom_array = ta.mse_to_met(atom_array)
                _converted_res["MSE"] = "MET"

            # ...generate and add bonds to the atom array bond list
            if add_bonds:
                atom_array = ta.add_bonds_to_bondlist_and_remove_leaving_atoms(
                    cif_block=data_dict["cif_block"],
                    atom_array=atom_array,
                    chain_info=data_dict["chain_info"],
                    keep_hydrogens=keep_hydrogens,  # needed for leaving group resolution
                    converted_res=_converted_res,  # needed for leaving group resolution
                    ignored_res=_ignored_res,
                    ccd_mirror_path=self.ccd_mirror_path,
                )

            if add_entity_annotations:
                atom_array = ta.add_id_and_entity_annotations(atom_array, data_dict["chain_info"])

            models.append(atom_array)

        # ...create an AtomArrayStack from the list of AtomArrays
        data_dict["asym_unit"] = struc.stack(models)

        # ...optionally, build assemblies and add assembly-specifc annotation (instance IDs)
        if exists(build_assembly):
            # ...assert that `build_assembly` is a valid option
            assert build_assembly in ["first", "all"] or isinstance(
                build_assembly, (list, tuple)
            ), "Invalid `build_assembly` option. Must be 'first', 'all', or a list/tuple of assembly IDs as strings."

            if "pdbx_struct_assembly" in data_dict["cif_block"].keys():
                # ...build the assemblies from the CIF file, adding the `iid` annotations as we do so
                assembly_gen_category = data_dict["cif_block"]["pdbx_struct_assembly_gen"]
                struct_oper_category = data_dict["cif_block"]["pdbx_struct_oper_list"]
            else:
                # ...if there are no assemblies, set the `assembly_gen_category` and `struct_oper_category` to identity operations
                assembly_gen_category = get_identity_assembly_gen_category(list(data_dict["chain_info"].keys()))
                struct_oper_category = get_identity_op_expr_category()

            data_dict["assemblies"] = process_assemblies(
                assembly_gen_category=assembly_gen_category,
                struct_oper_category=struct_oper_category,
                asym_unit_atom_array_stack=data_dict["asym_unit"],
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
        data_dict["ligand_info"] = get_ligand_of_interest_info(data_dict["cif_block"])

        # ...remove annotations that are no longer needed to save memory
        _remove_annotations = {
            "ins_code",
            "hetero",
            "leaving_atom_flag",
            "leaving_group",
            "index",
        }
        for annotation in _remove_annotations:
            _remove_annotation_if_exists(data_dict["asym_unit"], annotation)
            if "assemblies" in data_dict:
                for assembly in data_dict["assemblies"].values():
                    _remove_annotation_if_exists(assembly, annotation)

        # ...and subset to only the keys we want to return, verbosely for clarity
        _keep_keys = {"chain_info", "ligand_info", "asym_unit", "assemblies", "metadata", "extra_info"}
        return keyfilter(lambda k: k in _keep_keys, data_dict)

    def parse_from_pdb(self, filename: os.PathLike, **parse_from_cif_kwargs):
        """
        Parse a PDB file and return chain information, residue information, atom array, metadata, and legacy data.

        WARNING: We require that a single chain contains either polymer or non-polymer residues, but not both. Thus, if
        the PDB file contains a chain with both polymer and non-polymer residues, the non-polymer
        residues will be named with "$" appended to the chain ID (to not conflict with existing chains).

        WARNING: We assume that all residues are resolved (e.g., as is the case for computationally predicted structures). If not, use CIF files.

        Args:
            filename (str): Path to the PDB file.
            **parse_from_cif_kwargs: Additional keyword arguments to pass to `parse_from_cif`.

        Returns:
            dict: A dictionary containing the following keys:
                'chain_info': A dictionary mapping chain ID to sequence, type, RCSB entity,
                    EC number, and other information.
                'ligand_info': A dictionary containing ligand of interest information.
                'asym_unit': An AtomArrayStack instance representing the asymmetric unit.
                'assemblies': A dictionary mapping assembly IDs to AtomArrayStack instances.
                'metadata': A dictionary containing metadata about the structure
                    (e.g., resolution, deposition date, etc.).
        """
        # ...read the PDB file into a CIF block
        pdb_file = read_any(filename)
        atom_array_stack = pdb_file.get_structure(
            model=None,
            altloc="first",
            extra_fields=["b_factor", "occupancy", "charge", "atom_id"],
            include_bonds=True,
        )

        # ...if we have polymer and non-polymers on the same chain (as given by the HETATM field), we need to separate them for processing
        assert "hetero" in atom_array_stack.get_annotation_categories()

        hetero_atom_mask = atom_array_stack.get_annotation("hetero")
        if np.any(atom_array_stack.get_annotation("hetero")):
            # ...loop through chains and ensure the chain contains either polymer or non-polymer residues, but not both (as required by CIF files)
            original_chain_ids = np.unique(atom_array_stack.chain_id)
            chain_id_generator = create_chain_id_generator(unavailable_chain_ids=original_chain_ids)
            chain_ids = list(original_chain_ids)  # Creates a copy
            for chain_id in original_chain_ids:
                # ...check if we have blended `hetero` annotations in the chain
                chain_hetero_annotations = atom_array_stack.hetero[atom_array_stack.chain_id == chain_id]
                if np.any(chain_hetero_annotations) and np.any(~chain_hetero_annotations):
                    hetero_chain_id = next(chain_id_generator)
                    logger.warning(
                        f"Chain {chain_id} contains both polymer and non-polymer residues; separating them for processing, naming the non-polymer residues as {hetero_chain_id}."
                    )
                    atom_array_stack.chain_id[(atom_array_stack.chain_id == chain_id) & hetero_atom_mask] = (
                        hetero_chain_id
                    )

                    # Add the newly created chain ID to the list to avoid conflicts in future iterations
                    chain_ids.append(hetero_chain_id)

                # ...ensure we don't have blended `hetero` annotations
                updated_chain_hetero_annotations = atom_array_stack.hetero[atom_array_stack.chain_id == chain_id]
                assert np.all(updated_chain_hetero_annotations) or np.all(~updated_chain_hetero_annotations)

        cif_buffer = to_cif_buffer(atom_array_stack, id=Path(filename).stem)

        if (
            "assume_residues_all_resolved" in parse_from_cif_kwargs
            and not parse_from_cif_kwargs["assume_residues_all_resolved"]
        ):
            logger.warning(
                "PDB file detected; assuming all residues are resolved. We highly recommend using CIF files instead."
            )
        parse_from_cif_kwargs["assume_residues_all_resolved"] = True

        # ...parse the CIF block into a dictionary
        parse_from_cif_kwargs["file_type"] = "pdb"
        return self.parse_from_cif(filename=cif_buffer, **parse_from_cif_kwargs)


# Helper functions
def _remove_annotation_if_exists(atom_array: struc.AtomArray | struc.AtomArrayStack, annotation: str) -> None:
    """Safely remove an annotation from an AtomArray or AtomArrayStack if it exists."""
    if annotation in atom_array.get_annotation_categories():
        atom_array.del_annotation(annotation)
