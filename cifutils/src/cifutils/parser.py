"""Entrypoint for parsing atomic-level structure files (e.g., PDB, CIF) into Biotite-compatible data structures."""

from __future__ import annotations

import io
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
import pandas as pd
from biotite.file import InvalidFileError
from biotite.structure import AtomArrayStack
from toolz import keyfilter

import cifutils.transforms.atom_array as ta
from cifutils import __version__, template
from cifutils.common import exists, md5_hash_string
from cifutils.constants import CCD_MIRROR_PATH, CRYSTALLIZATION_AIDS, WATER_LIKE_CCDS
from cifutils.transforms.categories import (
    category_to_dict,
    extract_crystallization_details,
    get_ligand_of_interest_info,
    get_metadata_from_category,
    initialize_chain_info_from_category,
    load_monomer_sequence_information_from_category,
)
from cifutils.utils.assembly import build_assemblies_from_asym_unit
from cifutils.utils.ccd import check_ccd_codes_are_available
from cifutils.utils.chain import create_chain_id_generator
from cifutils.utils.io_utils import get_structure, infer_pdb_file_type, read_any, to_cif_buffer
from cifutils.utils.non_rcsb import (
    get_identity_assembly_gen_category,
    get_identity_op_expr_category,
    initialize_chain_info_from_atom_array,
)

logger = logging.getLogger("cifutils")

__all__ = ["parse"]


def parse(
    filename: os.PathLike | io.StringIO | io.BytesIO,
    *,
    file_type: Literal["cif", "pdb"] | None = None,
    ccd_mirror_path: os.PathLike | None = CCD_MIRROR_PATH,
    cache_dir: os.PathLike | None = None,
    save_to_cache: bool = False,
    load_from_cache: bool = False,
    add_missing_atoms: bool = True,
    add_id_and_entity_annotations: bool = True,
    add_bond_types_from_struct_conn: list[str] = ["covale"],
    remove_ccds: list[str] | None = None,
    remove_waters: bool = True,
    fix_ligands_at_symmetry_centers: bool = True,
    fix_arginines: bool = True,
    fix_formal_charges: bool = True,
    fix_bond_types: bool = True,
    convert_mse_to_met: bool = False,
    remove_hydrogens: bool | None = None,
    hydrogen_policy: Literal["keep", "remove", "infer"] = "keep",
    model: int | None = None,
    build_assembly: Literal["first", "all"] | list[str] | tuple[str] | None = "all",
    extra_fields: list[str] | Literal["all"] | None = None,
    keep_cif_block: bool = False,
) -> dict[str, Any]:
    """Entrypoint for general parsing of atomic-level structure files.

    Can either:
        - Directly load structure from file, using the specified keyword arguments; or,
        - Load the structure from a cached directory, re-building bioassemblies on-the-fly if necessary.

    We categorize arguments into two groups:
        - Wrapper arguments: Arguments that are used within the wrapping `parse` method (e.g., caching)
        - CIF parsing arguments: Arguments that control structure parsing and are ultimately are passed
            to the `parse_from_cif` method (regardless of file type, we convert to a CIF file before
            parsing)

    Args:
        filename (PathLike | io.StringIO | io.BytesIO): Path or buffer to the structural file. May be any format
            of atomic-evel structure (e.g. .cif, .bcif, .cif.gz, .pdb), Although .cif files are *strongly* recommended.

        *** Wrapper arguments ***
        file_type (Literal["cif", "pdb"] | None, optional): The file type of the structure file.
            If not provided, the file type will be inferred automatically.
        load_from_cache (bool, optional): Whether to load pre-compiled results from cache. Defaults to False.
        cache_dir (PathLike, optional): Directory path to save pre-compiled results. Defaults to None.
        save_to_cache (bool, optional): Whether to save the results to cache when building the structure. Defaults to False.

        *** CIF parsing arguments ***
        ccd_mirror_path (str, optional): Path to the local mirror of the Chemical Component Dictionary (recommended).
            If not provided, Biotite's built-in CCD will be used.
        add_missing_atoms (bool, optional): Whether to add missing atoms to the
            structure (from entirely or partially unresolved residues). Defaults to True.
        add_id_and_entity_annotations (bool, optional): Whether to add identifier and entity
            annotations to the structure. Defaults to True.
        add_bond_types_from_struct_conn (list, optional): A list of bond types to add to the structure
            from the `struct_conn` category. Defaults to `["covale"]`. This means that we will only
            add covalent bonds to the structure (excluding metal coordination and disulfide bonds).
        remove_ccds (list, optional): A list of CCD codes (e.g. `ALA`, `HEM`, ...) to remove from
            the structure. Defaults to crystallization aids. NOTE: Exclusion of polymer
            residues and common multi-chain ligands must be done with care to avoid sequence gaps.
        remove_waters (bool, optional): Whether to remove water molecules from the
            structure. Defaults to True.
        fix_ligands_at_symmetry_centers (bool, optional): Whether to patch non-polymer residues
            at symmetry centers that clash with themselves when transformed. Defaults to True.
        fix_arginines (bool, optional): Whether to fix arginine naming ambiguity, see the
            AF-3 supplement for details. Defaults to True.
        fix_formal_charges (bool, optional): Whether to fix formal charges on atoms involved in inter-residue bonds.
            Defaults to True.
        fix_bond_types (bool, optional): Whether to correct for nucleophilic additions on atoms involved in inter-residue bonds.
            Defaults to True.
        convert_mse_to_met (bool, optional): Whether to convert selenomethionine (MSE)
            residues to methionine (MET) residues. Defaults to False.
        remove_hydrogens (bool, optional): Whether to remove hydrogens from the structure
            (e.g., when adding missing atoms). Only has an effect if `add_missing_atoms` is True.
            WARNING: This parameter is deprecated and will be removed in a future release.
            Use `hydrogen_policy = 'remove'` instead.
        hydrogen_policy (Literal, optional): Whether to keep, remove or infer hydrogens using
            biotite-hydride (will remove existing hydrogens and infer fresh).
            Defaults to "keep". Options: "keep", "remove", "infer".
        model (int, optional): The model number to parse from the CIF file for NMR entries.
            Defaults to all models (None).
        build_assembly (string, list, or tuple, optional): Specifies which assembly to build, if any. Options are None
            (e.g., asymmetric unit), "first", "all", or a list or tuple of assembly IDs. Defaults to "all".
        extra_fields (list, optional): A list of extra fields to include in the AtomArrayStack. Defaults to None. "all" includes all fields.
            only support cif files.
        keep_cif_block (bool, optional): Whether to keep the CIF block in the result. Defaults to False.

    Returns:
        dict: A dictionary containing the following keys:
            chain_info: A dictionary mapping chain ID to sequence, type (as an IntEnum), RCSB entity,
                EC number, and other information.
            ligand_info: A dictionary containing ligand of interest information.
            asym_unit: An AtomArrayStack instance representing the asymmetric unit.
            assemblies: A dictionary mapping assembly IDs to AtomArrayStack instances.
            metadata: A dictionary containing metadata about the structure
                (e.g., resolution, deposition date, etc.).
            extra_info: A dictionary with information for cross-compatibility and caching.
                Should typically not be used directly.

    """
    # CCD mirror
    if exists(ccd_mirror_path) and not os.path.exists(ccd_mirror_path):
        logger.warning(
            f"Local mirror of the Chemical Component Dictionary does not exist: {ccd_mirror_path}. Falling back to Biotite's built-in CCD."
        )
        ccd_mirror_path = None

    # Set default value for remove_ccds if None
    if remove_ccds is None:
        remove_ccds = CRYSTALLIZATION_AIDS

    # Argument validation
    check_ccd_codes_are_available(remove_ccds, ccd_mirror_path=ccd_mirror_path, mode="warn")

    if load_from_cache and not cache_dir:
        raise ValueError("Must provide a cache directory to load from cache")

    if save_to_cache and not cache_dir:
        raise ValueError("Must provide a cache directory to save to cache")

    if fix_formal_charges and not add_missing_atoms:
        logger.warning(
            "We can't fix formal charges without building from templates, as we need to know the true number of "
            "hydrogens bonded to a given atom, not the inferred number. This may lead to occasional inaccuracies "
            "after adding inter-residue bonds. To avoid this and fix formal charges, set `add_missing_atoms = True`."
        )

    ## hydrogen policy checks
    # ... deprecation handling for `remove_hydrogens`
    if remove_hydrogens is not None:
        warnings.warn(
            "'remove_hydrogens' is deprecated. Use `hydrogen_policy = 'remove'` or 'keep'` instead.",
            category=DeprecationWarning,
            stacklevel=1,
        )
        hydrogen_policy = "remove" if remove_hydrogens else "keep"

    file_type = file_type or infer_pdb_file_type(filename)
    is_buffer = isinstance(filename, io.StringIO | io.BytesIO)

    # Only load from / save to cache if we are not using a buffer
    if cache_dir and not is_buffer:
        # Build the cache file path, if necessary
        cache_dir = Path(cache_dir)

        # Prepare readable arguments dict for metadata
        parse_arguments = {
            "ccd_mirror_path": ccd_mirror_path,
            "add_missing_atoms": add_missing_atoms,
            "add_id_and_entity_annotations": add_id_and_entity_annotations,
            "add_bond_types_from_struct_conn": add_bond_types_from_struct_conn,
            "remove_ccds": remove_ccds,
            "remove_waters": remove_waters,
            "fix_ligands_at_symmetry_centers": fix_ligands_at_symmetry_centers,
            "fix_arginines": fix_arginines,
            "fix_formal_charges": fix_formal_charges,
            "fix_bond_types": fix_bond_types,
            "convert_mse_to_met": convert_mse_to_met,
            "hydrogen_policy": hydrogen_policy,
        }
        # Compose args_string from parse_arguments values (in order)
        args_string = ",".join(str(parse_arguments[k]) for k in parse_arguments)
        args_hash = md5_hash_string(args_string, length=8)

        # ... generate assembly info
        assembly_info = ",".join(build_assembly) if isinstance(build_assembly, list | tuple) else build_assembly

        # ... construct the full cache file path
        cache_file_path = cache_dir / args_hash / f"{Path(filename).stem}_assembly_{assembly_info}.pkl.gz"

        # If we are loading from cache, try to load the result from the cache
        if load_from_cache:
            try:
                # Try to load the result from the cache
                if cache_file_path.exists():
                    # Load the result from the cache
                    result = pd.read_pickle(cache_file_path)

                    # Build assemblies
                    asym_unit = result["asym_unit"]
                    extra_info = result["extra_info"]
                    if "assembly_gen_category" in extra_info:
                        assemblies = build_assemblies_from_asym_unit(
                            assembly_gen_category=extra_info["assembly_gen_category"],
                            struct_oper_category=extra_info["struct_oper_category"],
                            asym_unit_atom_array_stack=asym_unit,
                            build_assembly=build_assembly,
                            fix_symmetry_centers=fix_ligands_at_symmetry_centers,
                        )
                    else:
                        assemblies = asym_unit

                    # Return updated result
                    result["assemblies"] = assemblies
                    return result
            except Exception as e:
                # Log an error, and continue to parse from CIF
                logger.error(f"Error loading from cache: {e}")

    if file_type == "pdb":
        result = _parse_from_pdb(
            filename=filename,
            ccd_mirror_path=ccd_mirror_path,
            add_missing_atoms=add_missing_atoms,
            add_id_and_entity_annotations=add_id_and_entity_annotations,
            add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
            remove_ccds=remove_ccds,
            remove_waters=remove_waters,
            fix_ligands_at_symmetry_centers=fix_ligands_at_symmetry_centers,
            fix_arginines=fix_arginines,
            fix_formal_charges=fix_formal_charges,
            fix_bond_types=fix_bond_types,
            convert_mse_to_met=convert_mse_to_met,
            remove_hydrogens=remove_hydrogens,
            hydrogen_policy=hydrogen_policy,
            model=model,
            build_assembly=build_assembly,
        )
    elif file_type in ("cif", "bcif"):
        result = _parse_from_cif(
            filename=filename,
            ccd_mirror_path=ccd_mirror_path,
            add_missing_atoms=add_missing_atoms,
            add_id_and_entity_annotations=add_id_and_entity_annotations,
            add_bond_types_from_struct_conn=add_bond_types_from_struct_conn,
            remove_ccds=remove_ccds,
            remove_waters=remove_waters,
            fix_ligands_at_symmetry_centers=fix_ligands_at_symmetry_centers,
            fix_arginines=fix_arginines,
            fix_formal_charges=fix_formal_charges,
            fix_bond_types=fix_bond_types,
            convert_mse_to_met=convert_mse_to_met,
            remove_hydrogens=remove_hydrogens,
            hydrogen_policy=hydrogen_policy,
            model=model,
            build_assembly=build_assembly,
            extra_fields=extra_fields,
            keep_cif_block=keep_cif_block,
        )
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    if not is_buffer and save_to_cache and cache_dir and (not cache_file_path.exists()):
        # We want our cache to include:
        #   (1) All keys in `result` excep the assemblies and
        #   (2) The information needed to rebuild the assembly(s), which is stored in `result["extra_info"]`
        #   (3) The parse_arguments and cifutils version

        # Add parse_arguments and version to metadata before saving
        result.setdefault("metadata", {}).update({"parse_arguments": parse_arguments, "cifutils_version": __version__})

        # Ensure all parent directories exist
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the result to the cache, excluding the assemblies
        result_to_cache = {k: v for k, v in result.items() if k != "assemblies"}
        pd.to_pickle(result_to_cache, cache_file_path)

    return result


def _parse_from_cif(filename: os.PathLike | io.StringIO | io.BytesIO, **kwargs) -> dict[str, Any]:
    """Parse the CIF file.

    Return chain information, residue information, atom array, and metadata.
    See `parse` for details on the arguments and return values.

    NOTE: This method is not intended to be called directly; use `parse` instead.
    """
    # (Handle default lists to avoid mutable default arguments)
    remove_ccds = [] if kwargs["remove_ccds"] is None else kwargs["remove_ccds"].copy()

    # (Default running dictionary, which we will populate through a series of Transforms)
    data_dict = {"extra_info": {}}

    # ... read the CIF file into the dictionary (we will clean up the dictionary before returning)
    cif_file = read_any(filename)
    data_dict["cif_block"] = cif_file.block

    # ... load metadata into "metadata" key (either from RCSB standard fields, or from the custom `extra_metadata` field)
    if isinstance(filename, io.StringIO | io.BytesIO):
        fallback_filename = next(iter(cif_file.keys()))
    else:
        fallback_filename = Path(filename).stem
    data_dict["metadata"] = get_metadata_from_category(cif_file.block, fallback_id=fallback_filename)

    # ... load structure into the "asym_unit" key using the RCSB labels for sequence ids, and later update for non-polymers
    common_extra_fields = [
        "label_entity_id",
        "auth_seq_id",  # for non-polymer residue indexing
        "atom_id",
        "b_factor",
        "occupancy",
        "charge",
    ]

    if kwargs["extra_fields"] is not None:
        if kwargs["extra_fields"] != "all":
            common_extra_fields += kwargs["extra_fields"]
        else:
            common_extra_fields = "all"

    try:
        asym_unit_stack = get_structure(
            cif_file,
            extra_fields=common_extra_fields,
            model=kwargs["model"],
            add_bond_types_from_struct_conn=kwargs["add_bond_types_from_struct_conn"],
            fix_bond_types=kwargs["fix_bond_types"],
        )
    except InvalidFileError:
        logger.info("Invalid file error encountered; loading with only one model")
        # Try again, choosing only the first model
        asym_unit_stack = get_structure(
            cif_file,
            extra_fields=common_extra_fields,
            model=1,
            add_bond_types_from_struct_conn=kwargs["add_bond_types_from_struct_conn"],
            fix_bond_types=kwargs["fix_bond_types"],
        )

    # If occupancy is not an annotation, add it, defaulting to 1.0
    if "occupancy" not in asym_unit_stack.get_annotation_categories():
        asym_unit_stack.set_annotation("occupancy", np.ones(asym_unit_stack.array_length()))

    # ... ensure we have an atom array stack (e.g., if we selected a specific model, we may get an AtomArray)
    if not isinstance(asym_unit_stack, AtomArrayStack):
        asym_unit_stack = struc.stack([asym_unit_stack])

        # ... remove any explicitly excluded residues (e.g., crystallization solvents, waters)
    if remove_ccds or kwargs["remove_waters"]:
        # NOTE: If the excluded residues are part of a polymer chain, or part of a
        #  multi-chain ligand, this may create sequence gaps!
        # ... remove the residues we don't want to keep
        remove_ccds = set(map(str.upper, remove_ccds))
        if kwargs["remove_waters"]:
            remove_ccds.update(WATER_LIKE_CCDS)

        asym_unit_stack = ta.remove_ccd_components(asym_unit_stack, remove_ccds)

    # ... initialize chain information from the first model (uses atom_array to build chain list)
    if "entity" in cif_file.block and "entity_poly" in cif_file.block:
        # We can get the chain entity-level information directly from the CIF file
        data_dict["chain_info"] = initialize_chain_info_from_category(cif_file.block, asym_unit_stack[0])
    else:
        # ... replace negative res_ids with auth_seq_id (as they are sometimes from AF-3 predictions)
        asym_unit_stack = ta.replace_negative_res_ids_with_auth_seq_id(asym_unit_stack)
        # ... infer the chain information from the AtomArray residue names (useful for inference; should not be used for RCSB files)
        data_dict["chain_info"] = initialize_chain_info_from_atom_array(
            asym_unit_stack[0], infer_chain_type=True, infer_chain_sequences=True
        )

    if "entity_poly_seq" in cif_file.block:
        # ... replace non-polymeric chain sequence ids with author sequence ids (since the non-polymer sequence ID's are not informative)
        asym_unit_stack = ta.update_nonpoly_seq_ids(asym_unit_stack, data_dict["chain_info"])

        # Use the `entity_poly_seq` category as ground-truth sequence for polymers, and the AtomArray as ground-truth for non-polymers
        data_dict["chain_info"] = load_monomer_sequence_information_from_category(
            cif_block=cif_file.block,
            chain_info_dict=data_dict["chain_info"],
            atom_array=asym_unit_stack,
            ccd_mirror_path=kwargs["ccd_mirror_path"],
        )

    # Handle sequence heterogeneity by selecting the residue that appears last
    asym_unit_stack = ta.keep_last_residue(asym_unit_stack)

    # ... add the is_polymer annotation to the AtomArray
    asym_unit_stack = ta.add_polymer_annotation(asym_unit_stack, data_dict["chain_info"])

    # ... add the ChainType annotation to the AtomArray
    asym_unit_stack = ta.add_chain_type_annotation(asym_unit_stack, data_dict["chain_info"])

    # (Most examples, except NMR studies and small molecules, will not have any hydrogens)
    if kwargs["hydrogen_policy"] == "keep":
        pass
    elif kwargs["hydrogen_policy"] == "remove":
        asym_unit_stack = ta.remove_hydrogens(asym_unit_stack)
    elif kwargs["hydrogen_policy"] == "infer":
        # infer hydrogens using biotite-hydride, will replace existing hydrogens
        asym_unit_stack = ta.add_hydrogen_atom_positions(asym_unit_stack)
    else:
        raise ValueError(f"Invalid hydrogen policy: {kwargs['hydrogen_policy']}. Must be 'keep', 'remove', or 'infer'.")

    models = []
    for model_idx in range(asym_unit_stack.stack_depth()):
        atom_array = asym_unit_stack[model_idx]

        # ... add any atoms that should be there based on the sequence information
        #     but may not be resolved. These will have occupancy 0.0 and `nan` coords.
        if kwargs["add_missing_atoms"]:
            if kwargs["extra_fields"] is not None:
                logger.warning(
                    "Adding missing atoms will erase extra fields. If you just want to load a structure with the given extra fields, "
                    "you should probably use the much faster 'load_any' function from cifutils.utils.io_utils instead of 'parse'. "
                    "Parse is meant for cleaning up structures from the RCSB PDB."
                )
            atom_array = template.add_missing_atoms(
                atom_array,
                chain_info_dict=data_dict["chain_info"],
                struct_conn_dict=category_to_dict(cif_file.block, "struct_conn"),
                add_bond_types_from_struct_conn=kwargs["add_bond_types_from_struct_conn"],
                remove_hydrogens=kwargs["hydrogen_policy"] == "remove",
                use_ccd_charges=True,
                fix_formal_charges=kwargs["fix_formal_charges"],
                fix_bond_types=kwargs["fix_bond_types"],
            )

        # ... resolve arginine naming ambiguity
        if kwargs["fix_arginines"]:
            atom_array = ta.resolve_arginine_naming_ambiguity(atom_array, raise_on_error=False)

        # ... convert MSE to MET
        if kwargs["convert_mse_to_met"]:
            atom_array = ta.mse_to_met(atom_array)

        # ... add identifiers and entity annotations
        if kwargs["add_id_and_entity_annotations"]:
            atom_array = ta.add_id_and_entity_annotations(atom_array)

        models.append(atom_array)

    # ... create an AtomArrayStack from the list of AtomArrays
    asym_unit_stack = struc.stack(models)

    # ... add the atomic number annotation (vs. element, which is a string)
    asym_unit_stack = ta.add_atomic_number_annotation(asym_unit_stack)

    # ... optionally, build assemblies and add assembly-specifc annotation (instance IDs)
    if exists(kwargs["build_assembly"]):
        assert kwargs["build_assembly"] in ["first", "all"] or isinstance(
            kwargs["build_assembly"], list | tuple
        ), "Invalid `build_assembly` option. Must be 'first', 'all', or a list/tuple of assembly IDs as strings."

        if "pdbx_struct_assembly" in data_dict["cif_block"]:
            # ... build the assemblies from the CIF file, adding the `iid` annotations as we do so
            assembly_gen_category = data_dict["cif_block"]["pdbx_struct_assembly_gen"]
            struct_oper_category = data_dict["cif_block"]["pdbx_struct_oper_list"]
        else:
            # If there are no assemblies, set the `assembly_gen_category` and `struct_oper_category` to identity operations
            assembly_gen_category = get_identity_assembly_gen_category(list(data_dict["chain_info"].keys()))
            struct_oper_category = get_identity_op_expr_category()

        data_dict["assemblies"] = build_assemblies_from_asym_unit(
            assembly_gen_category=assembly_gen_category,
            struct_oper_category=struct_oper_category,
            asym_unit_atom_array_stack=asym_unit_stack,
            build_assembly=kwargs["build_assembly"],
            fix_symmetry_centers=kwargs["fix_ligands_at_symmetry_centers"],
        )

        # Store the assembly generation and struct oper categories in extra_info for caching and future reference
        data_dict["extra_info"]["assembly_gen_category"] = assembly_gen_category
        data_dict["extra_info"]["struct_oper_category"] = struct_oper_category
    else:
        data_dict["assemblies"] = {}

    # Handle instances where ph information is included in crystallization conditions
    if "exptl_crystal_grow" in cif_file.block:
        crystal_key = "exptl_crystal_grow"
        crystal_dict = category_to_dict(cif_file.block, crystal_key)
        data_dict["metadata"]["crystallization_details"] = extract_crystallization_details(crystal_dict)
    else:
        # No crystal growth section available in the CIF
        data_dict["metadata"]["crystallization_details"] = {"pH": None}

    # ... get ligand of interest information
    data_dict["ligand_info"] = get_ligand_of_interest_info(data_dict["cif_block"])

    if "msa_paths_by_chain_id" in cif_file.block:
        # ... add the MSA information to the chain info dictionary
        logger.info("MSA paths detected in CIF file. Adding to chain information...")
        msa_paths_by_chain_id = category_to_dict(cif_file.block, "msa_paths_by_chain_id")
        for chain_id, msa_path in msa_paths_by_chain_id.items():
            data_dict["chain_info"][chain_id]["msa_path"] = Path(msa_path.item())

    # ... remove annotations that are no longer needed to save memory
    _remove_annotations = {
        "leaving_atom_flag",
        "is_leaving_atom",
        "is_n_terminal_atom",
        "is_c_terminal_atom",
        "index",
    }
    for annotation in _remove_annotations:
        _remove_annotation_if_exists(asym_unit_stack, annotation)
        if "assemblies" in data_dict:
            for assembly in data_dict["assemblies"].values():
                _remove_annotation_if_exists(assembly, annotation)
    data_dict["asym_unit"] = asym_unit_stack

    # ... subset to only the keys we want to return, verbosely for clarity
    _keep_keys = {"chain_info", "ligand_info", "asym_unit", "assemblies", "metadata", "extra_info"}
    if kwargs.get("keep_cif_block", False):
        _keep_keys.add("cif_block")
    return keyfilter(lambda k: k in _keep_keys, data_dict)


def _parse_from_pdb(filename: os.PathLike, **parse_from_cif_kwargs) -> dict[str, Any]:
    """Parse a PDB file and return chain information, residue information, atom array, metadata, and legacy data.

    WARNING: We require that a single chain contains either polymer or non-polymer residues, but not both. Thus, if
    the PDB file contains a chain with both polymer and non-polymer residues, the non-polymer
    residues will be named with "$" appended to the chain ID (to not conflict with existing chains).

    WARNING: We assume that all residues are resolved (e.g., as is the case for computationally predicted structures). If not, use CIF files.

    NOTE: This method is not intended to be called directly; use `parse` instead.
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
                atom_array_stack.chain_id[(atom_array_stack.chain_id == chain_id) & hetero_atom_mask] = hetero_chain_id

                # Add the newly created chain ID to the list to avoid conflicts in future iterations
                chain_ids.append(hetero_chain_id)

            # ...ensure we don't have blended `hetero` annotations
            updated_chain_hetero_annotations = atom_array_stack.hetero[atom_array_stack.chain_id == chain_id]
            assert np.all(updated_chain_hetero_annotations) or np.all(~updated_chain_hetero_annotations)

    cif_buffer = to_cif_buffer(atom_array_stack, id=Path(filename).stem)

    # ...parse the CIF block into a dictionary
    parse_from_cif_kwargs["file_type"] = "pdb"
    parse_from_cif_kwargs["extra_fields"] = None
    return _parse_from_cif(filename=cif_buffer, **parse_from_cif_kwargs)


# Helper functions
def _remove_annotation_if_exists(atom_array: struc.AtomArray | struc.AtomArrayStack, annotation: str) -> None:
    """Safely remove an annotation from an AtomArray or AtomArrayStack if it exists."""
    if annotation in atom_array.get_annotation_categories():
        atom_array.del_annotation(annotation)
