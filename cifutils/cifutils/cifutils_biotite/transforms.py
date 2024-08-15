"""
Transforms operating on the parsed CIF block. 
"""

from __future__ import annotations
import logging
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
from os import PathLike
from collections import Counter
from datetime import datetime
from pathlib import Path

from cifutils.cifutils_biotite.cifutils_biotite_utils import (
    apply_assembly_transformation,
    category_to_df,
    category_to_dict,
    deduplicate_iterator,
    fix_bonded_atom_charges,
    get_bond_type_from_order_and_is_aromatic,
    parse_transformations,
    read_cif_file,
    get_std_alt_atom_id_conversion,
    standardize_heavy_atom_ids,
    resolve_arginine_naming_ambiguity,
    mse_to_met,
    get_1_from_3_letter_code,
)
from cifutils.cifutils_biotite.common import exists

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx import CIFBlock, CIFCategory
from biotite.structure import AtomArray, Atom, AtomArrayStack
from biotite.file import InvalidFileError
from cifutils.cifutils_biotite.constants import CRYSTALLIZATION_AIDS, AF3_EXCLUDED_LIGANDS

logger = logging.getLogger(__name__)

def get_chain_info(cif_block: CIFBlock, atom_array: AtomArray) -> dict:
    """
    Extracts chain information from the CIF block.

    Args:
    - cif_block (CIFBlock): Parsed CIF block.
    - atom_array (AtomArray): Atom array containing the chain information.

    Returns:
    - dict: Dictionary containing the sequence details of each chain.
    """
    chain_info_dict = {}

    # If "entity" and "entity_poly" are not present, we cannot extract chain information, and return an empty dictionary
    if "entity" not in cif_block or "entity_poly" not in cif_block:
        return chain_info_dict

    # Step 1: Build a mapping of chain id to entity id from the `atom_site`
    chain_ids = atom_array.get_annotation("chain_id")
    entity_ids = atom_array.get_annotation("label_entity_id").astype(str)
    unique_chain_entity_map = {chain_id: entity_id for chain_id, entity_id in zip(chain_ids, entity_ids)}

    # Step 2: Load additional chain information
    entity_df = category_to_df(cif_block, "entity")
    entity_df["id"] = entity_df["id"].astype(str)
    entity_df.rename(columns={"type": "entity_type", "pdbx_ec": "ec_numbers"}, inplace=True)
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
        if chain_info.get("ec_numbers", "?") != "?":
            ec_numbers = [ec.strip() for ec in chain_info.get("ec_numbers", "").split(",")]
        else:
            ec_numbers = []

        chain_info_dict[chain_id] = {
            "entity_id": entity_id,
            "type": polymer_info.get("polymer_type", chain_info.get("entity_type", "")),
            "unprocessed_entity_canonical_sequence": polymer_info.get("canonical_sequence", "").replace("\n", ""),
            "unprocessed_entity_non_canonical_sequence": polymer_info.get("non_canonical_sequence", "").replace(
                "\n", ""
            ),
            "is_polymer": chain_info.get("entity_type") == "polymer",
            "ec_numbers": ec_numbers,
        }

    return chain_info_dict

def get_metadata(cif_block: CIFBlock, fallback_id: str = None) -> dict:
    """
    Extract metadata from the CIF block.

    Arguments:
        - cif_block (CIFBlock): The CIF block to extract metadata from.
        - fallback_id (str): A fallback ID to use if the `entry.id` field is not present in the CIF block.
    """
    metadata = {}

    # Assert that if the "entry.id" field is NOT present, a fallback ID is provided
    assert ("entry" in cif_block.keys() and "id" in cif_block["entry"].keys()) or fallback_id is not None, "No ID found in CIF block or provided as fallback."

    # Set the ID field, using the fallback if necessary
    metadata["id"] = cif_block["entry"]["id"].as_item().lower() if "entry" in cif_block.keys() and "id" in cif_block["entry"].keys() else fallback_id.lower()

    # +---------------- Look for standard RCSB metadata categories, default to None if not found ----------------+
    exptl = cif_block["exptl"] if "exptl" in cif_block.keys() else None

    status = cif_block["pdbx_database_status"] if "pdbx_database_status" in cif_block.keys() else None

    refine = cif_block["refine"] if "refine" in cif_block.keys() else None

    em_reconstruction = cif_block["em_3d_reconstruction"] if "em_3d_reconstruction" in cif_block.keys() else None

    # Method
    metadata["method"] = ",".join(exptl["method"].as_array()).replace(" ", "_") if exptl and "method" in exptl.keys() else None

    # Initial deposition date and release date to the PDB
    metadata["deposition_date"] = (
        status["recvd_initial_deposition_date"].as_item()
        if status and "recvd_initial_deposition_date" in status
        else None
    )

    # The relevant release date is the smallest `pdbx_audit_revision_history.revision_date` entry
    if "pdbx_audit_revision_history" in cif_block.keys() and "revision_date" in cif_block["pdbx_audit_revision_history"]:
        revision_dates = cif_block["pdbx_audit_revision_history"]["revision_date"].as_array()
    else:
        revision_dates = None

    if revision_dates is not None:
        # Convert string dates to datetime objects
        date_objects = [datetime.strptime(date, "%Y-%m-%d") for date in revision_dates]
        # Find the smallest date, convert back to string
        smallest_date = min(date_objects)
        metadata["release_date"] = smallest_date.strftime("%Y-%m-%d")
    else:
        metadata["release_date"] = None

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

    # Serialize the catch-all metadata cateogry, if it exists (we can later load with CIFCategory.deserialize() at will)
    metadata["extra_metadata"] = cif_block["metadata"].serialize() if "extra_metadata" in cif_block.keys() else None

    return metadata

def update_nonpoly_seq_ids(atom_array: AtomArray, chain_info_dict: dict) -> AtomArray:
    """
    Updates the sequence IDs of non-polymeric chains in the atom array.
    Additionally, adds an annotation to the atom array to indicate whether a chain is a polymer.

    Args:
    - atom_array (AtomArray): The atom array containing the chain information.
    - chain_info_dict (dict): Dictionary containing the sequence details of each chain.

    Returns:
    - AtomArray: The updated atom array with the sequence IDs updated for non-polymeric chains.
    """
    # For non-polymeric chains, we use the author sequence ids
    author_seq_ids = atom_array.get_annotation("auth_seq_id")
    chain_ids = atom_array.get_annotation("chain_id")

    # Create mask based on the is_polymer column
    non_polymer_mask = ~np.array([chain_info_dict[chain_id]["is_polymer"] for chain_id in chain_ids])

    # Update the atom_array_label with the (1-indexed) author sequence ids
    atom_array.res_id[non_polymer_mask] = author_seq_ids[non_polymer_mask]

    return atom_array

def load_monomer_sequence_information_from_category(
    cif_block: CIFBlock, chain_info_dict: dict, atom_array: AtomArray, known_residues: set
) -> dict:
    """
    Load monomer sequence information into a chain_info_dict, using:
        (a) The CIFCategory as the ground-truth for polymere.
        (b) The AtomArray as the ground-truth for non-polymers.

    For polymers, uses the 'entity_poly_seq' category in the CIF block to get the sequence.
    For non-polymers, uses the atom array to get the sequence.

    Args:
        cif_block (CIFBlock): The CIF block containing the monomer sequence information.
        chain_info_dict (dict): The dictionary where the monomer sequence information will be stored.
        atom_array (AtomArray): The atom array used to get the sequence for non-polymers.
        known_residues (set): The set of known residues to filter out unknown residues.

    Returns:
        The updated chain_info_dict with monomer sequence information.
    """

    # Assert that entity_poly_seq category is present
    assert "entity_poly_seq" in cif_block.keys(), "entity_poly_seq category not found in CIF block."


    # Handle polymers by using `entity_poly_seq`
    polymer_seq_df = category_to_df(cif_block, "entity_poly_seq")
    polymer_seq_df = polymer_seq_df.loc[:, ["entity_id", "num", "mon_id"]].rename(
        columns={"num": "residue_id", "mon_id": "residue_name"}
    )

    # Keep only the last occurrence of each residue
    duplicates = polymer_seq_df.duplicated(subset=["entity_id", "residue_id"], keep="last")
    entities_with_sequence_heterogeneity = polymer_seq_df[duplicates]["entity_id"].unique()
    if duplicates.any():
        logger.info("Sequence heterogeneity detected, keeping only the last occurrence of each residue.")
        polymer_seq_df = polymer_seq_df[~duplicates]

    # Filter out residues that are not in the precompiled CCD data (e.g., remove unknown residues)
    polymer_seq_df = polymer_seq_df[polymer_seq_df["residue_name"].apply(lambda x: x in known_residues)]

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
            residue_names = polymer_entity_id_to_residue_names_and_ids[entity_id]["residue_names"]
            chain_type = chain_info_dict[chain_id]["type"]
            if residue_names:
                chain_info_dict[chain_id]["residue_name_list"] = residue_names
                chain_info_dict[chain_id]["residue_id_list"] = polymer_entity_id_to_residue_names_and_ids[
                    entity_id
                ]["residue_ids"]

                # Create the processed single-letter sequence representations
                processed_entity_non_canonical_sequence = [
                    get_1_from_3_letter_code(residue, chain_type, use_closest_canonical=False)
                    for residue in residue_names
                ]
                processed_entity_canonical_sequence = [
                    get_1_from_3_letter_code(residue, chain_type, use_closest_canonical=True)
                    for residue in residue_names
                ]
                chain_info_dict[chain_id]["processed_entity_non_canonical_sequence"] = "".join(
                    processed_entity_non_canonical_sequence
                )
                chain_info_dict[chain_id]["processed_entity_canonical_sequence"] = "".join(
                    processed_entity_canonical_sequence
                )
        else:
            # For non-polymers, we must re-compute every time, since entities are not guaranteed to have the same monomer sequence (e.g., for H2O chains)
            chain_atom_array = atom_array[atom_array.chain_id == chain_id]
            residue_id_list, residue_name_list = struc.get_residues(chain_atom_array)
            # We don't need to filter out unmatched residues for non-polymers here, since we handled that by filtering the AtomArray earlier
            chain_info_dict[chain_id]["residue_name_list"] = residue_name_list
            chain_info_dict[chain_id]["residue_id_list"] = residue_id_list
            unique_residues.update(residue_name_list)

        chain_info_dict[chain_id]["has_sequence_heterogeneity"] = (
            str(entity_id) in entities_with_sequence_heterogeneity
        )

    # Remove entries from chain_info_dict that have no residues
    chain_info_dict = {
        chain_id: chain_info
        for chain_id, chain_info in chain_info_dict.items()
        if "residue_name_list" in chain_info
    }

    return chain_info_dict 