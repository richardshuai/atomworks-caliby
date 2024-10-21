"""
Transforms operating on Biotite's CIFBlock and CIFCategory objects.
These transforms are used to extract information from the CIFBlock and return a dictionary containing processed information.
"""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from datetime import datetime

from cifutils.utils.sequence_utils import (
    get_1_from_3_letter_code,
)
from cifutils.common import deduplicate_iterator, exists
from cifutils.enums import ChainType

import biotite.structure as struc
from biotite.structure.io.pdbx import CIFBlock
from biotite.structure import AtomArray
import toolz

logger = logging.getLogger("cifutils")


def category_to_df(cif_block: CIFBlock, category: str) -> pd.DataFrame | None:
    """Convert a CIF block to a pandas DataFrame."""
    return pd.DataFrame(category_to_dict(cif_block, category)) if category in cif_block.keys() else None


def category_to_dict(cif_block: CIFBlock, category: str) -> dict[str, np.ndarray]:
    """Convert a CIF block to a dictionary."""
    if exists(cif_block.get(category)):
        return toolz.valmap(lambda x: x.as_array(), dict(cif_block[category]))
    else:
        return {}


def get_chain_info_from_category(cif_block: CIFBlock, atom_array: AtomArray) -> dict:
    """
    Extracts chain information from the CIF block.

    Args:
        cif_block (CIFBlock): Parsed CIF block.
        atom_array (AtomArray): Atom array containing the chain information.

    Returns:
        dict: Dictionary containing the sequence details of each chain.
    """
    chain_info_dict = {}

    # Step 1: Build a mapping of chain id to entity id from the `atom_site`
    chain_ids = atom_array.get_annotation("chain_id")
    rcsb_entities = atom_array.get_annotation("label_entity_id").astype(str)
    unique_chain_entity_map = {chain_id: rcsb_entity for chain_id, rcsb_entity in zip(chain_ids, rcsb_entities)}

    # Step 2: Load additional chain information
    rcsb_entity_df = category_to_df(cif_block, "entity")
    rcsb_entity_df["id"] = rcsb_entity_df["id"].astype(str)
    rcsb_entity_df.rename(columns={"type": "entity_type", "pdbx_ec": "ec_numbers"}, inplace=True)
    rcsb_entity_dict = rcsb_entity_df.set_index("id").to_dict(orient="index")

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
    for chain_id, rscb_entity in unique_chain_entity_map.items():
        chain_info = rcsb_entity_dict.get(rscb_entity, {})
        polymer_info = polymer_dict.get(rscb_entity, {})
        if chain_info.get("ec_numbers", "?") != "?":
            ec_numbers = [ec.strip() for ec in chain_info.get("ec_numbers", "").split(",")]
        else:
            ec_numbers = []

        chain_info_dict[chain_id] = {
            "rcsb_entity": rscb_entity,
            "type": polymer_info.get(
                "polymer_type", chain_info.get("entity_type", "")
            ).lower(),  # Convert to lowercase to match ChainType enum
            "unprocessed_entity_canonical_sequence": polymer_info.get("canonical_sequence", "").replace("\n", ""),
            "unprocessed_entity_non_canonical_sequence": polymer_info.get("non_canonical_sequence", "").replace(
                "\n", ""
            ),
            "is_polymer": chain_info.get("entity_type") == "polymer",
            "ec_numbers": ec_numbers,
        }

    return chain_info_dict


def get_metadata_from_category(cif_block: CIFBlock, fallback_id: str = None) -> dict:
    """
    Extract metadata from the CIF block.
    If the `entry.id` field is not present in the CIF block, the `fallback_id` is used instead (e.g., the filename of the CIF).

    From RCSB CIF files, this function extracts:
        - ID (e.g., PDB ID)
        - Method (e.g., X-ray, NMR, etc.)
        - Deposition date (initial)
        - Release date (smallest revision date)
        - Resolution (e.g., 5.0, 3.0, etc.)

    For custom CIF files (e.g., distillation), this function extracts:
        - Extra metadata (all other categories)

    Arguments:
        cif_block (CIFBlock): The CIF block to extract metadata from.
        fallback_id (str): A fallback ID to use if the `entry.id` field is not present in the CIF block.
    """
    metadata = {}

    # Assert that if the "entry.id" field is NOT present, a fallback ID is provided
    assert (
        "entry" in cif_block.keys() and "id" in cif_block["entry"].keys()
    ) or fallback_id is not None, "No ID found in CIF block or provided as fallback."

    # Set the ID field, using the fallback if necessary
    metadata["id"] = (
        cif_block["entry"]["id"].as_item().lower()
        if "entry" in cif_block.keys() and "id" in cif_block["entry"].keys()
        else fallback_id.lower()
    )

    # +---------------- Look for standard RCSB metadata categories, default to None if not found ----------------+
    exptl = cif_block["exptl"] if "exptl" in cif_block.keys() else None

    status = cif_block["pdbx_database_status"] if "pdbx_database_status" in cif_block.keys() else None

    refine = cif_block["refine"] if "refine" in cif_block.keys() else None

    em_reconstruction = cif_block["em_3d_reconstruction"] if "em_3d_reconstruction" in cif_block.keys() else None

    # Method
    metadata["method"] = (
        ",".join(exptl["method"].as_array()).replace(" ", "_") if exptl and "method" in exptl.keys() else None
    )

    # Initial deposition date and release date to the PDB
    metadata["deposition_date"] = (
        status["recvd_initial_deposition_date"].as_item()
        if status and "recvd_initial_deposition_date" in status
        else None
    )

    # The relevant release date is the smallest `pdbx_audit_revision_history.revision_date` entry
    if (
        "pdbx_audit_revision_history" in cif_block.keys()
        and "revision_date" in cif_block["pdbx_audit_revision_history"]
    ):
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
    metadata["extra_metadata"] = (
        cif_block["extra_metadata"].serialize() if "extra_metadata" in cif_block.keys() else None
    )

    return metadata


def load_monomer_sequence_information_from_category(
    cif_block: CIFBlock, chain_info_dict: dict, atom_array: AtomArray, known_residues: list[str] | set[str]
) -> dict:
    """
    Load monomer sequence information into a chain_info_dict, using:
        (a) The CIFCategory 'entity_poly_seq' as the sequence ground-truth for polymere.
        (b) The AtomArray as the ground-truth for non-polymers.

    We must rely on the CIFCategory 'entity_poly_seq' for polymers, as the AtomArray may not contain the full sequence information (e.g., unresolved residues)
    For non-polymers, there's no standard equivalent to 'entity_poly_seq', so we must use the AtomArray to get the sequence information.

    When loading both polymer and non-polymer sequences, we also filter out unknown or otherwise ignored residues.

    Args:
        cif_block (CIFBlock): The CIF block containing the monomer sequence information.
        chain_info_dict (dict): The dictionary where the monomer sequence information will be stored.
        atom_array (AtomArray): The atom array used to get the sequence for non-polymers.
        known_residues (list): The set of known residues to filter out unknown residues.

    Returns:
        The updated chain_info_dict with monomer sequence information.
    """
    known_residues = set(known_residues)

    # Assert that entity_poly_seq category is present
    assert "entity_poly_seq" in cif_block.keys(), "entity_poly_seq category not found in CIF block."

    # Handle polymers by using `entity_poly_seq`
    polymer_seq_df = category_to_df(cif_block, "entity_poly_seq")
    polymer_seq_df = polymer_seq_df.loc[:, ["entity_id", "num", "mon_id"]].rename(
        columns={"entity_id": "rcsb_entity", "num": "residue_id", "mon_id": "residue_name"}
    )

    # Keep only the last occurrence of each residue
    duplicates = polymer_seq_df.duplicated(subset=["rcsb_entity", "residue_id"], keep="last")
    entities_with_sequence_heterogeneity = polymer_seq_df[duplicates]["rcsb_entity"].unique()
    if duplicates.any():
        logger.info("Sequence heterogeneity detected, keeping only the last occurrence of each residue.")
        polymer_seq_df = polymer_seq_df[~duplicates]

    # Map any polymer residues not in the precompiled CCD data to "UNK" (unknown polymer residue)
    polymer_seq_df["residue_name"] = polymer_seq_df["residue_name"].apply(lambda x: x if x in known_residues else "UNK")

    # Map rcsb_entity to lists of residue names and residue IDs
    polymer_seq_df["rcsb_entity"] = polymer_seq_df["rcsb_entity"].astype(float)
    polymer_entity_id_to_residue_names_and_ids = {
        rcsb_entity: {"residue_names": group["residue_name"].tolist(), "residue_ids": group["residue_id"].tolist()}
        for rcsb_entity, group in polymer_seq_df.groupby("rcsb_entity")
    }

    # Build up the chain_info_dict with the sequence information
    for chain_id in deduplicate_iterator(struc.get_chains(atom_array)):
        rcsb_entity = int(chain_info_dict[chain_id]["rcsb_entity"])
        if rcsb_entity in polymer_entity_id_to_residue_names_and_ids:
            # For polymers, we use the stored entity residue list
            residue_names = polymer_entity_id_to_residue_names_and_ids[rcsb_entity]["residue_names"]
            chain_type = ChainType.from_string(chain_info_dict[chain_id]["type"])
            if residue_names:
                chain_info_dict[chain_id]["residue_name_list"] = residue_names
                chain_info_dict[chain_id]["residue_id_list"] = polymer_entity_id_to_residue_names_and_ids[rcsb_entity][
                    "residue_ids"
                ]

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

            # Map any non-polymer residues not in the precompiled CCD data to "UNL" (unknown ligand)
            residue_name_list = [residue if residue in known_residues else "UNL" for residue in residue_name_list]

            chain_info_dict[chain_id]["residue_name_list"] = list(residue_name_list)
            chain_info_dict[chain_id]["residue_id_list"] = list(residue_id_list)

        chain_info_dict[chain_id]["has_sequence_heterogeneity"] = (
            str(rcsb_entity) in entities_with_sequence_heterogeneity
        )

    # Remove entries from chain_info_dict that have no residues
    chain_info_dict = {
        chain_id: chain_info for chain_id, chain_info in chain_info_dict.items() if "residue_name_list" in chain_info
    }

    return chain_info_dict


def get_ligand_of_interest_info(cif_block: CIFBlock) -> dict:
    """Extract ligand of interest information from a CIF block.

    Reference:
        - https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/small-molecule-ligands
    """

    # Extract binary flag for whether the ligand of interest is specified
    # NOTE: This is being used in addition to the below as it has slightly higher coverage across the PDB
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_pdbx_entry_details.has_ligand_of_interest.html
    has_loi = category_to_dict(cif_block, "pdbx_entry_details").get("has_ligand_of_interest", np.array(["N"]))[0] == "Y"

    # Extract which ligand is of interest if specified
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_pdbx_entity_instance_feature.feature_type.html
    entity_instance_feature = category_to_dict(cif_block, "pdbx_entity_instance_feature")
    comp_id_names = entity_instance_feature.get("comp_id", np.array([], dtype="<U3"))
    comp_id_mask = entity_instance_feature.get("feature_type", np.array([])) == "SUBJECT OF INVESTIGATION"

    return {
        "ligand_of_interest": list(comp_id_names[comp_id_mask]),
        "has_ligand_of_interest": has_loi | (len(comp_id_names) > 0),
    }
