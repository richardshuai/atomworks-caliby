"""
Pre-process mmCIF files into an "query PN units" dataframe.
From the "query PN units" dataframe, we will later generate an "interfaces" dataframe.
At training time, RoseTTAFold2 will select a query PN unit or interface from the appropriate DataFrame and crop the structure around it.

See the README for a term glosssary.
"""

from __future__ import annotations

import json
import logging
import re
from os import PathLike
from pathlib import Path
from typing import Any

import biotite.structure as struc
import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from cifutils.constants import CRYSTALLIZATION_AIDS
from cifutils.parser import CIFParser

import datahub.preprocessing.utils as dp  # to avoid circular imports
from datahub.common import exists
from datahub.preprocessing.constants import CELL_SIZE, ChainType, ClashSeverity
from datahub.utils.misc import hash_sequence

logger = logging.getLogger("preprocess")


class DataPreprocessor:
    PDB_REGEX = re.compile(r"^[0-9A-Za-z]{4}$")

    def __init__(
        self,
        *,
        # Cutoff distances
        close_distance: float = 30.0,
        contact_distance: float = 5,
        clash_distance: float = 1.0,
        # Paths
        # TODO: Initialize the base_cif_dir from Hydra
        base_cif_dir: PathLike = "/databases/rcsb/cif",
        # Misc
        ignore_residues: list[str] = [],
        # Efficiency
        polymer_pn_unit_limit: int = 2000,
        **kwargs,
    ):
        # Cutoff distances
        self.close_distance = close_distance
        self.contact_distance = contact_distance
        self.clash_distance = clash_distance

        # Paths
        self.base_cif_dir = base_cif_dir  # directory used to infer PDB path from PDB ID if no path is given

        # Misc
        self.ignore_residues = ignore_residues

        # Efficiency
        self.polymer_pn_unit_limit = polymer_pn_unit_limit

        # Arguments for parsing, with defaults
        self.add_missing_atoms = kwargs.get("add_missing_atoms", True)
        self.add_bonds = kwargs.get("add_bonds", True)
        self.remove_waters = kwargs.get("remove_waters", True)
        self.remove_crystallization_aids = kwargs.get("remove_crystallization_aids", True)
        self.patch_symmetry_centers = kwargs.get("patch_symmetry_centers", True)
        self.build_assembly = kwargs.get("build_assembly", "all")
        self.fix_arginines = kwargs.get("fix_arginines", True)
        self.convert_mse_to_met = kwargs.get("convert_mse_to_met", True)
        self.fix_arginines = kwargs.get("fix_arginines", True)

        # Initialize parser
        self.parser = CIFParser()

        logger.info(f"Initialized DataPreprocessor with the following parameters: {self.__dict__}")

    def _maybe_infer_path(self, path_or_pdb_id: PathLike | str) -> Path:
        """
        Given a path or PDB ID, return the path to the PDB entry.
        If the PDB ID is given, infer the path to the PDB entry.
        """
        if isinstance(path_or_pdb_id, str) and DataPreprocessor.PDB_REGEX.match(path_or_pdb_id):
            pdb_id = path_or_pdb_id.lower()
            path = Path(f"{self.base_cif_dir}/{pdb_id[1:3]}/{pdb_id}.cif.gz")
        else:
            path = Path(path_or_pdb_id)
        return path

    def _load_cif(self, path_or_pdb_id: PathLike | str) -> dict[str, Any]:
        """Load mmCIF file using CIFUtils Biotite parser."""
        path = self._maybe_infer_path(path_or_pdb_id)
        self.path = path  # for logging
        return self.parser.parse(
            filename=path,
            build_assembly="all",
            add_bonds=self.add_bonds,
            add_missing_atoms=self.add_missing_atoms,
            remove_waters=self.remove_waters,
            patch_symmetry_centers=self.patch_symmetry_centers,
            convert_mse_to_met=self.convert_mse_to_met,
            fix_arginines=self.fix_arginines,
            keep_hydrogens=False,
            residues_to_remove=CRYSTALLIZATION_AIDS if self.remove_crystallization_aids else [],
        )

    def _apply_filters(self, atom_array: AtomArray) -> AtomArray:
        """Apply filters to the AtomArray to remove non-biological bonds, ignore residues, and filter out atoms with zero occupancy."""
        # ----- Filter A: Filter out non-polymers with non-biological bonds to polymers ------
        # Check for non-biological bonds between the current non-polymer PN unit and any polymer (e.g., oxygen-oxygen, etc.)
        inter_pn_unit_bond_mask = DataPreprocessor.get_inter_pn_unit_bond_mask(atom_array)
        filtered_atom_array = atom_array
        if np.sum(inter_pn_unit_bond_mask) > 0:
            pn_units_with_non_biological_bonds = dp.get_pn_units_with_non_biological_bonds(
                atom_array, inter_pn_unit_bond_mask
            )
            if len(pn_units_with_non_biological_bonds) > 0:
                # Filter out non-polymer chains with non-biological bonds to polymer chains
                non_biological_mask = np.isin(atom_array.pn_unit_iid, pn_units_with_non_biological_bonds) & (
                    atom_array.chain_type == ChainType.NON_POLYMER
                )
                filtered_atom_array = atom_array[~non_biological_mask]
                logger.warning(f"{self.path}: Non-biological bonds detected between non-polymer and polymer PN units.")

        # ----- Filter B: Apply ignore residue list ------
        processed_ignore_residues = [c.strip() for c in self.ignore_residues]
        mask = ~np.isin(
            filtered_atom_array.res_name, processed_ignore_residues
        )  # Applying filter also remove impacted bonds
        filtered_atom_array = filtered_atom_array[mask]

        # ----- Filter C: Filter out atoms with zero occupancy ------
        filtered_atom_array = filtered_atom_array[filtered_atom_array.occupancy > 0.0]

        return filtered_atom_array

    @staticmethod
    def get_inter_pn_unit_bond_mask(atom_array: AtomArray) -> np.ndarray:
        """
        Returns a mask indicating which bonds in `atom_array.bonds` are between two distinct PN units.
        Because we are operating at the PN unit-level, such bonds cannot be bonds between non-polymers.

        Arguments:
        - atom_array (AtomArray): The full atom array. Must have PN unit-level annotations.

        Returns:
        - numpy.ndarray: A boolean mask indicating which bonds are between two PN units.
        """
        bond_pn_unit_a = atom_array.pn_unit_iid[atom_array.bonds.as_array()[:, 0]]
        bond_pn_unit_b = atom_array.pn_unit_iid[atom_array.bonds.as_array()[:, 1]]
        return bond_pn_unit_a != bond_pn_unit_b

    def get_rows(
        self,
        path_or_pdb_id: PathLike | str,
        ligand_scores: list[str] = [
            "RSCC",
            "RSR",
            "completeness",
            "intermolecular_clashes",
            "is_best_instance",
            "ranking_model_fit",
            "ranking_model_geometry",
        ],
    ) -> list[dict[str, Any]]:
        """
        Processes a PDB entry, applies filters, and generates a list of records to be loaded at train-time.
        We create a record for each PN unit (protein, nucleic acid, or non-polymer) in the PDB entry.
        Each record contains information about a query PN unit and its partner (contacting) PN units in the PDB entry.

        Args:
        - path_or_pdb_id (str): The path to the PDB entry (in cif format / cif.gz format) to process.
            OR the PDB ID of the entry to process. If the PDB ID is provided, the path to the entry will be inferred
            via the `self.base_cif_dir` directory.

        Returns:
        - list: A list of dictionaries. Each dictionary contains information about a query PN unit and its partner PN units.
        """
        result_dict = self._load_cif(path_or_pdb_id)
        id = result_dict["metadata"]["id"]
        self.id = id  # For logging

        # Query the RCSB for ligand validity scores
        if exists(ligand_scores) and len(ligand_scores) > 0:
            # Attempt fetching ligand validity scores
            ligand_validity_scores = dp.get_ligand_validity_scores_from_pdb_id(id)
            if exists(ligand_validity_scores) and len(ligand_validity_scores) > 0:
                # ... if successful, filter and format the data
                ligand_validity_scores = pd.DataFrame(ligand_validity_scores)
                ligand_validity_scores = ligand_validity_scores.set_index(["asym_id", "res_name"])[ligand_scores]

                # Ensure the index is sorted lexicographically
                ligand_validity_scores.sort_index(inplace=True)
            else:
                # ... if unsuccessful, log and set to None
                logger.debug(f"Failed to fetch ligand validity scores for ID {id}")
                ligand_validity_scores = None
        else:
            ligand_validity_scores = None

        # Process each assembly
        records = []
        for assembly_id in result_dict["assemblies"].keys():
            result = self._process_assembly(result_dict, assembly_id, ligand_validity_scores)
            if result is not None:
                records.extend(result)
        return records

    def _process_assembly(
        self, result_dict: dict, assembly_id: str, ligand_validity_scores: pd.DataFrame | None = None
    ) -> list[dict[str, Any]]:
        """
        Processes an atom array that represents a single assembly and generate a list of metadata records for each PN unit.

        Arguments:
        — result_dict (dict): The dictionary containing the output of CIFUtils Biotite parser.
        — assembly_id (str): The ID of the assembly to process.
        """
        id = result_dict["metadata"]["id"]
        full_atom_array = result_dict["assemblies"][assembly_id][0]  # Choose the first model

        chain_info_dict = result_dict["chain_info"]

        # ---------- Step 1: Upfront pre-processing ---------- #

        # Re-map PN unit IDs to integers, and keep a dictionary that maps back to the verbose PN unit IDs
        # We will use the integer IDs for memory efficiency downstream
        ids_to_remap = ["pn_unit_id", "pn_unit_iid"]
        id_map_dict = {}
        for id_to_remap in ids_to_remap:
            ids = np.unique(full_atom_array.get_annotation(id_to_remap))

            # ...create the new map
            mapped_ids = {old_id: new_id for new_id, old_id in enumerate(ids)}

            # ...apply the new map
            new_ids = np.array(
                [mapped_ids[old_id] for old_id in full_atom_array.get_annotation(id_to_remap)], dtype=np.int16
            )
            full_atom_array.del_annotation(id_to_remap)  # Remove the old annotation (so that we can change the type)
            full_atom_array.set_annotation(id_to_remap, new_ids)

            # ...set the reverse map so we can look up the verbose IDs later
            id_map_dict[id_to_remap] = {new_id: old_id for old_id, new_id in mapped_ids.items()}

        # ---------- Step 2: Apply filters to the AtomArray ---------- #
        filtered_atom_array = self._apply_filters(full_atom_array)

        # ---------- Step 3: Load entry-level criteria ---------- #
        loi_ligand_set = set(result_dict["ligand_info"]["ligand_of_interest"])
        num_polymer_pn_units = len(np.unique(filtered_atom_array.pn_unit_iid[filtered_atom_array.is_polymer]))

        if num_polymer_pn_units > self.polymer_pn_unit_limit:
            logger.warning(
                f"(PDB ID {self.id}): {num_polymer_pn_units} polymer PN units in entry; skipping for performance reasons."
            )
            return None

        # ---------- Step 4: Detect and resolve clashes ---------- #

        # Build cell list for rapid distance computations
        if len(filtered_atom_array) == 0:
            logger.warning(f"(PDB ID {self.id}): No atoms remaining after filtering.")
            return []
        cell_list = struc.CellList(filtered_atom_array, cell_size=CELL_SIZE)

        # Get the unique polymer/non-polymer unit IDs from the AtomArray (we will consider process each moving forward)
        pn_unit_iids_to_consider = np.unique(filtered_atom_array.pn_unit_iid)

        # Find inter-PN unit clashes
        clash_severity = ClashSeverity.NO_CLASH
        clashing_pn_units_set, clashing_pn_units_dict = dp.get_clashing_pn_units(
            pn_unit_iids_to_consider, filtered_atom_array, cell_list, clash_distance=self.clash_distance
        )
        if clashing_pn_units_set:
            logger.warning(
                f"(PDB ID {self.id}): Clash detected between PN units: {[id_map_dict['pn_unit_iid'][pn_unit] for pn_unit in clashing_pn_units_set]}"
            )
            filtered_atom_array, clash_severity = dp.handle_clashing_pn_units(
                clashing_pn_units_set, clashing_pn_units_dict, filtered_atom_array, id_map_dict["pn_unit_iid"]
            )

            # Remake the cell list, since we have removed atoms
            cell_list = struc.CellList(filtered_atom_array, cell_size=CELL_SIZE)

        # ---------- Step 5: Find contacting/partner PN units and build dataframes ---------- #
        # Loop through all considered query PN units, including proteins (single-chain), nucleic acids (single-chain), and ligands (single- or multi-chain)
        assembly_records = []

        # Filter out pn_units that have length 0 atom arrays
        pn_unit_iids_to_consider = [
            pn_unit_iid
            for pn_unit_iid in pn_unit_iids_to_consider
            if len(filtered_atom_array[filtered_atom_array.pn_unit_iid == pn_unit_iid]) > 0
        ]

        for query_pn_unit_iid in pn_unit_iids_to_consider:
            query_pn_unit_atom_array = filtered_atom_array[filtered_atom_array.pn_unit_iid == query_pn_unit_iid]

            assert len(query_pn_unit_atom_array) > 0, f"Query PN unit {query_pn_unit_iid} has zero atoms"

            query_pn_unit_type = ChainType(
                query_pn_unit_atom_array.chain_type[0]
            )  # All chains in a PN unit have the same type

            # Find contacting PN units, which we will use to construct interfaces
            contacting_pn_unit_iids = dp.get_contacting_pn_units(
                query_pn_unit=query_pn_unit_atom_array,
                filtered_atom_array=filtered_atom_array,
                cell_list=cell_list,
                contact_distance=self.contact_distance,
                min_contacts_required=1,
                calculate_min_distance=True,
            )

            # Find close PN units, which will be used to determine which PN units to load at train-time
            close_pn_unit_iids = dp.get_contacting_pn_units(
                query_pn_unit=query_pn_unit_atom_array,
                filtered_atom_array=filtered_atom_array,
                cell_list=cell_list,
                contact_distance=self.close_distance,
                min_contacts_required=1,
                calculate_min_distance=False,
            )
            close_pn_unit_iids = [id_map_dict["pn_unit_iid"][pn_unit["pn_unit_iid"]] for pn_unit in close_pn_unit_iids]

            # Sort contacting PN units by number of contacting atoms and then by minimum distance
            contacting_pn_unit_iids = sorted(
                contacting_pn_unit_iids,
                key=lambda x: (x["num_contacts"], -x["min_distance"]),
                reverse=True,
            )

            # Determine the primary polymer chain, which is the first partner polymer PN unit for non-polymers, or the query PN unit itself for polymers
            if query_pn_unit_type != ChainType.NON_POLYMER:
                primary_polymer_partner_pn_unit_iid = query_pn_unit_iid
            else:
                polymer_pn_unit_iids = np.unique(filtered_atom_array.pn_unit_iid[filtered_atom_array.is_polymer])
                # Find the first polymer PN unit in the contacting_pn_unit_iids sorted list
                partner_polymer_pn_units = [
                    partner for partner in contacting_pn_unit_iids if partner["pn_unit_iid"] in polymer_pn_unit_iids
                ]
                primary_polymer_partner_pn_unit_iid = (
                    partner_polymer_pn_units[0]["pn_unit_iid"] if len(partner_polymer_pn_units) > 0 else None
                )

            type_specific_criteria = {}
            # For non-polymers, calculate additional information
            if query_pn_unit_type == ChainType.NON_POLYMER:
                bonded_polymer_pn_units = dp.get_bonded_polymer_pn_units(query_pn_unit_iid, filtered_atom_array)

                # Check is SOI
                residue_names = np.unique(query_pn_unit_atom_array.res_name)
                is_loi = True if len(loi_ligand_set.intersection(residue_names)) > 0 else False

                # Check if metal
                is_metal = (
                    True if len(query_pn_unit_atom_array) == 1 and query_pn_unit_atom_array[0].is_metal else False
                )

                if exists(ligand_validity_scores):
                    _query_pn_unit_ligand_ids = sorted(
                        set(list(zip(query_pn_unit_atom_array.chain_id, query_pn_unit_atom_array.res_name)))
                    )

                    # Subset to the ids that have validity scores
                    _ligands_with_scores = [
                        _id for _id in _query_pn_unit_ligand_ids if _id in ligand_validity_scores.index
                    ]

                    ligand_validity = ligand_validity_scores.loc[_ligands_with_scores].to_dict()
                else:
                    ligand_validity = {}

                # Get the sequence of residues in the non-polymer PN unit
                non_polymer_res_names = struc.get_residues(query_pn_unit_atom_array)[
                    1
                ]  # get_residues returns a tuple of (ids, names)

                # Other options to consider for criteria:
                # -- Get the diameter of the PN unit
                # -- Get whether the query PN unit is coordinated
                # -- Get polar contacts
                # -- Get fraction of atoms in hull

                type_specific_criteria = {
                    "is_metal": is_metal,
                    "is_loi": is_loi,
                    "bonded_polymer_pn_units": bonded_polymer_pn_units,
                    "ligand_validity": ligand_validity,
                    "non_polymer_res_names": non_polymer_res_names,
                }
            elif query_pn_unit_type.is_polymer():
                chain_id = query_pn_unit_atom_array.chain_id[0]  # Polymers have only one chain
                ec_numbers = chain_info_dict[chain_id]["ec_numbers"]
                type_specific_criteria = {
                    "ec_numbers": json.dumps(ec_numbers),
                    "sequence_length": len(
                        chain_info_dict[chain_id]["processed_entity_canonical_sequence"]
                    ),  # We need to use the processed sequence
                    "processed_entity_canonical_sequence": chain_info_dict[chain_id][
                        "processed_entity_canonical_sequence"
                    ],
                    "processed_entity_non_canonical_sequence": chain_info_dict[chain_id][
                        "processed_entity_non_canonical_sequence"
                    ],
                }

            # Sequence information
            q_pn_unit_processed_entity_canonical_sequence = type_specific_criteria.get(
                "processed_entity_canonical_sequence", ""
            )
            q_pn_unit_processed_entity_non_canonical_sequence = type_specific_criteria.get(
                "processed_entity_non_canonical_sequence", ""
            )

            # Sequence hashes
            q_pn_unit_processed_entity_canonical_sequence_hash = (
                hash_sequence(q_pn_unit_processed_entity_canonical_sequence)
                if q_pn_unit_processed_entity_canonical_sequence
                else None
            )
            q_pn_unit_processed_entity_non_canonical_sequence_hash = (
                hash_sequence(q_pn_unit_processed_entity_non_canonical_sequence)
                if q_pn_unit_processed_entity_non_canonical_sequence
                else None
            )

            # Resolved residues (we already removed atoms with zero occupancy)
            num_resolved_residues = struc.get_residue_count(query_pn_unit_atom_array)

            pn_unit_record = {
                # Entry-level data
                "pdb_id": id,
                "assembly_id": assembly_id,
                "clash_severity": clash_severity,
                "resolution": result_dict["metadata"]["resolution"],
                "deposition_date": result_dict["metadata"]["deposition_date"],
                "release_date": result_dict["metadata"]["release_date"],
                "method": result_dict["metadata"]["method"],
                "num_polymer_pn_units": num_polymer_pn_units,
                "num_atoms": len(filtered_atom_array),
                # Query PN unit-level data
                "q_pn_unit_iid": id_map_dict["pn_unit_iid"][query_pn_unit_iid],
                "q_pn_unit_id": id_map_dict["pn_unit_id"][query_pn_unit_atom_array.pn_unit_id[0]],
                "q_pn_unit_type": query_pn_unit_type.value,
                "q_pn_unit_transformation_id": query_pn_unit_atom_array.transformation_id[
                    0
                ],  # All chains in a PN unit have the same transformation ID
                "q_pn_unit_num_atoms": len(query_pn_unit_atom_array),
                "q_pn_unit_is_multichain": len(np.unique(query_pn_unit_atom_array.chain_id)) > 1,
                "q_pn_unit_is_multiresidue": len(np.unique(query_pn_unit_atom_array.res_id)) > 1,
                "q_pn_unit_num_resolved_residues": num_resolved_residues,
                # Non-polymer type-specific criteria
                "q_pn_unit_is_metal": type_specific_criteria.get("is_metal", False),
                "q_pn_unit_is_loi": type_specific_criteria.get("is_loi", False),
                "q_pn_unit_ligand_validity": type_specific_criteria.get("ligand_validity", {}),
                "q_pn_unit_bonded_polymer_pn_units": {
                    id_map_dict["pn_unit_iid"][pn_unit]
                    for pn_unit in type_specific_criteria.get("bonded_polymer_pn_units", set())
                },  # Covalent modifications
                "q_pn_unit_non_polymer_res_names": ",".join(type_specific_criteria.get("non_polymer_res_names", [])),
                # Polymer type-specific criteria
                "q_pn_unit_ec_numbers": type_specific_criteria.get("ec_numbers", []),
                "q_pn_unit_sequence_length": type_specific_criteria.get("sequence_length", None),
                # Sequences
                "q_pn_unit_processed_entity_canonical_sequence": q_pn_unit_processed_entity_canonical_sequence,
                "q_pn_unit_processed_entity_non_canonical_sequence": q_pn_unit_processed_entity_non_canonical_sequence,
                # Hashes
                "q_pn_unit_processed_entity_canonical_sequence_hash": q_pn_unit_processed_entity_canonical_sequence_hash,
                "q_pn_unit_processed_entity_non_canonical_sequence_hash": q_pn_unit_processed_entity_non_canonical_sequence_hash,
                # Partners
                "q_pn_unit_primary_polymer_partner": id_map_dict["pn_unit_iid"][primary_polymer_partner_pn_unit_iid]
                if primary_polymer_partner_pn_unit_iid
                else None,
                "q_pn_unit_contacting_pn_unit_iids": json.dumps(
                    [
                        {
                            "pn_unit_iid": id_map_dict["pn_unit_iid"][partner["pn_unit_iid"]],
                            **{k: v for k, v in partner.items() if k != "pn_unit_iid"},
                        }
                        for partner in contacting_pn_unit_iids
                    ]
                ),
                "q_pn_unit_close_pn_unit_iids": json.dumps(close_pn_unit_iids),
            }
            assembly_records.append(pn_unit_record)
        return assembly_records
