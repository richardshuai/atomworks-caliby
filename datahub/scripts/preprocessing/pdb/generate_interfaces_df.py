from __future__ import annotations

import json
import logging
import re
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm

from datahub.common import generate_example_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_contacting_pn_unit_info(partner_list: str | None = None) -> dict:
    """Extracts partner polymer/non-polymer unit instance IDs and number of contacts from the partner list string."""
    if partner_list is None:
        return {
            "contacting_pn_unit_iids": [],
            "num_contacts": {},
        }

    partner_list = json.loads(partner_list)
    return {
        "contacting_pn_unit_iids": [partner["pn_unit_iid"] for partner in partner_list],
        "num_contacts": {partner["pn_unit_iid"]: partner["num_contacts"] for partner in partner_list},
    }


def process_pdb_id_group(group: pd.DataFrame) -> list[dict]:
    interfaces_list = []

    pdb_id = group["pdb_id"].iloc[0]

    # ...validate that the PDB ID is a four-letter alphanumeric code
    if not pdb_id.isalnum() or len(pdb_id) != 4:
        logger.warning(f"Invalid PDB ID: {pdb_id}")
        return interfaces_list

    pn_units = group["q_pn_unit_iid"].tolist()
    contacting_pn_units_info_dict = {
        row["q_pn_unit_iid"]: extract_contacting_pn_unit_info(row["q_pn_unit_contacting_pn_unit_iids"])
        for _, row in group.iterrows()
    }

    for pn_unit_1, pn_unit_2 in combinations(pn_units, 2):
        if (
            pn_unit_2 in contacting_pn_units_info_dict[pn_unit_1]["contacting_pn_unit_iids"]
            and pn_unit_1 in contacting_pn_units_info_dict[pn_unit_2]["contacting_pn_unit_iids"]
        ):
            # ...sort and rename; only process each pair combination once (due to `combinations`)
            sorted_pair = tuple(sorted([pn_unit_1, pn_unit_2]))
            pn_unit_1 = sorted_pair[0]
            pn_unit_2 = sorted_pair[1]

            # ...extract the PN unit data from the dataframe containing all PN units in the example
            pn_unit_1_data = group[group["q_pn_unit_iid"] == pn_unit_1]
            if pn_unit_1_data.shape[0] != 1:
                raise ValueError(f"Expected exactly one row for pn_unit_1, but got {pn_unit_1_data.shape[0]}")
            pn_unit_1_data = pn_unit_1_data.iloc[0]

            pn_unit_2_data = group[group["q_pn_unit_iid"] == pn_unit_2]
            if pn_unit_2_data.shape[0] != 1:
                raise ValueError(f"Expected exactly one row for pn_unit_2, but got {pn_unit_2_data.shape[0]}")
            pn_unit_2_data = pn_unit_2_data.iloc[0]

            # ...check if the interface is a covalent modification
            involves_covalent_modification = False
            if pn_unit_2 in eval(pn_unit_1_data["q_pn_unit_bonded_polymer_pn_units"]) or pn_unit_1 in eval(
                pn_unit_2_data["q_pn_unit_bonded_polymer_pn_units"]
            ):
                involves_covalent_modification = True

            # ...check if the interface is inter-molecule
            # (i.e., covalent modifications would not be considered inter-molecule)
            # NOTE: Theoretically, equivalent to `includes_covalent_modification`, but we keep general in the event other edge cases arise
            is_inter_molecule = pn_unit_1_data["q_pn_unit_molecule_iid"] != pn_unit_2_data["q_pn_unit_molecule_iid"]

            # ...count the number of contacts between the two interacting PN units
            num_contacts = contacting_pn_units_info_dict[pn_unit_1]["num_contacts"][pn_unit_2]
            assert num_contacts == contacting_pn_units_info_dict[pn_unit_2]["num_contacts"][pn_unit_1]

            # ...populate static fields
            # fmt: off
            interface = {
                # ...populate entry-wide fields (do not change between interfaces within the same example)
                "pdb_id": pdb_id,
                "assembly_id": pn_unit_1_data["assembly_id"],  # same entry
                "clash_severity": pn_unit_1_data["clash_severity"],  # same entry
                "resolution": pn_unit_1_data["resolution"],  # same entry
                "deposition_date": pn_unit_1_data["deposition_date"],  # same entry
                "release_date": pn_unit_1_data["release_date"],  # same entry
                "method": pn_unit_1_data["method"],  # same entry
                "num_polymer_pn_units": pn_unit_1_data["num_polymer_pn_units"],  # same entry

                # ...generate the unique example ID (used for debugging)
                "example_id": generate_example_id(
                    ["pdb", "interfaces"],
                    pdb_id,
                    pn_unit_1_data["assembly_id"],
                    [sorted_pair[0], sorted_pair[1]],
                ),

                # ...values for AF-3-style weighted sampling
                "n_prot": pn_unit_1_data["n_prot"] + pn_unit_2_data["n_prot"],
                "n_nuc": pn_unit_1_data["n_nuc"] + pn_unit_2_data["n_nuc"],
                "n_ligand": pn_unit_1_data["n_ligand"] + pn_unit_2_data["n_ligand"],

                # ...polymer/non-polymer unit information
                "pn_unit_1_iid": sorted_pair[0],
                "pn_unit_2_iid": sorted_pair[1],
                "pn_unit_1_non_polymer_res_names": pn_unit_1_data["q_pn_unit_non_polymer_res_names"],
                "pn_unit_2_non_polymer_res_names": pn_unit_2_data["q_pn_unit_non_polymer_res_names"],
                "pn_unit_1_type": pn_unit_1_data["q_pn_unit_type"],
                "pn_unit_2_type": pn_unit_2_data["q_pn_unit_type"],
                "pn_unit_1_num_atoms": pn_unit_1_data["q_pn_unit_num_atoms"],
                "pn_unit_2_num_atoms": pn_unit_2_data["q_pn_unit_num_atoms"],
                "pn_unit_1_is_multichain": pn_unit_1_data["q_pn_unit_is_multichain"],
                "pn_unit_2_is_multichain": pn_unit_2_data["q_pn_unit_is_multichain"],
                "pn_unit_1_is_multiresidue": pn_unit_1_data["q_pn_unit_is_multiresidue"],
                "pn_unit_2_is_multiresidue": pn_unit_2_data["q_pn_unit_is_multiresidue"],
                "pn_unit_1_sequence_length": pn_unit_1_data["q_pn_unit_sequence_length"],
                "pn_unit_2_sequence_length": pn_unit_2_data["q_pn_unit_sequence_length"],
                "pn_unit_1_num_resolved_residues": pn_unit_1_data["q_pn_unit_num_resolved_residues"],
                "on_unit_2_num_resolved_residues": pn_unit_2_data["q_pn_unit_num_resolved_residues"],

                # ...partners
                # (Must meet minimum number of sub-5A contacts to be considered a partner)
                "contacting_pn_unit_iids": json.dumps(
                    list(
                        set(
                            contacting_pn_units_info_dict[sorted_pair[0]]["contacting_pn_unit_iids"]
                            + contacting_pn_units_info_dict[sorted_pair[1]]["contacting_pn_unit_iids"]
                        )
                    )
                ),
                # (Associate each PN unit with a primary polymer partner)
                "pn_unit_1_primary_polymer_partner": pn_unit_1_data["q_pn_unit_primary_polymer_partner"],
                "pn_unit_2_primary_polymer_partner": pn_unit_2_data["q_pn_unit_primary_polymer_partner"],

                # ...add interface indicators
                "involves_covalent_modification": involves_covalent_modification,
                "is_inter_molecule": is_inter_molecule,
                "involves_loi": pn_unit_1_data["q_pn_unit_is_loi"] or pn_unit_2_data["q_pn_unit_is_loi"],
                "involves_metal": pn_unit_1_data["q_pn_unit_is_metal"] or pn_unit_2_data["q_pn_unit_is_metal"],
                "num_contacts": num_contacts,
            }
            # fmt: on

            # ...add columns with dynamic names (from clustering)
            pattern = re.compile(r".*(protein_cluster|nucleic_acid_cluster).*")
            cluster_columns = [col for col in pn_unit_1_data.index if pattern.match(col)]
            for col in cluster_columns:
                col_without_prefix = col.replace("q_pn_unit_", "")
                interface[f"pn_unit_1_{col_without_prefix}"] = pn_unit_1_data[col]
                interface[f"pn_unit_2_{col_without_prefix}"] = pn_unit_2_data[col]

            # ...existing cluster columns, if present
            if "cluster" in pn_unit_1_data.index:
                interface["pn_unit_1_cluster"] = pn_unit_1_data["cluster"]
            if "cluster" in pn_unit_2_data.index:
                interface["pn_unit_2_cluster"] = pn_unit_2_data["cluster"]

            # ...if both are present, add the joint cluster column
            if "cluster" in pn_unit_1_data.index and "cluster" in pn_unit_2_data.index:
                # If either is None, set to None (we will exclude with a dataset filter)
                if pn_unit_1_data["cluster"] is None or pn_unit_2_data["cluster"] is None:
                    interface["cluster"] = None
                else:
                    interface["cluster"] = f"{pn_unit_1_data['cluster']}+{pn_unit_2_data['cluster']}"

            # ...and finally, add the interface to the list
            interfaces_list.append(interface)
        elif (
            pn_unit_2 in contacting_pn_units_info_dict[pn_unit_1]
            or pn_unit_1 in contacting_pn_units_info_dict[pn_unit_2]
        ):
            raise ValueError(f"Missing interface for {pdb_id} between {pn_unit_1} and {pn_unit_2}")

    return interfaces_list


def generate_interfaces_df(df: pd.DataFrame) -> pd.DataFrame:
    entries_by_pdb_id_and_assembly_id = [group for _, group in df.groupby(["pdb_id", "assembly_id"])]
    num_workers = min(cpu_count(), 14)  # Adjust based on your system
    chunksize = min(
        2000, max(1, len(entries_by_pdb_id_and_assembly_id) // num_workers), len(entries_by_pdb_id_and_assembly_id)
    )

    logger.info(f"Generating interfaces dataframe using {num_workers} workers...")

    # Create a pool of workers
    with Pool(processes=num_workers) as pool:
        interfaces_list = tqdm(
            pool.imap(process_pdb_id_group, entries_by_pdb_id_and_assembly_id, chunksize=chunksize),
            total=len(entries_by_pdb_id_and_assembly_id),
        )

        # Flatten the list of lists
        interfaces_list = [item for sublist in interfaces_list for item in sublist]

        return pd.DataFrame(interfaces_list)


def generate_and_save_interfaces_df(input_path: str, output_path: str):
    # ...load the query PN unit DataFrame
    logger.info("Loading query PN unit dataframe...")
    query_pn_units_df = pd.read_parquet(input_path)

    # ...assert that we have the columns from clustering already
    # (Cluster columns include with "protein_cluster_" and "nucleic_acid_cluster_")
    assert any("protein_cluster_" in col for col in query_pn_units_df.columns), (
        "Missing protein cluster columns in the input DataFrame. "
        "Please run the clustering script before generating the interfaces DataFrame."
    )
    assert any("nucleic_acid_cluster_" in col for col in query_pn_units_df.columns), (
        "Missing nucleic acid cluster columns in the input DataFrame. "
        "Please run the clustering script before generating the interfaces DataFrame."
    )

    # ...create the interfaces DataFrame
    logger.info("Generating interfaces dataframe...")
    interfaces_df = generate_interfaces_df(query_pn_units_df)

    output_path = Path(output_path)
    # ...ensure the output file has a .parquet suffix
    if output_path.suffix != ".parquet":
        output_path = output_path.with_suffix(".parquet")

    # ...check for duplicate rows
    num_duplicates = interfaces_df.duplicated().sum()
    if num_duplicates > 0:
        logger.warning(f"Found {num_duplicates} duplicate rows in the interfaces DataFrame")
        # ...deduplicate (paranoia)
        interfaces_df = interfaces_df.drop_duplicates()

    # ...save the interfaces DataFrame to the specified location as a parquet (requires pyarrow)
    interfaces_df.to_parquet(output_path, index=False)
    logger.info(f"Interfaces DataFrame saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(generate_and_save_interfaces_df)
