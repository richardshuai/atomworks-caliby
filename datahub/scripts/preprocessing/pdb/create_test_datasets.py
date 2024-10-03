"""
Script to generate the test datasets for data loading and pre-processing.
These test datasets are a subset of the PDB, containing both (a) a set of diverse/difficult examples and (b) a set of random examples.

Example usage: python create_test_datasets.py --pn_units_df_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_07_01/pn_units_df.parquet
"""

import logging
import os
import random
from pathlib import Path

import fire
import pandas as pd

from scripts.preprocessing.clustering.annotate_clusters import (
    AF3_CLUSTER_CHOICE_SUFFIXES,
    add_pn_unit_cluster_column,
)
from scripts.preprocessing.clustering.cluster_sequences import cluster_all_sequences
from scripts.preprocessing.pdb.confscript import get_all_pdb_ids
from scripts.preprocessing.pdb.generate_interfaces_df import generate_and_save_interfaces_df
from scripts.preprocessing.pdb.generate_pn_units_df import generate_pn_units_df
from scripts.preprocessing.pdb.process_pdbs import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TEST_DATA_DIR = Path(os.path.dirname(__file__)).parent.parent.parent / "tests" / "data"
DEFAULT_BASE_CIF_DIR = Path("/databases/rcsb/cif")

# fmt: off
"""
PDB IDs that we will always include when building test datasets.
If we use a PDB ID in a Transforms test case, we should add it here.
"""
PDB_IDS_TO_INCLUDE_IN_TEST_DATASETS = [
    "1A80", "1IVO", "3K4A", "3KFA", "6WJC", "1EN2", "1CBN", "133D", "4JS1", 
    "1L2Y", "2K0A", "4CPA", "1ZY8", "6DMH", "1FU2", "6DMG", "1Y1W", "5XNL", 
    "2E2H", "4NDZ", "3NE7", "3NEZ", "1RXZ", "3J31", "7MUB", "1QK0", "1DYL", 
    "7SBV", "3EPC", "6O7K", "104D", "5X3O", "5GAM", "6A5J", "3NE2", "1MNA", 
    "1HGE", "3EJJ", "112M", "1A3G", "1A2N", "1A2Y", "1BDV", "184D", "4HF4", 
    "3LPV", "2NVZ", "7KF1", "7CJG", "7B1W", "6ZIE", "7NMJ", "6M2Z", "5OCM", 
    "3SJM", "4I7Z", "4OLB", "4RES", "6BGN", "6VET", "6ZSJ", "5RX1", "7D9H", 
    "5S4P", "4GQA", "7AH0", "4U4H", "2PNO", "1PFI", 
    "1S2K", # Polymer with < 4 resolved residues
    "6Z3R", # Unresolved backbone coordinates (and thus cannot built backbone frame)
]
# fmt: on


def create_test_datasets(
    num_pdbs: int = 3000,
    test_data_dir: Path = DEFAULT_TEST_DATA_DIR,
    base_cif_dir: Path = DEFAULT_BASE_CIF_DIR,
    pn_units_df_path: os.PathLike = None,
):
    """Process PDB IDs, concatenate CSV files, and generate interfaces dataframe.
    May be slow if no pre-existing PN units dataframe is provided (highly recommended).

    Args:
        num_pdbs (int): Number of PDB IDs to process.
        test_data_dir (Path, optional): Directory to store test data. Defaults to the current value.
        base_cif_dir (Path, optional): Directory containing CIF files. Defaults to /databases/cif.
        pn_units_df_file (Path, optional): Path to existing PN units dataframe. If given, will not re-process from CIF files. Defaults to None.
    """
    # Make the test data directory if it doesn't exist
    test_data_dir.mkdir(parents=True, exist_ok=True)

    master_pdb_id_list = PDB_IDS_TO_INCLUDE_IN_TEST_DATASETS

    pn_units_df_path = Path(pn_units_df_path) if pn_units_df_path else None

    # Check current number of PDB IDs in master list
    num_master_pdb_ids = len(master_pdb_id_list)
    logger.info(f"Aggregated {num_master_pdb_ids} PDB IDs from test cases.")

    # If num_pdbs is greater than the number in master_pdb_id_list, add random PDB IDs
    if num_pdbs > num_master_pdb_ids:
        # Calculate number of additional PDB IDs needed
        additional_pdbs_needed = num_pdbs - num_master_pdb_ids

        logger.info(f"Fetching all PDB IDs to select additional {additional_pdbs_needed} PDB IDs...")
        all_pdb_ids = set(get_all_pdb_ids(base_cif_dir=base_cif_dir))
        selected_pdb_ids = set(master_pdb_id_list)

        # Select additional PDB IDs not in master_pdb_id_list but present in all PDB ids
        available_pdb_ids = list(all_pdb_ids - selected_pdb_ids)
        if len(available_pdb_ids) < additional_pdbs_needed:
            raise ValueError("Not enough available PDB IDs to meet the requested number.")

        additional_pdb_ids = random.sample(available_pdb_ids, additional_pdbs_needed)
        master_pdb_id_list.extend(additional_pdb_ids)

    # Convert the master_pdb_id_list to a set to remove duplicates, then back to a list
    master_pdb_id_list = list(set(master_pdb_id_list))
    # To lowercase
    master_pdb_id_list = [pdb_id.lower() for pdb_id in master_pdb_id_list]

    new_pn_units_df_path = test_data_dir / "pn_units_df.parquet"
    if pn_units_df_path and pn_units_df_path.exists():
        logger.info(f"Using existing PN units dataframe: {pn_units_df_path}")
        pn_units_df = pd.read_parquet(pn_units_df_path)
        # Filter to only the PDB IDs in master_pdb_id_list
        pn_units_df = pn_units_df[pn_units_df["pdb_id"].isin(master_pdb_id_list)]
        # Save the filtered PN units dataframe
        pn_units_df.to_parquet(new_pn_units_df_path)
    else:
        logger.info("No existing PN units dataframe found. Generating new PN units dataframe...")

        run_pipeline(
            pdb_selection=master_pdb_id_list,
            out_dir=test_data_dir,
            timeout_seconds=60,
            log_errors=False,
        )

        logger.info("Concatenating csvs...")
        generate_pn_units_df(input_dir=test_data_dir / "csv", output_path=new_pn_units_df_path)

    # Add the cluster columns to the PN units dataframe, overwriting the existing PN units dataframe in-place
    # Cluster at 40% (for proteins) and 100% (for nucleic acids) identity, with 80% coverage
    cluster_modes = [
        {
            "cluster_identity": 0.4,
            "coverage": 0.8,
            "coverage_mode": 0,
        },
        {
            "cluster_identity": 1.0,
            "coverage": 0.8,
            "coverage_mode": 0,
        },
    ]
    cluster_all_sequences(
        pn_units_df=new_pn_units_df_path, output_path=new_pn_units_df_path, cluster_modes=cluster_modes
    )

    # Add the cluster column to the PN units dataframe
    add_pn_unit_cluster_column(
        new_pn_units_df_path, cluster_choice_suffixes=AF3_CLUSTER_CHOICE_SUFFIXES, replace_df=True
    )

    # Generate interfaces dataframe
    # NOTE: Also adds the interfaces cluster column, based on the PN unit clusters
    logger.info("Generating interfaces dataframe...")
    new_interfaces_df_path = test_data_dir / "interfaces_df.parquet"
    generate_and_save_interfaces_df(input_path=new_pn_units_df_path, output_path=new_interfaces_df_path)

    # Delete the CSV directory
    logger.info("Cleaning up...")
    csv_dir = test_data_dir / "csv"
    if csv_dir.exists():
        logger.info(f"Deleting {csv_dir}...")
        for file in csv_dir.iterdir():
            file.unlink()
        csv_dir.rmdir()

    logger.info("Done.")


if __name__ == "__main__":
    fire.Fire(create_test_datasets)
