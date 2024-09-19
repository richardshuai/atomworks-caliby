import logging
import os
import pickle
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm

from datahub.transforms.rdkit_utils import res_name_to_rdkit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_residue_to_rdkit_map(pn_units_df_path: str | os.PathLike, output_pickle_path: str | os.PathLike) -> None:
    """
    Processes the pn_units DataFrame to build a map from residue names to RD Kit molecule objects,
    and saves the result as a pickle file.

    Args:
        pn_units_path (Union[str, os.PathLike]): Path to the pn_units DataFrame.
        output_pickle_path (Union[str, os.PathLike]): Path to save the output pickle file.

    Example:
        python build_residue_to_rdkit_map.py /path/to/pn_units_df.parquet /path/to/output_directory/output.pkl
    """

    # Ensure paths are Path objects
    pn_units_df_path = Path(pn_units_df_path)
    output_pickle_path = Path(output_pickle_path)

    # Load the dataset with specific columns
    PN_UNITS_COLUMNS_TO_LOAD = [
        "pdb_id",
        "q_pn_unit_iid",
        "q_pn_unit_type",
        "q_pn_unit_non_polymer_res_names",
    ]
    pn_units_df = pd.read_parquet(pn_units_df_path, columns=PN_UNITS_COLUMNS_TO_LOAD)

    # Get all of the unique residues
    q_pn_unit_non_polymer_res_names = pn_units_df["q_pn_unit_non_polymer_res_names"].unique()
    logger.info(f"Number of unique residues (including multi-residue): {len(q_pn_unit_non_polymer_res_names)}")

    # Split each residue name by commas and make unique again
    q_pn_unit_non_polymer_res_names_split = [x.split(",") for x in q_pn_unit_non_polymer_res_names if x is not None]
    q_pn_unit_non_polymer_res_names_split_unique = set(
        [item for sublist in q_pn_unit_non_polymer_res_names_split for item in sublist]
    )
    logger.info(f"Number of unique residues (split by comma): {len(q_pn_unit_non_polymer_res_names_split_unique)}")

    # Build a map from the residue name to the RD Kit molecule object
    residue_name_to_rdkit_molecule = {}
    n_failures = 0

    logger.info("Building residue name to RD Kit molecule map...")
    for res_name in tqdm(q_pn_unit_non_polymer_res_names_split_unique, desc="Processing residues"):
        try:
            rdkit_molecule = res_name_to_rdkit(res_name, sanitize=False, attempt_fixing_corrupted_molecules=False)
            residue_name_to_rdkit_molecule[res_name] = rdkit_molecule
        except Exception as e:
            n_failures += 1
            logger.info(f"Failed to convert residue name to RD Kit molecule: {res_name}, with error: {e}")

    # Print number of failures
    logger.info(f"Number of conversion failures: {n_failures}")

    # Print the size of the pickle file before saving
    pickle_data = pickle.dumps(residue_name_to_rdkit_molecule)
    pickle_size = len(pickle_data)
    logger.info(f"Size of the pickle file: {pickle_size} bytes")

    # Save the dictionary to a pickle file
    with open(output_pickle_path, "wb") as f:
        f.write(pickle_data)


if __name__ == "__main__":
    fire.Fire(build_residue_to_rdkit_map)
