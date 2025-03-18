"""
Script that, given a list of directories containing MSA files named by sequence hashes and a dataframe with full sequence information (e.g, "pn_units_df"), finds the missing protein sequences in the MSA files. 

Example:
    python find_missing_msa_protein_sequences.py \
        --df_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/pn_units_df.parquet \
        --msa_dirs /projects/ml/RF2_allatom/data_preprocessing/msa/protein,/projects/msa/rf2aa_af3/2024_08_12/processed \
        --output_path /projects/ml/RF2_allatom/data_preprocessing/msa/missing_protein_sequences_08_21.csv
"""

import logging
import re
from os import PathLike
from pathlib import Path

import fire
import pandas as pd
from cifutils.enums import ChainType
from tqdm import tqdm

from datahub.utils.misc import hash_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_filenames(msa_dirs: list[Path]) -> set:
    """
    Collect all filenames in the MSA directories into a set for membership checking.

    Args:
        msa_dirs (list[Path]): The directories to search for filenames.

    Returns:
        set: A set of filenames.
    """
    filenames = set()
    for msa_dir in msa_dirs:
        filenames.update(
            {
                file.name
                for file in tqdm(msa_dir.rglob("*"), desc=f"Collecting files from {msa_dir.name}")
                if file.is_file()
            }
        )

        logger.info(f"Collected {len(filenames)} filenames from {msa_dir}.")
    return filenames


def extract_last_hash(text: str):
    """Extracts the last 11-character alphanumeric hash from a string."""
    pattern = r"\b[a-zA-Z0-9]{11}\b"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def find_missing_msa_protein_sequences(
    df_path: PathLike | str,
    msa_dirs: str,
    output_path: str,
    sequence_column: str = "q_pn_unit_processed_entity_non_canonical_sequence",
) -> pd.DataFrame:
    """
    Find the missing sequences in the MSA files.

    Args:
        df_path (PathLike | str): Path to the dataframe containing the full sequence information (e.g., "pn_units_df").
        msa_dirs (str): Comma-separated list of directories containing the MSA files named by sequence hashes.
        output_path (str): Path to save the output CSV file containing the missing sequences.
        sequence_column (str): The column name in the dataframe containing the sequences.

    Returns:
        pd.DataFrame: A dataframe containing the missing sequences.
    """

    # Convert msa_dirs string to a list of Path objects
    msa_dirs = [Path(dir_path) for dir_path in msa_dirs.split(",")]
    output_path = Path(output_path)
    df_path = Path(df_path)

    # Check paths upfront
    assert df_path.exists(), f"Dataframe {df_path} does not exist."
    for msa_dir in msa_dirs:
        assert msa_dir.exists(), f"MSA directory {msa_dir} does not exist."
    assert output_path.parent.exists(), f"Output directory {output_path.parent} does not exist."

    # Load dataframe
    df = pd.read_parquet(df_path)

    # Filter to only rows where the query PN unit is a protein
    df = df[
        (df["q_pn_unit_type"] == ChainType.POLYPEPTIDE_L.value)
        | (df["q_pn_unit_type"] == ChainType.POLYPEPTIDE_D.value)
    ]
    logger.info(f"Loaded dataframe with {len(df)} protein rows.")

    # Get the unique sequences in the df
    all_msa_sequences = df[sequence_column].unique()

    # Collect all filenames in the MSA directories into a set for membership checking
    logger.info("Collecting filenames in the MSA directories...")
    existing_filenames = collect_filenames(msa_dirs)
    logger.info(f"Collected {len(existing_filenames)} filenames.")

    logger.info("Extracting hashes from filenames...")
    existing_hashes = {extract_last_hash(filename) for filename in existing_filenames}
    logger.info(f"Extracted {len(existing_hashes)} unique hashes.")

    # Compute the missing sequences
    missing_sequences = []
    for sequence in tqdm(all_msa_sequences, desc="Checking for missing sequences"):
        if sequence:  # There are some empty sequences hanging around in the df
            sequence_hash = hash_sequence(sequence)
            if sequence_hash not in existing_hashes:
                missing_sequences.append(sequence)

    logger.info(f"Found {len(missing_sequences)} missing sequences. Saving to {output_path}.")

    # Save the missing sequences to a DataFrame
    missing_sequences_df = pd.DataFrame(missing_sequences, columns=[sequence_column])
    missing_sequences_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(find_missing_msa_protein_sequences)
