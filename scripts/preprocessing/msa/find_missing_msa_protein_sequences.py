"""
Script that, given a list of directories containing MSA files named by sequence hashes and a dataframe with full sequence information (e.g, "pn_units_df"), finds the missing protein sequences in the MSA files.

Example:
    python find_missing_msa_protein_sequences.py \
        --df_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/pn_units_df.parquet \
        --msa_dirs /projects/ml/RF2_allatom/data_preprocessing/msa/protein,/projects/msa/rf2aa_af3/2024_08_12/processed \
        --output_path /projects/ml/RF2_allatom/data_preprocessing/msa/missing_protein_sequences_08_21.csv \
"""

import logging
import random
from multiprocessing import Pool, cpu_count
from os import PathLike
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm

from atomworks.io import parse
from atomworks.enums import ChainType
from atomworks.ml.utils.misc import hash_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def shallow_collect_sequences(df: pd.DataFrame, sequence_column: str) -> list[str]:
    """
    Just gets the unique sequences from the PN units in the dataframe.
    Note this will NOT find chains that are in the structure but are not labelled as PN-units
    we care about!

    Args:
        df (pd.DataFrame): The dataframe containing the full sequence information.
        sequence_column (str): The column name in the dataframe containing the sequences.

    Returns:
        list[str]: A list of unique sequences from the PN units in the dataframe.
    """
    # Filter to only rows where the query PN unit is a protein
    df = df[
        (df["q_pn_unit_type"] == ChainType.POLYPEPTIDE_L.value)
        | (df["q_pn_unit_type"] == ChainType.POLYPEPTIDE_D.value)
    ]
    logger.info(f"Loaded dataframe with {len(df)} protein rows.")

    # Get the unique sequences in the df
    return df[sequence_column].unique()


def process_cif_file(cif_file_path: str) -> list[str]:
    """
    Process a single CIF file and extract sequences from all chains.

    Args:
        cif_file_path (str): Path to the CIF file.

    Returns:
        list[str]: A list of sequences from all chains in the structure.
    """
    sequences = []
    try:
        structure_dict = parse(cif_file_path)  # all default settings
        for chain in structure_dict["chain_info"].values():
            # only if they have the field we append. If not it usually means it's not a polymer/protein
            if (
                chain["chain_type"] == ChainType.POLYPEPTIDE_L.value
                and "processed_entity_non_canonical_sequence" in chain
            ):
                sequences.append(chain["processed_entity_non_canonical_sequence"])
    except Exception as e:
        logger.warning(f"Error parsing CIF file {cif_file_path}: {e}")
        return []
    return sequences


def deep_collect_sequences(
    df: pd.DataFrame,
    cif_file_path_column: str,
    num_workers: int | None = None,
) -> list[str]:
    """
    Collects all sequences from all chains in the structure. We actually have to load the CIF
    files because this stuff isn't stored in our parquet files ever.

    Args:
        df (pd.DataFrame): The dataframe containing the full sequence information.
        cif_file_path_column (str): The column name in the dataframe containing the CIF file paths.
        num_workers (int, optional): Number of workers for parallel processing. Defaults to None (uses CPU count).

    Returns:
        list[str]: A list of unique sequences from all chains in the structures.
    """
    unique_cif_file_paths = df[cif_file_path_column].unique()
    # randomize order so we spread out potentially really large files
    random.shuffle(unique_cif_file_paths)

    # Use all available CPUs if num_workers is not specified
    if num_workers is None:
        num_workers = cpu_count()

    all_sequences = []
    with Pool(processes=num_workers) as pool:
        for sequences in tqdm(
            pool.imap(process_cif_file, unique_cif_file_paths),
            total=len(unique_cif_file_paths),
            desc="Collecting sequences from CIF files",
        ):
            all_sequences.extend(sequences)

    return list(set(all_sequences))


def check_for_seq_in_dirs(
    inputs: tuple[str, list[Path]],
    shard_depths_to_check: list[int] = [0, 1, 2, 3],
    extensions_to_check: list[str] = [".a3m", ".a3m.gz"],
) -> bool:
    """
    Check if a sequence exists in any of the MSA directories.

    Args:
        inputs: a tuple containing:
            seq (str): The sequence to check for.
            msa_dirs (list[Path]): The directories to check for the sequence in.
        shard_depths_to_check (list[int], optional): The depths to check for the sequence in. Defaults to [0, 1, 2, 3].
        extensions_to_check (list[str], optional): The extensions to check for the sequence in. Defaults to [".a3m", ".a3m.gz"].

    Returns:
        bool: True if the sequence exists in any of the MSA directories, False otherwise.
    """
    seq, msa_dirs = inputs
    hashed_sequence = hash_sequence(seq)

    possible_file_paths = []
    for msa_dir in msa_dirs:
        for shard_depth in shard_depths_to_check:
            for extension in extensions_to_check:
                possible_file_path = "".join([f"{hashed_sequence[(i*2):(i+1)*2]}/" for i in range(shard_depth)])
                possible_file_paths.append(msa_dir / possible_file_path / f"{hashed_sequence}{extension}")

    return any(possible_file_path.exists() for possible_file_path in possible_file_paths)


def find_missing_msa_protein_sequences(
    df_path: PathLike | str,
    msa_dirs: str,
    output_path: str,
    sequence_column: str = "q_pn_unit_processed_entity_non_canonical_sequence",
    deep_search: bool = False,
    cif_file_path_column: str = "path",
    num_workers: int | None = None,
) -> pd.DataFrame:
    """
    Find the missing sequences in the MSA files.

    Args:
        df_path (PathLike | str): Path to the dataframe containing the full sequence information (e.g., "pn_units_df").
        msa_dirs (str): Comma-separated list of directories containing the MSA files named by sequence hashes.
        output_path (str): Path to save the output CSV file containing the missing sequences.
        sequence_column (str): The column name in the dataframe containing the sequences.
        deep_search (bool): Whether to search for all chains' sequences in the CIF files.
        cif_file_path_column (str): The column name in the dataframe containing the CIF file paths.
        num_workers (int, optional): Number of workers for parallel processing. Defaults to None (uses CPU count).
    Returns:
        pd.DataFrame: A dataframe containing the missing sequences.
    """

    # Use all available CPUs if num_workers is not specified
    if num_workers is None:
        num_workers = cpu_count()

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

    if deep_search:
        all_msa_sequences = deep_collect_sequences(df, cif_file_path_column, num_workers)
    else:
        all_msa_sequences = shallow_collect_sequences(df, sequence_column)

    missing_sequences_mask = []
    with Pool(processes=num_workers) as pool:
        args_list = [(seq, msa_dirs) for seq in all_msa_sequences]
        for does_sequence_exist in tqdm(
            pool.imap(check_for_seq_in_dirs, args_list), total=len(args_list), desc="Checking for missing sequences"
        ):
            missing_sequences_mask.append(does_sequence_exist)

    missing_sequences = [
        seq for seq, does_exist in zip(all_msa_sequences, missing_sequences_mask, strict=False) if not does_exist
    ]

    logger.info(f"Found {len(missing_sequences)} missing sequences. Saving to {output_path}.")

    # Save the missing sequences to a DataFrame
    missing_sequences_df = pd.DataFrame(missing_sequences, columns=[sequence_column])
    missing_sequences_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(find_missing_msa_protein_sequences)
