"""
Functions to cluster protein and nucleic acid sequences using MMseqs2.

NOTE: Not able to be run interactively due to `mmseqs` instability; must be run via command line.
NOTE: See `https://github.com/soedinglab/MMseqs2` for MMseqs2 installation instructions (`conda` recommended).
"""

import logging
import shutil
import subprocess
from os import PathLike, devnull
from pathlib import Path

import fire
import pandas as pd

from datahub.preprocessing.constants import NA_VALUES
from scripts.preprocessing.clustering.create_fasta_files_from_df import create_fasta_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_mmseqs2_clustering(
    input_fasta: str,
    cluster_identity: float = 0.4,
    coverage: float = 0.8,
    coverage_mode: int = 0,
    temp_dir: PathLike | str | None = None,
) -> pd.DataFrame:
    """
    Runs MMseqs2 clustering on the input FASTA file.

    Args:
        input_fasta (str): Path to the input FASTA file. The headers for the sequences should be unique, as they are used to identify the sequences in the output.
        cluster_identity (float): Sequence identity threshold for clustering. Default is 0.4 (40%) for proteins, per AF-3.
        coverage (float): Coverage threshold for clustering. Default is 0.8 (80%).
        coverage_mode (int): Mode for coverage calculation. Options are:
            0: (Default) coverage of query and target (bi-directional; most common default for full-length protein sequence comparisons)
            1: coverage of target
            2: coverage of query
            3: target seq. length has to be at least x% of query length
            4: query seq. length has to be at least x% of target length
            5: short seq. needs to be at least x% of the other seq. length
        tmp_dir (PathLike | str): Path to the temporary directory where MMseqs2 will write intermediate files. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the clustering results with columns ["cluster_rep_seq_hash", "seq_hash"].

    Example:
        cluster_rep_seq_hash, seq_hash
        afe56282ba3, afe56282ba3
        afe56282ba3, ee1f80a23f3
        afe56282ba3, 4a2caa18797
        afe56282ba3, 19f7ce1eed1

    References:
    - PDB clustering approach: https://www.rcsb.org/docs/grouping-structures/sequence-based-clustering
    - MMseqs2 documentation: https://github.com/soedinglab/mmseqs2/wiki
    - CLI documentation for the `easy-cluster` command: `mmseqs easy-cluster -h`
    """
    temp_dir = Path.cwd() / "temp" if temp_dir is None else Path(temp_dir)
    try:
        # Run MMseqs2 easy-cluster command
        logger.info(
            f"Running MMseqs2 easy-cluster with cluster_identity={cluster_identity}, coverage={coverage}, and coverage_mode={coverage_mode}..."
        )
        with open(devnull, "w") as fnull:
            subprocess.run(
                [
                    "mmseqs",
                    "easy-cluster",
                    input_fasta,
                    "result",
                    temp_dir,
                    "--min-seq-id",
                    str(
                        cluster_identity
                    ),  # Sequence identity threshold for clustering, typically 0.4 for proteins, and 1.0 for nucleic acids and peptides
                    "-c",
                    str(coverage),  # Coverage threshold for clustering, typically 0.8
                    "-s",
                    "8",  # MMseqs2's highest alignment sensitivity for clustering
                    "--cluster-mode",
                    "0",  # 0 = standard, 1 = Connected component algorithm, slower but capable of covering more remote homologs. See https://www.rcsb.org/docs/grouping-structures/sequence-based-clustering
                    "--cov-mode",
                    str(
                        coverage_mode
                    ),  # Bi-directional coverage requirements (0) likely best for full-length proteins (but possible failure mode for fragment vs. full protein)
                ],
                check=True,
                stdout=fnull,
                stderr=fnull,
            )

        logger.info("Clustering completed! Parsing TSV output file into a pandas DataFrame...")

        current_dir = Path.cwd()
        cluster_file = current_dir / "result_cluster.tsv"

        # Load the TSV output file into a DataFrame
        df = pd.read_csv(
            cluster_file, sep="\t", header=None, names=["cluster_rep_seq_hash", "seq_hash"], keep_default_na=NA_VALUES
        )

        logger.info(f"DataFrame created with {len(df)} rows!")

        return df

    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while running MMseqs2: {e}")


def cluster_all_sequences(
    pn_units_df: PathLike | str | pd.DataFrame,
    output_path: str | None = None,
    cluster_modes: list[dict] = [
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
    ],
):
    """
    Clusters protein and nucleic acid sequences from a DataFrame and merges the cluster information back into the DataFrame.

    This function performs the following steps:
    1. Creates (deduplicated) FASTA files for proteins and nucleic acids from the input DataFrame.
    2. Runs MMseqs2 clustering on the generated FASTA files for each specified parameter configuration.
    3. Merges the clustering results back into the original DataFrame.
    4. Saves the updated DataFrame with clustering information to a specified output path.

    Args:
        pn_units_df_path (PathLike | str | pd.DataFrame): Path to the input DataFrame stored as a Parquet file, or the DataFrame directly.
        output_path (str | None): Path to save the output DataFrame with clustering information. If None, returns the dataframe without saving.
        cluster_modes (list[dict]): List of dictionaries specifying the clustering configurations to run.

    Columns added to DataFrame:
        - protein_cluster_{configuration}_rep_seq_hash: Representative sequence hash for protein clusters with the given configuration.
        - nucleic_acid_cluster_{configuration}_rep_seq_hash: Representative sequence hash for nucleic acid clusters with the given configuration.

    Returns:
        None
    """
    current_dir = Path.cwd()
    temp_dir = current_dir / "tmp"

    if not isinstance(pn_units_df, pd.DataFrame):
        pn_units_df = Path(pn_units_df)
        df = pd.read_parquet(pn_units_df)
    else:
        df = pn_units_df

    # Create FASTA files for proteins and nucleic acids from the dataframe, and save them in the temp directory
    logger.info("Creating FASTA files for proteins and nucleic acids...")
    create_fasta_files(df, temp_dir)
    logger.info(f"FASTA files saved to {temp_dir}; will be cleaned up after clustering.")

    for cluster_mode in cluster_modes:
        # Run clustering for proteins
        protein_fasta = temp_dir / "protein_sequences.fasta"
        protein_cluster_df = run_mmseqs2_clustering(
            str(protein_fasta),
            cluster_identity=cluster_mode["cluster_identity"],
            coverage=cluster_mode["coverage"],
            coverage_mode=cluster_mode["coverage_mode"],
            temp_dir=temp_dir,
        )

        # Run clustering for nucleic acids
        nucleic_acid_fasta = temp_dir / "nucleic_acid_sequences.fasta"
        nucleic_acid_cluster_df = run_mmseqs2_clustering(
            str(nucleic_acid_fasta),
            cluster_identity=cluster_mode["cluster_identity"],
            coverage=cluster_mode["coverage"],
            coverage_mode=cluster_mode["coverage_mode"],
            temp_dir=temp_dir,
        )

        # Create a short string of the cluster mode for the column name
        cluster_mode_str = f"(id:{cluster_mode['cluster_identity']})(cov:{cluster_mode['coverage']})(cov_mode:{cluster_mode['coverage_mode']})".replace(
            ".", ","
        )

        logger.info("Merging clustering information into the master DataFrame...")
        # Merge protein clusters into the master DataFrame

        # ...drop the `cluster_mode_str` col, if it already exists
        protein_cluster_col = f"q_pn_unit_protein_cluster_{cluster_mode_str}_rep_seq_hash"
        if protein_cluster_col in df.columns:
            df.drop(columns=[protein_cluster_col], inplace=True)

        # ...merge and rename
        df = df.merge(
            protein_cluster_df[["seq_hash", "cluster_rep_seq_hash"]],
            left_on="q_pn_unit_processed_entity_canonical_sequence_hash",
            right_on="seq_hash",
            how="left",
        ).rename(columns={"cluster_rep_seq_hash": protein_cluster_col})
        logger.info(f"Merged protein clusters for {cluster_mode_str} configuration.")

        # Drop the redundant 'seq_hash' column from the merge
        if "seq_hash" in df.columns:
            df.drop(columns=["seq_hash"], inplace=True)

        # Merge nucleic acid clusters into the master DataFrame

        # ...drop the `cluster_mode_str` col, if it already exists
        nucleic_acid_cluster_col = f"q_pn_unit_nucleic_acid_cluster_{cluster_mode_str}_rep_seq_hash"
        if nucleic_acid_cluster_col in df.columns:
            df.drop(columns=[nucleic_acid_cluster_col], inplace=True)

        # ...merge and rename
        df = df.merge(
            nucleic_acid_cluster_df[["seq_hash", "cluster_rep_seq_hash"]],
            left_on="q_pn_unit_processed_entity_canonical_sequence_hash",
            right_on="seq_hash",
            how="left",
        ).rename(columns={"cluster_rep_seq_hash": nucleic_acid_cluster_col})
        logger.info(f"Merged nucleic acid clusters for {cluster_mode_str} sequence identity.")

        # Drop the redundant 'seq_hash' column from the merge
        if "seq_hash" in df.columns:
            df.drop(columns=["seq_hash"], inplace=True)

        logger.info(f"Clustering completed for {cluster_mode_str} configuration!")

    logger.info("Clusting complete!")
    if output_path is not None:
        # Save before cleaning up, in case of errors
        logger.info(f"Saving to {output_path}...")
        df.to_parquet(output_path, index=False)
        logger.info(f"DataFrame with clustering information saved to {output_path}")

    # Remove everything in the temp directory
    logger.info("Cleaning up...")
    try:
        shutil.rmtree(temp_dir)

        # Remove files created by MMseqs2 in the current directory
        current_dir = Path.cwd()
        filenames = ["result_cluster.tsv", "result_all_seqs.fasta", "result_rep_seq.fasta"]
        for filename in filenames:
            file_path = current_dir / filename
            if file_path.exists():
                file_path.unlink()

    except Exception as e:
        logger.error(f"Error removing temp directory {temp_dir}: {e}")

    if output_path is None:
        # Return after cleaning up (primarily used for testing)
        return df


if __name__ == "__main__":
    fire.Fire(cluster_all_sequences)
