"""Functions to cluster protein and nucleic acid sequences using MMseqs2.

NOTE: Not able to be run interactively due to `mmseqs` instability; must be run via command line.
NOTE: See `https://github.com/soedinglab/MMseqs2` for MMseqs2 installation instructions (`conda` recommended).

From the AF3 supplement:
    Chain-based clustering occur at 40% sequence homology for proteins, 100% homology for nucleic acids,
    100% homology for peptides (<10 residues), and according to CCD identity for small molecules (e.g., only identical molecules share a cluster)
"""

import logging
from os import PathLike
from pathlib import Path

import fire
import pandas as pd
from atomworks.io.enums import ChainType, ChainTypeInfo

from datahub.preprocessing.utils.clustering import MMSeqs2Config, cluster_all_sequences

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AF3_CLUSTER_CHOICE_SUFFIXES = {
    # cov defines the minimum coverage, which is defined by the number of aligned residue
    # pairs divided by either the maximum of the length of query/centre and target/non-centre
    # sequences alnRes/max(qLen,tLen) (default mode, cov-mode 0), or by the length of
    # the target/non-centre sequence alnRes/tLen (cov-mode 1), or by the length of the
    # query/centre alnRes/qLen (--cov-mode 2).
    # Reference: https://mmseqs.com/latest/userguide.pdf
    "protein": "protein_cluster_(id:0,4)(cov:0,8)(cov_mode:0)_rep_seq_hash",
    "peptide": "protein_cluster_(id:1,0)(cov:0,8)(cov_mode:0)_rep_seq_hash",
    "nucleic_acid": "nucleic_acid_cluster_(id:1,0)(cov:0,8)(cov_mode:0)_rep_seq_hash",
    "ligand": "non_polymer_res_names",
}


def cluster_proteins_and_nucleic_acids(
    pn_units_df: PathLike | str | pd.DataFrame,
    output_path: str | None = None,
    clustering_configs: list[MMSeqs2Config] = [MMSeqs2Config(), MMSeqs2Config(cluster_identity=1.0)],
    sequence_col_name: str = "q_pn_unit_processed_entity_canonical_sequence",
    sequence_hash_col_name: str = "q_pn_unit_processed_entity_canonical_sequence_hash",
) -> pd.DataFrame | None:
    """Clusters protein and nucleic acid sequences using MMseqs2.

    Args:
        pn_units_df (PathLike | str | pd.DataFrame): Path to the input DataFrame or the DataFrame itself.
        output_path (str | None): Path to save the output DataFrame. If None, the modified DataFrame is returned.
        sequence_col_name (str): Name of the column containing the sequences to be clustered.
        sequence_hash_col_name (str): Name of the column containing hashes of the sequences to be clustered.
        clustering_configs (list[MMSeqs2Config]): List of MMSeqs2Config objects specifying the clustering configurations to run.

    Returns:
        pd.DataFrame | None: The modified DataFrame if `output_path` is None, otherwise returns None.
    """

    # Get input DataFrame
    if not isinstance(pn_units_df, pd.DataFrame):
        pn_units_df = Path(pn_units_df)
        df = pd.read_parquet(pn_units_df)
    else:
        df = pn_units_df

    # Record original length for sanity check
    original_df_length = len(df)

    # Build protein dataframe
    proteins_df = df[df["q_pn_unit_type"].isin([chain_type.value for chain_type in ChainTypeInfo.PROTEINS])]

    # Build nucleic acid DataFrame
    nucleic_acids_df = df[df["q_pn_unit_type"].isin([chain_type.value for chain_type in ChainTypeInfo.NUCLEIC_ACIDS])]

    # Compute protein clusters
    if len(proteins_df) > 0:
        df_with_protein_clusters = cluster_all_sequences(
            proteins_df,
            sequence_col_name,
            sequence_hash_col_name,
            output_col_prefix="q_pn_unit_protein_cluster",
            set_to_cluster_col=False,
            output_path=None,
            clustering_configs=clustering_configs,
        )

    # Compute nucleic acid clusters
    if len(nucleic_acids_df) > 0:
        df_with_nucleic_acid_clusters = cluster_all_sequences(
            nucleic_acids_df,
            sequence_col_name,
            sequence_hash_col_name,
            output_col_prefix="q_pn_unit_nucleic_acid_cluster",
            set_to_cluster_col=False,
            output_path=None,
            clustering_configs=clustering_configs,
        )

    # Merge the cluster information back into the full dataframe
    if len(proteins_df) > 0:
        protein_merge_cols = df.columns.intersection(df_with_protein_clusters.columns).tolist()
        df = df.merge(df_with_protein_clusters, how="left", on=protein_merge_cols)

    if len(nucleic_acids_df) > 0:
        nucleic_acid_merge_cols = df.columns.intersection(df_with_nucleic_acid_clusters.columns).tolist()
        df = df.merge(df_with_nucleic_acid_clusters, how="left", on=nucleic_acid_merge_cols)

    assert len(df) == original_df_length

    # Save or return the modified DataFrame
    if output_path is not None:
        df.to_parquet(output_path)
    else:
        return df


def add_pn_unit_cluster_column(
    pn_units_df: str | PathLike | pd.DataFrame,
    cluster_choice_suffixes: dict = AF3_CLUSTER_CHOICE_SUFFIXES,
    replace_df: bool = False,
) -> pd.DataFrame:
    """Add a cluster column to the PN units dataframe based on the query PN unit type.

    Note that the cluster column for interfaces is handled by `generate_interfaces_df.py`.

    Args:
        pn_units_df (str | PathLike | DataFrame): Path to the PN units dataframe in parquet format or a DataFrame.
        suffixes (dict): Dictionary of suffixes for protein, nucleic acid, and ligand.
        replace_df (bool): Whether to replace the input DataFrame with the updated DataFrame.

    Returns:
        DataFrame | None: The updated PN units dataframe with the cluster column added, or None if replace_df is True.
    """
    if isinstance(pn_units_df, (str, PathLike)):
        df = pd.read_parquet(pn_units_df)
    else:
        df = pn_units_df

    q_pn_unit_type_col = "q_pn_unit_type"
    q_pn_unit_sequence_length_col = "q_pn_unit_sequence_length"
    pn_unit_prefix = "q_pn_unit_"

    logger.info("Adding cluster column to PN units dataframe...")

    df["cluster"] = df.apply(
        lambda x: x[f"{pn_unit_prefix}{cluster_choice_suffixes['peptide']}"]
        if ChainType(x[q_pn_unit_type_col]).is_protein() and x[q_pn_unit_sequence_length_col] < 10
        else x[f"{pn_unit_prefix}{cluster_choice_suffixes['protein']}"]
        if ChainType(x[q_pn_unit_type_col]).is_protein()
        else x[f"{pn_unit_prefix}{cluster_choice_suffixes['nucleic_acid']}"]
        if ChainType(x[q_pn_unit_type_col]).is_nucleic_acid()
        else x[f"{pn_unit_prefix}{cluster_choice_suffixes['ligand']}"],
        axis=1,
    )

    # Save the updated DataFrame if replace_df is True and the input was a file
    if replace_df:
        if isinstance(pn_units_df, (str, PathLike)):
            logger.info(f"Saving updated PN units dataframe to {pn_units_df}...")
            df.to_parquet(pn_units_df)
        else:
            raise ValueError("Cannot replace DataFrame if it was not loaded from a file.")
    else:
        logger.info("Cluster column added to PN units dataframe. Returning the modified dataframe.")
        return df


def cluster_and_annotate_clusters(
    pn_units_df: PathLike | str | pd.DataFrame,
    clustering_configs: list[MMSeqs2Config] = [MMSeqs2Config(), MMSeqs2Config(cluster_identity=1.0)],
    cluster_choice_suffixes: dict = AF3_CLUSTER_CHOICE_SUFFIXES,
    sequence_col_name: str = "q_pn_unit_processed_entity_canonical_sequence",
    sequence_hash_col_name: str = "q_pn_unit_processed_entity_canonical_sequence_hash",
    output_path: PathLike | None = None,
):
    """Main entry point for AF3-like clustering and annotation of clusters. The resulting DataFrame is either returned or saved to disk.

    Args:
        pn_units_df (PathLike | str | pd.DataFrame): Path to the input DataFrame stored as a Parquet file, or the DataFrame directly.
        clustering_configs (list[MMSeqs2Config]): List of MMSeqs2Config objects specifying the clustering configurations to run.
        cluster_choice_suffixes (dict): Dictionary of suffixes for protein, nucleic acid, and ligand.
        sequence_col_name (str): Name of the column containing the sequences to be clustered.
        sequence_hash_col_name (str): Name of the column containing hashes of the sequences to be clustered.
        output_path (PathLike | None): Path to save the output DataFrame with clustering information. If None, returns the dataframe without saving.

    Returns:
        DataFrame | None: Returns the updated PN units dataframe with the cluster column added if output_path is None,
            or returns None otherwise.
    """

    df_with_specific_cluster_columns = cluster_proteins_and_nucleic_acids(
        pn_units_df=pn_units_df,
        sequence_col_name=sequence_col_name,
        sequence_hash_col_name=sequence_hash_col_name,
        clustering_configs=clustering_configs,
    )
    df_with_generic_cluster_column = add_pn_unit_cluster_column(
        df_with_specific_cluster_columns, cluster_choice_suffixes=cluster_choice_suffixes
    )

    if output_path is None:
        return df_with_generic_cluster_column
    else:
        logger.info(f"DataFrame with clustering information saved to {output_path}")
        df_with_generic_cluster_column.to_parquet(output_path)


if __name__ == "__main__":
    fire.Fire(cluster_and_annotate_clusters)
