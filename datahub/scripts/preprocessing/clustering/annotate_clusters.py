"""
Adds a cluster column to the PN units dataframe.
Note that the cluster column for interfaces is handled by `generate_interfaces_df.py`, after annotating the PN units dataframe with clusters.
"""

import logging
from os import PathLike

import fire
import pandas as pd
from cifutils.enums import ChainType

logger = logging.getLogger(__name__)

"""
From the AF3 supplement:
    Chain-based clustering occur at 40% sequence homology for proteins, 100% homology for nucleic acids, 
    100% homology for peptides (<10 residues), and according to CCD identity for small molecules (e.g., only identical molecules share a cluster)
"""
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


def add_pn_unit_cluster_column(
    pn_units_df: str | PathLike | pd.DataFrame,
    cluster_choice_suffixes: dict = AF3_CLUSTER_CHOICE_SUFFIXES,
    replace_df: bool = False,
) -> pd.DataFrame:
    """
    Add a cluster column to the PN units dataframe based on the query PN unit type.

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


if __name__ == "__main__":
    fire.Fire(add_pn_unit_cluster_column)
