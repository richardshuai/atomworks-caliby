from os import PathLike
from pathlib import Path

import fire
import pandas as pd


def add_and_merge_token_counts(
    input_df_path: PathLike, output_df_path: PathLike, all_token_counts_path: PathLike
) -> None:
    """
    Adds a 'n_tokens_total' column to the all_token_counts DataFrame, merges it with the input DataFrame,
    and saves the result to the specified output path.

    Args:
        input_df_path (PathLike): Path to the input DataFrame.
        output_df_path (PathLike): Path to save the output DataFrame.
        all_token_counts_path (PathLike): Path to the all_token_counts DataFrame.

    Example:
        python add_and_merge_token_counts.py /path/to/input_df.parquet /path/to/output_df.parquet /path/to/all_token_counts.parquet
    """

    # Ensure paths are Path objects
    input_df_path = Path(input_df_path)
    output_df_path = Path(output_df_path)
    all_token_counts_path = Path(all_token_counts_path)

    # Load all_token_counts DataFrame
    all_token_counts_concat = pd.read_parquet(all_token_counts_path)

    # Add n_tokens_total column, if it doesn't already exist
    if "n_tokens_total" not in all_token_counts_concat.columns:
        all_token_counts_concat["n_tokens_total"] = (
            all_token_counts_concat["n_atomized_tokens"] + all_token_counts_concat["n_non_atomized_tokens"]
        )

    # Load the input DataFrame (e.g., `pn_units_df`)
    input_df = pd.read_parquet(input_df_path)

    # Merge input_df with all_token_counts_concat on 'pdb_id' and 'assembly_id' (which must be present in both DataFrames, and named the same)
    merged_df = input_df.merge(all_token_counts_concat, on=["pdb_id", "assembly_id"])

    # Save the merged DataFrame to disk
    merged_df.to_parquet(output_df_path)


if __name__ == "__main__":
    fire.Fire(add_and_merge_token_counts)
