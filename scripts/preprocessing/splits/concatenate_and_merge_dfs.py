import logging
from os import PathLike
from pathlib import Path

import fire
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def concatenate_parquet_files(input_dir: PathLike, output_file: PathLike) -> pd.DataFrame:
    """
    Concatenates all Parquet files in the specified directory into a single DataFrame.

    Args:
        input_dir (str): Directory containing the Parquet files to concatenate.

    Returns:
        pd.DataFrame: The concatenated DataFrame.
    """
    input_dir = Path(input_dir)
    parquet_files = list(input_dir.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No Parquet files found in directory: {input_dir}")

    logger.info(f"Concatenating {len(parquet_files)} Parquet files in {input_dir}")
    dfs = [pd.read_parquet(file) for file in parquet_files]
    concatenated_df = pd.concat(dfs, ignore_index=True)

    # Save the concatenated DataFrame to the output_file
    output_file = Path(output_file)
    concatenated_df.to_parquet(output_file)
    logger.info(f"Concatenated DataFrame saved to {output_file}")


def merge_dfs(
    primary_file: PathLike, secondary_file: PathLike, merge_columns: list, output_file: PathLike = None
) -> None:
    """
    Merges two Parquet files on the specified columns and saves the resulting DataFrame.

    Args:
        primary_file (PathLike): Path to the primary Parquet file.
        secondary_file (PathLike): Path to the secondary Parquet file.
        merge_columns (list): List of column names to merge on.
        output_file (PathLike): Path to save the merged DataFrame.
    """
    primary_file = Path(primary_file)
    secondary_file = Path(secondary_file)
    output_file = Path(output_file) if output_file else primary_file.with_name(primary_file.stem + "_merged.parquet")

    # Load the primary and secondary DataFrames
    primary_df = pd.read_parquet(primary_file)
    secondary_df = pd.read_parquet(secondary_file)

    # Merge the DataFrames
    merged_df = pd.merge(primary_df, secondary_df, on=merge_columns, how="left")

    # Save the merged DataFrame to the output file, if given, or to the primary_file with '_merged' appended
    merged_df.to_parquet(output_file)
    logger.info(f"Merged DataFrame saved to {output_file}")


def main():
    fire.Fire({"concatenate": concatenate_parquet_files, "merge": merge_dfs})


if __name__ == "__main__":
    main()
