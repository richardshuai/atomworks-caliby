import pandas as pd
from pathlib import Path
import fire

def concatenate_parquet_files(input_dir: str) -> pd.DataFrame:
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

    dfs = [pd.read_parquet(file) for file in parquet_files]
    concatenated_df = pd.concat(dfs, ignore_index=True)
    return concatenated_df

def concatenate_and_merge(input_dir: str, pn_units_df_path: str, merge_columns: list, output_file: str = None) -> None:
    """
    Concatenates all Parquet files in the specified directory, merges the result with the pn_units_df on specified columns,
    and saves the merged DataFrame to the specified output file or to the pn_units_df_path with '_merged' appended if no output file is specified.

    Args:
        input_dir (str): Directory containing the Parquet files to concatenate.
        pn_units_df_path (str): Path to the pn_units DataFrame Parquet file.
        merge_columns (list): List of column names to merge on.
        output_file (str, optional): Path to save the merged DataFrame. Defaults to None.

    Returns:
        None
    """
    input_dir = Path(input_dir)
    pn_units_df_path = Path(pn_units_df_path)

    # Concatenate Parquet files
    concatenated_df = concatenate_parquet_files(input_dir)

    # Save the concatenated DataFrame to the input_dir
    concatenated_df.to_parquet(input_dir / "concatenated.parquet")

    # Load the pn_units DataFrame
    pn_units_df = pd.read_parquet(pn_units_df_path)

    # Merge with pn_units_df
    merged_df = pd.merge(pn_units_df, concatenated_df, on=merge_columns, how='left')

    # Determine the output file path
    if output_file is None:
        output_file = pn_units_df_path.with_name(pn_units_df_path.stem + "_merged.parquet")

    # Save the merged DataFrame to the output file
    merged_df.to_parquet(output_file)
    print(f"Merged DataFrame saved to {output_file}")

if __name__ == "__main__":
    """
    Example usage:
    python concatenate_and_merge.py --input_dir /path/to/input/directory --pn_units_df_path /path/to/pn_units_df.parquet --merge_columns pdb_id assembly_id --output_file /path/to/output/file.parquet

    If no output file is specified, the merged DataFrame will be saved to the same directory as the pn_units_df_path with '_merged' appended to the filename.
    """
    fire.Fire(concatenate_and_merge)
