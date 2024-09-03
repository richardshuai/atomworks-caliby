"""
Genereate a single PN unit-level DataFrame, `pn_units_df`, from multiple CSV files output by the pipeline.
After concatenating the CSV files, performs some basic preprocessing to generate the `pn_units_df` DataFrame.

Example usage:
```bash
python generate_pn_units_df.py --input_dir /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_07_01/csv --output_path /projects/ml/RF2_allatom/data_preprocessing/PDB_2024_07_01/pn_units_df
```
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from os import PathLike
from pathlib import Path

import fire
import pandas as pd
from cifutils.enums import ChainType
from tqdm import tqdm

from datahub.common import generate_example_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_csv_file(csv_path: PathLike):
    """Reads a single CSV file into a DataFrame, ensuring correct data types."""

    # Specify the data types for the relevant columns to ensure they are read as strings
    dtype_spec = {
        "pdb_id": str,
        "assembly_id": str,
        "q_pn_unit_transformation_id": str,
        "q_pn_unit_non_polymer_res_names": str,
        "q_pn_unit_ec_numbers": str,
        "q_pn_unit_processed_entity_canonical_sequence_hash": str,
        "q_pn_unit_processed_entity_non_canonical_sequence_hash": str,
    }

    # Read the CSV file with the specified data types
    df = pd.read_csv(csv_path, dtype=dtype_spec)

    return df


def concatenate_csv_files(input_dir: PathLike, output_path: PathLike = None, max_workers: int = 8, timeout: int = 30):
    """
    Concatenate a directory of CSV files into a single DataFrame.
    Each CSV file should be created by the `process_pdbs` pipeline.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path) if output_path is not None else None

    # Check if input_dir exists and is a directory
    if not input_dir.is_dir():
        raise ValueError(f"{input_dir} is not a valid directory")

    # List all CSV files in the directory
    csv_files = list(input_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {input_dir}. Concatenating...")

    # Read CSV files in parallel using ThreadPoolExecutor
    dataframes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_csv_file, csv_file): csv_file for csv_file in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading CSV files"):
            try:
                dataframes.append(future.result(timeout=timeout))
            except TimeoutError:
                logger.warning(f"Timeout error: {futures[future]} took longer than {timeout} seconds to read.")
            except Exception as e:
                logger.warning(f"Error reading file {futures[future]}: {e}")

    # Remove any None values that may have been appended due to errors or timeouts
    dataframes = [df for df in dataframes if df is not None]

    # Concatenate all DataFrames
    concatenated_df = pd.concat(dataframes, axis=0)

    logger.info(
        f"Concatenated {len(dataframes)} DataFrames into a single DataFrame with {len(concatenated_df)} rows. Saving to file..."
    )

    return concatenated_df


def generate_pn_units_df(
    input_dir: PathLike, output_path: PathLike | None = None, max_workers: int = 8, timeout: int = 30
):
    # Convert to Path, if given
    output_path = Path(output_path) if output_path is not None else None

    # Concatenate the CSV files in the input directory
    concatenated_df = concatenate_csv_files(input_dir, output_path, max_workers, timeout)

    q_pn_unit_type_col = "q_pn_unit_type"
    # Initialize columns for n_prot, n_nuc, n_ligand (required for AF-3 data sampling strategy)
    concatenated_df["n_prot"] = (
        concatenated_df[q_pn_unit_type_col].apply(lambda x: ChainType(x).is_protein()).astype(int)
    )
    concatenated_df["n_nuc"] = (
        concatenated_df[q_pn_unit_type_col].apply(lambda x: ChainType(x).is_nucleic_acid()).astype(int)
    )
    concatenated_df["n_ligand"] = (
        concatenated_df[q_pn_unit_type_col].apply(lambda x: ChainType(x).is_non_polymer()).astype(int)
    )

    # Add the example_id column (required for testing and reproducibility)
    concatenated_df["example_id"] = concatenated_df.apply(
        lambda x: generate_example_id(
            ["pdb", "pn_units"],
            x["pdb_id"],
            x["assembly_id"],
            [x["q_pn_unit_iid"]],
        ),
        axis=1,
    )

    # Turn the deposition_date and release_date columns into datetime objects
    concatenated_df["deposition_date"] = pd.to_datetime(concatenated_df["deposition_date"])
    concatenated_df["release_date"] = pd.to_datetime(concatenated_df["release_date"])

    # Count duplicates
    n_duplicates = concatenated_df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates} duplicate rows in the concatenated DataFrame")
        # Deduplicate
        concatenated_df = concatenated_df.drop_duplicates()

    if output_path is not None:
        # Save the concatenated DataFrame to the specified location...
        if output_path.suffix != ".parquet":
            output_path = output_path.with_suffix(".parquet")

        # Save the concatenated DataFrame to the specified location as a parquet (requires pyarrow)
        concatenated_df.to_parquet(output_path, index=False)
        logger.info(f"Concatenated DataFrame saved to {output_path}")
    else:
        # ...or return the DataFrame if no output path is specified
        return concatenated_df


if __name__ == "__main__":
    fire.Fire(generate_pn_units_df)
