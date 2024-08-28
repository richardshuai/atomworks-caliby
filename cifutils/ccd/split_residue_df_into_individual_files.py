"""
Script to split a residue library into individual files.

Example usage:
```bash
python split_residue_library_into_individual_files.py \
    --df_path "/projects/ml/RF2_allatom/cifutils_biotite/ligands_by_residue_ideal_v2024_06_10.pkl" \
    --output_dir "/projects/ml/RF2_allatom/cifutils_biotite/ccd_library"
``` 
"""

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import fire


def save_dataframe_rows(df_path: str, output_dir: str) -> None:
    """
    Save each row of the DataFrame into a separate file in the specified directory.

    Args:
        df_path (str): Path to the DataFrame stored as a pickle file.
        output_dir (str): Path to the directory where the files should be saved.
    """
    # Load the DataFrame from the pickle file
    df = pd.read_pickle(df_path)

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate over the rows of the DataFrame
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Saving rows"):
        # Get the ID for the file name
        file_id = row["id"]
        # Define the file path
        file_path = os.path.join(output_dir, f"{file_id}.pkl")
        # Save the row to a file
        row.to_pickle(file_path)


if __name__ == "__main__":
    fire.Fire(save_dataframe_rows)
