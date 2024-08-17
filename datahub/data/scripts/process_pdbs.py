import logging
import random
import time
from pathlib import Path
from typing import List, Union

import fire

from data.data_preprocessor import DataPreprocessor
from data.scripts.confscript import get_all_pdb_ids, process_pdb_ids

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(
    base_cif_dir: str = "/databases/rcsb/cif",
    pdb_selection: Union[str, List[str]] = "all",
    out_dir: str = "/projects/ml/RF2_allatom/data_preprocessing",
    task_id: int = 0,
    num_tasks: int = 1,
    log_errors: bool = True,
    print_progress: bool = True,
    timeout_seconds: int = 30 * 60,  # Timeout parameter in seconds (default 30 minutes)
    **kwargs,
):
    logger.info(f"Launching task {task_id + 1} of {num_tasks}.")

    # We first get a list of all PDB IDs from the CIF files.
    all_pdb_ids = get_all_pdb_ids(base_cif_dir=Path(base_cif_dir), file_extension=".cif.gz")

    # Determine the PDB IDs to process based on the selection mode
    if pdb_selection == "all":
        pdb_ids_to_process = all_pdb_ids
    elif isinstance(pdb_selection, int) or (isinstance(pdb_selection, str) and pdb_selection.isdigit()):
        num_samples = int(pdb_selection)
        random.seed(42)
        pdb_ids_to_process = random.sample(all_pdb_ids, num_samples)
    elif isinstance(pdb_selection, list):
        pdb_ids_to_process = pdb_selection
    else:
        pdb_ids_to_process = pdb_selection.split(",")

    # Initialize the DataPreprocessor with remaining kwargs
    preprocessor = DataPreprocessor(base_cif_dir=base_cif_dir, **kwargs)

    # Create the output directory, if it doesn't already exist
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Ensure the output directory exists
    if not out_dir_path.exists():
        raise Exception(f"Output directory {out_dir_path} does not exist.")

    # Filter out PDB IDs that have been already processed
    processed_pdb_ids = set(get_all_pdb_ids(out_dir_path / "csv", file_extension=".csv"))
    unprocessed_pdb_ids = [pdb_id for pdb_id in pdb_ids_to_process if pdb_id not in processed_pdb_ids]
    logger.info(
        f"Task {task_id}: {len(unprocessed_pdb_ids):,} PDB IDs to process (vs. {len(pdb_ids_to_process):,} originally)."
    )

    # Randomly shuffle the list of PDB IDs according to the task ID
    seed = task_id + time.time()
    logger.info(f"Task {task_id}: Random seed: {seed}")
    random.seed(seed)
    random.shuffle(unprocessed_pdb_ids)

    # Create name for log file
    error_log_path = out_dir_path / f"error_log_{task_id}.txt"

    # Create a "csv" directory within out_dir, if it doesn't already exist
    csv_dir = out_dir_path / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Process the list of PDB IDs, saving the output of each entry as a separate dataframe (we later merge)
    start_time = time.time()
    process_pdb_ids(
        pdb_ids=unprocessed_pdb_ids,
        preprocessor=preprocessor,
        csv_dir=csv_dir,
        print_progress=print_progress,
        error_log_file=error_log_path,
        log_errors=log_errors,
        timeout_seconds=timeout_seconds,
    )
    end_time = time.time()
    logger.info(f"Task {task_id}: Time taken to process PDB IDs: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    fire.Fire(run_pipeline)
