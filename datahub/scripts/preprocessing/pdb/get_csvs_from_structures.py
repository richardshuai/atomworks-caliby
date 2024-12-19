import logging
import random
import time
from os import PathLike
from pathlib import Path
from typing import List, Union

import fire

from datahub.preprocessing.get_pn_unit_data_from_structure import DataPreprocessor
from scripts.preprocessing.pdb.confscript import generate_csv_files_from_paths, get_all_files_in_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(
    out_dir: str,
    base_dir: PathLike = "/databases/rcsb/cif",
    # If a list or comma-separated string, selection should contain full file paths
    selection: Union[int, str, List[PathLike]] = "all",
    task_id: int = 0,
    num_tasks: int = 1,
    log_errors: bool = True,
    print_progress: bool = True,
    timeout_seconds: int = 30 * 60,  # Timeout parameter in seconds (default 30 minutes)
    from_rcsb: bool = True,
    file_extension: str = ".cif.gz",
    **kwargs,
):
    logger.info(f"Launching task {task_id + 1} of {num_tasks}.")

    # We first get a list of all file paths in the directory.
    all_example_paths = get_all_files_in_dir(Path(base_dir), file_extension, only_stem=False)

    # Determine the examples to process based on the selection mode
    if selection == "all":
        example_paths_to_process = all_example_paths
    elif isinstance(selection, int) or (isinstance(selection, str) and selection.isdigit()):
        num_samples = int(selection)
        random.seed(42)
        example_paths_to_process = random.sample(all_example_paths, num_samples)
    elif isinstance(selection, list):
        example_paths_to_process = [Path(structure_path) for structure_path in selection]
    else:
        example_paths_to_process = [Path(structure_path) for structure_path in selection.split(",")]

    # Initialize the DataPreprocessor with remaining kwargs
    preprocessor = DataPreprocessor(from_rcsb=from_rcsb, **kwargs)

    # Create the output directory, if it doesn't already exist
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Ensure the output directory exists
    if not out_dir_path.exists():
        raise Exception(f"Output directory {out_dir_path} does not exist.")

    # Filter out examples that have been already processed
    processed_example_stems = set(get_all_files_in_dir(out_dir_path / "csv", file_extension=".csv", only_stem=True))
    unprocessed_example_paths = [
        example for example in example_paths_to_process if example.stem.split(".")[0] not in processed_example_stems
    ]
    logger.info(
        f"Task {task_id}: {len(unprocessed_example_paths):,} examples to process (vs. {len(example_paths_to_process):,} originally)."
    )

    # Randomly shuffle the list of examples according to the task ID
    seed = task_id + time.time()
    logger.info(f"Task {task_id}: Random seed: {seed}")
    random.seed(seed)
    random.shuffle(unprocessed_example_paths)

    # Create name for log file
    error_log_path = out_dir_path / f"error_log_{task_id}.txt"

    # Create a "csv" directory within out_dir, if it doesn't already exist
    csv_dir = out_dir_path / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Process the list of examples, saving the output of each entry as a separate dataframe (we later merge)
    start_time = time.time()
    generate_csv_files_from_paths(
        example_paths=unprocessed_example_paths,
        preprocessor=preprocessor,
        csv_dir=csv_dir,
        print_progress=print_progress,
        error_log_file=error_log_path,
        log_errors=log_errors,
        timeout_seconds=timeout_seconds,
    )
    end_time = time.time()
    logger.info(f"Task {task_id}: Time taken to process examples: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    fire.Fire(run_pipeline)
