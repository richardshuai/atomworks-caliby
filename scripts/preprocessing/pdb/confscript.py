import logging
import signal
import time
import traceback
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Never

import pandas as pd

from atomworks.ml.preprocessing.constants import ENTRIES_TO_EXCLUDE_FOR_PRE_PROCESSING
from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import DataPreprocessor

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame) -> Never:
    raise TimeoutException


def generate_csv_files_from_paths(
    example_paths: list[PathLike],
    preprocessor: DataPreprocessor,
    csv_dir: str,
    print_progress: bool = True,
    log_errors: bool = False,
    error_log_file: str | None = None,
    timeout_seconds: int = 30 * 60,  # Timeout parameter in seconds (default 30 minutes)
) -> None:
    """
    Process a list of paths to structure files and save each entry as a CSV file.

    Arguments:
    - example_paths (List[str]): List of paths to structure files to process.
    - preprocessor (DataPreprocessor): An instance of the DataPreprocessor class.
    - error_log_file (str): Path to the log file for recording PDB IDs that raise exceptions.
    - csv_dir (str): Path to the directory where the CSV files will be saved, one for each entry.
    - print_progress (bool): Whether to log progress. Default is True.
    - timeout_seconds (int): Timeout for processing each PDB ID in seconds. Default is 1800 seconds (30 minutes).

    Returns: None; saves examples as CSV files.
    """
    if log_errors:
        assert error_log_file is not None, "Error log file must be provided if logging errors."
        log_file_path = Path(error_log_file)
        log_file_path.touch(exist_ok=True)
        logger.info(f"Error log file created at {log_file_path}")

    total_example_paths = len(example_paths)
    num_examples_processed = 0
    num_errors = 0
    start_time = time.time()

    def generate_single_csv_from_path(example_path: Path, index: int) -> None:
        nonlocal num_examples_processed  # Explicitly declaring nonlocal variable
        nonlocal num_errors

        try:
            example_stem = example_path.stem.split(".")[0]
            csv_out_path = csv_dir / f"{example_stem}.csv"
            if (csv_out_path).exists():
                return
            else:
                try:
                    if print_progress:
                        logger.info(
                            f"#----- Processing example {example_path}: {index}/{total_example_paths} ({num_examples_processed} parsed by this worker) -----# ({datetime.now()})"
                        )

                    if example_stem in ENTRIES_TO_EXCLUDE_FOR_PRE_PROCESSING:
                        logger.warning(f"Skipping example {example_path} because it is in the exclusion list.")
                    else:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout_seconds)

                        try:
                            records = preprocessor.get_rows(example_path)
                            if records is not None:
                                entry_df = pd.DataFrame(records)
                                entry_df.to_csv(csv_out_path, index=False)

                                del entry_df
                            del records

                        finally:
                            signal.alarm(0)  # Disable the alarm

                except TimeoutException:
                    logger.warning(
                        f"Timeout: Processing example {example_path} exceeded the allotted time. Moving to the next."
                    )
                    if log_errors:
                        with open(error_log_file, "a") as error_log_f:
                            error_log_f.write(
                                f"Timeout: Processing example {example_path} exceeded the allotted time.\n"
                            )
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"Error processing example {example_path}: {e}\n{tb}")
                    if log_errors:
                        with open(error_log_file, "a") as error_log_f:
                            error_log_f.write(f"Error processing example {example_path}: {e}\n{tb}\n")
                else:
                    end_time = time.time()
                    num_examples_processed += 1

                    if (num_examples_processed + 1) % 10 == 0:
                        # Print examples per second
                        seconds_per_example = (end_time - start_time) / num_examples_processed
                        logger.info(
                            f"Processed {num_examples_processed} examples ({seconds_per_example:.2f} seconds/example)"
                        )

        except Exception as main_e:
            logger.error(f"Exception occurred during processing example {example_path}: {main_e}")
            num_examples_processed += 1
            num_errors += 1
            logger.info(
                f"Total number of errors on this worker: {num_errors}, " f"or {num_errors / num_examples_processed:.2%}"
            )

    for index, path in enumerate(example_paths):
        generate_single_csv_from_path(Path(path), index)


def get_all_files_in_dir(
    base_dir: Path,
    file_extension: str = ".cif.gz",
    only_stem: bool = True,
) -> list[Path]:
    """Get all file paths or file stems with a given extension from a given directory."""
    files = base_dir.glob(f"**/*{file_extension}")
    if only_stem:
        return [file.stem.split(".")[0] for file in files]
    else:
        return list(files)
