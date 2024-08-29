import logging
import signal
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from data.data_preprocessing_constants import ENTRIES_TO_EXCLUDE_FOR_PRE_PROCESSING
from data.data_preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def process_pdb_ids(
    pdb_ids: List[str],
    preprocessor: DataPreprocessor,
    csv_dir: str,
    print_progress: bool = True,
    log_errors: bool = False,
    error_log_file: str = None,
    timeout_seconds: int = 30 * 60,  # Timeout parameter in seconds (default 30 minutes)
) -> None:
    """
    Process a list of PDB IDs and save each entry as a CSV file.

    Arguments:
    - pdb_ids (List[str]): List of PDB IDs to process.
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

    total_pdb_ids = len(pdb_ids)
    num_examples_processed = 0
    start_time = time.time()

    def process_single_pdb_id(pdb_id: str, index: int) -> None:
        nonlocal num_examples_processed  # Explicitly declaring nonlocal variable

        try:
            if (csv_dir / f"{pdb_id}.csv").exists():
                return
            else:
                try:
                    if print_progress:
                        logger.info(
                            f"#----- Processing PDB ID {pdb_id}: {index}/{total_pdb_ids} ({num_examples_processed} parsed by this worker) -----# ({datetime.now()})"
                        )

                    if pdb_id in ENTRIES_TO_EXCLUDE_FOR_PRE_PROCESSING:
                        logger.warning(f"Skipping PDB ID {pdb_id} because it is in the exclusion list.")
                    else:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout_seconds)

                        try:
                            records = preprocessor.get_rows(pdb_id)
                            if records is not None:
                                entry_df = pd.DataFrame(records)
                                entry_df.to_csv(csv_dir / f"{pdb_id}.csv", index=False)

                                del entry_df
                            del records

                        finally:
                            signal.alarm(0)  # Disable the alarm

                except TimeoutException:
                    logger.warning(
                        f"Timeout: Processing PDB ID {pdb_id} exceeded the allotted time. Moving to the next."
                    )
                    if log_errors:
                        with open(error_log_file, "a") as error_log_f:
                            error_log_f.write(f"Timeout: Processing PDB ID {pdb_id} exceeded the allotted time.\n")
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"Error processing PDB ID {pdb_id}: {e}\n{tb}")
                    if log_errors:
                        with open(error_log_file, "a") as error_log_f:
                            error_log_f.write(f"Error processing PDB ID {pdb_id}: {e}\n{tb}\n")
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
            logger.error(f"Exception occurred during processing: {main_e}")

    for index, pdb_id in enumerate(pdb_ids):
        process_single_pdb_id(pdb_id, index)


def get_all_pdb_ids(
    base_cif_dir: Path,
    file_extension: str = ".cif.gz",
) -> List[str]:
    """Get all PDB IDs from a directory of PDB files with a given extension."""
    files = base_cif_dir.glob(f"**/*{file_extension}")
    pdb_ids = [file.stem.split(".")[0] for file in files]
    return pdb_ids
