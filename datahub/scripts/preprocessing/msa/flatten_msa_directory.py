"""
Example usage:
    python flatten_msa_directory.py --source_dir /projects/msa/rf2aa_af3/2024_08_12/processed --target_dir /projects/msa/rf2aa_af3/missing_msas_through_2024_08_12 --extension .msa0.a3m --compress
"""

import gzip
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

import fire
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_file(file_args: dict):
    """
    Process a single file by copying it to the target directory and optionally compressing it.

    Args:
        file_args (dict): Dictionary containing the source file, target directory, and compression flags.
    """
    source_file = file_args["source_file"]
    target_path = file_args["target_path"]
    compress = file_args["compress"]
    is_compressed = file_args["is_compressed"]

    new_file_path = target_path / source_file.name

    # ...if file already exists, skip it
    if new_file_path.exists():
        return 0

    # ...otherwise
    if compress and not is_compressed:
        # ...compress and copy the file
        compressed_path = new_file_path.with_suffix(new_file_path.suffix + ".gz")

        # ...check if the compressed file already exists
        if compressed_path.exists():
            return 0
        else:
            with source_file.open("rb") as f_in, gzip.open(str(compressed_path), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        # ...or copy the file without compression, if it doesn't exist
        if new_file_path.exists():
            return 0

        shutil.copy2(str(source_file), str(new_file_path))

    return 1  # Return 1 to count the processed file


def flatten_directory(source_dir: str, target_dir: str, extension: str = ".a3m", compress: bool = False):
    """
    Flatten the directory structure, moving all files with the specified extension to the target directory.
    Optionally compress files if they're not already compressed based on the extension.

    Args:
        source_dir (str): Path to the source directory containing subdirectories.
        target_dir (str): Path to the target directory where files will be moved.
        extension (str): File extension to search for (default: '.a3m').
        compress (bool): Whether to compress files if they're not already compressed (default: False).
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # ...ensure the extension starts with a dot
    if not extension.startswith("."):
        extension = "." + extension

    # ...check if the extension indicates the file is already compressed
    compressed_extensions = [".gz", ".bz2", ".xz", ".zip"]
    is_compressed = any(extension.lower().endswith(ext) for ext in compressed_extensions)

    # ...create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Counter for moved and compressed files
    moved_files = 0

    # ...use recursive glob to find all files with the specified extension
    files_to_process = list(source_path.glob(f"**/*{extension}"))
    logger.info(f"Found {len(files_to_process)} files with extension '{extension}' in {source_dir}")

    # ...prepare arguments for each file
    file_arg_list = [
        {"source_file": file, "target_path": target_path, "compress": compress, "is_compressed": is_compressed}
        for file in files_to_process
    ]

    # ...define multiprocessing paramters
    num_workers = min(cpu_count(), 20)
    chunksize = min(100, max(1, len(file_arg_list) // num_workers), len(file_arg_list))

    # ...and process the files in parallel
    with Pool(processes=num_workers) as pool:
        moved_files = sum(
            tqdm(
                pool.imap(process_file, file_arg_list, chunksize=chunksize),
                total=len(file_arg_list),
            )
        )

    logger.info(f"Moved {moved_files} files with extension '{extension}' to {target_dir}")
    if compress and not is_compressed:
        logger.info(f"Compressed {moved_files} files")


if __name__ == "__main__":
    fire.Fire(flatten_directory)
