import logging
import shutil
from os import PathLike
from pathlib import Path

import fire
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nest_files(directory: PathLike) -> None:
    """
    Nest files in a directory based on the first two and the next two characters of the filenames.

    Args:
        directory (Union[Path, os.PathLike]): The path to the directory containing the files to be nested.
    """
    directory = Path(directory)  # Ensure directory is a Path object

    # Ensure the directory exists
    if not directory.exists():
        print(f"Directory {directory} does not exist.")
        return

    # Get list of files in the directory
    logger.info(f"Scanning directory {directory} for files...")
    files = [f for f in directory.iterdir() if f.is_file()]

    logger.info(f"Found {len(files)} files in the directory.")
    logger.info("Nesting files...")

    for file in tqdm(files, desc="Nesting files"):
        # Extract the first two and the next two characters from the filename
        first_two = file.name[:2]
        next_two = file.name[2:4]

        # Create the nested directory path
        nested_dir = directory / first_two / next_two

        # Create the nested directories if they do not exist
        nested_dir.mkdir(parents=True, exist_ok=True)

        # Move the file to the new nested directory
        shutil.move(str(file), str(nested_dir / file.name))

    logger.info("Files have been nested successfully.")


if __name__ == "__main__":
    fire.Fire(nest_files)
