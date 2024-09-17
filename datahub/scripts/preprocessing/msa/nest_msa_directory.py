import logging
import shutil
from os import PathLike
from pathlib import Path

import fire
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def nest_files(input_dir: PathLike, output_dir: PathLike = None, action: str = "copy") -> None:
    """
    Nest files in a directory based on the first two and the next two characters of the filenames.

    Args:
        directory (Union[Path, os.PathLike]): The path to the directory containing the files to be nested.
        action (str): The action to perform on the files - "move" or "copy". Default is "move".
    """
    input_dir = Path(input_dir)  # Ensure directory is a Path object
    output_dir = (
        Path(output_dir) if output_dir else input_dir
    )  # Use input directory if output directory is not provided
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Ensure the directory exists
    if not input_dir.exists():
        print(f"Directory {input_dir} does not exist.")
        return

    # Validate the action argument
    if action not in ["move", "copy"]:
        print(f"Invalid action: {action}. Use 'move' or 'copy'.")
        return

    # Get list of files in the directory
    logger.info(f"Scanning directory {input_dir} for files...")
    files = [f for f in input_dir.iterdir() if f.is_file()]

    logger.info(f"Found {len(files)} files in the directory.")
    logger.info(f"Nesting files with {action}")

    for file in tqdm(files, desc="Nesting files"):
        # Extract the first two and the next two characters from the filename
        first_two = file.name[:2]
        next_two = file.name[2:4]

        # Create the nested directory path
        nested_dir = output_dir / first_two / next_two

        # Create the nested directories if they do not exist
        nested_dir.mkdir(parents=True, exist_ok=True)

        # Move or copy the file to the new nested directory
        if action == "move":
            shutil.move(str(file), str(nested_dir / file.name))
        elif action == "copy":
            shutil.copy2(str(file), str(nested_dir / file.name))

    logger.info(f"Files have been {action}d successfully.")


if __name__ == "__main__":
    fire.Fire(nest_files)
