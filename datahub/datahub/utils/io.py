import gzip
import hashlib
import pickle
from functools import wraps
from os import PathLike
from pathlib import Path
from typing import TextIO

from datahub.utils.misc import logger


def open_file(filename: PathLike) -> TextIO:
    """Open a file, handling gzipped files if necessary."""
    filename = Path(filename)
    # ...assert that the file exists
    assert filename.exists(), f"File {filename} does not exist"
    # ...open the file for reading, accepting either gzipped or plaintext files
    if filename.suffix == ".gz":
        return gzip.open(filename, "rt")
    return filename.open("r")


def cache_to_disk_as_pickle(cache_dir: PathLike | None = None, use_gzip: bool = True, directory_depth: int = 2):
    """
    A decorator to cache the results of a function to disk as a pickle file.

    Creates a unique cached pickle file for each set of function arguments using an MD5 hash.
    If the cache file exists, the result is loaded from the file. Otherwise, the
    function is called, and the result is saved to the cache file.

    If `cache_dir` is `None`, caching is disabled and the function is always executed.

    Args:
        cache_dir (PathLike or None): The directory where cache files will be stored, or
            `None` to disable caching.
        use_gzip (bool): Whether to use gzip compression for the cache files.
        directory_depth (int): The depth of the directory structure for sharding cache files.

    Returns:
        function: The wrapped function with optional disk caching enabled.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if cache_dir is None:
                # If caching is disabled, always execute the function
                return func(self, *args, **kwargs)

            # ...create cache directory if it doesn't exist
            cache_dir_path = Path(cache_dir)
            cache_dir_path.mkdir(parents=True, exist_ok=True)

            # ...create a unique cache file path based on the MD5 hash of function arguments
            args_repr = f"{args}_{kwargs}"
            hash_hex = hashlib.md5(args_repr.encode()).hexdigest()
            file_extension = ".pkl.gz" if use_gzip else ".pkl"
            cache_file = get_sharded_file_path(cache_dir, hash_hex, file_extension, directory_depth)

            # ...check if cache file exists
            open_func = gzip.open if use_gzip else open
            if cache_file.exists():
                try:
                    # ...try to load the result from cache file
                    with open_func(cache_file, "rb") as f:
                        result = pickle.load(f)
                    return result

                except Exception as e:
                    # ...fallback to executing the function, with a warning
                    logger.error(f"Error loading cache file {cache_file}: {e}")

            # ...if cache file doesn't exist, execute the function
            result = func(self, *args, **kwargs)

            # ...and save the result to cache file
            with open_func(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator


def get_sharded_file_path(base_dir: Path, file_hash: str, extension: str, depth: int) -> Path:
    """
    Construct a nested file path based on the directory depth.

    Args:
        base_dir (Path): The base directory where the files are stored.
        file_hash (str): The hash of the file content or identifier.
        extension (str): The file extension.
        depth (int): The directory nesting depth.

    Returns:
        Path: The constructed path to the file.

    Example:
        >>> get_sharded_file_path("/path/to/cache", "abcdef123456", ".pkl", 2)
        Path("/path/to/cache/ab/cd/abcdef123456.pkl")
    """
    nested_path = Path(base_dir)
    for i in range(depth):
        nested_path /= Path(file_hash[2 * i : 2 * (i + 1)])
    return (nested_path / file_hash).with_suffix(extension)
