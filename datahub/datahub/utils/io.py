import gzip
import hashlib
import pickle
from functools import wraps
from os import PathLike
from pathlib import Path
from typing import Any, Callable, TextIO

import biotite.structure as struc
import numpy as np
from cifutils.constants import (
    AA_LIKE_CHEM_TYPES,
    ATOMIC_NUMBER_TO_UPPERCASE_ELEMENT,
    DNA_LIKE_CHEM_TYPES,
    POLYPEPTIDE_D_CHEM_TYPES,
    POLYPEPTIDE_L_CHEM_TYPES,
    RNA_LIKE_CHEM_TYPES,
)
from cifutils.utils.residue_utils import get_chem_comp_type

from datahub.utils.misc import convert_pn_unit_iids_to_pn_unit_ids, extract_transformation_id_from_pn_unit_iid, logger


def open_file(filename: PathLike) -> TextIO:
    """Open a file, handling gzipped files if necessary."""
    filename = Path(filename)
    # ...assert that the file exists
    assert filename.exists(), f"File {filename} does not exist"
    # ...open the file for reading, accepting either gzipped or plaintext files
    if filename.suffix == ".gz":
        return gzip.open(filename, "rt")
    return filename.open("r")


def cache_based_on_subset_of_args(cache_keys: list[str], maxsize: int | None = None) -> Callable:
    """
    Decorator to cache function results based on a subset of its keyword arguments.
    Most helpful when some arguments may be unhashable types (e.g., dictionaries, AtomArray).

    If the value of any of the cache keys is None, the function is executed and the result is not cached.

    Note:
        The wrapped function must use keyword arguments for those specified in `cache_keys`.
        Positional arguments are not supported for cache key extraction.

    Args:
        cache_keys (List[str]): The names of the keyword arguments to use as the cache key.
        maxsize (Optional[int]): The maximum number of entries to store in the cache.
            If None, the cache size is unlimited.

    Returns:
        Callable: A decorator that caches the function results based on the specified keyword arguments.

    Example:
        @cache_based_on_subset_of_args(['arg1'], maxsize=2)
        def function(*, arg1, arg2):
            return arg1 + arg2

        result1 = function(arg1=1, arg2=2)  # Caches with key 1
        result2 = function(arg1=1, arg2=3)  # Retrieves from cache
    """

    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_order: list[tuple[Any, ...]] = []  # To track the order of keys for eviction

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract the cache key values from kwargs
            key_values = tuple(kwargs.get(key) for key in cache_keys)

            # Check if any of the key values are None
            if None in key_values:
                # Bypass caching if any key value is None
                return func(*args, **kwargs)

            # Use the key values to form a unique cache key
            cache_key = tuple(key_values)

            if cache_key not in cache:
                # Evict the oldest entry if the cache is full
                if maxsize is not None and len(cache) >= maxsize:
                    oldest_key = cache_order.pop(0)
                    del cache[oldest_key]

                # Cache the result
                cache[cache_key] = func(*args, **kwargs)
                cache_order.append(cache_key)
            return cache[cache_key]

        return wrapper

    return decorator


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

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
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

            # ...and save the result to cache file, creating directories if necessary
            cache_file.parent.mkdir(parents=True, exist_ok=True)
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


def convert_af3_model_output_to_atom_array(
    atom_to_token_map: np.ndarray[int],
    pn_unit_iids: np.ndarray[str],
    decoded_restypes: np.ndarray[str],
    xyz: np.ndarray,
    elements: np.ndarray[int | str],
    token_is_atomized: np.ndarray[bool] = None,
) -> struc.AtomArray:
    """
    Create an AtomArray from AlphaFold-3-type model outputs.
    Specific to AF-3; may not work with other formats.

    Parameters:
        - atom_to_token_map (np.ndarray): Mapping from atoms to tokens [n_atom]
        - pn_unit_iids (np.ndarray): PN unit IID's for each token [n_token]
        - decoded_restype (np.ndarray): Decoded residue types for each token [n_token]
        - xyz (np.ndarray): Coordinates of atoms [n_atom, 3]
        - elements (np.ndarray): Element types for each atom [n_atom]
        - token_is_atomized (np.ndarray, optional): Flags indicating if tokens are atomized [n_token]. If not provided
            or None, residues with a single atom are considered atomized.

    Returns:
        - AtomArray: Constructed AtomArray.
    """
    atom_array = None
    chain_iid_residue_counts = {}

    # If elements are integers, convert them to strings (since that's what we get from the CCD, and it better matches what CIF files expect)
    if np.issubdtype(type(elements[0]), np.integer):
        elements = np.array([ATOMIC_NUMBER_TO_UPPERCASE_ELEMENT[element] for element in elements])

    for global_res_idx, res_name in enumerate(decoded_restypes):
        # Get atoms corresponding to the residue
        atom_indices_in_residue = np.where(atom_to_token_map == global_res_idx)[0]

        # ...check if we're dealing with an atomized token
        if token_is_atomized is not None:
            is_atom = token_is_atomized[global_res_idx]
        else:
            is_atom = len(atom_indices_in_residue) == 1

        #  ...compute the residue ID
        pn_unit_iid = pn_unit_iids[global_res_idx]
        if pn_unit_iid not in chain_iid_residue_counts:
            chain_iid_residue_counts[pn_unit_iid] = 1
        elif not is_atom:
            # Only increment the residue count if we're not dealing with an atomized token (we put all atomized tokens in the same residue, like the PDB)
            chain_iid_residue_counts[pn_unit_iid] += 1
        res_id = chain_iid_residue_counts[pn_unit_iid]

        if is_atom:
            coord = xyz[atom_indices_in_residue.item()]

            # UNL is "Unknown Ligand" in the CCD
            element = elements[atom_indices_in_residue].item()

            def atom_name_exists(atom_name: str) -> bool:
                return (
                    atom_array[
                        (atom_array.pn_unit_iid == pn_unit_iid)
                        & (atom_array.res_id == res_id)
                        & (atom_array.atom_name == atom_name)
                    ].array_length()
                    > 0
                )

            # Create the atom name and ensure it's unique within the residue (so that we can give all the atoms the same ID)
            atom_name = element
            if atom_name_exists(atom_name):
                atom_name = next(
                    f"{element}{atom_count}"
                    for atom_count in range(2, len(atom_array) + 1)
                    if not atom_name_exists(f"{element}{atom_count}")
                )

            atom = struc.Atom(coord, res_name="UNL", element=element, atom_name=atom_name)
            residue_atom_array = struc.array([atom])
        else:
            chem_type = get_chem_comp_type(res_name)

            # Get the atom array of the residue from the CCD
            residue_atom_array = struc.info.residue(res_name)

            # Set the elements to uppercase for consistency
            residue_atom_array.element = np.array([x.upper() for x in residue_atom_array.element])

            # If needed, remove type-specific atoms (e.g., OXT in polypeptides, O3' in RNA or DNA) for residues participating in inter-residue bonds
            # If we are at a terminal residue, we don't want to remove these leaving groups
            residue_atom_array = filter_residue_atoms(
                residue_atom_array=residue_atom_array, chem_type=chem_type, elements=elements[atom_indices_in_residue]
            )

            # Empty coordinates to avoid unexpected behavior
            residue_atom_array.coord = np.full((residue_atom_array.array_length(), 3), np.nan)

            # Wipe the bond information (we are better off letting PyMOL infer the bonds)
            residue_atom_array.bonds = None

        # Set the coordinates
        residue_atom_array.coord = xyz[atom_indices_in_residue]

        # Get the chain_iid, chain_id, and transformation_id
        pn_unit_id = convert_pn_unit_iids_to_pn_unit_ids([pn_unit_iid])[0]
        transformation_id = extract_transformation_id_from_pn_unit_iid(pn_unit_iid)

        # Set the annotations (for our purposes, chains and pn_units are the same)
        residue_atom_array.set_annotation("chain_id", np.full(residue_atom_array.array_length(), pn_unit_id))
        residue_atom_array.set_annotation("pn_unit_id", np.full(residue_atom_array.array_length(), pn_unit_id))
        residue_atom_array.set_annotation("chain_iid", np.full(residue_atom_array.array_length(), pn_unit_iid))
        residue_atom_array.set_annotation("pn_unit_iid", np.full(residue_atom_array.array_length(), pn_unit_iid))
        residue_atom_array.set_annotation(
            "transformation_id", np.full(residue_atom_array.array_length(), transformation_id)
        )

        # Everything is full occupancy
        residue_atom_array.set_annotation("occupancy", np.full(residue_atom_array.array_length(), 1.0))

        # Set the residue ID
        residue_atom_array.set_annotation("res_id", np.full(residue_atom_array.array_length(), res_id))

        if atom_array is None:
            atom_array = residue_atom_array
        else:
            atom_array += residue_atom_array

    return atom_array


def filter_residue_atoms(
    residue_atom_array: struc.AtomArray, chem_type: str, elements: np.ndarray[str]
) -> struc.AtomArray:
    """
    Filter out unwanted atoms from a residue (e.g.., hydrogens, leaving groups)

    Parameters:
        - residue_atom_array (struc.AtomArray): The AtomArray to filter.
        - chem_type (str): Type of the chemical chain.
        - elements (np.array): Element types (as strings, e.g., "C") for each atom in the residue.

    Returns:
        - struc.AtomArray: Filtered AtomArray.
    """
    # ...capitalize the chemical type
    chem_type = chem_type.upper()

    # ...remove hydrogens and deuteriums
    residue_atom_array = residue_atom_array[(residue_atom_array.element != "H") & (residue_atom_array.element != "D")]

    # If the arrays match, we return the residue as-is
    if len(residue_atom_array) == len(elements) and all(elements == residue_atom_array.element):
        return residue_atom_array

    # ...otherwise, we will try to remove specific atoms until the arrays match
    if (
        chem_type in AA_LIKE_CHEM_TYPES
        or chem_type in POLYPEPTIDE_L_CHEM_TYPES
        or chem_type in POLYPEPTIDE_D_CHEM_TYPES
    ):
        # ...try removing OXT in non-terminal polypeptides
        candidate_residue_atom_array = residue_atom_array[residue_atom_array.atom_name != "OXT"]
        if len(candidate_residue_atom_array) == len(elements) and all(elements == candidate_residue_atom_array.element):
            return candidate_residue_atom_array

    elif chem_type in RNA_LIKE_CHEM_TYPES or chem_type in DNA_LIKE_CHEM_TYPES:
        # ...try removing OP3 in RNA or DNA
        candidate_residue_atom_array = residue_atom_array[residue_atom_array.atom_name != "OP3"]
        if len(candidate_residue_atom_array) == len(elements) and all(elements == candidate_residue_atom_array.element):
            return candidate_residue_atom_array

    # ...as a last resort, try and match the elements by sliding a window over the residue
    for start in range(len(residue_atom_array) - len(elements) + 1):
        current_slice = residue_atom_array[start : start + len(elements)]
        if all(elements == current_slice.element):
            return current_slice

    raise ValueError(
        f"Could not find a matching AtomArray for residue {residue_atom_array.res_name[0]} with elements {elements}"
    )
