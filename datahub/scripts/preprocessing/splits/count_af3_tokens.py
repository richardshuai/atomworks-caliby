"""
Example usage: python count_af3_tokens.py --pn_units_df_path /projects/ml/RF2_allatom/datasets/pdb/2024_09_10/pn_units_df.parquet
"""

import logging
from functools import partial
from multiprocessing import Pool
from os import PathLike
from pathlib import Path

import fire
import pandas as pd
from biotite.structure import AtomArray
from cifutils import parse
from cifutils.constants import AF3_EXCLUDED_LIGANDS, STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from tqdm import tqdm

from datahub.common import exists
from datahub.datasets.parsers.base import DEFAULT_CIF_PARSER_ARGS
from datahub.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from datahub.transforms.base import Compose
from datahub.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from datahub.transforms.filters import (
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveUnresolvedPNUnits,
)
from datahub.utils.testing import get_digs_path
from datahub.utils.token import get_token_starts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("count_af3_tokens.log")],
)
logger = logging.getLogger(__name__)


def count_af3_style_tokens_from_atom_array(atom_array: AtomArray) -> dict:
    """
    Counts the number of polymer and non-polymer tokens in an AtomArray after processing it through AF-3 style transforms.

    Args:
        atom_array (AtomArray): The AtomArray to process and count tokens in.

    Returns:
        dict: A dictionary with keys 'n_atomized_tokens' and 'n_non_atomized_tokens' representing
              the number of polymer (atomized) tokens and non-polymer (non-atomized) tokens, respectively.
    """

    # ...construct the pipeline
    transforms = [
        RemoveHydrogens(),
        RemoveUnresolvedPNUnits(),  # Remove PN units that are unresolved early (and also after cropping)
        HandleUndesiredResTokens(AF3_EXCLUDED_LIGANDS),  # e.g., non-standard residues
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
    ]

    # ...run the pipeline on the AtomArray
    pipeline = Compose(transforms)
    atom_array = pipeline({"atom_array": atom_array})["atom_array"]

    # ... get token-level array
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]

    # ...count the number of polymer and non-polymer tokens
    n_atomized_tokens = len(token_level_array[token_level_array.atomize])
    n_non_atomized_tokens = len(token_level_array[~token_level_array.atomize])

    return {"n_atomized_tokens": n_atomized_tokens, "n_non_atomized_tokens": n_non_atomized_tokens}


def count_af3_style_tokens_in_atom_array_from_file(
    file_path: PathLike, assembly_ids: list[str], cache_dir: PathLike | None = None
) -> dict[str, dict[str, int]]:
    """Counts AF3-style tokens in assemblies from an atom array parsed from a file.

    Args:
        file_path (PathLike): Path to the file containing atom array data.
        assembly_ids (list[str]): List of assembly IDs to process.
        cache_dir (PathLike | None, optional): Directory for loading cached parsing data. Defaults to None.

    Returns:
        dict[str, dict[str, int]]: A dictionary with assembly IDs as keys. Each value is another dictionary with
        keys 'n_atomized_tokens' and 'n_non_atomized_tokens' (see `count_af3_style_tokens_from_atom_array`).
    """

    # Merge DEFAULT_CIF_PARSER_ARGS with cif_parser_args, overriding with the keys present in cif_parser_args
    cif_parser_args = {
        "build_assembly": assembly_ids,
        "load_from_cache": exists(cache_dir),
        "save_to_cache": False,  # we don't want to save the cache here
        "cache_dir": cache_dir,
    }
    cif_parser_args = {**DEFAULT_CIF_PARSER_ARGS, **cif_parser_args}

    try:
        # ...load the file
        result_dict = parse(
            filename=file_path,
            **cif_parser_args,
        )

        # ...loop through the assemblies and count tokens
        counts = {}
        for assembly_id in assembly_ids:
            atom_array = result_dict["assemblies"][assembly_id][0]  # First model
            counts[assembly_id] = count_af3_style_tokens_from_atom_array(atom_array)

        return counts
    except Exception as e:
        logger.error(f"ERROR processing {file_path}: {e}")
        return {}


def count_tokens_for_pdb_id(row: pd.Series, cache_dir: PathLike) -> list[dict]:
    """
    Process a single pdb_id and its associated assembly_ids to count AF3-style tokens.

    Args:
        row (pd.Series): A row of the DataFrame containing 'pdb_id' and 'assembly_id'.
        cache_dir (PathLike): The directory to use for caching parsed data.

    Returns:
        list[dict]: A list of dictionaries with token counts for each assembly_id.
    """
    pdb_id = row["pdb_id"]
    file_path = get_digs_path(pdb_id)
    assembly_ids = row["assembly_id"]
    counts = count_af3_style_tokens_in_atom_array_from_file(file_path, assembly_ids, cache_dir=cache_dir)
    results = []
    for assembly_id in assembly_ids:
        try:
            results.append(
                {
                    "pdb_id": pdb_id,
                    "assembly_id": assembly_id,
                    "n_atomized_tokens": counts[assembly_id]["n_atomized_tokens"],
                    "n_non_atomized_tokens": counts[assembly_id]["n_non_atomized_tokens"],
                }
            )
        except KeyError:
            logger.error(f"Assembly ID {assembly_id} not found in {pdb_id}")
    return results


def generate_af3_token_counts_df(
    pn_units_df_path: PathLike,
    tokens_df_output_path: PathLike | None = None,
    existing_tokens_df_path: PathLike | None = None,
    cache_dir: PathLike | None = "/projects/ml/RF2_allatom/cache/msa",
    num_workers: int = 2,
    task_id: int = 0,
    num_tasks: int = 1,
) -> None:
    """
    Generates a DataFrame with AF3-style token counts for each unique (pdb_id, assembly_id) pair in the input DataFrame.

    Args:
        pn_units_df_path (PathLike): Path to the input Parquet file containing the pn_units DataFrame.
        tokens_df_output_path (PathLike | None): Path to save the token counts DataFrame. If None, saves to the same directory as the input file with a unique name. Defaults to None.
        existing_tokens_df_path (PathLike | None): Path to an existing tokens DataFrame to be updated. Defaults to None.
        cache_dir (PathLike | None): Directory to cache intermediate files. Defaults to "/projects/ml/RF2_allatom/cache/msa".
        num_workers (int): Number of worker processes to use for parallel processing. Defaults to 16.
        task_id (int): The ID of the current task in the job array. Defaults to 0.
        num_tasks (int): The total number of tasks in the job array. Defaults to 1.

    Returns:
        None
    """
    logger.info(f"Counting tokens with arguments: num_workers={num_workers}, task_id={task_id}, num_tasks={num_tasks}")

    # Load the DataFrame
    pn_units_df_path = Path(pn_units_df_path)
    pn_units_df = pd.read_parquet(pn_units_df_path)

    # Deduplicate the DataFrame
    subset = ["pdb_id", "assembly_id"]
    deduped_df = pn_units_df[subset].drop_duplicates()

    # If an existing tokens DataFrame path is provided, load it and filter out existing entries
    if existing_tokens_df_path:
        existing_tokens_df_path = Path(existing_tokens_df_path)
        if existing_tokens_df_path.exists():
            existing_tokens_df = pd.read_parquet(existing_tokens_df_path)
            initial_length = len(deduped_df)

            # Filter out rows that already exist in the existing tokens DataFrame
            deduped_df = deduped_df.merge(existing_tokens_df[subset], on=subset, how="left", indicator=True)
            deduped_df = deduped_df[deduped_df["_merge"] == "left_only"].drop(columns=["_merge"])

            new_length = len(deduped_df)
            logger.info(f"Filtered out {initial_length - new_length} existing entries from the input DataFrame.")
        else:
            logger.info("No existing tokens DataFrame found. Proceeding with all input entries.")

    # Group by pdb_id and get a list of assembly_ids for each pdb_id (so we don't need to load multiple CIFs for each assembly)
    grouped = deduped_df.groupby("pdb_id")["assembly_id"].apply(list).reset_index()
    grouped = [group for _, group in grouped.iterrows()]

    # Sort by PDB ID (for reproducibility)
    grouped = sorted(grouped, key=lambda x: x["pdb_id"])

    # Determine the slice of the DataFrame to process
    total_rows = len(grouped)
    slice_size = (total_rows + num_tasks - 1) // num_tasks  # Ceiling division
    logger.info(f"Total rows: {total_rows}, slice size: {slice_size}")
    start_index = task_id * slice_size
    end_index = min(start_index + slice_size, total_rows)
    grouped = grouped[start_index:end_index]
    logger.info(f"Processing rows {start_index} to {end_index} of {total_rows}...")

    # Determine the chunk size for multiprocessing
    chunksize = min(100, max(1, len(grouped) // num_workers), len(grouped))

    logger.info(f"Counting tokens for each example using {num_workers} workers...")

    # Process each pdb_id and count tokens
    aggregated_results = []
    partial_process_pdb_id = partial(count_tokens_for_pdb_id, cache_dir=cache_dir)

    with Pool(processes=num_workers) as pool:
        results_generator = pool.imap(partial_process_pdb_id, grouped, chunksize=chunksize)
        for results in tqdm(results_generator, total=len(grouped)):
            aggregated_results.extend(results)

    # Create a DataFrame from the results
    token_counts_df = pd.DataFrame(aggregated_results)

    # Save the token counts DataFrame to disk
    if tokens_df_output_path is None:
        output_dir = pn_units_df_path.parent / "af3_token_counts"
        output_dir.mkdir(exist_ok=True)
        tokens_df_output_path = output_dir / f"af3_token_counts_{start_index}_{end_index}.parquet"
    tokens_df_output_path = Path(tokens_df_output_path)
    logger.info(f"Saving the token counts DataFrame to {tokens_df_output_path}...")
    token_counts_df.to_parquet(tokens_df_output_path)


if __name__ == "__main__":
    fire.Fire(generate_af3_token_counts_df)
