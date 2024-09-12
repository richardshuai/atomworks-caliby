"""
Example usage: python count_af3_tokens.py --pn_units_df_path /projects/ml/RF2_allatom/datasets/pdb/2024_09_10/pn_units_df.parquet
"""

import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from os import PathLike
from pathlib import Path

import fire
import pandas as pd
from biotite.structure import AtomArray
from cifutils.constants import AF3_EXCLUDED_LIGANDS, STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from cifutils.parser import CIFParser
from tqdm import tqdm

from datahub.common import exists
from datahub.transforms.atom_array import (
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveUnresolvedPNUnits,
)
from datahub.transforms.atomize import AtomizeResidues, FlagNonPolymersForAtomization
from datahub.transforms.base import Compose
from datahub.transforms.covalent_modifications import FlagAndReassignCovalentModifications
from datahub.utils.token import get_token_starts
from tests.conftest import get_digs_path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('count_af3_tokens.log')
                    ])
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
        AtomizeResidues(
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
    file_path: PathLike, assembly_ids: list[str], parser: CIFParser, cache_dir: PathLike | None = None
) -> dict[str, dict[str, int]]:
    """
    Counts AF3-style tokens in assemblies from an atom array parsed from a file.

    Args:
        file_path (PathLike): Path to the file containing atom array data.
        assembly_ids (list[str]): List of assembly IDs to process.
        parser (CIFParser): Parser to use for reading the file.
        cache_dir (PathLike | None, optional): Directory for loading cached parsing data. Defaults to None.

    Returns:
        dict[str, dict[str, int]]: A dictionary with assembly IDs as keys. Each value is another dictionary with
        keys 'n_atomized_tokens' and 'n_non_atomized_tokens' (see `count_af3_style_tokens_from_atom_array`).
    """

    try:
        # ...load the file
        result_dict = parser.parse(
            filename=file_path,
            build_assembly=assembly_ids,
            add_bonds=True,
            add_missing_atoms=True,
            remove_waters=True,
            patch_symmetry_centers=True,
            convert_mse_to_met=True,
            fix_arginines=True,
            keep_hydrogens=False,
            model=1,  # First model
            load_from_cache=exists(cache_dir),
            save_to_cache=False,  # we don't want to save the cache here
            cache_dir=cache_dir,
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


def process_pdb_id(row: pd.Series, parser: CIFParser, cache_dir: PathLike) -> list[dict]:
    """
    Process a single pdb_id and its associated assembly_ids to count AF3-style tokens.

    Args:
        row (pd.Series): A row of the DataFrame containing 'pdb_id' and 'assembly_id'.
        parser (CIFParser): The CIFParser instance to use for parsing files.
        cache_dir (PathLike): The directory to use for caching parsed data.

    Returns:
        list[dict]: A list of dictionaries with token counts for each assembly_id.
    """
    pdb_id = row["pdb_id"]
    file_path = get_digs_path(pdb_id, base="mirror")
    assembly_ids = row["assembly_id"]
    counts = count_af3_style_tokens_in_atom_array_from_file(file_path, assembly_ids, parser, cache_dir=cache_dir)
    results = []
    for assembly_id in assembly_ids:
        results.append(
            {
                "pdb_id": pdb_id,
                "assembly_id": assembly_id,
                "n_atomized_tokens": counts[assembly_id]["n_atomized_tokens"],
                "n_non_atomized_tokens": counts[assembly_id]["n_non_atomized_tokens"],
            }
        )
    return results


def add_af3_style_token_counts_to_pn_units_df(
    pn_units_df_path: PathLike, cache_dir: PathLike | None = "/projects/ml/RF2_allatom/cache/msa"
) -> pd.DataFrame:
    # ...load the pn_units_df
    # (We must load the entire parquet into memory such that we can later create a new column and re-save)
    pn_units_df_path = Path(pn_units_df_path)
    pn_units_df = pd.read_parquet(pn_units_df_path)

    # ...deduplicate
    subset = ["pdb_id", "assembly_id"]
    deduped_df = pn_units_df[subset].drop_duplicates()

    # ...instantiate the CIFParser
    parser = CIFParser()

    # ...group by pdb_id to get the relevant assembly IDs
    grouped = deduped_df.groupby("pdb_id")["assembly_id"].apply(list).reset_index()
    grouped = [group for _, group in grouped.iterrows()]

    # ...multiprocessing parameters
    num_workers = min(cpu_count(), 16)  # Adjust based on your system
    chunksize = min(100, max(1, len(grouped) // num_workers), len(grouped))

    logger.info(f"Counting tokens for each example using {num_workers} workers...")

    # ...get the token counts for each entry, indeed by assembly_id
    aggregated_results = []
    partial_process_pdb_id = partial(process_pdb_id, parser=parser, cache_dir=cache_dir)

    with Pool(processes=num_workers) as pool:
        results_generator = pool.imap(partial_process_pdb_id, grouped, chunksize=chunksize)
        for results in tqdm(results_generator, total=len(grouped)):
            aggregated_results.extend(results)

    # ...create a DataFrame from the results
    token_counts_df = pd.DataFrame(aggregated_results)

    # ...merge the token counts back to the original DataFrame
    new_pn_units_df = pd.merge(
        pn_units_df,
        token_counts_df[["pdb_id", "assembly_id", "n_atomized_tokens", "n_non_atomized_tokens"]],
        on=["pdb_id", "assembly_id"],
        how="left",
    )

    # ...save the DataFrame back to disk
    # Add "af3_token_counts" to the path
    pn_units_df_path = pn_units_df_path.with_name(pn_units_df_path.stem + "_af3_token_counts" + pn_units_df_path.suffix)
    logger.info(f"Saving the updated `pn_units_df` to {pn_units_df_path}...")
    new_pn_units_df.to_parquet(pn_units_df_path)

    logger.info("Done!")


if __name__ == "__main__":
    fire.Fire(add_af3_style_token_counts_to_pn_units_df)
