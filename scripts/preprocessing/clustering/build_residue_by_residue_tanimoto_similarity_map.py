import logging
import pickle
from functools import partial
from itertools import product
from multiprocessing import Pool
from os import PathLike
from pathlib import Path

import fire
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_similarity(pair: tuple[str, str], residue_to_rdkit: dict) -> tuple[str, float] | None:
    """Calculates the Tanimoto similarity between two residues."""
    res_name_1, res_name_2 = pair
    mol_1_fingerprint = residue_to_rdkit.get(res_name_1)["morgan_fingerprint"]
    mol_2_fingerprint = residue_to_rdkit.get(res_name_2)["morgan_fingerprint"]

    similarity = Chem.DataStructs.TanimotoSimilarity(mol_1_fingerprint, mol_2_fingerprint)
    return res_name_1, similarity


def build_maximum_similarity_map_for_residue_lists(
    residue_list_1: list[str], residue_list_2: list[str], residue_to_rdkit: dict, num_workers: int = 1
) -> dict[tuple, float]:
    """
    Build a maximum Tanimoto similarity map for two lists of residue names.
    This function calculates the Tanimoto similarity for each pair of residues (one from each list)
    and returns a dictionary mapping each residue in residue_list_1 to its maximum similarity score
    to any residue in residue_list_2.

    Args:
        residue_list_1 (list[str]): First list of residue names.
        residue_list_2 (list[str]): Second list of residue names.
        residue_to_rdkit (dict): Dictionary mapping residue names to RDKit molecule objects and Morgan fingerprints.
        num_workers (int, optional): Number of workers to use for multiprocessing. Defaults to 1.

    Returns:
        dict[str, float]: Dictionary mapping residue names from residue_list_1 to their maximum Tanimoto similarity
                          to any residue in residue_list_2.
    """
    # ...create a list of residue pairs to calculate similarity for
    pairs = list(product(residue_list_1, residue_list_2))

    # ...determine the optimal chunk size for multiprocessing
    chunksize = min(20_000_000, max(1, len(pairs) // num_workers), len(pairs))

    # ...create a partial function to calculate similarity (so that we can use Pool.imap)
    partial_calculate_similarity = partial(calculate_similarity, residue_to_rdkit=residue_to_rdkit)

    # ...calculate the similarity for each pair of residues
    max_similarity_map = {res: 0.0 for res in residue_list_1}
    with Pool(processes=num_workers) as pool:
        results_generator = pool.imap(partial_calculate_similarity, pairs, chunksize=chunksize)
        for result in tqdm(results_generator, total=len(pairs)):
            if result:
                res_name_1, similarity = result
                if similarity > max_similarity_map[res_name_1]:
                    max_similarity_map[res_name_1] = similarity

    return max_similarity_map


def build_residue_by_residue_tanimoto_similarity_map_from_paths(
    pn_units_df_1_path: PathLike,
    pn_units_df_2_path: PathLike,
    residue_name_to_info_path: PathLike,
    output_path: PathLike = None,
    num_workers: int = 1,
) -> None:
    """
    Builds a Tanimoto similarity map between residues from two dataframes using RDKit molecule objects from a dictionary,
    and saves the result as a dictionary.

    Args:
        pn_units_df_1_path (PathLike): Path to the first PN Unit dataframe.
        pn_units_df_2_path (PathLike): Path to the second PN Unit dataframe.
        residue_name_to_info_path (PathLike): Path to the dictionary mapping residue names to RDKit molecule objects and fingerprints.
        output_path (PathLike, optional): Path to save the output Tanimoto similarity map. If not provided, a descriptive filename will be generated.
        num_workers (int, optional): Number of workers to use for multiprocessing.
    """
    # ...load the dataframes and residue dictionary
    pn_units_df_1 = pd.read_parquet(pn_units_df_1_path)
    pn_units_df_2 = pd.read_parquet(pn_units_df_2_path)

    with open(residue_name_to_info_path, "rb") as f:
        residue_name_to_rdkit_molecule = pickle.load(f)

    def extract_unique_residues(df: pd.DataFrame) -> list[str]:
        """Extracts the unique residue names from a DataFrame."""
        residues = df["q_pn_unit_non_polymer_res_names"].unique()
        residues_split = [x.split(",") for x in residues if x is not None]
        return list(set(item for sublist in residues_split for item in sublist))

    residue_list_1 = extract_unique_residues(pn_units_df_1)
    residue_list_2 = extract_unique_residues(pn_units_df_2)

    # ...subset to only the residues that have RDKit molecule objects (e.g., some might have failed sanitization)
    residue_list_1 = [res for res in residue_list_1 if res in residue_name_to_rdkit_molecule]
    residue_list_2 = [res for res in residue_list_2 if res in residue_name_to_rdkit_molecule]

    logger.info(f"Number of unique residues in list 1: {len(residue_list_1)}")
    logger.info(f"Number of unique residues in list 2: {len(residue_list_2)}")

    # ...build a dictionary mapping each residue in residue_list_1 to its maximum similarity score to any residue in residue_list_2
    max_similarity_map = build_maximum_similarity_map_for_residue_lists(
        residue_list_1, residue_list_2, residue_name_to_rdkit_molecule, num_workers=num_workers
    )

    # ...save the output to disk
    if output_path is None:
        residue_dir = Path(residue_name_to_info_path).parent
        output_path = residue_dir / "tanimoto_similarity_with_train_set.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(max_similarity_map, f)

    logger.info(f"Saved Tanimoto similarity map to: {output_path}")


if __name__ == "__main__":
    fire.Fire(build_residue_by_residue_tanimoto_similarity_map_from_paths)
