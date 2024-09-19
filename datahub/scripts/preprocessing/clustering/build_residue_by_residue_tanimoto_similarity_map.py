import logging

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from datahub.transforms.rdkit_utils import calculate_tanimoto_similarity_between_two_rdkit_mols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_similarity_map_for_residue_lists(
    residue_list_1: list[str], residue_list_2: list[str], residue_to_rdkit: dict[str, Chem.Mol]
) -> dict[tuple, float]:
    """
    Build a Tanimoto similarity map for two lists of residue names.

    Args:
        residue_list_1 (list[str]): First list of residue names.
        residue_list_2 (list[str]): Second list of residue names.
        residue_to_rdkit (dict[str, Chem.Mol]): Dictionary mapping residue names to RDKit molecule objects.

    Returns:
        dict[tuple, float]: Dictionary mapping residue name pairs to their Tanimoto similarity.
    """
    tanimoto_similarity_map = {}

    for res_name_1 in tqdm(residue_list_1, desc="Processing residues..."):
        mol1 = residue_to_rdkit.get(res_name_1)
        if mol1 is None:
            logger.info(f"RDKit molecule not found for residue: {res_name_1}")
            continue

        for res_name_2 in residue_list_2:
            mol2 = residue_to_rdkit.get(res_name_2)
            if mol2 is None:
                logger.info(f"RDKit molecule not found for residue: {res_name_2}")
                continue

            try:
                similarity = calculate_tanimoto_similarity_between_two_rdkit_mols(mol1, mol2)
                tanimoto_similarity_map[(res_name_1, res_name_2)] = similarity
            except Exception as e:
                logger.info(f"Failed to calculate similarity between {res_name_1} and {res_name_2}, with error: {e}")

    return tanimoto_similarity_map


def build_residue_by_residue_tanimo_similarity_map_from_pn_units_dfs(
    pn_units_df_1: pd.DataFrame, pn_units_df_2: pd.DataFrame, residue_name_to_rdkit_molecule: dict[str, Chem.Mol]
) -> dict[str, dict[str, float]]:
    """
    Builds a Tanimoto similarity map between residues from two dataframes using RDKit molecule objects from a dictionary,
    and returns the result as a dictionary.

    Args:
        pn_units_df_1 (pd.DataFrame): The first PN Unit dataframe.
        pn_units_df_2 (pd.DataFrame): The second PN Unit dataframe.
        residue_name_to_rdkit_molecule (Dict[str, Chem.Mol]): Dictionary mapping residue names to RDKit molecule objects.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary representing the Tanimoto similarity map.
    """

    # Extract unique residue names from the dataframes
    def extract_unique_residues(df: pd.DataFrame) -> list[str]:
        residues = df["q_pn_unit_non_polymer_res_names"].unique()
        residues_split = [x.split(",") for x in residues if x is not None]
        return list(set(item for sublist in residues_split for item in sublist))

    residue_list_1 = extract_unique_residues(pn_units_df_1)
    residue_list_2 = extract_unique_residues(pn_units_df_2)

    logger.info(f"Number of unique residues in list 1: {len(residue_list_1)}")
    logger.info(f"Number of unique residues in list 2: {len(residue_list_2)}")

    # Build the similarity map
    tanimoto_similarity_map = build_similarity_map_for_residue_lists(
        residue_list_1, residue_list_2, residue_name_to_rdkit_molecule
    )

    return tanimoto_similarity_map
