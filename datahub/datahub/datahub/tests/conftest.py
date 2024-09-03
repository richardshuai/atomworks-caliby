"""Shared test utils and fixtures for all tests"""

import os
from copy import deepcopy
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray
from cifutils import CIFParser

from datahub.datasets.base import NamedConcatDataset
from datahub.datasets.dataframe_parsers import InterfacesDFParser, PNUnitsDFParser
from datahub.datasets.pdb_dataset import PDBDataset
from datahub.encoding_definitions import TokenEncoding
from datahub.pipelines.rf2aa import build_rf2aa_transform_pipeline
from datahub.preprocessing.constants import SUPPORTED_CHAIN_TYPES
from datahub.transforms.msa._msa_featurizing_utils import uniformly_select_rows

SUPPORTED_CHAIN_TYPES_INTS = [type.value for type in SUPPORTED_CHAIN_TYPES]

# Directory containing pn_units_df and interfaces_df
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
PN_UNITS_DF = pd.read_parquet(f"{TEST_DATA_DIR}/pn_units_df.parquet")
INTERFACES_DF = pd.read_parquet(f"{TEST_DATA_DIR}/interfaces_df.parquet")

PROTEIN_MSA_DIR = Path("/projects/ml/RF2_allatom/data_preprocessing/msa/protein")
RNA_MSA_DIR = Path("/projects/ml/RF2_allatom/data_preprocessing/msa/rna")

CIF_PARSER = CIFParser()
CANONICAL_AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
RNA_RESIDUES = ["A", "C", "G", "U"]
DNA_RESIDUES = ["DA", "DC", "DG", "DT"]

SHARED_TEST_FILTERS = [
    "deposition_date < '2022-01-01'",
    "resolution < 5.0 and ~method.str.contains('NMR')",
    "num_polymer_pn_units <= 20",  # To ensure we don't freeze loading a massive example
    "cluster.notnull()",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
]

TEST_PN_UNITS_FILTERS = [
    f"q_pn_unit_type in {SUPPORTED_CHAIN_TYPES_INTS}",  # Limit query PN units to proteins, RNA, DNA, and ligands (i.e., exclude RNA/DNA hybrids)
]

TEST_INTERFACES_FILTERS = [
    f"pn_unit_1_type in {SUPPORTED_CHAIN_TYPES_INTS}",  # Limit interface PN units to proteins, RNA, DNA, and ligands (i.e., exclude RNA/DNA hybrids)
    f"pn_unit_2_type in {SUPPORTED_CHAIN_TYPES_INTS}",
]

# Define the PDB datasets with their respective parsers...
PN_UNITS_DATASET = PDBDataset(
    name="pn_units",
    dataset_path=PN_UNITS_DF,
    cif_parser=CIF_PARSER,
    filters=SHARED_TEST_FILTERS + TEST_PN_UNITS_FILTERS,
    dataset_parser=PNUnitsDFParser(),
    id_column="example_id",
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dir=PROTEIN_MSA_DIR,
        rna_msa_dir=RNA_MSA_DIR,
        n_recycles=5,
        crop_size=256,
        crop_contiguous_probability=1 / 3,
        crop_spatial_probability=2 / 3,
    ),
    unpack_data_dict=False,
)

INTERFACES_DATASET = PDBDataset(
    name="interfaces",
    dataset_path=INTERFACES_DF,
    cif_parser=CIF_PARSER,
    filters=SHARED_TEST_FILTERS + TEST_INTERFACES_FILTERS,
    dataset_parser=InterfacesDFParser(),
    id_column="example_id",
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dir=PROTEIN_MSA_DIR,
        rna_msa_dir=RNA_MSA_DIR,
        n_recycles=5,
        crop_size=256,
        crop_spatial_probability=1.0,
        crop_contiguous_probability=0.0,
    ),
    unpack_data_dict=False,
)

# ...build the ConcatDataset
PDB_DATASET = NamedConcatDataset(datasets=[PN_UNITS_DATASET, INTERFACES_DATASET], name="pdb")  # NOTE: Order matters!


def get_digs_path(pdbid: str, base: Literal["trrosetta", "mirror"] = "trrosetta") -> str:
    if base == "trrosetta":
        base_dir = Path("/databases/TrRosetta/cif")
    elif base == "mirror":
        base_dir = Path("/databases/rcsb/cif")
    else:
        raise ValueError(f"Invalid base: {base}")
    pdbid = pdbid.lower()
    filename = f"{base_dir}/{pdbid[1:3]}/{pdbid}.cif.gz"
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename


@cache
def _cached_parse(
    pdb_id: str,
    base: Literal["trrosetta", "mirror"] = "trrosetta",
    convert_mse_to_met: bool = True,
    remove_waters: bool = True,
    build_assembly: str = "first",
    keep_hydrogens: bool = True,
    fix_arginines: bool = True,
    patch_symmetry_centers: bool = True,
) -> AtomArray:
    data = CIF_PARSER.parse(
        filename=get_digs_path(pdb_id, base),
        convert_mse_to_met=convert_mse_to_met,
        remove_waters=remove_waters,
        build_assembly=build_assembly,
        keep_hydrogens=keep_hydrogens,
        fix_arginines=fix_arginines,
        patch_symmetry_centers=patch_symmetry_centers,
    )
    if "atom_array" not in data:
        assembly_ids = list(data["assemblies"].keys())
        data["atom_array"] = data["assemblies"][assembly_ids[0]][0]
    data["pdb_id"] = pdb_id
    return data


def cached_parse(pdb_id: str, **kwargs) -> AtomArray:
    """Wrapper around _cached_parse with caching to return an immutable copy of the output dict"""
    return deepcopy(_cached_parse(pdb_id, **kwargs))


def assert_equal_atom_arrays(
    arr1: AtomArray,
    arr2: AtomArray,
    compare_coords: bool = True,
    compare_bonds: bool = True,
    annotations_to_compare: list[str] | None = None,
    _n_mismatches_to_show: int = 20,
):
    """Asserts that two AtomArray objects are equal.

    Args:
        arr1 (AtomArray): The first AtomArray to compare.
        arr2 (AtomArray): The second AtomArray to compare.
        compare_coords (bool, optional): Whether to compare coordinates. Defaults to True.
        compare_bonds (bool, optional): Whether to compare bonds. Defaults to True.
        annotations_to_compare (list[str] | None, optional): List of annotation categories to compare.
            Defaults to None, in which case all annotations are compared.
        _n_mismatches_to_show (int, optional): Number of mismatches to show. Defaults to 20.

    Raises:
        AssertionError: If the AtomArray objects are not equal.
    """
    assert isinstance(arr1, AtomArray), f"arr1 is not an AtomArray but has type {type(arr1)}"
    assert isinstance(arr2, AtomArray), f"arr2 is not an AtomArray but has type {type(arr2)}"

    if compare_coords:
        assert (
            arr1.coord.shape == arr2.coord.shape
        ), f"Coord shapes do not match: {arr1.coord.shape} != {arr2.coord.shape}"
        if not np.allclose(arr1.coord, arr2.coord, equal_nan=True, atol=1e-3, rtol=1e-3):
            mismatch_idxs = np.where(arr1.coord != arr2.coord)[0]
            msg = f"Coords do not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
            for idx in mismatch_idxs[:_n_mismatches_to_show]:
                msg += f"\t{idx}: {arr1.coord[idx]} != {arr2.coord[idx]}\n"
            raise AssertionError(msg)

    if compare_bonds:
        assert arr1.bonds is not None, "arr1.bonds is None"
        assert arr2.bonds is not None, "arr2.bonds is None"
        if not np.array_equal(arr1.bonds.as_array(), arr2.bonds.as_array()):
            mismatch_idxs = np.where(arr1.bonds.as_array() != arr2.bonds.as_array())[0]
            msg = f"Bonds do not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
            for idx in mismatch_idxs[:_n_mismatches_to_show]:
                msg += f"\t{idx}: {arr1.bonds.as_array()[idx]} != {arr2.bonds.as_array()[idx]}\n"
            raise AssertionError(msg)

    if annotations_to_compare is None:
        arr1_annotation_keys = arr1.get_annotation_categories()
        arr2_annotation_keys = arr2.get_annotation_categories()
        missing_in_arr1 = set(arr2_annotation_keys) - set(arr1_annotation_keys)
        missing_in_arr2 = set(arr1_annotation_keys) - set(arr2_annotation_keys)
        assert len(missing_in_arr1) == 0, f"Annotations missing in arr1: {missing_in_arr1}"
        assert len(missing_in_arr2) == 0, f"Annotations missing in arr2: {missing_in_arr2}"
        annotations_to_compare = arr1_annotation_keys

    for annotation in annotations_to_compare:
        if annotation not in arr1.get_annotation_categories():
            raise AssertionError(f"Annotation {annotation} not in arr1.")
        if annotation not in arr2.get_annotation_categories():
            raise AssertionError(f"Annotation {annotation} not in arr2.")
        if not np.array_equal(arr1.get_annotation(annotation), arr2.get_annotation(annotation)):
            mismatch_idxs = np.where(arr1.get_annotation(annotation) != arr2.get_annotation(annotation))[0]
            msg = (
                f"Annotation {annotation} does not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
            )
            for idx in mismatch_idxs[:_n_mismatches_to_show]:
                msg += f"\t{idx}: {arr1.get_annotation(annotation)[idx]} != {arr2.get_annotation(annotation)[idx]}\n"
            raise AssertionError(msg)


def generate_synthetic_msa(
    encoding: TokenEncoding, n_rows: int, n_tokens_across_chains: int, n_msa_cluster_representatives: int
) -> dict[str, torch.Tensor]:
    """
    Generate synthetic Multiple Sequence Alignment (MSA) data.

    Args:
        encoding (TokenEncoding): An object containing token encoding information.
        n_rows (int): Number of rows in the MSA.
        n_tokens_across_chains (int): Number of tokens across all chains.
        n_msa_cluster_representatives (int): Number of MSA cluster representatives to select.

    Returns:
        SyntheticMSAData: A dictionary containing various components of the synthetic MSA.
    """

    def get_token_range(token_list: list[int]) -> tuple[int, int]:
        return min(token_list), max(token_list) + 1

    def generate_msa_segment(token_range: tuple[int, int], shape: tuple[int, int]) -> torch.Tensor:
        return torch.randint(token_range[0], token_range[1], shape)

    amino_acid_tokens = [encoding.token_to_idx[res] for res in CANONICAL_AMINO_ACIDS + ["UNK"]]
    rna_tokens = [encoding.token_to_idx[res] for res in RNA_RESIDUES + ["X"]]
    dna_tokens = [encoding.token_to_idx[res] for res in DNA_RESIDUES + ["DX"]]
    atom_tokens = [encoding.token_to_idx[res] for res in [13, 33, 79, 5, 4, 35, 6, 20, 17, 27, 24, 29, 9, 26, 80, 53]]
    mask_token = encoding.token_to_idx["<M>"]

    # Generate full MSA profile
    full_msa_profile = torch.rand(n_tokens_across_chains, encoding.n_tokens)
    full_msa_profile[:, mask_token] = 0
    full_msa_profile /= full_msa_profile.sum(dim=1, keepdim=True)

    # Generate MSA segments
    example_protein_msa = generate_msa_segment(
        get_token_range(amino_acid_tokens), (n_rows, n_tokens_across_chains // 2)
    )
    example_rna_msa = generate_msa_segment(get_token_range(rna_tokens), (n_rows, n_tokens_across_chains // 10))
    example_dna_msa = generate_msa_segment(get_token_range(dna_tokens), (1, n_tokens_across_chains // 10)).repeat(
        n_rows, 1
    )
    example_atom_1_msa = generate_msa_segment(get_token_range(atom_tokens), (1, n_tokens_across_chains // 10)).repeat(
        n_rows, 1
    )
    example_atom_2_msa = generate_msa_segment(
        get_token_range(atom_tokens), (1, 2 * n_tokens_across_chains // 10)
    ).repeat(n_rows, 1)

    # Concatenate into a single MSA
    encoded_msa = torch.cat(
        [example_protein_msa, example_rna_msa, example_dna_msa, example_atom_1_msa, example_atom_2_msa], dim=1
    )

    # Generate masks
    msa_is_padded_mask = torch.randint(0, 2, (n_rows, n_tokens_across_chains)).bool()
    token_idx_has_msa = torch.zeros(n_tokens_across_chains, dtype=torch.bool)
    token_idx_has_msa[: (example_protein_msa.shape[1] + example_rna_msa.shape[1])] = True

    # Break apart the MSA into selected and not selected indices
    selected_indices, not_selected_indices = uniformly_select_rows(n_rows, n_msa_cluster_representatives)

    return {
        "encoded_msa": encoded_msa,
        "msa_is_padded_mask": msa_is_padded_mask,
        "token_idx_has_msa": token_idx_has_msa,
        "full_msa_profile": full_msa_profile,
        "selected_indices": selected_indices,
        "not_selected_indices": not_selected_indices,
        "example_protein_msa": example_protein_msa,
        "example_rna_msa": example_rna_msa,
        "example_dna_msa": example_dna_msa,
        "example_atom_1_msa": example_atom_1_msa,
        "example_atom_2_msa": example_atom_2_msa,
    }


def all_different(tensor_list: list[torch.Tensor]) -> bool:
    """
    Check if all tensors in the list are unique.

    Args:
        tensor_list (list): List of tensors to compare.

    Returns:
        bool: True if all tensors are different, False if any are equal.
    """
    for i in range(len(tensor_list)):
        for j in range(i + 1, len(tensor_list)):
            if torch.equal(tensor_list[i], tensor_list[j]):
                return False
    return True


def similar_stats(tensor_list: list[torch.Tensor], mean_lower: float=0.3, mean_upper: float=1.3, std_lower: float=0.7, std_upper: float=1.3) -> bool:
    """Check if tensor statistics are similar within specified ranges.

    Args:
        tensor_list (list): List of tensors to compare.
        mean_lower (float, optional): Lower bound for mean. Defaults to 0.3.
        mean_upper (float, optional): Upper bound for mean. Defaults to 1.3.
        std_lower (float, optional): Lower bound for std dev. Defaults to 0.7.
        std_upper (float, optional): Upper bound for std dev. Defaults to 1.3.

    Returns:
        bool: True if all tensors have similar stats, False otherwise.
    """
    means = [t.float().mean().item() for t in tensor_list]
    stds = [t.float().std().item() for t in tensor_list]
    mean_mean, mean_std = sum(means) / len(means), sum(stds) / len(stds)

    return all(
        mean_lower * mean_mean <= m <= mean_upper * mean_mean and std_lower * mean_std <= s <= std_upper * mean_std
        for m, s in zip(means, stds)
    )
