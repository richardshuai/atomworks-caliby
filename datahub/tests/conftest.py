"""Shared test utils and fixtures for all tests"""

import os
from copy import deepcopy
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from cifutils import parse

# Directory containing pn_units_df and interfaces_df
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
PN_UNITS_DF = pd.read_parquet(f"{TEST_DATA_DIR}/pn_units_df.parquet")
INTERFACES_DF = pd.read_parquet(f"{TEST_DATA_DIR}/interfaces_df.parquet")
AF2_DISTILLATION_FACEBOOK_DF = pd.read_parquet(f"{TEST_DATA_DIR}/test_af2_distillation_facebook.parquet")

# The validation dataset is small, so we don't need to use a subset
AF3_VALIDATION_DF = pd.read_parquet(
    "/projects/ml/RF2_allatom/datasets/af3_splits/2024_10_18/entry_level_val_df.parquet"
)

PROTEIN_MSA_DIRS = [
    {
        "dir": "/projects/msa/rf2aa_af3/rf2aa_paper_model_protein_msas",
        "extension": ".a3m.gz",
        "directory_depth": 2,
    },
    {
        "dir": "/projects/msa/rf2aa_af3/missing_msas_through_2024_08_12",
        "extension": ".msa0.a3m.gz",
        "directory_depth": 2,
    },
]

RNA_MSA_DIRS = [
    {"dir": "/projects/msa/rf2aa_af3/rf2aa_paper_model_rna_msas", "extension": ".afa", "directory_depth": 0}
]


def get_digs_path(pdbid: str, base_dir: str = "/projects/ml/frozen_pdb_copies/2024_12_01_pdb") -> str:
    # Assert that the base directory exists
    assert os.path.exists(base_dir), f"Base directory {base_dir} does not exist"

    # Build the path to the file
    pdbid = pdbid.lower()
    filename = f"{base_dir}/{pdbid[1:3]}/{pdbid}.cif.gz"
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename


@cache
def _cached_parse(
    pdb_id: str,
    **kwargs,
) -> AtomArray:
    data = parse(
        filename=get_digs_path(pdb_id),
        **kwargs,
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

        # Check if the arrays contain floating-point numbers (in which case, we allow NaN == NaN)
        if np.issubdtype(arr1.get_annotation(annotation).dtype, np.floating) and np.issubdtype(
            arr2.get_annotation(annotation).dtype, np.floating
        ):
            arrays_equal = np.array_equal(
                arr1.get_annotation(annotation), arr2.get_annotation(annotation), equal_nan=True
            )
        else:
            arrays_equal = np.array_equal(
                arr1.get_annotation(annotation), arr2.get_annotation(annotation), equal_nan=False
            )

        if not arrays_equal:
            mismatch_idxs = np.where(arr1.get_annotation(annotation) != arr2.get_annotation(annotation))[0]
            msg = (
                f"Annotation {annotation} does not match at {len(mismatch_idxs)} indices. First few mismatches:" + "\n"
            )
            for idx in mismatch_idxs[:_n_mismatches_to_show]:
                msg += f"\t{idx}: {arr1.get_annotation(annotation)[idx]} != {arr2.get_annotation(annotation)[idx]}\n"
            raise AssertionError(msg)
