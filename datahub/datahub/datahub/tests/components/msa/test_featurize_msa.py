import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from cifutils.enums import ChainType

from datahub.datasets.dataframe_parsers import PNUnitsDFParser, load_from_row
from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING
from datahub.tests.conftest import (
    CANONICAL_AMINO_ACIDS,
    CIF_PARSER,
    DNA_RESIDUES,
    PN_UNITS_DF,
    PROTEIN_MSA_DIR,
    RNA_MSA_DIR,
    RNA_RESIDUES,
)
from datahub.transforms.atom_array import (
    AddWithinPolyResIdxAnnotation,
    RemoveHydrogens,
)
from datahub.transforms.atomize import AtomizeResidues
from datahub.transforms.base import Compose, ConvertToTorch
from datahub.transforms.encoding import EncodeAtomArray
from datahub.transforms.msa._msa_featurizing_utils import (
    assign_extra_rows_to_cluster_representatives,
    build_indices_should_be_counted_masks,
    build_msa_index_can_be_masked,
    encode_msa_like_RF2AA,
    mask_msa_like_bert,
    summarize_clusters,
    uniformly_select_msa_cluster_representatives,
)
from datahub.transforms.msa.msa import (
    EncodeMSA,
    FeaturizeMSALikeRF2AA,
    FillFullMSAFromEncoded,
    LoadPolymerMSAs,
    PairAndMergePolymerMSAs,
)
from datahub.utils.rng import create_rng_state_from_seeds, rng_state
from datahub.utils.token import token_iter

ENCODE_MSA_LIKE_RF2AA_TEST_CASES = [
    # Protein test cases
    (np.array([[b"A", b"G"], [b"S", b"T"]]), ChainType.POLYPEPTIDE_L, np.array([[0, 7], [15, 16]])),
    (
        # Test that ambiguous ("B") and unknown ("X") amino acids are encoded as unknown (20)
        # NOTE: In the future, we may adjust behavior for "B" and other ambiguous amino acids
        np.array([[b"K", b"B"], [b"X", b"V"]]),
        ChainType.POLYPEPTIDE_L,
        np.array([[11, 20], [20, 19]]),
    ),
    # RNA test cases
    (np.array([[b"A", b"U"], [b"C", b"X"]]), ChainType.RNA, np.array([[27, 30], [28, 31]])),
    # DNA test cases
    (np.array([[b"A", b"T"], [b"X", b"G"]]), ChainType.DNA, np.array([[22, 25], [26, 24]])),
]


@pytest.mark.parametrize("test_case", ENCODE_MSA_LIKE_RF2AA_TEST_CASES)
def test_encode_msa_like_RF2AA(test_case):
    """
    Test that the MSA is encoded correctly according to the RF2AA_ATOM36_ENCODING.
    """
    msa, chain_type, expected_output = test_case
    output = encode_msa_like_RF2AA(msa, chain_type, encoding=RF2AA_ATOM36_ENCODING)
    assert np.array_equal(output, expected_output), f"Expected {expected_output}, but got {output}"


FILL_FULL_MSA_FROM_ENCODED_TEST_CASES = ["3ejj", "1mna", "1hge"]


@pytest.mark.parametrize("pdb_id", FILL_FULL_MSA_FROM_ENCODED_TEST_CASES)
def test_fill_full_msa_from_encoded(pdb_id):
    """
    Test if the full MSA is filled correctly from the encoded MSA.

    In particular, we want to ensure:
    - The padding is carried over correctly (i.e., the padding in the encoded MSA is reflected in the full MSA)
    - The corresponding MSA columns match (i.e., after fancy indexing)
    - The insertions match (i.e., we didn't lose any)
    """
    row = PN_UNITS_DF[PN_UNITS_DF["pdb_id"] == pdb_id].iloc[0]  # Get the first row; we don't care which we choose
    data = load_from_row(row, PNUnitsDFParser(), cif_parser=CIF_PARSER)

    encoding = RF2AA_ATOM36_ENCODING
    res_names_to_ignore = encoding.tokens[
        encoding.tokens != "ASP"
    ]  # Atomize aspartate so we can test atomization and MSA indexing
    pad_token = RF2AA_ATOM36_ENCODING.token_to_idx["UNK"]

    # Apply initial transforms
    # fmt: off
    pipeline = Compose([
        AddWithinPolyResIdxAnnotation(),
        LoadPolymerMSAs(protein_msa_dir=PROTEIN_MSA_DIR, rna_msa_dir=RNA_MSA_DIR, max_msa_sequences=100),
        PairAndMergePolymerMSAs(),
        AtomizeResidues(
            atomize_by_default=True, res_names_to_ignore=res_names_to_ignore, move_atomized_part_to_end=False
        ),
        EncodeAtomArray(encoding),
        # MSA featurize workflow
        EncodeMSA(encoding_function=encode_msa_like_RF2AA),
        FillFullMSAFromEncoded(pad_token=pad_token),
    ], track_rng_state=False)
    # fmt: on

    output = pipeline(data)
    atom_array = output["atom_array"]

    # Iterate through all tokens
    atomized_indices = []
    for index, token_atom_array in enumerate(token_iter(atom_array)):
        if token_atom_array.atomize[0]:
            # If this residue is atomized, ensure that the entire MSA column (other than the query sequence) is padding...
            assert np.all(
                output["encoded"]["msa"][1:, index] == pad_token
            ), f"MSA column for atomized residue {index} is not padding"

            # ...and the padding is represented in the full MSA details
            assert not output["full_msa_details"]["token_idx_has_msa"][index], "Token index has MSA when it should not"
            assert np.all(
                output["full_msa_details"]["msa_is_padded_mask"][1:, index]
            ), "MSA is not padded when it should be"
            atomized_indices.append(index)
        else:
            # If this residue is not atomized, ensure that the MSA matches with the pre-atomized MSA...
            within_poly_res_idx = token_atom_array.within_poly_res_idx[0]
            chain_id = token_atom_array.chain_id[0]
            encoded_old_msa = output["polymer_msas_by_chain_id"][chain_id]["encoded_msa"]
            msa_column_old = encoded_old_msa[:, within_poly_res_idx]
            msa_column_new = output["encoded"]["msa"][:, index]
            assert np.array_equal(
                msa_column_old, msa_column_new
            ), f"MSA column for non-atomized residue {index} does not match"

            # ...and that we are noting that this token has MSA
            assert output["full_msa_details"]["token_idx_has_msa"][
                index
            ], "Token index does not have MSA when it should"

    # Check that there are no insertions where there is MSA padding...
    msa_raw_ins = output["full_msa_details"]["msa_raw_ins"]
    msa_is_padded_mask = output["full_msa_details"]["msa_is_padded_mask"]
    assert np.sum(msa_raw_ins * msa_is_padded_mask) == 0, "There should be no insertions where there is MSA padding"

    # ...AND that there are no insertions where there are atomized tokens
    assert (
        np.sum(msa_raw_ins[:, atomized_indices]) == 0
    ), "There should be no insertions where there are atomized tokens"


ASSIGN_EXTRA_ROWS_TEST_CASES = [
    # Test case 1: No masking, but ignore specific tokens
    {
        "mask_position": torch.tensor([[False, False], [False, False], [False, False]], dtype=torch.bool),
        "encoded_msa": torch.tensor([[1, 2], [3, 4], [1, 4]], dtype=torch.int),
        "selected_indices": torch.tensor([0, 1], dtype=torch.int),  # Main MSA
        "not_selected_indices": torch.tensor([2], dtype=torch.int),  # Extra MSA -- we will match to a cluster
        "token_idx_has_msa": torch.tensor([True, True], dtype=torch.bool),  # Include all columns (tokens)
        "tokens_to_ignore": torch.tensor([1], dtype=torch.int),  # Ignore the "1" token
        "expected_assignment": torch.tensor(
            [1], dtype=torch.int
        ),  # Since we are ignoring the "1" token, we should assign to the "2" token (2 Hamming distance vs. 1 Hamming distance)
    },
    # Test case 2: Simplified example from the docstring, no masking, or ignoring tokens
    {
        "mask_position": torch.zeros((6, 5), dtype=bool),
        "encoded_msa": torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 2, 2],
                [2, 2, 1, 0, 0],
                [3, 3, 3, 2, 2],
                [1, 1, 3, 3, 3],
            ],
            dtype=torch.int,
        ),
        "selected_indices": torch.tensor([0, 1, 2], dtype=torch.int),
        "not_selected_indices": torch.tensor([3, 4, 5], dtype=torch.int),
        "token_idx_has_msa": torch.tensor([True, True, True, True, True], dtype=torch.bool),
        "tokens_to_ignore": torch.tensor([], dtype=torch.int),
        "expected_assignment": torch.tensor([1, 2, 0], dtype=torch.int),
    },
    # Test case 3: Testing mask_position functionality
    {
        # Simulate masking a block of the MSA due to unpaired sequences
        "mask_position": torch.tensor(
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
                [True, True, False, False, False],
                [True, True, False, False, False],
            ],
            dtype=torch.bool,
        ),
        "encoded_msa": torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 2, 2],
                [2, 2, 1, 0, 0],
                [3, 3, 3, 2, 2],
                [1, 1, 3, 3, 3],
            ],  # With the mask, most similar to row index 2 (previously was row index 0)
            dtype=torch.int,
        ),
        "selected_indices": torch.tensor([0, 1, 2], dtype=torch.int),
        "not_selected_indices": torch.tensor([3, 4, 5], dtype=torch.int),
        "token_idx_has_msa": torch.tensor([True, True, True, True, True], dtype=torch.bool),
        "tokens_to_ignore": torch.tensor([], dtype=torch.int),
        "expected_assignment": torch.tensor([1, 2, 2], dtype=torch.int),
    },
]


@pytest.mark.parametrize("test_case", ASSIGN_EXTRA_ROWS_TEST_CASES)
def test_assign_extra_rows_to_cluster_representatives(test_case):
    """
    Tests assignment of extra MSA rows to rows within the main MSA by Hamming distance.

    Involves two functions:
    (1) `build_indices_should_be_counted_mask` to identify which indices should count towards the agreement sum
    (2) `assign_extra_rows_to_cluster_representatives` to assign the extra rows to the main MSA rows, based on the agreement sum
    """
    mask_position = test_case["mask_position"]
    encoded_msa = test_case["encoded_msa"]
    selected_indices = test_case["selected_indices"]
    not_selected_indices = test_case["not_selected_indices"]
    token_idx_has_msa = test_case["token_idx_has_msa"]
    tokens_to_ignore = test_case["tokens_to_ignore"]
    expected_assignment = test_case["expected_assignment"]

    index_should_be_counted_mask = build_indices_should_be_counted_masks(
        encoded_msa=encoded_msa,
        mask_position=mask_position,
        tokens_to_ignore=tokens_to_ignore,
        token_idx_has_msa=token_idx_has_msa,
    )  # [n_rows, n_tokens_across_chains] (bool)

    assignments = assign_extra_rows_to_cluster_representatives(
        cluster_representatives_msa=encoded_msa[selected_indices],
        clust_reps_should_be_counted_mask=index_should_be_counted_mask[selected_indices],
        extra_msa=encoded_msa[not_selected_indices],
        extra_msa_should_be_counted_mask=index_should_be_counted_mask[not_selected_indices],
    )

    assert torch.equal(assignments, expected_assignment), f"Expected {expected_assignment}, but got {assignments}"


SUMMARIZE_CLUSTERS_TEST_CASES = [
    # Test case 1: All mask_position is False
    {
        "encoded_msa": torch.tensor([[0, 1, 2], [1, 0, 2], [0, 1, 0], [2, 0, 2]], dtype=torch.int64),
        "msa_raw_ins": torch.tensor([[0, 0, 2], [1, 0, 0], [0, 1, 0], [4, 0, 1]], dtype=torch.int),
        "mask_position": torch.tensor(
            [[False, False, False], [False, False, False], [False, False, False], [False, False, False]],
            dtype=torch.bool,
        ),
        "assignments": torch.tensor([0, 1], dtype=torch.int64),
        "selected_indices": torch.tensor([0, 1], dtype=torch.int),
        "not_selected_indices": torch.tensor([2, 3], dtype=torch.int),
        "msa_is_padded_mask": torch.tensor(
            [[False, False, False], [False, False, False], [False, False, False], [False, False, False]],
            dtype=torch.bool,
        ),
        "expected_profiles": torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]], [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
        ),
        "expected_insertions": torch.tensor([[0.0, 0.5, 1.0], [5 / 2, 0, 0.5]]),
    },
    # Test case 2: Example from docstring
    {
        "encoded_msa": torch.tensor([[0, 1, 2], [1, 0, 2], [0, 1, 0], [2, 0, 2]], dtype=torch.int64),
        "msa_raw_ins": torch.tensor([[0, 0, 2], [1, 0, 0], [0, 1, 0], [4, 0, 1]], dtype=torch.int),
        "mask_position": torch.tensor(
            [[False, False, False], [True, False, False], [False, False, False], [False, False, False]],
            dtype=torch.bool,
        ),
        "assignments": torch.tensor([0, 1], dtype=torch.int64),
        "selected_indices": torch.tensor([0, 1], dtype=torch.int),
        "not_selected_indices": torch.tensor([2, 3], dtype=torch.int),
        "msa_is_padded_mask": torch.tensor(
            [[False, False, False], [False, False, False], [False, False, True], [False, False, True]], dtype=torch.bool
        ),
        "expected_profiles": torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
        ),
        "expected_insertions": torch.tensor([[0.0, 0.5, 2.0], [4, 0.0, 0.0]]),
    },
]


@pytest.mark.parametrize("test_case", SUMMARIZE_CLUSTERS_TEST_CASES)
def test_summarize_clusters(test_case):
    encoded_msa = test_case["encoded_msa"]
    msa_raw_ins = test_case["msa_raw_ins"]
    mask_position = test_case["mask_position"]
    assignments = test_case["assignments"]
    selected_indices = test_case["selected_indices"]
    not_selected_indices = test_case["not_selected_indices"]
    msa_is_padded_mask = test_case["msa_is_padded_mask"]
    expected_profiles = test_case["expected_profiles"]
    expected_insertions = test_case["expected_insertions"]

    msa_cluster_profiles, msa_cluster_ins = summarize_clusters(
        encoded_msa=encoded_msa,
        msa_raw_ins=msa_raw_ins,
        mask_position=mask_position,
        assignments=assignments,
        selected_indices=selected_indices,
        not_selected_indices=not_selected_indices,
        msa_is_padded_mask=msa_is_padded_mask,
        n_tokens=3,
    )

    assert torch.allclose(
        msa_cluster_profiles, expected_profiles, atol=1e-4
    ), f"Expected profiles {expected_profiles}, but got {msa_cluster_profiles}"
    assert torch.allclose(
        msa_cluster_ins, expected_insertions, atol=1e-4
    ), f"Expected insertions {expected_insertions}, but got {msa_cluster_ins}"


def test_mask_msa_like_bert():
    """
    Tests the generation and application of the BERT-style masking to the MSA.
    Only the main MSA is masked; the extra MSA is left unchanged.

    Includes:
    - Assertions to sanity-check outputs
    - Regression test to ensure that the output is consistent across runs
    """
    # Set the seed for reproducibility
    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        encoding = RF2AA_ATOM36_ENCODING

        # TODO: Generalize and move generation of synthetic data to `test_utils.py`

        ############## Utility definitions to generate synthetic data ##############

        amino_acid_tokens = [encoding.token_to_idx[res] for res in CANONICAL_AMINO_ACIDS + ["UNK"]]
        rna_tokens = [encoding.token_to_idx[res] for res in RNA_RESIDUES + ["X"]]
        dna_tokens = [encoding.token_to_idx[res] for res in DNA_RESIDUES + ["DX"]]
        mask_token = encoding.token_to_idx["<M>"]
        atom_tokens = [
            encoding.token_to_idx[res] for res in [13, 33, 79, 5, 4, 35, 6, 20, 17, 27, 24, 29, 9, 26, 80, 53]
        ]  # Not exhaustive

        ############## Generate synthetic data ##############
        n_rows = 50
        n_msa_cluster_representatives = 40
        n_tokens_across_chains = 40
        n_tokens = encoding.n_tokens

        mask_behavior_probs = {"replace_with_random_aa": 0.1, "replace_with_msa_profile": 0.1, "do_not_replace": 0.1}

        mask_probability = 0.15
        full_msa_profile = torch.rand(n_tokens_across_chains, n_tokens)  # [n_tokens_across_chains, n_tokens]
        # Set probability for the mask token to 0
        full_msa_profile[:, mask_token] = 0
        # Make the full MSA profile sum to 1
        full_msa_profile = full_msa_profile / full_msa_profile.sum(dim=1, keepdim=True)

        # Protein example
        example_protein_msa = torch.randint(
            min(amino_acid_tokens), max(amino_acid_tokens) + 1, (n_rows, n_tokens_across_chains // 2)
        )  # Half amino acids

        # RNA example
        example_rna_msa = torch.randint(
            min(rna_tokens), max(rna_tokens) + 1, (n_rows, n_tokens_across_chains // 10)
        )  # 1/10 RNA

        # DNA example
        first_row_dna_msa = torch.randint(
            min(dna_tokens), max(dna_tokens) + 1, (n_tokens_across_chains // 10,)
        )  # 1/10 DNA
        example_dna_msa = torch.zeros((n_rows, n_tokens_across_chains // 10), dtype=torch.int)
        example_dna_msa[0] = first_row_dna_msa

        # Atomized example - #1
        first_row_atom_1_msa = torch.randint(
            min(atom_tokens), max(atom_tokens) + 1, (n_tokens_across_chains // 10,)
        )  # 1/10 atomized
        example_atom_1_msa = torch.zeros((n_rows, n_tokens_across_chains // 10), dtype=torch.int)
        example_atom_1_msa[0] = first_row_atom_1_msa

        # Atomized example - #2
        first_row_atom_2_msa = torch.randint(
            min(atom_tokens), max(atom_tokens) + 1, (2 * n_tokens_across_chains // 10,)
        )  # 1/10 atomized
        example_atom_2_msa = torch.zeros((n_rows, 2 * n_tokens_across_chains // 10), dtype=torch.int)
        example_atom_2_msa[0] = first_row_atom_2_msa

        # Concatenate into a single MSA
        encoded_msa = torch.cat(
            [example_protein_msa, example_rna_msa, example_dna_msa, example_atom_1_msa, example_atom_2_msa], dim=1
        )  # [n_rows, n_tokens_across_chains]

        # Break apart the MSA into selected and not selected indices
        selected_indices, not_selected_indices = uniformly_select_msa_cluster_representatives(
            n_rows, n_msa_cluster_representatives
        )

        # Build a mask to indicate which positions can be masked
        msa_is_padded_mask = torch.randint(0, 2, (n_rows, n_tokens_across_chains)).bool()

        # Create a mask which is 1's across protein and RNA, and 0's across DNA and atomized
        token_idx_has_msa = torch.zeros(n_tokens_across_chains, dtype=torch.bool)
        token_idx_has_msa[: (example_protein_msa.shape[1] + example_rna_msa.shape[1])] = True

    ############## Run the functions ##############

    # Reset the seed (since we may have changed it when generating synthetic data)
    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        index_can_be_masked = build_msa_index_can_be_masked(
            msa_is_padded_mask=msa_is_padded_mask,
            token_idx_has_msa=token_idx_has_msa,
            encoded_msa=encoded_msa,
            encoding=encoding,
        )

        new_partial_msa, mask_position = mask_msa_like_bert(
            encoding=encoding,
            mask_behavior_probs=mask_behavior_probs,
            mask_probability=mask_probability,
            full_msa_profile=full_msa_profile,
            encoded_msa=encoded_msa[selected_indices],
            index_can_be_masked=index_can_be_masked[selected_indices],
        )
        new_encoded_msa = encoded_msa.clone()
        new_encoded_msa[selected_indices] = new_partial_msa

        ############## Assertions ##############

        # Step 1: Ensure things that weren't suppose to change, didn't change

        # Check that padding positions remain unchanged
        assert torch.equal(encoded_msa[msa_is_padded_mask], new_encoded_msa[msa_is_padded_mask])

        # Check that the extra MSA columns remained unchanged
        assert torch.equal(encoded_msa[not_selected_indices], new_encoded_msa[not_selected_indices])

        # Step 2: Check that mask_position holds the correct values

        # Check that no masking occurs where we didn't want any
        assert torch.all(~mask_position[~index_can_be_masked[selected_indices]])

        # Check that there is masking where we did want some
        assert torch.any(mask_position[index_can_be_masked[selected_indices]])

        # Create a tensor to indicate protein columns
        protein_columns = torch.zeros(n_tokens_across_chains, dtype=torch.int)

        # Fill the protein columns with 1s
        protein_columns[: example_protein_msa.shape[1]] = 1

        # Ensure mask_position is False for all non-protein columns
        assert torch.all(~mask_position[:, ~protein_columns.bool()])

        # Step 2: Check that we have approximately the right number of mask tokens

        # Create a mask that indicates the possible positions of masked tokens
        num_could_be_masked = index_can_be_masked[selected_indices].sum().item()

        # Check the number of mask tokens is approximately mask_probability * (1 - sum of mask_behavior_probs)
        expected_num_mask_applied = int(mask_probability * num_could_be_masked)
        actual_num_mask_applied = mask_position.sum().item()

        # Calculate the standard deviation of the binomial distribution = sqrt(n * p * (1 - p))
        std_dev = (num_could_be_masked * mask_probability * (1 - mask_probability)) ** 0.5

        # Check that the actual number of masks is within 2 standard deviations of the expected number
        assert abs(actual_num_mask_applied - expected_num_mask_applied) <= 2 * std_dev

        # Check that the number of mask tokens is close to the expected number
        mask_token_probability = 1 - sum(list(mask_behavior_probs.values()))
        expected_num_mask_tokens = actual_num_mask_applied * mask_token_probability
        actual_num_mask_tokens = (new_partial_msa == mask_token).sum().item()

        # Check that the number of mask tokens is within 2 standard deviations of the expected number
        std_dev = (actual_num_mask_applied * mask_token_probability * (1 - mask_token_probability)) ** 0.5
        assert abs(actual_num_mask_tokens - expected_num_mask_tokens) <= 2 * std_dev

        ############## Regression test ##############

        # Save in the test directory
        SAVED_RESULT_PATH = Path(__file__).resolve().parents[2] / "data" / "mask_msa_regression_test.json"

        # Uncomment to save new_encoded_msa for regression tests, as a JSON
        # with open(SAVED_RESULT_PATH, "w") as f:
        #     json.dump(new_encoded_msa.tolist(), f)

        # Check that the new_encoded_msa matches the saved results
        old_results = json.load(open(SAVED_RESULT_PATH))
        assert torch.allclose(new_encoded_msa, torch.tensor(old_results), atol=1e-4, rtol=1e-4)


MSA_FEATURIZE_PIPELINE_TEST_CASES = ["3ejj"]


@pytest.mark.parametrize("pdb_id", MSA_FEATURIZE_PIPELINE_TEST_CASES)
def test_msa_featurize_full_pipeline(pdb_id):
    # Hyperparameters (to be defined in Hydra)
    encoding = RF2AA_ATOM36_ENCODING
    n_msa_cluster_representatives = 20
    n_extra_rows = 20
    n_recycles = 5  # We choose 5 recycles to ensure we would find any drift across recycles
    probs = {
        "replace_with_random_aa": 0.1,
        "replace_with_msa_profile": 0.1,
        "do_not_replace": 0.1,
    }
    mask_probability = 0.15
    PAD_TOKEN = encoding.token_to_idx["UNK"]

    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        # Execution code
        row = PN_UNITS_DF[PN_UNITS_DF["pdb_id"] == pdb_id].iloc[0]  # Get the first row; we don't care which we choose
        data = load_from_row(row, PNUnitsDFParser(), cif_parser=CIF_PARSER)
        pipeline = Compose(
            [
                RemoveHydrogens(),
                AddWithinPolyResIdxAnnotation(),
                LoadPolymerMSAs(protein_msa_dir=PROTEIN_MSA_DIR, rna_msa_dir=RNA_MSA_DIR, max_msa_sequences=1000),
                PairAndMergePolymerMSAs(),
                AtomizeResidues(
                    atomize_by_default=True, res_names_to_ignore=encoding.tokens, move_atomized_part_to_end=True
                ),
                EncodeAtomArray(encoding),
                # MSA featurize workflow
                EncodeMSA(encoding_function=encode_msa_like_RF2AA),
                FillFullMSAFromEncoded(pad_token=PAD_TOKEN),
                ConvertToTorch(keys=["polymer_msas_by_chain_id", "encoded", "full_msa_details"]),
                FeaturizeMSALikeRF2AA(
                    n_recycles=n_recycles,
                    n_msa_cluster_representatives=n_msa_cluster_representatives,
                    n_extra_rows=n_extra_rows,
                    mask_behavior_probs=probs,
                    mask_probability=mask_probability,
                    encoding=encoding,
                    polymer_token_indices=torch.arange(
                        32
                    ),  # NOTE: This is hard-coded for the AA and NA tokens in the RF2AA Encoding (all non-atom tokens)
                ),
            ],
            track_rng_state=False,
        )
        output = pipeline(data)
        assert output is not None

        ############## Assertions ##############
        features_by_recycle = output["features_per_recycle_dict"]

        # Helper function to check if all elements in a list are different
        def all_different(tensor_list):
            for i in range(len(tensor_list)):
                for j in range(i + 1, len(tensor_list)):
                    if torch.equal(tensor_list[i], tensor_list[j]):
                        return False
            return True

        # Helper function to check if means and standard deviations are within a certain range
        def similar_stats(tensor_list):
            means = [tensor.float().mean().item() for tensor in tensor_list]
            stds = [tensor.float().std().item() for tensor in tensor_list]
            mean_mean = sum(means) / len(means)
            mean_std = sum(stds) / len(stds)
            for m, s in zip(means, stds):
                if not (0.3 * mean_mean <= m <= 1.3 * mean_mean) or not (0.7 * mean_std <= s <= 1.3 * mean_std):
                    return False
            return True

        # List of keys to check for being different and having similar sums
        keys_to_check = [
            "first_row_of_msa",  # NOTE: This will fail if our test example has no polymers
            "cluster_representatives_msa_ground_truth",
            "cluster_representatives_msa_masked",
            "cluster_representatives_has_insertion",
            "cluster_representatives_insertion_value",
            "cluster_insertion_mean",
            "cluster_profile",
            "extra_msa",
            "extra_msa_has_insertion",
            "extra_msa_insertion_value",
            "bert_mask_position",
        ]

        for key in keys_to_check:
            tensor_list = features_by_recycle[key]
            assert all_different(tensor_list), f"{key} elements are not all different"
            assert similar_stats(tensor_list), f"{key} elements do not have similar means and standard deviations"

        ############## Regression test ##############

        # Save in the test directory
        SAVED_RESULT_PATH = Path(__file__).resolve().parents[2] / "data" / f"{pdb_id}_featurize_msa_regression_test.pkl"

        # Uncomment to save output['features_per_recycle_dict] for regression tests, as a pickle (JSON is too slow)
        # with open(SAVED_RESULT_PATH, "wb") as f:
        #     pickle.dump(output["features_per_recycle_dict"], f)

        # Check that the new_encoded_msa matches the saved results
        with open(SAVED_RESULT_PATH, "rb") as f:
            old_results = pickle.load(f)

        # For each key in the dictionary, check that the values match
        for key, old_values in old_results.items():
            new_values = output["features_per_recycle_dict"][key]
            assert torch.allclose(
                torch.stack(new_values), torch.stack(old_values), atol=1e-4, rtol=1e-4
            ), f"Failed at key: {key}. Difference: {set(new_values) - set(old_values)}"


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=WARNING", __file__])
