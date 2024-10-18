"""
Includes tests to:
    - Run through the data loading pipeline to ensure examples can process without error
    - Benchmark the pipeline on a pre-defined set of examples to facilitate performance comparisons (slow; should be run separately)
"""

import logging
import multiprocessing
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.autonotebook import tqdm

from datahub.datasets.datasets import ConcatDatasetWithID, get_row_and_index_by_example_id
from tests.datasets.conftest import (
    AF3_AF2FB_DISTILLATION_DATASET,
    AF3_PDB_DATASET,
    AF3_VALIDATION_DATASET,
    RF2AA_AF2FB_DISTILLATION_DATASET,
    RF2AA_PDB_DATASET,
    RF2AA_VALIDATION_DATASET,
)

logger = logging.getLogger(__name__)


DATASETS_TO_TEST = [
    {
        "dataset": RF2AA_VALIDATION_DATASET,
        "type": "validation",
        "num_examples": 1,
    },
    {
        "dataset": AF3_VALIDATION_DATASET,
        "type": "validation",
        "num_examples": 2,
    },
    {
        "dataset": RF2AA_PDB_DATASET,
        "type": "train",
        "num_examples": 5,
    },
    {
        "dataset": AF3_PDB_DATASET,
        "type": "train",
        "num_examples": 5,
    },
    {
        "dataset": RF2AA_AF2FB_DISTILLATION_DATASET,
        "type": "train",
        "num_examples": 1,
    },
    {
        "dataset": AF3_AF2FB_DISTILLATION_DATASET,
        "type": "train",
        "num_examples": 1,
    },
]


def identity_collate_fn(batch):
    return batch


@pytest.mark.parametrize("dataset_to_test", DATASETS_TO_TEST)
@pytest.mark.slow
def test_data_loading_pipeline_with_multiple_workers(dataset_to_test: dict):
    """Test random examples using a DataLoader with basic smoke tests."""
    dataset = dataset_to_test["dataset"]
    dataset_type = dataset_to_test["type"]

    # Set the seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    def worker_init_fn(worker_id):
        # For reproducibility when using multiple workers
        seed = 0 + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    # Select deterministic examples to profile
    # NOTE: TEST_FILTERS ensures we don't end up with any huge examples that would slow down the test
    deterministic_indices = np.random.choice(len(dataset), dataset_to_test["num_examples"], replace=False)

    # Create a Subset of the dataset with the selected indices
    subset = Subset(dataset, deterministic_indices)

    # Create a DataLoader for the subset
    # We must include an identity collate_fn to avoid errors with unrecognized types (AtomArray)
    available_cores = multiprocessing.cpu_count()
    data_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=min(available_cores, 4, dataset_to_test["num_examples"])
        if dataset_to_test["num_examples"] > 1
        else 0,  # Don't spawn more workers than examples
        worker_init_fn=worker_init_fn,
        collate_fn=identity_collate_fn,
    )

    for i, sample in enumerate(tqdm(data_loader, desc="Loading examples", total=len(deterministic_indices))):
        example_id = sample[0]["example_id"]
        logger.info(f"Loaded example_id: {example_id} ({i+1}/{len(deterministic_indices)})")
        row = get_row_and_index_by_example_id(dataset, example_id)[
            "row"
        ]  # Check if we can reverse-engineer the row from the example_id
        assert row is not None, f"Failed to get row from example_id for example_id: {example_id}"
        assert sample is not None, f"Sample is None, with example_id: {example_id}"

        # For validation datasets, also check that the "ground_truth" key contains information on which chains/interfaces to score, and the map from token index to `chain_iid`
        if dataset_type == "validation":
            assert "ground_truth" in sample[0], f"Missing 'ground_truth' key in sample with example_id: {example_id}"
            assert (
                "chain_iid_token_lvl" in sample[0]["ground_truth"]
            ), f"Missing 'chain_iid_token_lvl' key in sample with example_id: {example_id}"


BENCHMARK_EXAMPLE_IDS = [
    "{['pdb', 'pn_units']}{7d9h}{2}{['B_1']}",
    "{['pdb', 'pn_units']}{5gam}{1}{['C_1']}",  # Large MSA
    "{['pdb', 'interfaces']}{6m2z}{1}{['A_4', 'B_1']}",
    "{['pdb', 'interfaces']}{7kf1}{2}{['F_1', 'F_3']}",
    "{['pdb', 'interfaces']}{7cjg}{1}{['C_1', 'H_1']}",
    "{['pdb', 'pn_units']}{7b1w}{1}{['N_1']}",
    "{['pdb', 'pn_units']}{6zie}{1}{['E_1']}",
    "{['pdb', 'interfaces']}{7nmj}{1}{['D_1', 'L_1']}",
]


@pytest.mark.parametrize("dataset", DATASETS_TO_TEST)
@pytest.mark.benchmark
@pytest.mark.very_slow
def test_data_loading_benchmark(benchmark, dataset: ConcatDatasetWithID):
    """Benchmark a pre-defined set of examples to profile the data loading pipeline."""
    logger.info("Starting benchmarking of data loading pipeline")

    # Get the indices of the pre-selected examples
    indices = []
    for example_id in BENCHMARK_EXAMPLE_IDS:
        index = get_row_and_index_by_example_id(dataset, example_id)["index"]
        indices.append(index)

    def load_samples():
        for idx in indices:
            sample = dataset[idx]
            assert sample is not None, f"Sample at index {idx} is None"

    # Benchmark the loading of the samples
    benchmark.pedantic(load_samples, iterations=2, rounds=1)


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=INFO", "-m slow", __file__])
