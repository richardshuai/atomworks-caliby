"""
Includes tests to:
- Run through the data loading pipeline and ensure the examples come out with the
  correct shapes to plug into RF2AA
"""

import logging
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datahub.pipelines.rf2aa import assert_satisfies_rf2aa_assumptions
from tests.datasets.conftest import RF2AA_PDB_DATASET

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_satisfies_rf2aa_assumptions(pdb_dataset=RF2AA_PDB_DATASET):
    NUM_RANDOM_EXAMPLES = 10

    # Set the seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Select deterministic examples to profile
    # NOTE: TEST_FILTERS ensures we don't end up with any huge examples that would slow down the test
    deterministic_indices = np.random.choice(len(pdb_dataset), NUM_RANDOM_EXAMPLES, replace=False)

    # Create a Subset of the dataset with the selected indices
    subset = Subset(pdb_dataset, deterministic_indices)

    # Create a DataLoader for the subset
    data_loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
    )

    for sample in tqdm(data_loader):
        example_id = sample["example_id"][0]
        try:
            assert_satisfies_rf2aa_assumptions(sample)
        except AssertionError as e:
            # Update message with sample index
            rng_state_dict = pdb_dataset.get_dataset_by_idx(
                pdb_dataset.id_to_idx(example_id)
            ).transform.latest_rng_state_dict
            raise AssertionError(f"Assertion failed for sample {example_id}." + "\n" + f"{rng_state_dict}") from e


if __name__ == "__main__":
    pytest.main(["-s", "-v", "-m", "slow", __file__])
