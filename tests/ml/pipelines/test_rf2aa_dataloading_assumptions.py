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

from atomworks.ml.transforms.rf2aa_assumptions import assert_satisfies_rf2aa_assumptions

logger = logging.getLogger(__name__)


def identity_collate_fn(batch):
    return batch


@pytest.mark.skip
def test_satisfies_rf2aa_assumptions(rf2aa_pdb_dataset):
    """
    NOTE: This test is stochastic; it's results should only be interpreted in-context.
    """
    NUM_RANDOM_EXAMPLES = 5

    # Set the seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Select deterministic examples to profile
    # NOTE: TEST_FILTERS ensures we don't end up with any huge examples that would slow down the test
    deterministic_indices = np.random.choice(len(rf2aa_pdb_dataset), NUM_RANDOM_EXAMPLES, replace=False)

    for index in deterministic_indices:
        sample = rf2aa_pdb_dataset[index]
        example_id = sample["example_id"]

        try:
            assert_satisfies_rf2aa_assumptions(sample["feats"])
        except AssertionError as e:
            # Update message with sample index
            rng_state_dict = rf2aa_pdb_dataset.get_dataset_by_idx(
                rf2aa_pdb_dataset.id_to_idx(example_id)
            ).transform.latest_rng_state_dict
            raise AssertionError(f"Assertion failed for sample {example_id}." + "\n" + f"{rng_state_dict}") from e


if __name__ == "__main__":
    pytest.main(["-s", "-v", "-m", "slow", __file__])
