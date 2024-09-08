import random

import numpy as np
import pytest
import torch

from datahub.utils.rng import (
    create_rng_state_from_seeds,
    rng_state,
    serialize_rng_state_dict,
)


def test_rng_state():
    # Collect initial random values outside the context manager
    initial_numpy = np.random.random(3)
    initial_torch = torch.rand(3)
    initial_python = [random.random() for _ in range(3)]

    # Inside the context manager with fixed seeds
    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)) as rng_state_dict:
        my_state = serialize_rng_state_dict(rng_state_dict)
        numpy_inside = np.random.random(3)
        torch_inside = torch.rand(3)
        python_inside = [random.random() for _ in range(3)]

    # Collect random values again outside the context manager
    final_numpy = np.random.random(3)
    final_torch = torch.rand(3)
    final_python = [random.random() for _ in range(3)]

    # Inside the context manager with fixed seeds again
    with rng_state(eval(my_state)):
        numpy_inside_again = np.random.random(3)
        torch_inside_again = torch.rand(3)
        python_inside_again = [random.random() for _ in range(3)]

    # Assertions
    NUMPY_INSIDE = np.array([0.37454012, 0.95071431, 0.73199394])
    TORCH_INSIDE = torch.tensor([0.88226926, 0.91500396, 0.38286376])
    PYTHON_INSIDE = [0.6394267984578837, 0.025010755222666936, 0.27502931836911926]

    assert not np.array_equal(
        initial_numpy, final_numpy
    ), "NumPy values should be different outside the context manager"
    assert not torch.equal(initial_torch, final_torch), "PyTorch values should be different outside the context manager"
    assert initial_python != final_python, "Python random values should be different outside the context manager"

    assert np.array_equal(
        numpy_inside, numpy_inside_again
    ), "NumPy values should be the same inside the context manager"
    assert torch.equal(torch_inside, torch_inside_again), "PyTorch values should be the same inside the context manager"
    assert python_inside == python_inside_again, "Python random values should be the same inside the context manager"

    assert np.allclose(numpy_inside, NUMPY_INSIDE), "NumPy values should be the same inside the context manager"
    assert torch.allclose(torch_inside, TORCH_INSIDE), "PyTorch values should be the same inside the context manager"
    assert python_inside == PYTHON_INSIDE, "Python random values should be the same inside the context manager"


if __name__ == "__main__":
    # Dev setup:
    # Run all tests in the file with verbose output and stop after first failure
    pytest.main(["-v", "-x", __file__])
