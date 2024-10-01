import pytest
import torch
import pickle
from pathlib import Path

from datahub.transforms.diffusion.edm import sample_t_edm, sample_noise_edm

def test_edm_t_sampling():
    torch.manual_seed(0)
    diffusion_batch_size = 1000
    sigma_data = 1.0

    t = sample_t_edm(sigma_data, diffusion_batch_size)
    assert t.shape == (diffusion_batch_size,)

    # regression test; does the distribution match?

    SAVED_RESULT_PATH = (
        Path(__file__).resolve().parents[2] / "data" / f"edm_t_sampled.pkl"
    )
    # Uncomment to save t for regression tests, as a pickle (JSON is too slow)
    #with open(SAVED_RESULT_PATH, "wb") as f:
        #pickle.dump(t, f)

    with open(SAVED_RESULT_PATH, "rb") as f:
        expected_t = pickle.load(f)
    assert torch.eq(t, expected_t).all()


def test_edm_noise_sampling():
    torch.manual_seed(0)
    num_atoms = 10
    diffusion_batch_size = 1000
    sigma_data = 1.0

    t = torch.randn(diffusion_batch_size) * sigma_data # spoofing t sampling to decouple tests
    noise = sample_noise_edm(t, num_atoms)
    assert noise.shape == (diffusion_batch_size, num_atoms, 3)

    # regression test; does the distribution match?

    SAVED_RESULT_PATH = (
        Path(__file__).resolve().parents[2] / "data" / f"edm_noise_sampled.pkl"
    )
    # Uncomment to save noise for regression tests, as a pickle (JSON is too slow)
    #with open(SAVED_RESULT_PATH, "wb") as f:
        #pickle.dump(noise, f)

    with open(SAVED_RESULT_PATH, "rb") as f:
        expected_noise = pickle.load(f)
    assert torch.eq(noise, expected_noise).all()

def test_sample_edm_noise_transform():
    torch.manual_seed(0)
    num_atoms = 10
    diffusion_batch_size = 1000
    sigma_data = 1.0

    from datahub.transforms.diffusion.edm import SampleEDMNoise

    transform = SampleEDMNoise(sigma_data, diffusion_batch_size)
    data = {"ground_truth": {"coord_atom_lvl": torch.randn(num_atoms, 3)}}

    data = transform(data)
    assert "t" in data
    assert "noise" in data
    assert data["t"].shape == (diffusion_batch_size,)
    assert data["noise"].shape == (diffusion_batch_size, num_atoms, 3)
