import torch

from datahub.transforms._checks import check_contains_keys
from datahub.transforms.base import Transform


def sample_t_edm(sigma_data, diffusion_batch_size):
    # Reference for h-params: NVIDIA EDM Paper (https://arxiv.org/pdf/2206.00364)
    t = sigma_data * torch.exp(-1.2 + 1.5 * torch.normal(mean=0, std=1, size=(diffusion_batch_size,)))
    return t


def sample_noise_edm(t, num_atoms):
    t_tiled = t[:, None, None].tile(1, num_atoms, 3)
    return torch.normal(mean=0, std=1, size=t_tiled.shape) * t_tiled


class SampleEDMNoise(Transform):
    def __init__(self, sigma_data, diffusion_batch_size, **kwargs):
        super().__init__(**kwargs)
        self.sigma_data = sigma_data
        self.diffusion_batch_size = diffusion_batch_size

    def check_input(self, data):
        check_contains_keys(data, ["ground_truth"])
        check_contains_keys(data["ground_truth"], ["coord_atom_lvl"])

    def forward(self, data):
        t = sample_t_edm(self.sigma_data, self.diffusion_batch_size)
        noise = sample_noise_edm(t, data["ground_truth"]["coord_atom_lvl"].shape[0])
        data["t"] = t
        data["noise"] = noise
        return data
