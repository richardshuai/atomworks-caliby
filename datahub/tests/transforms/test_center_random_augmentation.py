import torch

import pytest
from datahub.transforms.center_random_augmentation import center, random_augmentation


def test_center():
    torch.manual_seed(0)

    coord_atom_lvl = torch.randn(1, 10, 3)
    mask_atom_lvl = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])[None].bool()

    coord_atom_lvl_center = center(coord_atom_lvl, mask_atom_lvl)
    assert torch.allclose(coord_atom_lvl_center[mask_atom_lvl].mean(0), torch.zeros(3), atol=1e-6, rtol=1e-6)

def test_random_augmentation():
    torch.manual_seed(0)

    coord_atom_lvl = torch.randn(1, 10, 3)
    batch_size = 1

    coord_atom_lvl_augmented = random_augmentation(coord_atom_lvl, batch_size=batch_size)

    assert coord_atom_lvl_augmented.shape == (batch_size, 10, 3)
    assert not torch.allclose(coord_atom_lvl, coord_atom_lvl_augmented)

def test_center_random_augmentation():
    torch.manual_seed(0)

    coord_atom_lvl = torch.randn(1, 10, 3)
    mask_atom_lvl = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])[None].bool()
    batch_size = 1

    from datahub.transforms.center_random_augmentation import CenterRandomAugmentation

    transform = CenterRandomAugmentation(batch_size)
    data = {"ground_truth": {"coord_atom_lvl": coord_atom_lvl, "mask_atom_lvl": mask_atom_lvl}}

    data = transform(data)

    assert data["ground_truth"]["coord_atom_lvl"].shape == (batch_size, 10, 3)
    # make sure the structure was translated
    assert not torch.allclose(data["ground_truth"]["coord_atom_lvl"][mask_atom_lvl].mean(0), torch.zeros(3), atol=1e-6, rtol=1e-6)
    # make sure the structure was rotated
    assert not torch.allclose(coord_atom_lvl, data["ground_truth"]["coord_atom_lvl"], atol=1e-6, rtol=1e-6)
