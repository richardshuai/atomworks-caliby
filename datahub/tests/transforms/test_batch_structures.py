import pytest

from datahub.transforms.batch_structures import BatchStructures


def test_batch_structures():

    batch_size = 2
    coord_atom_lvl = torch.randn(10, 3)
    mask_atom_lvl = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]).bool()

    transform = BatchStructures(batch_size)
    data = {"ground_truth": {"coord_atom_lvl": coord_atom_lvl, "mask_atom_lvl": mask_atom_lvl}}

    data = transform(data)

    assert data["ground_truth"]["coord_atom_lvl"].shape == (batch_size, 10, 3)
    assert data["ground_truth"]["mask_atom_lvl"].shape == (batch_size, 10)
    assert torch.allclose(data["ground_truth"]["coord_atom_lvl"][0], data["ground_truth"]["coord_atom_lvl"][1])
    assert torch.allclose(data["ground_truth"]["mask_atom_lvl"][0], data["ground_truth"]["mask_atom_lvl"][1])