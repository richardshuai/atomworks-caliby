from datahub.transforms._checks import check_contains_keys
from datahub.transforms.base import Transform
from datahub.utils.geometry import apply_batched_rigid, get_random_rigid


def center(coord_atom_lvl, mask_atom_lvl):
    atoms = coord_atom_lvl[mask_atom_lvl]
    center = atoms.mean(0)
    coord_atom_lvl = coord_atom_lvl - center
    return coord_atom_lvl


def random_augmentation(coord_atom_lvl, batch_size, s=1.0):
    rigid = get_random_rigid(batch_size, scale=s)

    # get random rigid squeezes dimension for batch_size=1
    if batch_size == 1:
        rigid = rigid[0].unsqueeze(0), rigid[1].unsqueeze(0)
    return apply_batched_rigid(rigid, coord_atom_lvl)


class CenterRandomAugmentation(Transform):
    """
    Centers coordinates and then randomly rotates and translates the input coordinates.
    """

    requires_previous_transforms = ["BatchStructures"]

    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def check_input(self, data):
        check_contains_keys(data, ["ground_truth"])
        check_contains_keys(data["ground_truth"], ["coord_atom_lvl", "mask_atom_lvl"])
        assert (
            data["ground_truth"]["coord_atom_lvl"].shape[0] == self.batch_size
        ), "must batch coordinates before applying this transform"
        assert (
            data["ground_truth"]["mask_atom_lvl"].shape[0] == self.batch_size
        ), "must batch mask before applying this transform"

    def forward(self, data):
        coord_atom_lvl = data["ground_truth"]["coord_atom_lvl"]
        mask_atom_lvl = data["ground_truth"]["mask_atom_lvl"]
        coord_atom_lvl = center(coord_atom_lvl, mask_atom_lvl)
        coord_atom_lvl = random_augmentation(coord_atom_lvl, batch_size=self.batch_size)
        data["ground_truth"]["coord_atom_lvl"] = coord_atom_lvl
        return data
