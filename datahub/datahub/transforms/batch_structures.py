from datahub.transforms._checks import check_contains_keys
from datahub.transforms.base import Transform


class BatchStructures(Transform):
    """
    Tiles the input structures to match the batch size.
    """

    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def check_input(self, data):
        check_contains_keys(data, ["ground_truth", "atom_array"])
        check_contains_keys(data["ground_truth"], ["coord_atom_lvl", "mask_atom_lvl"])
        assert len(data["ground_truth"]["coord_atom_lvl"]) == len(
            data["atom_array"]
        ), "structure must not be batched yet"
        assert len(data["ground_truth"]["mask_atom_lvl"]) == len(data["atom_array"]), "mask must not be batched yet"

    def forward(self, data):
        data["ground_truth"]["coord_atom_lvl"] = data["ground_truth"]["coord_atom_lvl"].repeat(self.batch_size, 1, 1)
        data["ground_truth"]["mask_atom_lvl"] = data["ground_truth"]["mask_atom_lvl"].repeat(self.batch_size, 1)
        return data
