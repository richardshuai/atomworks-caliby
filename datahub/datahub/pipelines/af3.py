import numpy as np
from cifutils.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA

from datahub.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddWithinChainInstanceResIdx,
    RemoveHydrogens,
)
from datahub.transforms.atomize import AtomizeResidues
from datahub.transforms.base import Compose
from datahub.transforms.crop import CropSpatialLikeAF3


def build_af3_pipeline(
    crop_size: int = 256,
    crop_center_cutoff_distance: float = 15.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
):
    assert np.isclose(
        crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
    ), "Crop probabilities must sum to 1.0"
    assert crop_size > 0, "Crop size must be greater than 0"
    assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"

    pipeline = Compose(
        [
            RemoveHydrogens(),
            AddGlobalAtomIdAnnotation(),
            AtomizeResidues(
                atomize_by_default=True,
                res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
                move_atomized_part_to_end=False,
                validate_atomize=False,
            ),
            AddWithinChainInstanceResIdx(),
            CropSpatialLikeAF3(),
        ]
    )

    return pipeline
