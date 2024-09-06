import numpy as np
from cifutils.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA

from datahub.transforms.af3_reference_molecule import GetAF3ReferenceMoleculeFeatures
from datahub.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddWithinChainInstanceResIdx,
    RemoveHydrogens,
)
from datahub.transforms.atomize import AtomizeResidues
from datahub.transforms.base import Compose, RandomRoute
from datahub.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from datahub.transforms.encoding import EncodeAF3TokenLevelFeatures


def build_af3_pipeline(
    crop_size: int = 384,
    crop_center_cutoff_distance: float = 15.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    conformer_generation_timeout: float = 10.0,
):
    assert np.isclose(
        crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
    ), "Crop probabilities must sum to 1.0"
    assert crop_size > 0, "Crop size must be greater than 0"
    assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"

    transforms = [
        RemoveHydrogens(),
        AddGlobalAtomIdAnnotation(),
        AtomizeResidues(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        AddWithinChainInstanceResIdx(),
    ]

    # Crop
    contiguous_crop_transform = CropContiguousLikeAF3(crop_size=crop_size, keep_uncropped_atom_array=True)
    spatial_crop_transform = CropSpatialLikeAF3(
        crop_size=crop_size, crop_center_cutoff_distance=crop_center_cutoff_distance, keep_uncropped_atom_array=True
    )
    if crop_contiguous_probability > 0 and crop_spatial_probability > 0:
        transforms += [
            # ...crop around our query pn_unit(s) early, since we don't need the full structure moving forward
            RandomRoute(
                transforms=[
                    contiguous_crop_transform,
                    spatial_crop_transform,
                ],
                probs=[crop_contiguous_probability, crop_spatial_probability],
            ),
        ]
    elif crop_contiguous_probability > 0:
        transforms.append(contiguous_crop_transform)
    elif crop_spatial_probability > 0:
        transforms.append(spatial_crop_transform)

    transforms += []

    transforms += [
        EncodeAF3TokenLevelFeatures(),
        GetAF3ReferenceMoleculeFeatures(
            conformer_generation_timeout=conformer_generation_timeout,
        ),
    ]

    # ... compose final pipelien
    pipeline = Compose(transforms)
    return pipeline
