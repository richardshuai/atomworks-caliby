import numpy as np
import torch
from cifutils.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from cifutils.enums import ChainType

from datahub.encoding_definitions import AF3SequenceEncoding
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
from datahub.transforms.template import AddRFTemplates, FeaturizeTemplatesLikeAF3


def build_af3_pipeline(
    # Crop params
    crop_size: int = 384,
    crop_center_cutoff_distance: float = 15.0,
    crop_contiguous_probability: float = 0.5,
    crop_spatial_probability: float = 0.5,
    # Conformer generation params
    conformer_generation_timeout: float = 10.0,
    # Template params
    n_template: int = 5,
    pick_top_templates: bool = False,
    template_max_seq_similarity: float = 60.0,
    template_min_seq_similarity: float = 10.0,
    template_min_length: int = 10,
    template_allowed_chain_types: list[ChainType] = [ChainType.POLYPEPTIDE_L, ChainType.RNA],
    template_distogram_bins: torch.Tensor = torch.linspace(3.25, 50.75, 38),
    template_default_token: str = "<G>",
):
    """Build the AF3 pipeline with specified parameters.

    This function constructs a pipeline of transforms for processing protein structures
    in a manner similar to AlphaFold 3. The pipeline includes steps for removing hydrogens,
    adding annotations, atomizing residues, cropping, adding templates, encoding features,
    and generating reference molecule features.

    Args:
        crop_size (int, optional): The size of the crop. Defaults to 384.
        crop_center_cutoff_distance (float, optional): The cutoff distance for spatial cropping.
            Defaults to 15.0.
        crop_contiguous_probability (float, optional): The probability of using contiguous cropping.
            Defaults to 0.5.
        crop_spatial_probability (float, optional): The probability of using spatial cropping.
            Defaults to 0.5.
        conformer_generation_timeout (float, optional): The timeout for conformer generation in seconds.
            Defaults to 10.0.

    Returns:
        Transform: A composed pipeline of transforms.

    Raises:
        AssertionError: If the crop probabilities do not sum to 1.0, if the crop size is not positive,
        or if the crop center cutoff distance is not positive.

    Note:
        The cropping method is chosen randomly based on the provided probabilities.
        The pipeline includes steps for processing the structure, adding annotations,
        and generating features required for AF3-like predictions.

    References:
        - AlphaFold 3 Supplementary Information.
          https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
    """
    assert np.isclose(
        crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
    ), "Crop probabilities must sum to 1.0"
    assert crop_size > 0, "Crop size must be greater than 0"
    assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"

    af3_sequence_encoding = AF3SequenceEncoding()

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
        EncodeAF3TokenLevelFeatures(sequence_encoding=af3_sequence_encoding),
        GetAF3ReferenceMoleculeFeatures(
            conformer_generation_timeout=conformer_generation_timeout,
        ),
        AddRFTemplates(
            max_n_template=n_template,
            pick_top=pick_top_templates,
            max_seq_similarity=template_max_seq_similarity,
            min_seq_similarity=template_min_seq_similarity,
            min_template_length=template_min_length,
        ),
        FeaturizeTemplatesLikeAF3(
            n_templates=n_template,
            sequence_encoding=af3_sequence_encoding,
            gap_token=template_default_token,
            allowed_chain_type=template_allowed_chain_types,
            distogram_bins=template_distogram_bins,
        ),
    ]

    # ... compose final pipelien
    pipeline = Compose(transforms)
    return pipeline
