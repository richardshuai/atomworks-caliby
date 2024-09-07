import pytest
import torch

from datahub.encoding_definitions import RF2_ATOM36_ENCODING, RF2AA_ATOM36_ENCODING
from datahub.transforms.atom_array import (
    AddGlobalTokenIdAnnotation,
    AddWithinPolyResIdxAnnotation,
    FilterToProteins,
    RemoveHydrogens,
    RemoveTerminalOxygen,
)
from datahub.transforms.base import Compose
from datahub.transforms.encoding import EncodeAtomArray, TokenEncoding
from datahub.transforms.template import AddRFTemplates, FeaturizeTemplatesLikeRF2AA, RF2AATemplate
from datahub.utils.rng import create_rng_state_from_seeds, rng_state
from tests.conftest import cached_parse

TEST_CASES = [
    {
        "pdb_id": "5ocm",  # multi-chain template
        "n_templates": {"A": 446, "B": 446, "C": 446, "D": 446, "E": 446, "F": 446},
    },
    {
        "pdb_id": "6lyz",  # single-chain template
        "n_templates": {"A": 383},
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES)
def test_add_rf_templates(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)
    data = AddRFTemplates(
        max_n_template=1000, pick_top=False, min_seq_similarity=10, max_seq_similarity=100, min_template_length=10
    )(data)

    for chain, n_templates in test_case["n_templates"].items():
        assert (
            len(data["template"][chain]) == n_templates
        ), f"For {pdb_id}-{chain}: Expected {n_templates} templates, got {len(data['template'][chain])}"


@pytest.mark.parametrize("test_case", TEST_CASES)
@pytest.mark.parametrize("min_seq_similarity, max_seq_similarity", [(30, 55), (20, 50), (10, 80)])
@pytest.mark.parametrize("min_template_length", [10, 60])
def test_add_rf_templates_filters(
    test_case: dict, min_seq_similarity: int, max_seq_similarity: int, min_template_length: int
):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)
    transform = AddRFTemplates(
        max_n_template=5,
        pick_top=True,
        min_seq_similarity=min_seq_similarity,
        max_seq_similarity=max_seq_similarity,
        min_template_length=min_template_length,
    )
    data = transform(data)

    for chain, templates in data["template"].items():
        assert len(templates) > 0, f"No templates found for {pdb_id}-{chain}"
        assert len(templates) <= 5, f"Expected 5 templates, got {len(templates)} for {pdb_id}-{chain}"
        for template in templates:
            assert (
                min_seq_similarity <= template["seq_similarity"] <= max_seq_similarity
            ), f"Expected seq similarity between {min_seq_similarity} and {max_seq_similarity}, got {template['seq_similarity']} for {pdb_id}-{chain}-{template['id']}"
            assert (
                template["n_res"] >= min_template_length
            ), f"Expected at least {min_template_length} residues, got {template['n_res']} for {pdb_id}-{chain}-{template['id']}"


@pytest.mark.parametrize("test_case", TEST_CASES)
@pytest.mark.parametrize("encoding", [RF2_ATOM36_ENCODING, RF2AA_ATOM36_ENCODING])
def test_featurize_rf_templates(test_case: dict, encoding: TokenEncoding, n_template: int = 3):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)
    pipe = Compose(
        [
            RemoveHydrogens(),
            RemoveTerminalOxygen(),
            AddWithinPolyResIdxAnnotation(),
            FilterToProteins(),
            AddGlobalTokenIdAnnotation(),
            AddRFTemplates(
                max_n_template=2, pick_top=False, min_seq_similarity=20, max_seq_similarity=60, min_template_length=10
            ),
            FeaturizeTemplatesLikeRF2AA(
                n_template=n_template,
                encoding=encoding,
                mask_token_idx=21,
                init_coords=RF2AATemplate.RF2AA_INIT_TEMPLATE_COORDINATES,
            ),
            EncodeAtomArray(encoding=encoding),
        ],
        track_rng_state=False,
    )
    with rng_state(create_rng_state_from_seeds(12345)):
        data = pipe(data)

    xyz_encoded = data["encoded"]["xyz"]
    mask_encoded = data["encoded"]["mask"]
    seq_encoded = torch.nn.functional.one_hot(torch.tensor(data["encoded"]["seq"]), encoding.n_tokens)

    xyz_template = data["template_feat"]["xyz"]
    mask_template = data["template_feat"]["mask"]
    t1d_template = data["template_feat"]["t1d"]

    # Check the template features are of the correct shape
    assert xyz_template.shape[0] == n_template
    assert mask_template.shape[0] == n_template
    assert t1d_template.shape[0] == n_template

    assert xyz_template[0].shape == xyz_encoded.shape
    assert mask_template[0].shape == mask_encoded.shape
    assert t1d_template[0].shape == seq_encoded.shape

    # Check that the t1d last axis adds up to one when excluding the last dimension
    assert torch.all(
        t1d_template[..., :-1].sum(dim=-1) == 1
    ), f"Expected t1d last axis to add up to one (one-hot encoded), but got {t1d_template[..., :-1].sum(dim=-1)}"

    # Check that alignment confidences exist for all non-masked tokens
    is_masked = t1d_template[..., 21] == 1
    assert torch.all(
        t1d_template[..., -1][~is_masked] > 0
    ), "Expected t1d last axis to be greater than zero (alignment confidence)"

    # Check that all not-masked tokens have finite coordinates
    assert torch.all(torch.isfinite(xyz_template[mask_template])), "Expected non-masked template xyz to be finite"

    # Ensure that at least something was filled in
    assert torch.any(mask_template), "Expected at least one token to be filled in"
