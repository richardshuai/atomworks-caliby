import copy

import pytest
import torch

from datahub.encoding_definitions import RF2_ATOM36_ENCODING, RF2AA_ATOM36_ENCODING
from datahub.transforms.atom_array import (
    AddGlobalTokenIdAnnotation,
    AddWithinPolyResIdxAnnotation,
)
from datahub.transforms.base import Compose
from datahub.transforms.encoding import EncodeAtomArray, TokenEncoding
from datahub.transforms.filters import FilterToProteins, RemoveHydrogens, RemoveTerminalOxygen
from datahub.transforms.template import (
    AddRFTemplates,
    FeaturizeTemplatesLikeRF2AA,
    RandomSubsampleTemplates,
    RF2AATemplate,
)
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
def test_subsample_template(test_case: dict):
    pdb_id = test_case["pdb_id"]
    data = cached_parse(pdb_id)

    # Create a list of the number of templates for each chain
    template_counts = []

    out_before_subsampling = AddRFTemplates(
        max_n_template=20, pick_top=False, min_seq_similarity=10, max_seq_similarity=100, min_template_length=10
    )(data)

    with rng_state(create_rng_state_from_seeds(12345)):
        for _ in range(100):
            # Sample 10 times
            out_after_subsampling = RandomSubsampleTemplates(n_template=10)(copy.deepcopy(out_before_subsampling))
            template_counts.extend(len(templates) for templates in out_after_subsampling["template"].values())

    # Assert that at least one template has < n_template
    assert any(count < 10 for count in template_counts), "Expected at least one template to have less than 10 templates"

    # Assert that no template has > n_template
    assert all(count <= 10 for count in template_counts), "Expected no template to have more than 10 templates"

    # Assert the mean is what we would expect (~7.5 = 0.5 * 10 + 0.5 * 5)
    assert (
        7.5 - 1 < sum(template_counts) / len(template_counts) < 7.5 + 1
    ), f"Expected mean to be around 7.5. Found {sum(template_counts) / len(template_counts)}. This is a stochastic test, so running again may fix this."


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

    atom_array = data["atom_array"]
    n_tokens = len(atom_array[atom_array.atom_name == "CA"])
    len_token_vocabulary = len(encoding.token_atoms)

    xyz_encoded = data["encoded"]["xyz"]
    mask_encoded = data["encoded"]["mask"]
    seq_encoded = torch.nn.functional.one_hot(torch.tensor(data["encoded"]["seq"]), encoding.n_tokens)

    xyz_template = data["template_feat"]["xyz"]
    mask_template = data["template_feat"]["mask"]
    t1d_template = data["template_feat"]["t1d"]

    # Check the template features are of the correct shape
    assert xyz_template.shape == (n_template, n_tokens, 36, 3)
    assert mask_template.shape == (n_template, n_tokens, 36)
    assert t1d_template.shape == (
        n_template,
        n_tokens,
        len_token_vocabulary - 1 + 1,
    )  # -1 for removing mask, +1 for alignment confidence

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
