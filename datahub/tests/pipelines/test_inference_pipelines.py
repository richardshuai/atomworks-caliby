"""Tests for the AF3 inference pipeline to ensure proper transformation of protein structures."""

import pytest
import torch
from cifutils.tools.inference import components_to_atom_array, read_chai_fasta
from cifutils.utils.non_rcsb import infer_chain_info_from_atom_array

from datahub.pipelines.af3 import build_af3_transform_pipeline
from datahub.pipelines.rf2aa import build_rf2aa_transform_pipeline
from tests.conftest import PROTEIN_MSA_DIRS, RNA_MSA_DIRS, TEST_DATA_DIR


def test_af3_pipeline_from_chai_fasta():
    """Test the AF3 transformation pipeline with different configurations.

    Tests loading FASTA files and running them through the AF3 pipeline with different settings
    to ensure proper transformation of protein structures.
    """
    # Load chai fasta
    fasta_path = TEST_DATA_DIR / "inference_like_chai_fasta.fasta"
    inference_input_components = read_chai_fasta(fasta_path)
    atom_array = components_to_atom_array(inference_input_components)
    chain_info = infer_chain_info_from_atom_array(atom_array)

    assert atom_array is not None, "Failed to load atom array from FASTA file"

    # Build and run af3 inference pipeline
    pipeline = build_af3_transform_pipeline(
        is_inference=True,
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
    )

    transformed_data = pipeline(
        data={
            "example_id": str(fasta_path),
            "atom_array": atom_array,
            "chain_info": chain_info,
        }
    )

    # Basic validation checks
    assert "feats" in transformed_data, "Missing feats in pipeline output."
    # Check that none of the feats is `nan`
    for feat_name, feat in transformed_data["feats"].items():
        assert (
            feat.isfinite().all() if isinstance(feat, torch.Tensor) else True
        ), f"Found NaN in feats: {feat_name=}, {feat=}"


def test_rf2aa_pipeline_from_chai_fasta():
    """Test the RF2AA transformation pipeline with different configurations."""
    # Load chai fasta
    fasta_path = TEST_DATA_DIR / "inference_like_chai_fasta.fasta"
    inference_input_components = read_chai_fasta(fasta_path)
    atom_array = components_to_atom_array(inference_input_components)
    chain_info = infer_chain_info_from_atom_array(atom_array)

    # Build and run rf2aa inference pipeline
    pipeline = build_rf2aa_transform_pipeline(
        is_inference=True,
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
    )

    transformed_data = pipeline(
        data={
            "example_id": str(fasta_path),
            "atom_array": atom_array,
            "chain_info": chain_info,
        }
    )

    # Basic validation checks
    assert "feats" in transformed_data, "Missing feats in pipeline output."


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
