"""Shared test utils and fixtures for all tests"""

from pathlib import Path

import pandas as pd
import pytest
from cifutils.constants import AF3_EXCLUDED_LIGANDS_REGEX

from datahub.datasets.datasets import ConcatDatasetWithID, PandasDataset, StructuralDatasetWrapper
from datahub.datasets.parsers import (
    GenericDFParser,
    InterfacesDFParser,
    PNUnitsDFParser,
    ValidationDFParserLikeAF3,
)
from datahub.pipelines.af3 import build_af3_transform_pipeline
from datahub.pipelines.rf2aa import build_rf2aa_transform_pipeline
from datahub.preprocessing.constants import TRAINING_SUPPORTED_CHAIN_TYPES_INTS
from datahub.utils.io import read_parquet_with_metadata

##########################################################################################
# + ------------------------------------ Constants ------------------------------------- +
##########################################################################################

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
VALIDATION_DATA_PATH = "/projects/ml/RF2_allatom/datasets/af3_splits/2024_10_18/entry_level_val_df.parquet"

PROTEIN_MSA_DIRS = [
    {
        "dir": "/projects/msa/rf2aa_af3/rf2aa_paper_model_protein_msas",
        "extension": ".a3m.gz",
        "directory_depth": 2,
    },
    {
        "dir": "/projects/msa/rf2aa_af3/missing_msas_through_2024_08_12",
        "extension": ".msa0.a3m.gz",
        "directory_depth": 2,
    },
    {
        "dir": "/projects/msa/nvidia_renamed_with_seq_hash/maxseq_10k",
        "extension": ".a3m.gz",
        "directory_depth": 2,
    },
]

RNA_MSA_DIRS = [
    {"dir": "/projects/msa/rf2aa_af3/rf2aa_paper_model_rna_msas", "extension": ".afa", "directory_depth": 0}
]

##########################################################################################
# + ----------------------------------- Dataframes ------------------------------------- +
##########################################################################################


# Interfaces/PN Units
@pytest.fixture(scope="session")
def pn_units_df():
    return read_parquet_with_metadata(f"{TEST_DATA_DIR}/pn_units_df.parquet")


@pytest.fixture(scope="session")
def interfaces_df():
    return read_parquet_with_metadata(f"{TEST_DATA_DIR}/interfaces_df.parquet")


# AF2 Distillation Facebook, with and without table-wide metadata (to test metadata handling)
@pytest.fixture(scope="session")
def af2_distillation_facebook_df_no_metadata():
    return pd.read_parquet(f"{TEST_DATA_DIR}/test_af2_distillation_facebook.parquet")


@pytest.fixture(scope="session")
def af2_distillation_facebook_df_with_metadata():
    return read_parquet_with_metadata(f"{TEST_DATA_DIR}/test_af2_distillation_facebook.parquet")


# Validation
@pytest.fixture(scope="session")
def af3_validation_df():
    return read_parquet_with_metadata(VALIDATION_DATA_PATH)


##########################################################################################
# + ------------------------------------ Datasets -------------------------------------- +
##########################################################################################

SHARED_TEST_FILTERS = [
    "deposition_date < '2022-01-01'",
    "resolution < 5.0 and ~method.str.contains('NMR')",
    "num_polymer_pn_units <= 20",  # To ensure we don't freeze loading a massive example
    "cluster.notnull()",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
]

TEST_PN_UNITS_FILTERS = [
    f"q_pn_unit_type in {TRAINING_SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
]

TEST_INTERFACES_FILTERS = [
    f"pn_unit_1_type in {TRAINING_SUPPORTED_CHAIN_TYPES_INTS}",
    f"pn_unit_2_type in {TRAINING_SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
    f"~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
]

TEST_DIFFUSION_BATCH_SIZE = 32  # Set to a value other than default (48) for testing

# +--------------------------------------------------------------------------+
# Base PandasDataset fixtures
# +--------------------------------------------------------------------------+


@pytest.fixture(scope="session")
def pn_units_pandas_dataset(pn_units_df):
    return PandasDataset(
        name="pn_units",
        id_column="example_id",
        data=pn_units_df,
        filters=SHARED_TEST_FILTERS + TEST_PN_UNITS_FILTERS,
        columns_to_load=None,  # Load all columns
    )


@pytest.fixture(scope="session")
def interfaces_pandas_dataset(interfaces_df):
    return PandasDataset(
        name="interfaces",
        id_column="example_id",
        data=interfaces_df,
        filters=SHARED_TEST_FILTERS + TEST_INTERFACES_FILTERS,
        columns_to_load=None,  # Load all columns
    )


@pytest.fixture(scope="session")
def validation_pandas_dataset(af3_validation_df):
    return PandasDataset(
        name="validation",
        data=af3_validation_df,
        id_column="example_id",
        columns_to_load=None,  # Load all columns
    )


@pytest.fixture(scope="session")
def distillation_pandas_dataset_no_metadata(af2_distillation_facebook_df_no_metadata):
    return PandasDataset(
        data=af2_distillation_facebook_df_no_metadata,
        id_column="example_id",
        name="af2fb_distillation",
        columns_to_load=["example_id", "sequence_hash", "path"],
    )


@pytest.fixture(scope="session")
def distillation_pandas_dataset_with_metadata(af2_distillation_facebook_df_with_metadata):
    return PandasDataset(
        data=af2_distillation_facebook_df_with_metadata,
        id_column="example_id",
        name="af2fb_distillation",
        columns_to_load=["example_id", "sequence_hash", "path"],
    )


# +--------------------------------------------------------------------------+
# RF2AA Dataset fixtures
# +--------------------------------------------------------------------------+


@pytest.fixture(scope="session")
def rf2aa_pn_units_dataset(pn_units_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=PNUnitsDFParser(),
        transform=build_rf2aa_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            n_recycles=5,
            crop_size=256,
            crop_contiguous_probability=1 / 3,
            crop_spatial_probability=2 / 3,
            convert_feats_to_rf2aa_input_tuple=False,
            assert_rf2aa_assumptions=False,
        ),
        dataset=pn_units_pandas_dataset,
        cif_parser_args={"cache_dir": None},
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def rf2aa_interfaces_dataset(interfaces_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=InterfacesDFParser(),
        transform=build_rf2aa_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            n_recycles=5,
            crop_size=256,
            crop_spatial_probability=1.0,
            crop_contiguous_probability=0.0,
            assert_rf2aa_assumptions=False,
            convert_feats_to_rf2aa_input_tuple=False,
        ),
        dataset=interfaces_pandas_dataset,
        cif_parser_args={"cache_dir": None},
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def rf2aa_pdb_dataset(rf2aa_pn_units_dataset, rf2aa_interfaces_dataset):
    return ConcatDatasetWithID(datasets=[rf2aa_pn_units_dataset, rf2aa_interfaces_dataset])  # NOTE: Order matters!


@pytest.fixture(scope="session")
def rf2aa_validation_dataset(validation_pandas_dataset):
    """Create a StructuralDatasetWrapper for RF2AA validation."""
    return StructuralDatasetWrapper(
        dataset_parser=ValidationDFParserLikeAF3(),
        transform=build_rf2aa_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            n_recycles=5,
            crop_size=256,
            crop_spatial_probability=0.0,  # NOTE: Zero probability for cropping; we don't crop during validation
            crop_contiguous_probability=0.0,  # NOTE: Zero probability for cropping ; we don't crop during validation
            assert_rf2aa_assumptions=False,
            convert_feats_to_rf2aa_input_tuple=False,
        ),
        dataset=validation_pandas_dataset,
        save_failed_examples_to_dir=None,
    )


# +--------------------------------------------------------------------------+
# AF3 Dataset fixtures
# +--------------------------------------------------------------------------+


@pytest.fixture(scope="session")
def af3_pn_units_dataset(pn_units_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=PNUnitsDFParser(),
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            is_inference=False,
            n_recycles=5,
            crop_size=256,
            crop_contiguous_probability=1 / 3,
            crop_spatial_probability=2 / 3,
            diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
        ),
        dataset=pn_units_pandas_dataset,
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_interfaces_dataset(interfaces_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=InterfacesDFParser(),
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            is_inference=False,
            n_recycles=5,
            crop_size=256,
            crop_spatial_probability=1.0,
            crop_contiguous_probability=0.0,
            diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
        ),
        dataset=interfaces_pandas_dataset,
        cif_parser_args={"cache_dir": None},
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_pdb_dataset(af3_pn_units_dataset, af3_interfaces_dataset):
    return ConcatDatasetWithID(datasets=[af3_pn_units_dataset, af3_interfaces_dataset])  # NOTE: Order matters!


@pytest.fixture(scope="session")
def af3_validation_dataset(validation_pandas_dataset):
    return StructuralDatasetWrapper(
        dataset_parser=ValidationDFParserLikeAF3(),
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            is_inference=True,
            n_recycles=5,
            crop_size=256,
            crop_spatial_probability=0.0,  # NOTE: Zero probability for cropping; we don't crop during validation
            crop_contiguous_probability=0.0,  # NOTE: Zero probability for cropping; we don't crop during validation
        ),
        dataset=validation_pandas_dataset,
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_af2fb_distillation_dataset_no_metadata(distillation_pandas_dataset_no_metadata):
    return StructuralDatasetWrapper(
        dataset=distillation_pandas_dataset_no_metadata,
        dataset_parser=GenericDFParser(
            base_path="/squash/af2_distillation_facebook/cif",
            extension=".cif",
        ),
        cif_parser_args={},
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=[
                {"dir": "/squash/af2_distillation_facebook/msa", "extension": ".a3m", "directory_depth": 2}
            ],
            rna_msa_dirs=[],
            diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
            is_inference=False,
        ),
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_af2fb_distillation_dataset_with_metadata(distillation_pandas_dataset_with_metadata):
    return StructuralDatasetWrapper(
        dataset=distillation_pandas_dataset_with_metadata,
        dataset_parser=GenericDFParser(),
        cif_parser_args={},
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=[
                {"dir": "/squash/af2_distillation_facebook/msa", "extension": ".a3m", "directory_depth": 2}
            ],
            rna_msa_dirs=[],
            diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
            is_inference=False,
        ),
        save_failed_examples_to_dir=None,
    )


@pytest.fixture(scope="session")
def af3_af2fb_distillation_concat_dataset(af3_af2fb_distillation_dataset_no_metadata):
    return ConcatDatasetWithID(datasets=[af3_af2fb_distillation_dataset_no_metadata])
