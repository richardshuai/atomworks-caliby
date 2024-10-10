from cifutils.constants import AF3_EXCLUDED_LIGANDS_REGEX

from datahub.datasets.datasets import ConcatDatasetWithID, PandasDataset, StructuralDatasetWrapper
from datahub.datasets.parsers import (
    AF2FB_DistillationParser,
    InterfacesDFParser,
    PNUnitsDFParser,
    ValidationDFParserLikeAF3,
)
from datahub.pipelines.af3 import build_af3_transform_pipeline
from datahub.pipelines.rf2aa import build_rf2aa_transform_pipeline
from datahub.preprocessing.constants import SUPPORTED_CHAIN_TYPES_INTS
from tests.conftest import (
    CIF_PARSER,
    INTERFACES_DF,
    PN_UNITS_DF,
    PROTEIN_MSA_DIRS,
    RNA_MSA_DIRS,
    VALIDATION_DF,
)

SHARED_TEST_FILTERS = [
    "deposition_date < '2022-01-01'",
    "resolution < 5.0 and ~method.str.contains('NMR')",
    "num_polymer_pn_units <= 20",  # To ensure we don't freeze loading a massive example
    "cluster.notnull()",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
]

TEST_PN_UNITS_FILTERS = [
    f"q_pn_unit_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
]

TEST_INTERFACES_FILTERS = [
    f"pn_unit_1_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"pn_unit_2_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
    f"~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",
]


# Define the PDB datasets with their respective parsers...
RF2AA_PN_UNITS_DATASET = StructuralDatasetWrapper(
    dataset_parser=PNUnitsDFParser(),
    cif_parser=CIF_PARSER,
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
    dataset=PandasDataset(
        name="pn_units",
        id_column="example_id",
        data=PN_UNITS_DF,
        filters=SHARED_TEST_FILTERS + TEST_PN_UNITS_FILTERS,
        columns_to_load=None,  # Load all columns
    ),
    cif_parser_args={"cache_dir": "/projects/ml/RF2_allatom/cache/cif"},
    save_failed_examples_to_dir=None,
)

TEST_DIFFUSION_BATCH_SIZE = 32  # NOTE: setting to a value other than the default for testing purposes
AF3_PN_UNITS_DATASET = StructuralDatasetWrapper(
    dataset_parser=PNUnitsDFParser(),
    cif_parser=CIF_PARSER,
    transform=build_af3_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        n_recycles=5,
        crop_size=256,
        crop_contiguous_probability=1 / 3,
        crop_spatial_probability=2 / 3,
        diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
    ),
    dataset=PandasDataset(
        name="pn_units",
        id_column="example_id",
        data=PN_UNITS_DF,
        filters=SHARED_TEST_FILTERS + TEST_PN_UNITS_FILTERS,
        columns_to_load=None,  # Load all columns
    ),
    save_failed_examples_to_dir=None,
)

RF2AA_INTERFACES_DATASET = StructuralDatasetWrapper(
    dataset_parser=InterfacesDFParser(),
    cif_parser=CIF_PARSER,
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
    dataset=PandasDataset(
        name="interfaces",
        id_column="example_id",
        data=INTERFACES_DF,
        filters=SHARED_TEST_FILTERS + TEST_INTERFACES_FILTERS,
        columns_to_load=None,  # Load all columns
    ),
    cif_parser_args={"cache_dir": "/projects/ml/RF2_allatom/cache/cif"},
    save_failed_examples_to_dir=None,
)

AF3_INTERFACES_DATASET = StructuralDatasetWrapper(
    dataset_parser=InterfacesDFParser(),
    cif_parser=CIF_PARSER,
    transform=build_af3_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        n_recycles=5,
        crop_size=256,
        crop_spatial_probability=1.0,
        crop_contiguous_probability=0.0,
        diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
    ),
    dataset=PandasDataset(
        name="interfaces",
        id_column="example_id",
        data=INTERFACES_DF,
        filters=SHARED_TEST_FILTERS + TEST_INTERFACES_FILTERS,
        columns_to_load=None,  # Load all columns
    ),
    cif_parser_args={"cache_dir": "/projects/ml/RF2_allatom/cache/cif"},
    save_failed_examples_to_dir=None,
)

# ...build the ConcatDataset
RF2AA_PDB_DATASET = ConcatDatasetWithID(
    datasets=[RF2AA_PN_UNITS_DATASET, RF2AA_INTERFACES_DATASET]
)  # NOTE: Order matters!
AF3_PDB_DATASET = ConcatDatasetWithID(datasets=[AF3_PN_UNITS_DATASET, AF3_INTERFACES_DATASET])  # NOTE: Order matters!

AF3_AF2FB_DISTILLATION_DATASET = StructuralDatasetWrapper(
    dataset=PandasDataset(
        data="/squash/af2_distillation_facebook/af2_distillation_facebook.parquet",
        id_column="example_id",
        name="af2fb_distillation",
        columns_to_load=["example_id", "sequence_hash"],
    ),
    dataset_parser=AF2FB_DistillationParser(),
    cif_parser=CIF_PARSER,
    cif_parser_args={"assume_residues_all_resolved": True},
    transform=build_af3_transform_pipeline(
        protein_msa_dirs=[{"dir": "/squash/af2_distillation_facebook/msa", "extension": ".a3m", "directory_depth": 2}],
        rna_msa_dirs=[],
        diffusion_batch_size=TEST_DIFFUSION_BATCH_SIZE,
    ),
    save_failed_examples_to_dir=None,
)
AF3_AF2FB_DISTILLATION_CONCAT_DATASET = ConcatDatasetWithID(datasets=[AF3_AF2FB_DISTILLATION_DATASET])
RF2AA_AF2FB_DISTILLATION_DATASET = StructuralDatasetWrapper(
    dataset=PandasDataset(
        data="/squash/af2_distillation_facebook/af2_distillation_facebook.parquet",
        id_column="example_id",
        name="af2fb_distillation",
        columns_to_load=["example_id", "sequence_hash"],
    ),
    dataset_parser=AF2FB_DistillationParser(),
    cif_parser=CIF_PARSER,
    cif_parser_args={"assume_residues_all_resolved": True},
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dirs=[{"dir": "/squash/af2_distillation_facebook/msa", "extension": ".a3m", "directory_depth": 2}],
        rna_msa_dirs=[],
    ),
)

# Validation datasets

# ...for RF2AA
RF2AA_VALIDATION_DATASET = StructuralDatasetWrapper(
    dataset_parser=ValidationDFParserLikeAF3(),
    cif_parser=CIF_PARSER,
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
    dataset=PandasDataset(
        name="validation",
        data=VALIDATION_DF,
        id_column="example_id",
        columns_to_load=None,  # Load all columns
    ),
    save_failed_examples_to_dir=None,
)

# ...for AF3
AF3_VALIDATION_DATASET = StructuralDatasetWrapper(
    dataset_parser=ValidationDFParserLikeAF3(),
    cif_parser=CIF_PARSER,
    transform=build_af3_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        n_recycles=5,
        crop_size=256,
        crop_spatial_probability=0.0,  # NOTE: Zero probability for cropping; we don't crop during validation
        crop_contiguous_probability=0.0,  # NOTE: Zero probability for cropping ; we don't crop during validation
    ),
    dataset=PandasDataset(
        name="validation",
        id_column="example_id",
        data=VALIDATION_DF,
        columns_to_load=None,  # Load all columns
    ),
    save_failed_examples_to_dir=None,
)
