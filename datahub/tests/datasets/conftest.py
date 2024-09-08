from cifutils.constants import AF3_EXCLUDED_LIGANDS_REGEX

from datahub.datasets.base import NamedConcatDataset
from datahub.datasets.dataframe_parsers import InterfacesDFParser, PNUnitsDFParser
from datahub.datasets.pdb_dataset import PDBDataset
from datahub.pipelines.af3 import build_af3_transform_pipeline
from datahub.pipelines.rf2aa import build_rf2aa_transform_pipeline
from datahub.preprocessing.constants import SUPPORTED_CHAIN_TYPES_INTS
from tests.conftest import (
    CIF_PARSER,
    INTERFACES_DF,
    PN_UNITS_DF,
    PROTEIN_MSA_DIRS,
    RNA_MSA_DIRS,
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
    f"~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",  # TODO: Double check with NATE
]

TEST_INTERFACES_FILTERS = [
    f"pn_unit_1_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"pn_unit_2_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",  # TODO: Double check with NATE
    f"~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}', regex=True))",  # TODO: Double check with NATE
]


# Define the PDB datasets with their respective parsers...
RF2AA_PN_UNITS_DATASET = PDBDataset(
    name="pn_units",
    dataset_path=PN_UNITS_DF,
    dataset_parser=PNUnitsDFParser(),
    filters=SHARED_TEST_FILTERS + TEST_PN_UNITS_FILTERS,
    cif_parser=CIF_PARSER,
    columns_to_load=None,  # Load all columns
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
    id_column="example_id",
    return_key="feats",
    save_failed_examples_to_dir=None,
    cif_cache_dir="/projects/ml/RF2_allatom/cache/cif",
)

AF3_PN_UNITS_DATASET = PDBDataset(
    name="pn_units",
    dataset_path=PN_UNITS_DF,
    dataset_parser=PNUnitsDFParser(),
    filters=SHARED_TEST_FILTERS + TEST_PN_UNITS_FILTERS,
    cif_parser=CIF_PARSER,
    columns_to_load=None,  # Load all columns
    transform=build_af3_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        n_recycles=5,
        crop_size=256,
        crop_contiguous_probability=1 / 3,
        crop_spatial_probability=2 / 3,
    ),
    id_column="example_id",
    return_key=None,
    save_failed_examples_to_dir=None,
    cif_cache_dir="/projects/ml/RF2_allatom/cache/cif",
)

RF2AA_INTERFACES_DATASET = PDBDataset(
    name="interfaces",
    dataset_path=INTERFACES_DF,
    dataset_parser=InterfacesDFParser(),
    filters=SHARED_TEST_FILTERS + TEST_INTERFACES_FILTERS,
    cif_parser=CIF_PARSER,
    columns_to_load=None,  # Load all columns
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
    id_column="example_id",
    return_key="feats",
    save_failed_examples_to_dir=None,
    cif_cache_dir="/projects/ml/RF2_allatom/cache/cif",
)

AF3_INTERFACES_DATASET = PDBDataset(
    name="interfaces",
    dataset_path=INTERFACES_DF,
    dataset_parser=InterfacesDFParser(),
    filters=SHARED_TEST_FILTERS + TEST_INTERFACES_FILTERS,
    cif_parser=CIF_PARSER,
    columns_to_load=None,  # Load all columns
    transform=build_af3_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        n_recycles=5,
        crop_size=256,
        crop_spatial_probability=1.0,
        crop_contiguous_probability=0.0,
    ),
    id_column="example_id",
    return_key=None,
    save_failed_examples_to_dir=None,
    cif_cache_dir="/projects/ml/RF2_allatom/cache/cif",
)

# ...build the ConcatDataset
RF2AA_PDB_DATASET = NamedConcatDataset(
    datasets=[RF2AA_PN_UNITS_DATASET, RF2AA_INTERFACES_DATASET], name="pdb"
)  # NOTE: Order matters!
AF3_PDB_DATASET = NamedConcatDataset(
    datasets=[AF3_PN_UNITS_DATASET, AF3_INTERFACES_DATASET], name="pdb"
)  # NOTE: Order matters!
