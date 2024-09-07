
from cifutils.constants import AF3_EXCLUDED_LIGANDS

from datahub.datasets.base import NamedConcatDataset
from datahub.datasets.dataframe_parsers import InterfacesDFParser, PNUnitsDFParser
from datahub.datasets.pdb_dataset import PDBDataset
from datahub.pipelines.rf2aa import build_rf2aa_transform_pipeline
from tests.conftest import (
    CIF_PARSER,
    INTERFACES_DF,
    PN_UNITS_DF,
    PROTEIN_MSA_DIRS,
    RNA_MSA_DIRS,
    SUPPORTED_CHAIN_TYPES_INTS,
)

SHARED_TEST_FILTERS = [
    "deposition_date < '2022-01-01'",
    "resolution < 5.0 and ~method.str.contains('NMR')",
    "num_polymer_pn_units <= 20",  # To ensure we don't freeze loading a massive example
    "cluster.notnull()",
    "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
]

af3_excluded_ligand_pattern = "(?:^|,)\s*(?:" + "|".join(AF3_EXCLUDED_LIGANDS) + ")\s*(?:,|$)"

TEST_PN_UNITS_FILTERS = [
    f"q_pn_unit_type in {SUPPORTED_CHAIN_TYPES_INTS}",  # Limit query PN units to proteins, RNA, DNA, and ligands (i.e., exclude RNA/DNA hybrids)
    f"~(q_pn_unit_non_polymer_res_names.str.contains('{af3_excluded_ligand_pattern}'))",
]

TEST_INTERFACES_FILTERS = [
    f"pn_unit_1_type in {SUPPORTED_CHAIN_TYPES_INTS}",  # Limit interface PN units to proteins, RNA, DNA, and ligands (i.e., exclude RNA/DNA hybrids)
    f"pn_unit_2_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(pn_unit_1_non_polymer_res_names.str.contains('{af3_excluded_ligand_pattern}'))",
    f"~(pn_unit_2_non_polymer_res_names.str.contains('{af3_excluded_ligand_pattern}'))",
]


# Define the PDB datasets with their respective parsers...
PN_UNITS_DATASET = PDBDataset(
    name="pn_units",
    dataset_path=PN_UNITS_DF,
    cif_parser=CIF_PARSER,
    filters=SHARED_TEST_FILTERS + TEST_PN_UNITS_FILTERS,
    dataset_parser=PNUnitsDFParser(),
    id_column="example_id",
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        n_recycles=5,
        crop_size=256,
        crop_contiguous_probability=1 / 3,
        crop_spatial_probability=2 / 3,
    ),
    unpack_data_dict=False,
)

INTERFACES_DATASET = PDBDataset(
    name="interfaces",
    dataset_path=INTERFACES_DF,
    cif_parser=CIF_PARSER,
    filters=SHARED_TEST_FILTERS + TEST_INTERFACES_FILTERS,
    dataset_parser=InterfacesDFParser(),
    id_column="example_id",
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        n_recycles=5,
        crop_size=256,
        crop_spatial_probability=1.0,
        crop_contiguous_probability=0.0,
    ),
    unpack_data_dict=False,
)

# ...build the ConcatDataset
PDB_DATASET = NamedConcatDataset(datasets=[PN_UNITS_DATASET, INTERFACES_DATASET], name="pdb")  # NOTE: Order matters!
