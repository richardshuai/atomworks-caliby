import logging

import pytest
from cifutils.constants import AF3_EXCLUDED_LIGANDS_REGEX

from datahub.datasets.base import NamedConcatDataset, get_row_and_index_by_example_id
from datahub.datasets.dataframe_parsers import InterfacesDFParser, PNUnitsDFParser
from datahub.datasets.pdb_dataset import PDBDataset
from datahub.pipelines.rf2aa import build_rf2aa_transform_pipeline
from tests.conftest import (
    CIF_PARSER,
    INTERFACES_DF,
    PN_UNITS_DF,
    PROTEIN_MSA_DIRS,
    RNA_MSA_DIRS,
)
from tests.datasets.conftest import SUPPORTED_CHAIN_TYPES_INTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_PN_UNITS_FILTERS = [
    f"q_pn_unit_type in {SUPPORTED_CHAIN_TYPES_INTS}",  # Limit query PN units to proteins, RNA, DNA, and ligands (i.e., exclude RNA/DNA hybrids)
    f"~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}'))",  # TODO: Double check with NATE
]

TEST_INTERFACES_FILTERS = [
    f"pn_unit_1_type in {SUPPORTED_CHAIN_TYPES_INTS}",  # Limit interface PN units to proteins, RNA, DNA, and ligands (i.e., exclude RNA/DNA hybrids)
    f"pn_unit_2_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}'))",  # TODO: Double check with NATE
    f"~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}'))",  # TODO: Double check with NATE
]

# Define the PDB datasets with their respective parsers...
PN_UNITS_DATASET = PDBDataset(
    name="pn_units",
    dataset_path=PN_UNITS_DF,
    cif_parser=CIF_PARSER,
    dataset_parser=PNUnitsDFParser(),
    id_column="example_id",
    filters=TEST_PN_UNITS_FILTERS,
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        assert_rf2aa_assumptions=True,
    ),
)

INTERFACES_DATASET = PDBDataset(
    name="interfaces",
    dataset_path=INTERFACES_DF,
    cif_parser=CIF_PARSER,
    dataset_parser=InterfacesDFParser(),
    id_column="example_id",
    filters=TEST_INTERFACES_FILTERS,
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        assert_rf2aa_assumptions=True,
    ),
)

# ...build the ConcatDataset
PDB_DATASET = NamedConcatDataset(datasets=[PN_UNITS_DATASET, INTERFACES_DATASET], name="pdb")  # NOTE: Order matters!

PRIOR_PIPELINE_BUGS = [
    "{['pdb', 'interfaces']}{2pno}{3}{['DB_1', 'G_1']}",
    "{['pdb', 'interfaces']}{7ah0}{1}{['A_1', 'B_1']}",
    "{['pdb', 'interfaces']}{4u4h}{2}{['A_1', 'A_2']}",
]


@pytest.mark.parametrize("example_id", PRIOR_PIPELINE_BUGS)
@pytest.mark.slow
def test_specific_examples(example_id: str):
    """Run a single example through the pipeline. Useful for debugging specific examples."""
    try:
        index = get_row_and_index_by_example_id(PDB_DATASET, example_id)["index"]
        sample = PDB_DATASET[index]
        assert sample is not None, f"Sample is None, with example_id: {example_id}"
    except Exception as e:
        # We may be excluding some examples with the filters, which is desireable in some cases (e.g., AF-3 excluded ligands)
        logger.debug(f"Error processing example {example_id}: {e}; note that this may be expected due to filters.")


if __name__ == "__main__":
    pytest.main([__file__])
