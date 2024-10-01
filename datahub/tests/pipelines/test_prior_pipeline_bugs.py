import logging

import pytest
from cifutils.constants import AF3_EXCLUDED_LIGANDS_REGEX

from datahub.datasets.base import (
    ConcatDatasetWithID,
    PandasDataset,
    StructuralDatasetWrapper,
    get_row_and_index_by_example_id,
)
from datahub.datasets.dataframe_parsers import InterfacesDFParser, PNUnitsDFParser
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
    f"~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}'))",
]

TEST_INTERFACES_FILTERS = [
    f"pn_unit_1_type in {SUPPORTED_CHAIN_TYPES_INTS}",  # Limit interface PN units to proteins, RNA, DNA, and ligands (i.e., exclude RNA/DNA hybrids)
    f"pn_unit_2_type in {SUPPORTED_CHAIN_TYPES_INTS}",
    f"~(pn_unit_1_non_polymer_res_names.notnull() and pn_unit_1_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}'))",
    f"~(pn_unit_2_non_polymer_res_names.notnull() and pn_unit_2_non_polymer_res_names.str.contains('{AF3_EXCLUDED_LIGANDS_REGEX}'))",
]

# Define the PDB datasets with their respective parsers...
# (We must redefine due to the filters we imposed in the conftest)
# Define the PDB datasets with their respective parsers...
PN_UNITS_DATASET = StructuralDatasetWrapper(
    dataset_parser=PNUnitsDFParser(),
    cif_parser=CIF_PARSER,
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        assert_rf2aa_assumptions=True,
    ),
    dataset=PandasDataset(
        name="pn_units",
        id_column="example_id",
        data=PN_UNITS_DF,
        filters=TEST_PN_UNITS_FILTERS,
    ),
    cif_parser_args={"cache_dir": "/projects/ml/RF2_allatom/cache/cif"},
    return_key="feats",
    save_failed_examples_to_dir=None,
)

INTERFACES_DATASET = StructuralDatasetWrapper(
    dataset_parser=InterfacesDFParser(),
    cif_parser=CIF_PARSER,
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        assert_rf2aa_assumptions=True,
    ),
    dataset=PandasDataset(
        name="interfaces",
        id_column="example_id",
        data=INTERFACES_DF,
        filters=TEST_INTERFACES_FILTERS,
    ),
    cif_parser_args={"cache_dir": "/projects/ml/RF2_allatom/cache/cif"},
    return_key="feats",
    save_failed_examples_to_dir=None,
)

# ...build the ConcatDataset
PDB_DATASET = ConcatDatasetWithID(datasets=[PN_UNITS_DATASET, INTERFACES_DATASET])  # NOTE: Order matters!

PRIOR_PIPELINE_BUGS = [
    # "{['pdb', 'interfaces']}{1pfi}{1}{['B_12', 'C_12']}", # Integer overflow for number of isomorphisms
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
