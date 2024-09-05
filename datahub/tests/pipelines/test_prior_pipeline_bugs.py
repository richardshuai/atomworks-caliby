import pandas as pd
import pytest

from datahub.datasets.base import NamedConcatDataset, get_row_and_index_by_example_id
from datahub.datasets.dataframe_parsers import InterfacesDFParser, PNUnitsDFParser
from datahub.datasets.pdb_dataset import PDBDataset
from datahub.pipelines.rf2aa import build_rf2aa_transform_pipeline
from tests.conftest import (
    CIF_PARSER,
    PROTEIN_MSA_DIRS,
    RNA_MSA_DIRS,
)

# ...load the (full) dataframes
# NOTE: We can use the test dataframes if note all of the prior bugs in `PDB_IDS_TO_INCLUDE_IN_TEST_DATASETS ` in `create_test_datasets`.
PN_UNITS_DF = pd.read_parquet("/projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/pn_units_df.parquet")
INTERFACES_DF = pd.read_parquet("/projects/ml/RF2_allatom/data_preprocessing/PDB_2024_08_16/interfaces_df.parquet")

# Define the PDB datasets with their respective parsers...
PN_UNITS_DATASET = PDBDataset(
    name="pn_units",
    dataset_path=PN_UNITS_DF,
    cif_parser=CIF_PARSER,
    dataset_parser=PNUnitsDFParser(),
    id_column="example_id",
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
    ),
    unpack_data_dict=False,
)

INTERFACES_DATASET = PDBDataset(
    name="interfaces",
    dataset_path=INTERFACES_DF,
    cif_parser=CIF_PARSER,
    dataset_parser=InterfacesDFParser(),
    id_column="example_id",
    transform=build_rf2aa_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
    ),
    unpack_data_dict=False,
)

# ...build the ConcatDataset
PDB_DATASET = NamedConcatDataset(datasets=[PN_UNITS_DATASET, INTERFACES_DATASET], name="pdb")  # NOTE: Order matters!

PRIOR_PIPELINE_BUGS = ["{['pdb', 'interfaces']}{7ah0}{1}{['A_1', 'B_1']}"]


@pytest.mark.parametrize("example_id", PRIOR_PIPELINE_BUGS)
def test_specific_examples(example_id: str):
    """Run a single example through the pipeline. Useful for debugging specific examples."""
    index = get_row_and_index_by_example_id(PDB_DATASET, example_id)["index"]
    assert index is not None, f"Failed to get row from example_id for example_id: {example_id}"
    sample = PDB_DATASET[index]
    assert sample is not None, f"Sample is None, with example_id: {example_id}"


if __name__ == "__main__":
    pytest.main([__file__])
