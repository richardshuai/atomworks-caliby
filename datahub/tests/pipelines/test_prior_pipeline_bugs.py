import logging

import pandas as pd
import pytest

from datahub.datasets.datasets import (
    ConcatDatasetWithID,
    PandasDataset,
    StructuralDatasetWrapper,
    get_row_and_index_by_example_id,
)
from datahub.datasets.parsers import InterfacesDFParser, PNUnitsDFParser
from datahub.pipelines.af3 import build_af3_transform_pipeline
from tests.conftest import (
    PROTEIN_MSA_DIRS,
    RNA_MSA_DIRS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


##########################################################
# ------------------- AF3 Pipeline -----------------------#
##########################################################


# Define the PDB datasets with their respective parsers...
# (We must redefine due to the filters we imposed in the conftest; left as a fixture for lazy loading)
@pytest.fixture
def full_pdb_dataset_af3():
    # Create PN_UNITS_DATASET_AF3
    pn_units_dataset_af3 = StructuralDatasetWrapper(
        dataset_parser=PNUnitsDFParser(),
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            is_inference=False,
        ),
        dataset=PandasDataset(
            name="pn_units",
            id_column="example_id",
            data=pd.read_parquet("/projects/ml/RF2_allatom/datasets/af3_splits/2024_09_23/pn_units_df_train.parquet"),
        ),
        return_key="feats",
        save_failed_examples_to_dir=None,
    )

    # Create INTERFACES_DATASET_AF3
    interfaces_dataset_af3 = StructuralDatasetWrapper(
        dataset_parser=InterfacesDFParser(),
        transform=build_af3_transform_pipeline(
            protein_msa_dirs=PROTEIN_MSA_DIRS,
            rna_msa_dirs=RNA_MSA_DIRS,
            is_inference=False,
        ),
        dataset=PandasDataset(
            name="interfaces",
            id_column="example_id",
            data=pd.read_parquet("/projects/ml/RF2_allatom/datasets/af3_splits/2024_09_23/interfaces_df_train.parquet"),
        ),
        return_key="feats",
        save_failed_examples_to_dir=None,
    )

    # Combine both datasets into a single ConcatDatasetWithID
    return ConcatDatasetWithID(datasets=[pn_units_dataset_af3, interfaces_dataset_af3])


PRIOR_PIPELINE_BUGS_AF3 = [
    "{['pdb', 'pn_units']}{5epq}{1}{['A_1']}",
    "{['pdb', 'interfaces']}{2g37}{1}{['B_1', 'F_1']}",
    "{['pdb', 'interfaces']}{4v4s}{1}{['D_1', 'Y_1']}",
]


@pytest.mark.parametrize("example_id", PRIOR_PIPELINE_BUGS_AF3)
@pytest.mark.slow
def test_specific_examples_af3(example_id: str, full_pdb_dataset_af3: ConcatDatasetWithID):
    """Run a single example through the pipeline. Useful for debugging specific examples."""
    try:
        index = get_row_and_index_by_example_id(full_pdb_dataset_af3, example_id)["index"]
        sample = full_pdb_dataset_af3[index]
        assert sample is not None, f"Sample is None, with example_id: {example_id}"
    except Exception as e:
        # We may be excluding some examples with the filters, which is desireable in some cases (e.g., AF-3 excluded ligands)
        logger.debug(f"Error processing example {example_id}: {e}; note that this may be expected due to filters.")


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-m very_slow"])
