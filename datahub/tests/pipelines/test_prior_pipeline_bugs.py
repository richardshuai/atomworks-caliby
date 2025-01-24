import logging

import pandas as pd
import pytest

from datahub.datasets.datasets import (
    ConcatDatasetWithID,
    PandasDataset,
    StructuralDatasetWrapper,
)
from datahub.datasets.parsers import InterfacesDFParser, PNUnitsDFParser
from datahub.pipelines.af3 import build_af3_transform_pipeline
from datahub.utils.testing import cached_parse
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


PRIOR_PIPELINE_BUGS_AF3 = ["7qbs", "5epq", "2g37", "4v4s"]


@pytest.mark.parametrize("pdb_id", PRIOR_PIPELINE_BUGS_AF3)
@pytest.mark.slow
def test_prior_pipeline_bugs_af3(pdb_id: str):
    """Run a single example through the pipeline. Useful for debugging specific examples."""
    input = cached_parse(pdb_id)
    input["example_id"] = pdb_id
    pipe = build_af3_transform_pipeline(
        protein_msa_dirs=PROTEIN_MSA_DIRS,
        rna_msa_dirs=RNA_MSA_DIRS,
        is_inference=False,
    )
    output = pipe(input)

    assert output is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-m not very_slow"])
