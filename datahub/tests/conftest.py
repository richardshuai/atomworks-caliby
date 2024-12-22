"""Shared test utils and fixtures for all tests"""

from pathlib import Path

import pandas as pd

# Directory containing pn_units_df and interfaces_df
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
PN_UNITS_DF = pd.read_parquet(f"{TEST_DATA_DIR}/pn_units_df.parquet")
INTERFACES_DF = pd.read_parquet(f"{TEST_DATA_DIR}/interfaces_df.parquet")
AF2_DISTILLATION_FACEBOOK_DF = pd.read_parquet(f"{TEST_DATA_DIR}/test_af2_distillation_facebook.parquet")

# The validation dataset is small, so we don't need to use a subset
AF3_VALIDATION_DF = pd.read_parquet(
    "/projects/ml/RF2_allatom/datasets/af3_splits/2024_10_18/entry_level_val_df.parquet"
)

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
]

RNA_MSA_DIRS = [
    {"dir": "/projects/msa/rf2aa_af3/rf2aa_paper_model_rna_msas", "extension": ".afa", "directory_depth": 0}
]
