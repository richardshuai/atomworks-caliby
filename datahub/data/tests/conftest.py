"""Shared test utils"""

import data.data_preprocessor as data_preprocessor

DATA_PREPROCESSOR = data_preprocessor.DataPreprocessor(
    polymer_pn_unit_limit=50,  # Set to 50 for processing speed during testing
)
