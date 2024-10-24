"""Shared test utils"""

import datahub.preprocessing.process as data_preprocessor

DATA_PREPROCESSOR = data_preprocessor.DataPreprocessor(
    polymer_pn_unit_limit=50,  # Set to 50 for processing speed during testing
)
