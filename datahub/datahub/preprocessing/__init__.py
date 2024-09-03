import logging
import os

from .process import DataPreprocessor  # noqa: F401

logger = logging.getLogger("preprocess")
_log_level = os.environ.get("PREPROCESS_LOG_LEVEL", "WARNING").upper()
logger.setLevel(_log_level)
