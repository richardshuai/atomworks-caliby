import logging
import os
from cifutils.parser import CIFParser  # noqa: F401

# Set global logging level to `WARNING` if not set by user
logger = logging.getLogger("cifutils")
_log_level = os.environ.get("CIFUTILS_LOG_LEVEL", "WARNING").upper()
logger.setLevel(_log_level)
