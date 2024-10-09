import logging
import os
from cifutils.parser import CIFParser  # noqa: F401

# Set global logging level to `WARNING` if not set by user
logger = logging.getLogger("cifutils")
_log_level = os.environ.get("CIFUTILS_LOG_LEVEL", "WARNING").upper()
logger.setLevel(_log_level)

# Monkey patch biotite
import biotite.structure as struc  # noqa: E402
from cifutils.utils.selection_utils import get_residue_starts  # noqa: E402

struc.get_residue_starts = get_residue_starts
