"""
atomworks - Unified package for biological data I/O and machine learning.

This package combines functionality from atomworks.io (I/O operations) and atomworks.ml (ML utilities)
into a unified interface for biological data processing and machine learning.
"""

import logging
import os
import warnings

# Global logging configuration
logger = logging.getLogger("atomworks")
_log_level = os.environ.get("ATOMWORKS_LOG_LEVEL", "WARNING").upper()
logger.setLevel(_log_level)

# Ensure that deprecation warnings are not repeated
warnings.filterwarnings("once", category=DeprecationWarning)

# Import version information
# Import subpackages
from . import io, ml

# Re-export key functionality from subpackages for convenience
# This maintains backward compatibility and provides a clean top-level API
# Key I/O functionality
from .io.parser import parse
from .io.utils.query import monkey_patch_atomarray
from .version import __version__, __version_tuple__

# Apply monkey patching to extend AtomArray functionality
monkey_patch_atomarray()

__all__ = [
    "__version__",
    "__version_tuple__",
    "io",
    "ml",
    "monkey_patch_atomarray",
    "parse",
]
