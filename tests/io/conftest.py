"""IO-specific test fixtures and utilities for atomworks.io tests."""

import logging
import os
from pathlib import Path

from atomworks.io.utils.testing import get_pdb_path  # noqa: F401

TEST_DATA_DIR = Path(os.path.dirname(__file__)) / "data"

logger = logging.getLogger(__name__)
