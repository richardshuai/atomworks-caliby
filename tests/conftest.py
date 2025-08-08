"""Test fixtures and utilities for atomworks tests."""

import logging
import pathlib

logger = logging.getLogger(__name__)

TEST_DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
