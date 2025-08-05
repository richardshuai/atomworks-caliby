"""IO-specific test fixtures and utilities for atomworks.io tests."""

import logging
import os
import socket
from pathlib import Path

import pytest

from atomworks.io.enums import ChainType
from atomworks.io.utils.testing import get_pdb_path  # noqa: F401

logger = logging.getLogger(__name__)

TEST_DATA_DIR = Path(os.path.dirname(__file__)) / "data"


def _has_internet_connection() -> bool:
    try:
        # Try to connect to a well-known DNS server (Google's)
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


requires_internet = pytest.mark.skipif(not _has_internet_connection(), reason="Test requires an internet connection.")

CHAIN_TYPE_TEST_CASES = [
    {
        # Simple polymer & non-polymers
        "pdb_id": "6qhp",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.POLYPEPTIDE_L,
            "C": ChainType.NON_POLYMER,  # fluoroacetic acid
            "D": ChainType.NON_POLYMER,  # fluoroacetic acid
        },
    },
    {
        # DNA and RNA, separately
        "pdb_id": "1fix",
        "chain_types": {
            "A": ChainType.RNA,
            "B": ChainType.DNA,
        },
    },
    {
        # DNA and RNA hybrid
        "pdb_id": "1d9d",
        "chain_types": {
            "A": ChainType.DNA_RNA_HYBRID,
            "B": ChainType.POLYPEPTIDE_L,
            "C": ChainType.NON_POLYMER,  # Zinc ion
            "D": ChainType.NON_POLYMER,
            "E": ChainType.NON_POLYMER,
            "F": ChainType.NON_POLYMER,  # Magnesium ion
            "G": ChainType.NON_POLYMER,
            "H": ChainType.NON_POLYMER,  # Solvent
            "I": ChainType.NON_POLYMER,  # Solvent
            "J": ChainType.NON_POLYMER,  # Solvent
            "K": ChainType.NON_POLYMER,  # Solvent
        },
    },
    {
        # Oligosaccharides
        "pdb_id": "1ivo",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.POLYPEPTIDE_L,
            "C": ChainType.POLYPEPTIDE_L,
            "D": ChainType.POLYPEPTIDE_L,
            "E": ChainType.BRANCHED,  # Oligosaccharide
            "F": ChainType.NON_POLYMER,  # Monosaccharide
            "G": ChainType.NON_POLYMER,  # Monosaccharide
            "H": ChainType.NON_POLYMER,
            "I": ChainType.NON_POLYMER,
            "J": ChainType.NON_POLYMER,
            "K": ChainType.NON_POLYMER,
            "L": ChainType.NON_POLYMER,
            "M": ChainType.NON_POLYMER,
        },
    },
    {
        # Covalently bonded ligands
        "pdb_id": "3ne7",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.NON_POLYMER,  # Nickel
            "C": ChainType.NON_POLYMER,  # CoA
            "D": ChainType.NON_POLYMER,
            "E": ChainType.NON_POLYMER,
            "F": ChainType.NON_POLYMER,
            "G": ChainType.NON_POLYMER,
            "H": ChainType.NON_POLYMER,
        },
    },
]
