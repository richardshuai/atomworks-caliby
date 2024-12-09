"""Shared test utils and fixtures for all tests"""

import logging
import os
from pathlib import Path

import pytest

from cifutils.constants import CCD_MIRROR_PATH, PDB_MIRROR_PATH
from cifutils.parser import CIFParser as CIFParserBiotite

TEST_DATA_DIR = Path(os.path.dirname(__file__)) / "data"
CIF_PARSER_BIOTITE = CIFParserBiotite()

logger = logging.getLogger(__name__)


def get_pdb_path(pdbid: str) -> str:
    pdbid = pdbid.lower()
    filename = os.path.join(PDB_MIRROR_PATH, pdbid[1:3], f"{pdbid}.cif.gz")
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename
