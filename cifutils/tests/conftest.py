"""Shared test utils and fixtures for all tests"""

import os
from pathlib import Path
from cifutils.parser import CIFParser as CIFParserBiotite

TEST_DATA_DIR = Path(os.path.dirname(__file__)) / "data"
CIF_PARSER_BIOTITE = CIFParserBiotite()


def get_digs_path(pdbid: str) -> str:
    pdbid = pdbid.lower()
    filename = f"/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz"
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename
