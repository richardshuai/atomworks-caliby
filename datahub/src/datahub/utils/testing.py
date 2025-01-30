import os

from cifutils import parse
from cifutils.common import immutable_lru_cache
from cifutils.constants import PDB_MIRROR_PATH


def get_pdb_mirror_path(pdbid: str, base_dir: str = PDB_MIRROR_PATH) -> str:
    """Convenience util to get the path to a CIF file on the DIGS"""
    # Assert that the base directory exists
    assert os.path.exists(base_dir), f"Base directory {base_dir} does not exist"

    # Build the path to the file
    pdbid = pdbid.lower()
    filename = f"{base_dir}/{pdbid[1:3]}/{pdbid}.cif.gz"
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename


@immutable_lru_cache(maxsize=1000)
def cached_parse(pdb_id: str, **kwargs) -> dict:
    """Wrapper around _cached_parse with caching to return an immutable copy of the output dict"""
    data = parse(filename=get_pdb_mirror_path(pdb_id), **kwargs)
    if "atom_array" not in data:
        assembly_ids = list(data["assemblies"].keys())
        data["atom_array"] = data["assemblies"][assembly_ids[0]][0]
    data["pdb_id"] = pdb_id
    return data
