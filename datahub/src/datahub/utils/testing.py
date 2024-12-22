import os
from copy import deepcopy
from functools import cache

from biotite.structure import AtomArray
from cifutils import parse


def get_digs_path(pdbid: str, base_dir: str = "/projects/ml/frozen_pdb_copies/2024_12_01_pdb") -> str:
    """Convenience util to get the path to a CIF file on the DIGS"""
    # Assert that the base directory exists
    assert os.path.exists(base_dir), f"Base directory {base_dir} does not exist"

    # Build the path to the file
    pdbid = pdbid.lower()
    filename = f"{base_dir}/{pdbid[1:3]}/{pdbid}.cif.gz"
    if not os.path.exists(filename):
        raise ValueError(f"File {filename} does not exist")
    return filename


@cache
def _cached_parse(
    pdb_id: str,
    **kwargs,
) -> AtomArray:
    """Convenience util that caches and formats parsing output in a DataHub-compatible format"""
    data = parse(
        filename=get_digs_path(pdb_id),
        **kwargs,
    )
    if "atom_array" not in data:
        assembly_ids = list(data["assemblies"].keys())
        data["atom_array"] = data["assemblies"][assembly_ids[0]][0]
    data["pdb_id"] = pdb_id
    return data


def cached_parse(pdb_id: str, **kwargs) -> AtomArray:
    """Wrapper around _cached_parse with caching to return an immutable copy of the output dict"""
    return deepcopy(_cached_parse(pdb_id, **kwargs))
