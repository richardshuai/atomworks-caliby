import time

import pytest
from conftest import get_pdb_path

from cifutils.parser import parse
from cifutils.utils.atom_matching_utils import assert_same_atom_array

TEST_CASES = [
    "1A7J",  # Contains an unusual operation expression for assembly building
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_caching(pdb_id: str, tmp_path):
    path = get_pdb_path(pdb_id)

    # First, we load normally, tracking how long it takes...
    start_time = time.time()
    normal_result = parse(
        # Caching arguments
        load_from_cache=False,
        save_to_cache=False,
        cache_dir=None,
        # Standard arguments
        filename=path,
        build_assembly="all",
    )
    end_time = time.time()
    normal_elapsed_time = end_time - start_time
    assert normal_result is not None  # Check if processing runs through
    assert normal_elapsed_time > 0  # Check if processing time is non-zero

    # ...then we load, saving to the cache
    save_cache_result = parse(
        # Caching arguments
        load_from_cache=False,
        save_to_cache=True,
        cache_dir=tmp_path,
        # Standard arguments
        filename=path,
        build_assembly="all",
    )

    # ...then, we load from the cache, and keep track of how long it takes
    start_time = time.time()
    cached_result = parse(
        # Caching arguments
        load_from_cache=True,
        save_to_cache=False,
        cache_dir=tmp_path,
        # Standard arguments
        filename=path,
        build_assembly="all",
    )
    end_time = time.time()
    cached_elapsed_time = end_time - start_time

    # Assert that the assembly data is the same
    annotations_to_compare = ["chain_id", "res_name", "res_id", "atom_name", "chain_iid", "pn_unit_id", "pn_unit_iid"]
    for assembly_id in normal_result["assemblies"]:
        assert_same_atom_array(
            normal_result["assemblies"][assembly_id], cached_result["assemblies"][assembly_id], annotations_to_compare
        )
        assert_same_atom_array(
            save_cache_result["assemblies"][assembly_id],
            cached_result["assemblies"][assembly_id],
            annotations_to_compare,
        )

    # Assert that the cached result is at least twice as fast as the normal result
    assert cached_elapsed_time < normal_elapsed_time / 2


if __name__ == "__main__":
    pytest.main([__file__])
