import pytest
from tests.conftest import get_digs_path, CIF_PARSER_BIOTITE, assert_same_atom_array
import time

TEST_CASES = [
    "1A7J",  # Contains an unusual operation expression for assembly building
]


@pytest.mark.parametrize("pdb_id", TEST_CASES)
def test_caching(pdb_id: str, tmp_path):
    path = get_digs_path(pdb_id)

    # First, we load normally, tracking how long it takes...
    start_time = time.time()
    normal_result = CIF_PARSER_BIOTITE.parse(
        # Caching arguments
        load_from_cache=False,
        save_to_cache=False,
        cache_dir=None,
        # Standard arguments
        filename=path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        build_assembly="all",
        patch_symmetry_centers=True,
        fix_arginines=True,
    )
    end_time = time.time()
    normal_elapsed_time = end_time - start_time
    assert normal_result is not None  # Check if processing runs through
    assert normal_elapsed_time > 0  # Check if processing time is non-zero

    # ...then we load, saving to the cache
    save_cache_result = CIF_PARSER_BIOTITE.parse(
        # Caching arguments
        load_from_cache=False,
        save_to_cache=True,
        cache_dir=tmp_path,
        # Standard arguments
        filename=path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        build_assembly="all",
        patch_symmetry_centers=True,
        fix_arginines=True,
    )

    # ...then, we load from the cache, and keep track of how long it takes
    start_time = time.time()
    cached_result = CIF_PARSER_BIOTITE.parse(
        # Caching arguments
        load_from_cache=True,
        save_to_cache=False,
        cache_dir=tmp_path,
        # Standard arguments
        filename=path,
        add_missing_atoms=True,
        add_bonds=True,
        remove_waters=True,
        build_assembly="all",
        patch_symmetry_centers=True,
        fix_arginines=True,
    )
    end_time = time.time()
    cached_elapsed_time = end_time - start_time

    # Assert that the assembly data is the same
    for assembly_id in normal_result["assemblies"]:
        assert_same_atom_array(normal_result["assemblies"][assembly_id], cached_result["assemblies"][assembly_id])
        assert_same_atom_array(save_cache_result["assemblies"][assembly_id], cached_result["assemblies"][assembly_id])

    # Assert that the cached result is at least twice as fast as the normal result
    assert cached_elapsed_time < normal_elapsed_time / 2


if __name__ == "__main__":
    pytest.main([__file__])
