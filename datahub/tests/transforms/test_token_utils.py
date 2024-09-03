import biotite.structure as struc
import numpy as np
import pytest
from cifutils.utils.atom_matching_utils import assert_same_atom_array

from datahub.encoding_definitions import RF2AA_ATOM36_ENCODING
from datahub.transforms.atom_array import AddGlobalAtomIdAnnotation, AddGlobalTokenIdAnnotation
from datahub.transforms.atomize import AtomizeResidues
from datahub.transforms.base import Compose
from datahub.utils.token import (
    apply_segment_wise_2d,
    get_token_count,
    get_token_starts,
    token_iter,
)
from tests.conftest import cached_parse


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_tokens_are_residues_without_atomization(pdb_id: str):
    data = cached_parse(pdb_id)
    atom_array = data["atom_array"]

    assert get_token_count(atom_array) == struc.get_residue_count(atom_array)
    assert np.all(get_token_starts(atom_array) == struc.get_residue_starts(atom_array))
    assert np.all(
        get_token_starts(atom_array, add_exclusive_stop=True)
        == struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    )
    for res_1, res_2 in zip(struc.residue_iter(atom_array), token_iter(atom_array)):
        assert_same_atom_array(res_1, res_2)


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_tokens_are_atoms_with_full_atomization(pdb_id: str):
    data = cached_parse(pdb_id)
    data = AtomizeResidues(atomize_by_default=True)(data)
    atom_array = data["atom_array"]
    assert get_token_count(atom_array) == len(atom_array)
    assert np.all(get_token_starts(atom_array) == np.arange(len(atom_array)))
    assert np.all(get_token_starts(atom_array, add_exclusive_stop=True) == np.arange(len(atom_array) + 1))


def test_apply_segment_wise_2d():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    segment_start_end_idxs = np.array([0, 2, 3])
    assert np.all(
        apply_segment_wise_2d(array, segment_start_end_idxs, reduce_func=np.sum) == np.array([[12, 9], [15, 9]])
    )


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_add_global_token_id_annotation_when_fully_atomized(pdb_id):
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AtomizeResidues(atomize_by_default=True),  # atomize all residues
            AddGlobalTokenIdAnnotation(),
        ],
        track_rng_state=False,
    )

    data = cached_parse(pdb_id)
    data = pipe(data)

    atom_array = data["atom_array"]

    assert "atom_id" in atom_array.get_annotation_categories()
    assert "token_id" in atom_array.get_annotation_categories()
    assert np.all(
        atom_array.atom_id == atom_array.token_id
    ), "atom_id and token_id should be the same for a fully atomized atom_array"

    # cross check by iterating over the tokens
    # ... via token starts
    counter = 0
    token_start_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
    for start, end in zip(token_start_idxs[:-1], token_start_idxs[1:]):
        token = atom_array[start:end]
        assert len(token) == 1, f"token should have length 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"
        counter += 1

    # ... via token_iter
    counter = 0
    for token in token_iter(atom_array):
        assert len(token) == 1, f"token should have length 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"
        counter += 1

    # ... via atom iter (since when fully atomized, tokens are atoms)
    counter = 0
    for token in atom_array:
        assert token.token_id == counter, f"token_id should be {counter} but is {token.token_id}"
        counter += 1


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_add_global_token_id_annotation_when_not_atomized(pdb_id):
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AddGlobalTokenIdAnnotation(),
        ],
        track_rng_state=False,
    )

    data = cached_parse(pdb_id)
    data = pipe(data)

    atom_array = data["atom_array"]

    assert "atom_id" in atom_array.get_annotation_categories()
    assert "token_id" in atom_array.get_annotation_categories()
    assert atom_array.atom_id[-1] > atom_array.token_id[-1], "There should be more atom_ids than token_ids."

    # cross check by iterating over the tokens
    # ... via token starts
    counter = 0
    token_start_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
    for start, end in zip(token_start_idxs[:-1], token_start_idxs[1:]):
        token = atom_array[start:end]
        assert len(token) >= 1, f"token should have length at least 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"
        counter += 1

    # ... via token_iter
    counter = 0
    for token in token_iter(atom_array):
        assert len(token) >= 1, f"token should have length at least 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"
        counter += 1

    # ... via residue iter (since when not atomizing, tokens are residues)
    counter = 0
    for residue in struc.residue_iter(atom_array):
        assert len(residue) >= 1, f"residue should have length at least 1 but has length {len(residue)}"
        assert np.all(residue.token_id == counter), f"token_id should be {counter} but is {residue.token_id}"
        counter += 1


@pytest.mark.parametrize("pdb_id", ["6lyz", "5ocm"])
def test_add_global_token_id_annotation_when_partially_atomized(pdb_id):
    pipe = Compose(
        [
            AddGlobalAtomIdAnnotation(),
            AtomizeResidues(atomize_by_default=True, res_names_to_ignore=RF2AA_ATOM36_ENCODING.tokens),
            AddGlobalTokenIdAnnotation(),
        ],
        track_rng_state=False,
    )

    data = cached_parse(pdb_id)
    data = pipe(data)

    atom_array = data["atom_array"]

    assert "atom_id" in atom_array.get_annotation_categories()
    assert "token_id" in atom_array.get_annotation_categories()
    assert atom_array.atom_id[-1] > atom_array.token_id[-1], "There should be more atom_ids than token_ids."

    # cross check by iterating over the tokens
    # ... via token starts
    counter = 0
    token_start_idxs = get_token_starts(atom_array, add_exclusive_stop=True)
    for start, end in zip(token_start_idxs[:-1], token_start_idxs[1:]):
        token = atom_array[start:end]
        assert len(token) >= 1, f"token should have length at least 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"
        counter += 1

    # ... via token_iter
    counter = 0
    for token in token_iter(atom_array):
        assert len(token) >= 1, f"token should have length at least 1 but has length {len(token)}"
        assert np.all(token.token_id == counter), f"token_id should be {counter} but is {token.token_id}"
        counter += 1


if __name__ == "__main__":
    pytest.main(["-v", "-x", __file__])
