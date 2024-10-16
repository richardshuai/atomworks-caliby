import copy

import numpy as np

from datahub.encoding_definitions import (
    RF2AA_ATOM36_ENCODING,
)
from datahub.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from datahub.transforms.base import Compose
from datahub.transforms.encoding import AddTokenAnnotation, EncodeAtomArray
from datahub.transforms.featurize_unresolved_residues import (
    MaskResiduesWithUnresolvedBackboneAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
    mask_residues_with_unresolved_backbone_atoms,
)
from datahub.utils.token import (
    apply_and_spread_token_wise,
    get_af3_token_representative_coords,
    get_af3_token_representative_masks,
    token_iter,
)
from tests.conftest import assert_equal_atom_arrays, cached_parse


def test_mask_residues_with_unresolved_backbone_atoms():
    data = cached_parse("6wtf")
    atom_array = data["atom_array"]

    # ...manually set the occupancy of a CA atom to zero
    resolved_ca_atoms = (atom_array.atom_name == "CA") & (atom_array.occupancy > 0)

    # ...set the first CA atom to zero occupancy
    atom_array.occupancy[resolved_ca_atoms] = np.array([0.0] + [1.0] * (np.sum(resolved_ca_atoms) - 1))
    changed_atom = atom_array[resolved_ca_atoms][0]

    # ...apply the transform
    updated_atom_array = mask_residues_with_unresolved_backbone_atoms(atom_array)

    # ...assert that the manually set CA atom's residue is masked
    changed_residue_mask = (updated_atom_array.chain_id == changed_atom.chain_id) & (
        updated_atom_array.res_id == changed_atom.res_id
    )
    assert np.all(updated_atom_array.occupancy[changed_residue_mask] == 0)

    # ...assert that the rest of the residues are unchanged
    unchanged_residue_mask = ~changed_residue_mask
    assert np.all(updated_atom_array.occupancy[unchanged_residue_mask] == atom_array.occupancy[unchanged_residue_mask])


def test_place_unresolved_token_atoms_on_representative_atom():
    pdb_id = "6wtf"

    data = cached_parse(pdb_id)
    atom_array = data["atom_array"]

    # ...check for unresolved polymer atoms (there will be lots of unresolved hydrogens, so we leave them in  as a test case)
    unresolved_polymer_atoms = atom_array[(atom_array.is_polymer) & (atom_array.occupancy == 0)]

    # ...same thing for unresolved non-polymer atoms (hydrogens will be unresolved)
    unresolved_non_polymer_atoms = atom_array[(~atom_array.is_polymer) & (atom_array.occupancy == 0)]

    assert len(unresolved_polymer_atoms) > 0
    assert len(unresolved_non_polymer_atoms) > 0

    encoding = RF2AA_ATOM36_ENCODING
    pipe = Compose(
        [
            MaskResiduesWithUnresolvedBackboneAtoms(),
            FlagNonPolymersForAtomization(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=encoding.tokens),
            AddTokenAnnotation(encoding),
            EncodeAtomArray(encoding),
            PlaceUnresolvedTokenAtomsOnRepresentativeAtom(),
        ]
    )
    output = pipe(data)
    output_atom_array = output["atom_array"]

    # ...loop through each unresolved polymer token, and ensure that the unresolved atoms have the same coordinates as the representative atom
    for chain_id in np.unique(unresolved_polymer_atoms.chain_id):
        chain_atom_array = output_atom_array[output_atom_array.chain_id == chain_id]
        for res_id in np.unique(chain_atom_array.res_id):
            residue_atom_array = chain_atom_array[chain_atom_array.res_id == res_id]
            unresolved_atom_mask = residue_atom_array.occupancy == 0
            representative_atom_mask = get_af3_token_representative_masks(residue_atom_array)
            representative_atom_idx = np.where(representative_atom_mask)[0]

            output_atom_array_residue = output_atom_array[
                (output_atom_array.chain_id == chain_id) & (output_atom_array.res_id == res_id)
            ]
            assert np.allclose(
                residue_atom_array.coord[unresolved_atom_mask],
                output_atom_array_residue.coord[representative_atom_idx],
                atol=1e-6,
                equal_nan=True,
            )

    # ...loop through each unresolved non-polymer token, and ensure that nothing changed
    for chain_id in np.unique(unresolved_non_polymer_atoms.chain_id):
        output_chain_atom_array = output_atom_array[output_atom_array.chain_id == chain_id]
        input_chain_atom_array = atom_array[atom_array.chain_id == chain_id]
        assert_equal_atom_arrays(output_chain_atom_array, input_chain_atom_array)


def test_place_unresolved_token_on_closest_resolved_token_in_sequence():
    pdb_id = "6wtf"
    data = cached_parse(pdb_id)

    encoding = RF2AA_ATOM36_ENCODING
    pipe = Compose(
        [
            MaskResiduesWithUnresolvedBackboneAtoms(),
            FlagNonPolymersForAtomization(),
            AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=encoding.tokens),
            AddTokenAnnotation(encoding),
            EncodeAtomArray(encoding),
            PlaceUnresolvedTokenAtomsOnRepresentativeAtom(),
        ],
        track_rng_state=False,
    )
    output = pipe(data)
    input_atom_array = copy.deepcopy(data["atom_array"])

    # ...apply the transform
    output = PlaceUnresolvedTokenOnClosestResolvedTokenInSequence()(output)
    output_atom_array = output["atom_array"]

    for chain_id in np.unique(output_atom_array.chain_id):
        chain_atom_array = output_atom_array[output_atom_array.chain_id == chain_id]
        resolved_token_mask = apply_and_spread_token_wise(chain_atom_array, chain_atom_array.occupancy, function=np.any)

        # ...ensure that resolved tokens are unchanged
        input_atom_array_chain = input_atom_array[input_atom_array.chain_id == chain_id]
        assert np.allclose(
            chain_atom_array.coord[resolved_token_mask],
            input_atom_array_chain.coord[resolved_token_mask],
            atol=1e-6,
            equal_nan=True,
        )

        representative_tokens_coordinates = get_af3_token_representative_coords(input_atom_array_chain)

        # ...ensure that unresolved tokens have their tokens placed on the closest resolved token
        for idx, token in enumerate(token_iter(chain_atom_array)):
            # ...skip tokens with any resolved atoms
            if np.any(token.occupancy > 0):
                continue

            # ...assert all coordinates are the same within the token
            assert np.all(np.all(token.coord == token.coord[0]))

            # ...find the index of the closest resolved token
            # (Check below)
            lower_index = -float("inf")
            for i in range(idx - 1, -1, -1):
                if not np.isnan(representative_tokens_coordinates[i]).any():
                    lower_index = i
                    break
            # (Check above)
            upper_index = float("inf")
            for i in range(idx + 1, len(representative_tokens_coordinates)):
                if not np.isnan(representative_tokens_coordinates[i]).any():
                    upper_index = i
                    break

            # ...calculate the distance in sequence space to both the lower and upper resolved tokens
            if abs(idx - lower_index) <= abs(upper_index - idx):
                # ...assert that the closest resolved token is the lower one
                assert np.allclose(token.coord, representative_tokens_coordinates[lower_index], equal_nan=True)
            else:
                # ...assert that the closest resolved token is the upper one
                assert np.allclose(token.coord, representative_tokens_coordinates[upper_index], equal_nan=True)


if __name__ == "__main__":
    test_place_unresolved_token_atoms_on_representative_atom()
