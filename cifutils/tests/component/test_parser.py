from __future__ import annotations
import pytest
from cifutils.cifutils_biotite import cifutils_biotite
from cifutils.cifutils_legacy import cifutils_legacy
import biotite.structure as struc
import os
import numpy as np
from tests.conftest import get_digs_path


def parse_with_cifutils_legacy(filename, cif_parser_legacy):
    return cif_parser_legacy.parse(filename)


def parse_with_cifutils_biotite(filename, cifutils_biotite_parser):
    return cifutils_biotite_parser.parse(filename)


def convert_cifutils_biotite_to_legacy(result_dict):
    """
    Converts the result dictionary from cifutils_biotite_parser to the legacy format.
    NOTE: This function is slow; it is not optimized for performance, and should only be used for testing.
    """
    modres_legacy = None
    metadata_legacy = result_dict["metadata"]
    atom_array = result_dict["atom_array"]

    chains = {}
    for chain_id, chain_data in result_dict["chain_info"].items():
        legacy_atoms = {}
        chain_atoms = atom_array[atom_array.chain_id == chain_id]
        for i in range(len(chain_atoms)):
            atom = chain_atoms.get_atom(i)
            atom_id = (chain_id, str(atom.res_id), atom.res_name, atom.atom_name)
            legacy_atoms[atom_id] = cifutils_legacy.Atom(
                name=atom_id,
                element=int(atom.element),
                xyz=[round(coord, 3) for coord in atom.coord.tolist()],
                occ=round(float(atom.occupancy), 2),
                bfac=round(float(atom.b_factor), 2),
                charge=atom.charge,
                leaving=atom.leaving_atom_flag,
                leaving_group=atom.leaving_group,
                parent=None,  # We don't have this information in biotite, but could if necessary
                metal=atom.is_metal,
                hyb=atom.hyb,
                nhyd=atom.nhyd,
                hvydeg=atom.hvydeg,
                align=atom.align,
                hetero=None,  # We don't have this information in biotite, but could if necessary
            )

        bonds = []
        for bond in chain_atoms.bonds.as_array():
            a_atom = chain_atoms.get_atom(bond[0])
            b_atom = chain_atoms.get_atom(bond[1])
            bond_type = bond[2]
            bond = cifutils_legacy.Bond(
                a=(chain_id, str(a_atom.res_id), a_atom.res_name, a_atom.atom_name),
                b=(chain_id, str(b_atom.res_id), b_atom.res_name, b_atom.atom_name),
                aromatic=bond_type > 4,
                in_ring=False,
                order=bond_type if bond_type < 5 else bond_type - 4,
                intra=a_atom.res_id == b_atom.res_id and a_atom.chain_id == b_atom.chain_id,
                length=0,  # We don't have this information in biotite, but could if necessary
            )
            bonds.append(bond)

        # Build chirals, automorphisms, planars
        residue_info = result_dict["residue_info"]
        chirals = []
        automorphisms = []
        planars = []
        ids, names = struc.get_residues(chain_atoms)
        for residue_id, residue_name in zip(ids, names):
            # Chirals
            for chiral_center_list in residue_info[residue_name]["chirals"]:
                chiral_center = [(chain_id, str(residue_id), residue_name, atom) for atom in chiral_center_list]
                if chiral_center:
                    chirals.append(chiral_center)

            # Automorphisms
            automorphism_list = residue_info[residue_name]["automorphisms"]
            residue_automorphism_list = []
            for automorphism in automorphism_list:
                automorphism = [(chain_id, str(residue_id), residue_name, atom) for atom in automorphism]
                if automorphism:
                    residue_automorphism_list.append(automorphism)
            if len(residue_automorphism_list) > 0:
                automorphisms.append(residue_automorphism_list)

            # Planars
            for planar_list in residue_info[residue_name]["planars"]:
                planar = [(chain_id, str(residue_id), residue_name, atom) for atom in planar_list]
                if planar:
                    planars.append(planar)

        # Filter chirals. If we can't match all atoms, we discard the chiral
        chirals = [c for c in chirals if all([ci in legacy_atoms.keys() for ci in c])]

        # Filter planar. If we can't match all atoms, we discard the planar
        planars = [c for c in planars if all([ci in legacy_atoms.keys() for ci in c])]

        chain = cifutils_legacy.Chain(
            id=chain_id,
            type=chain_data["type"],
            sequence=chain_data["canonical_sequence"],
            atoms=legacy_atoms,
            bonds=bonds,
            chirals=chirals,
            planars=planars,
            automorphisms=automorphisms,
        )
        chains[chain_id] = chain

    return chains, modres_legacy, metadata_legacy


def validate_modified_residues(modres_legacy, converted_modres):
    """
    Validates that the converted_modres dictionary can be transformed to match the modres_legacy dictionary.
    We must handle cases where a modified residue is derived from multiple canonical residues, which the legacy format does not support.

    Args:
    - modres_legacy (dict): The legacy dictionary mapping modified residue names to their canonical names.
    - converted_modres (dict): The dictionary with lists of canonical residue names for each modified residue.
    """
    derived_modres = {}
    for key, value_list in converted_modres.items():
        mod_res_name = key[2]  # key: (chain_id, res_id, mod_res_name)
        sorted_list = sorted(value_list)
        last_element = sorted_list[-1]
        derived_modres[mod_res_name] = last_element

    assert derived_modres == modres_legacy


def validate_chains(pdb_id, chains_legacy, converted_chains):
    for chain_id, converted_chain in converted_chains.items():
        legacy_chain = chains_legacy[chain_id]
        converted_atoms = converted_chain.atoms
        legacy_atoms = legacy_chain.atoms

        legacy_atom_ids = set(legacy_atoms.keys())
        converted_atom_ids = set(converted_atoms.keys())

        # Assert number of atoms
        assert (
            len(legacy_atom_ids) == len(converted_atom_ids)
        ), f"Number of atoms mismatch for chain {chain_id} within PDB ID {pdb_id}: {len(legacy_atom_ids)} vs {len(converted_atom_ids)}.\n Missing atoms: {legacy_atom_ids - converted_atom_ids}\n Extra atoms: {converted_atom_ids - legacy_atom_ids}"
        # Assert atom IDs match
        assert (
            legacy_atom_ids == converted_atom_ids
        ), f"Atom IDs mismatch for chain {chain_id} within PDB ID {pdb_id}. Difference: {legacy_atom_ids.symmetric_difference(converted_atom_ids)}"
        # Assert the attributes match (except parent, which we ignore)
        for atom_id in legacy_atom_ids:
            legacy_atom = legacy_atoms[atom_id]
            converted_atom = converted_atoms[atom_id]
            assert (
                legacy_atom.name == converted_atom.name
            ), f"Name mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.element == converted_atom.element
            ), f"Element mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.name == converted_atom.name
            ), f"Name mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.occ == converted_atom.occ
            ), f"Occupancy mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.leaving == converted_atom.leaving
            ), f"Leaving atom flag mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert sorted(legacy_atom.leaving_group) == sorted(
                converted_atom.leaving_group
            ), f"Leaving group mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.metal == converted_atom.metal
            ), f"Metal flag mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.hyb == converted_atom.hyb
            ), f"Hybridization mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.nhyd == converted_atom.nhyd
            ), f"Number of hydrogens mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.hvydeg == converted_atom.hvydeg
            ), f"Heavy degree mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.align == converted_atom.align
            ), f"Alignment flag mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_atom.charge == converted_atom.charge
            ), f"Charge mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"

            # We need to handle the situation where the occupancy is 0.5 and there is an equally-occupied alternative location
            # Biotite will not pick between the two alternative locations deterministically
            # In that instance, the b-factors may be uncorrelated, so we don't check
            if legacy_atom.occ == 0.5 and converted_atom.occ == 0.5:
                assert np.allclose(
                    legacy_atom.xyz, converted_atom.xyz, atol=10
                ), f"(Partial occupancy) Approximate XYZ mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
            else:
                assert (
                    legacy_atom.xyz == converted_atom.xyz
                ), f"XYZ mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"
                assert (
                    legacy_atom.bfac == converted_atom.bfac
                ), f"B-factor mismatch for atom {atom_id} within chain {chain_id} in PDB ID {pdb_id}"

        legacy_bonds = legacy_chain.bonds
        converted_bonds = converted_chain.bonds
        legacy_bond_dict = {tuple(sorted((bond.a, bond.b))): bond for bond in legacy_bonds}
        converted_bond_dict = {tuple(sorted((bond.a, bond.b))): bond for bond in converted_bonds}

        # Assert number of bonds
        assert len(set(legacy_bond_dict.keys())) == len(
            set(converted_bond_dict.keys())
        ), f"Number of bonds mismatch for chain {chain_id} within PDB ID: {pdb_id}"

        legacy_bond_set = set(legacy_bond_dict.keys())
        converted_bond_set = set(converted_bond_dict.keys())
        # Assert bond IDs match
        assert (
            legacy_bond_set == converted_bond_set
        ), f"Bond IDs mismatch for chain {chain_id} within PDB ID {pdb_id}. Difference: {legacy_bond_set.symmetric_difference(converted_bond_set)}"

        # Compare bonds one by one
        for bond_id in legacy_bond_set:
            legacy_bond = legacy_bond_dict[bond_id]
            converted_bond = converted_bond_dict[bond_id]
            assert (
                legacy_bond.order == converted_bond.order
            ), f"Bond order mismatch for bond {bond_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_bond.aromatic == converted_bond.aromatic
            ), f"Aromatic flag mismatch for bond {bond_id} within chain {chain_id} in PDB ID {pdb_id}"
            assert (
                legacy_bond.intra == converted_bond.intra
            ), f"Intra-residue flag mismatch for bond {bond_id} within chain {chain_id} in PDB ID {pdb_id}"

        # Compare chirals
        legacy_chain_chirals = {tuple(sorted(chiral_center)) for chiral_center in legacy_chain.chirals}
        converted_chain_chirals = {tuple(sorted(chiral_center)) for chiral_center in converted_chain.chirals}
        assert legacy_chain_chirals == converted_chain_chirals, "Chirals mismatch."

        # Compare automorphisms
        legacy_chain_automorphisms = sorted(
            [sorted(map(tuple, inner_list)) for inner_list in legacy_chain.automorphisms]
        )
        converted_chain_automorphisms = sorted(
            [sorted(map(tuple, inner_list)) for inner_list in converted_chain.automorphisms]
        )
        assert legacy_chain_automorphisms == converted_chain_automorphisms, "Automorphisms mismatch."

        # Compare planars
        legacy_chain_planars = set(frozenset(planars) for planars in legacy_chain.planars)
        converted_chain_planars = set(frozenset(planars) for planars in converted_chain.planars)
        assert legacy_chain_planars == converted_chain_planars, "Planars mismatch."


@pytest.fixture(scope="module")
def cif_parser_legacy():
    return cifutils_legacy.CIFParser()


@pytest.fixture(scope="module")
def cifutils_biotite_parser():
    return cifutils_biotite.CIFParser(add_bonds=True, add_missing_atoms=True)


@pytest.mark.parametrize(
    "pdb_id",
    [
        "2k0a",
        "3k4a",
        "3kfa",
        "4az0",
        "2ejf",
        "5tmc",
        "6dru",
        "6s7t",
        "4xo3",
        "6tt7",
        "1khz",
        "1adl",
        "1nte",
        "3dpm",
        "1bs3",
        "2b4b",
        "1etu",
        "4ztt",
        "1brx",
        "3nez",
        "4ndz",
        "1lys",
        "6dmg",
        "1a8o",
        "6wjc",
        "4js1",
        "1ivo",
        "1fu2",
        "1cbn",
        "1en2",
        "1y1w",
        "133d",
        "5xnl",
        "6wtf",
        "1azx",
        "2e2h",
        "1q1k",
        "3ne7",
    ],
)
def test_parsing(pdb_id, cif_parser_legacy, cifutils_biotite_parser):
    """
    Compare the results of parsing a CIF file with cifutils_legacy and cifutils_biotite.

    Coverage for:
    - Atoms
    - Bonds
    - Metadata

    Does not compare:
    - Assembly (handled by test_bioassemblies.py)
    """
    filename = get_digs_path(pdb_id)

    # Parse with cifutils_legacy
    chains_legacy, asmb_legacy, covale_legacy, meta_legacy, modres_legacy = parse_with_cifutils_legacy(
        filename, cif_parser_legacy
    )

    # Parse with cifutils_biotite
    result_dict = parse_with_cifutils_biotite(filename, cifutils_biotite_parser)

    # Compare cifutils_legacy with biotite atom locations
    converted_chains, converted_modres, converted_metadata = convert_cifutils_biotite_to_legacy(result_dict)

    # Validate chains
    validate_chains(pdb_id, chains_legacy, converted_chains)

    # Test metadata
    assert result_dict["metadata"]["method"] == meta_legacy["method"], "Method mismatch."
    assert result_dict["metadata"]["resolution"] == meta_legacy["resolution"], "Resolution mismatch."
    assert result_dict["metadata"]["date"] == meta_legacy["date"], "Date mismatch."

    # Test modified residue dict
    validate_modified_residues(modres_legacy, result_dict["modified_residues"])


def test_unmatched_atom_types():
    """
    Ensure that unmatched atom types are handled correctly. For cifutils_biotite, that means masking the residue with the unmathced atom with 0 occupancy.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(script_dir, "../test_data/1a8o_modified.cif")

    # Parse with cifutils_biotite
    result_dict = parse_with_cifutils_biotite(
        filename, cifutils_biotite.CIFParser(add_bonds=True, add_missing_atoms=True)
    )

    # Ensure that residue 2 has no occupancy
    atom_array = result_dict["atom_array"]
    residue_2 = atom_array[(atom_array.chain_id == "A") & (atom_array.res_id == 2)]
    assert np.sum(residue_2.occupancy) == 0


if __name__ == "__main__":
    # Test a single example
    cif_parser_legacy = cifutils_legacy.CIFParser()
    cifutils_biotite_parser = cifutils_biotite.CIFParser(add_bonds=True, add_missing_atoms=True)
    test_parsing("2k0a", cif_parser_legacy, cifutils_biotite_parser)
