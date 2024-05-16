import pytest
import gzip
import io
from cifutils import cifutils_extended, cifutils_legacy, parser_utils
from Bio.PDB.MMCIFParser import MMCIFParser, MMCIF2Dict
import logging
import numpy as np
import collections


# For common Bond representation
Bond = collections.namedtuple('Bond', [
    'a', 'b', 'aromatic', 'in_ring', 'order', 'intra', 'length'
])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_legacy_bonds(chains_legacy, covale_legacy):
    legacy_bonds = {}
    for chain in chains_legacy.values():
        for bond in chain.bonds:
            id = tuple(sorted([bond.a, bond.b]))
            legacy_bonds[id] = (
                bond.aromatic,
                bond.in_ring,
                bond.order,
                bond.intra,
                bond.length
            )
    for bond in covale_legacy:
        id = tuple(sorted([bond.a, bond.b]))
        legacy_bonds[id] = (
            bond.aromatic,
            bond.in_ring,
            bond.order,
            bond.intra,
            bond.length
        )
    return legacy_bonds
    
def compare_bonds(chains_legacy, structure_cifutils_extended, covale_legacy):
    legacy_bonds = extract_legacy_bonds(chains_legacy, covale_legacy)
    extended_bonds = extract_extended_bonds(structure_cifutils_extended)

    # Compute the difference between the two bond sets
    for id, bond_legacy in legacy_bonds.items():
        assert id in extended_bonds, f"Bonds mismatch: {id}"
        bond_extended = extended_bonds[id]
        # Assert tuples are equal
        assert bond_legacy == bond_extended, f"Bond attribute mismatch: {id} {bond_legacy} vs {bond_extended}"
    
    # Compute the opposite direction
    for id, bond_extended in extended_bonds.items():
        assert id in legacy_bonds, f"Bonds mismatch: {id}"
        bond_legacy = legacy_bonds[id]
        # Assert tuples are equal
        assert bond_legacy == bond_extended, f"Bond attribute mismatch: {id} {bond_legacy} vs {bond_extended}"
    
    assert len(legacy_bonds) == len(extended_bonds), "Bonds mismatch"

    return True
    
def extract_extended_bonds(structure_cifutils_extended):
    extended_bonds = {}
    for model in structure_cifutils_extended.get_models():
        for bond in model.get_bonds():
            atom_a = (bond.atom_a.get_parent_chain_id(), bond.atom_a.get_parent_residue_id(), bond.atom_a.get_parent_residue_name(), bond.atom_a.name)
            atom_b = (bond.atom_b.get_parent_chain_id(), bond.atom_b.get_parent_residue_id(), bond.atom_b.get_parent_residue_name(), bond.atom_b.name)
            id = tuple(sorted([atom_a, atom_b]))
            extended_bonds[id] = (
                bond.is_aromatic,
                bond.in_ring,
                bond.order,
                bond.type == 'intra_residue',
                bond.length
            )
    return extended_bonds

def extract_extended_atoms(structure):
    extended_atoms = {}
    for model in structure.get_models():
        for chain in model.get_chains():
            for residue in chain.get_residues():
                for atom in residue.get_atoms():
                    atom_id = (chain.id, residue.id, atom.name)
                    extended_atoms[atom_id] = {
                        'element': atom.element,
                        'xyz': tuple(atom.xyz),
                        'occupancy': atom.occupancy,
                        'bfactor': atom.bfactor,
                        'charge': atom.charge,
                        'leaving_atom_flag': atom.leaving_atom_flag,
                        'leaving_group': tuple(atom.leaving_group),
                        'parent_heavy_atom': atom.parent_heavy_atom,
                        'is_metal': atom.is_metal,
                        'hyb': atom.hyb,
                        'nhyd': atom.nhyd,
                        'hvydeg': atom.hvydeg,
                        'align': atom.align,
                        'hetero': atom.hetero,
                    }
    return extended_atoms

def extract_legacy_atoms(chains):
    legacy_atoms = {}
    for chain in chains.values():
        for atom in chain.atoms.values():
            atom_id = (atom.name[0], atom.name[1], atom.name[3])
            legacy_atoms[atom_id] = {
                'element': atom.element,
                'xyz': tuple(atom.xyz),
                'occupancy': atom.occ,
                'bfactor': atom.bfac,
                'charge': atom.charge,
                'leaving_atom_flag': atom.leaving,
                'leaving_group': tuple(atom.leaving_group),
                'parent_heavy_atom': atom.parent,
                'is_metal': atom.metal,
                'hyb': atom.hyb,
                'nhyd': atom.nhyd,
                'hvydeg': atom.hvydeg,
                'align': atom.align,
                'hetero': atom.hetero,
            }
    return legacy_atoms

def compare_atoms(legacy_atoms, extended_atoms):
    for atom_id, legacy_atom in legacy_atoms.items():
        assert atom_id in extended_atoms, f"Atom {atom_id} not found in extended representation"
        extended_atom = extended_atoms[atom_id]
        assert legacy_atom == extended_atom, f"Atom {atom_id} mismatch: {legacy_atom} vs {extended_atom}"
    
    for atom_id, extended_atom in extended_atoms.items():
        assert atom_id in legacy_atoms, f"Atom {atom_id} not found in legacy representation"
        legacy_atom = legacy_atoms[atom_id]
        assert legacy_atom == extended_atom, f"Atom {atom_id} mismatch: {legacy_atom} vs {extended_atom}"
    
    assert len(legacy_atoms) == len(extended_atoms), "Atom count mismatch"

def parse_with_cifutils_legacy(filename):
    cif_parser_legacy = cifutils_legacy.CIFParser()
    return cif_parser_legacy.parse(filename)

def parse_with_cifutils_extended(filename):
    cif_parser_extended = cifutils_extended.CIFParser()
    return cif_parser_extended.parse(filename)

def parse_with_biopython(filename, pdb_id):
    biopython_parser = MMCIFParser(QUIET=True)
    with gzip.open(filename, 'rt') as file:
        structure = biopython_parser.get_structure(pdb_id, file)
    return structure

def compare_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    
    for key in dict1:
        if not compare_items(dict1[key], dict2[key]):
            return False
    
    return True

def compare_items(item1, item2):
    if type(item1) != type(item2):
        return False
    
    if isinstance(item1, dict):
        return compare_dicts(item1, item2)
    
    if isinstance(item1, list) or isinstance(item1, tuple):
        if len(item1) != len(item2):
            return False
        return all(compare_items(i1, i2) for i1, i2 in zip(item1, item2))
    
    if isinstance(item1, np.ndarray):
        return np.array_equal(item1, item2)
    
    return item1 == item2
    
def compare_cifutils_legacy_and_extended(chains_legacy, asmb_legacy, covale_legacy, meta_legacy, modres_legacy, structure_cifutils_extended, asmb_extended, modified_residues_extended):
    # We only consider the first model
    structure_cifutils_extended.child_list = structure_cifutils_extended.child_list[:1]
    # Compare asmb
    assert compare_dicts(asmb_legacy, asmb_extended), "ASMB mismatch"

    # Compare modified residues
    assert compare_dicts(modres_legacy, modified_residues_extended), "MODRES mismatch"

    # Compare bonds
    assert compare_bonds(chains_legacy, structure_cifutils_extended, covale_legacy), "Bonds mismatch"

    # Extract and compare atoms
    legacy_atoms = extract_legacy_atoms(chains_legacy)
    extended_atoms = extract_extended_atoms(structure_cifutils_extended)
    assert compare_atoms(legacy_atoms, extended_atoms), "Atoms mismatch"

def compare_parsers(pdb_id, parser_1_all_xyz, parser_1_heavy_xyz, parser_2_all_xyz, parser_2_heavy_xyz, parser_1_common_atoms, parser_2_common_atoms, parser_1_original_atoms, parser_2_original_atoms):
    # Asserting that XYZ coordinates match
    all_xyz_mismatches = parser_1_all_xyz.symmetric_difference(parser_2_all_xyz)
    heavy_xyz_mismatches = parser_1_heavy_xyz.symmetric_difference(parser_2_heavy_xyz)

    # Log warning for non-heavy XYZ mismatches
    if all_xyz_mismatches:
        # Loop through mismatches
        for mismatch in all_xyz_mismatches:
            biopython_atom = parser_2_original_atoms[mismatch] if mismatch in parser_2_original_atoms else None
            cifutils_atom = parser_1_original_atoms[mismatch] if mismatch in parser_1_original_atoms else None
            logger.warning(f"XYZ coordinate mismatch at {mismatch}: BioPython {biopython_atom} vs CIFParser {cifutils_atom}")
        logger.warning(f"Non-heavy XYZ coordinate mismatches ({len(all_xyz_mismatches)} errors), e.g., {next(iter(all_xyz_mismatches), None)}")

    assert not heavy_xyz_mismatches, f"Heavy XYZ coordinate mismatches ({len(heavy_xyz_mismatches)} errors), e.g., {next(iter(heavy_xyz_mismatches), None)}"

    # Mismatch counters
    mismatches = {
        "chain_id": 0,
        "charge": 0
    }

    # Compare the atomic data for common elements
    for xyz in parser_1_heavy_xyz & parser_2_heavy_xyz:
        cif_atom = parser_1_common_atoms[xyz]
        bio_atom = parser_2_common_atoms[xyz]
        
        # Check all attributes, handle charge and chain_id separately without assertions
        for key in mismatches.keys():
            if key in ["charge", "chain_id"]:  # Check without assertions
                if cif_atom[key] != bio_atom[key]:
                    mismatches[key] += 1
            else:  # Use assertions for other keys
                assert cif_atom[key] == bio_atom[key], f"{key.capitalize()} mismatch at {xyz}: {cif_atom[key]} vs {bio_atom[key]}"

    # Report all mismatches
    for key, count in mismatches.items():
        if count > 0:
            logger.warning(f"{pdb_id}: Total atoms with {key} mismatched: {count}")
    
    # If no mismatches in mismatches, report that the test passed
    if all(count == 0 for count in mismatches.values()):
        logger.info("All atomic data matches between parsers for PDB ID: %s", pdb_id)

def create_common_atom(periodic_table, chain_id, residue_name, atom_name, element, xyz_coords, occ, bfac, charge):
    formatted_xyz = tuple(round(float(coord), 3) for coord in xyz_coords)
    formatted_residue_name = residue_name.strip()
    formatted_atom_name = atom_name.strip()
    formatted_chain_id = chain_id.strip()
    
    if isinstance(element, str):
        element_number = periodic_table.get_atomic_number(element.strip())
    elif isinstance(element, int):
        element_number = element
    
    formatted_occ = round(float(occ), 1)
    formatted_bfac = round(float(bfac), 2)
    formatted_charge = int(charge) if charge is not None else 0

    return {
        "chain_id": formatted_chain_id,
        "residue_name": formatted_residue_name,
        "atom_name": formatted_atom_name,
        "element": element_number,
        "xyz": formatted_xyz,
        "occ": formatted_occ,
        "bfac": formatted_bfac,
        "charge": formatted_charge
    }


def extract_extended_cifutils_atoms(structure, periodic_table):
    common_atoms = {}
    all_xyz = set()
    heavy_xyz = set()
    for model in structure.get_models():
        for atom in model.get_atoms():
            atom_data = create_common_atom(
                periodic_table,
                chain_id=atom.get_parent_chain_id(),
                residue_name=atom.parent.name,
                atom_name=atom.name,
                element=atom.element,
                xyz_coords=atom.xyz,
                occ=atom.occupancy,
                bfac=atom.bfactor,
                charge=atom.charge
            )
            if atom_data['xyz'] != (0.000, 0.000, 0.000):
                all_xyz.add(atom_data['xyz'])
                if atom_data['element'] > 1:
                    heavy_xyz.add(atom_data['xyz'])
                common_atoms[atom_data['xyz']] = atom_data
    return common_atoms, all_xyz, heavy_xyz

def extract_legacy_cifutils_atoms(chains, periodic_table):
    common_atoms = {}
    all_xyz = set()
    heavy_xyz = set()
    for chain in chains.values():
        for atom in chain.atoms.values():
            atom_data = create_common_atom(
                periodic_table,
                chain_id=atom.name[0],
                residue_name=atom.name[2],
                atom_name=atom.name[3],
                element=atom.element,
                xyz_coords=atom.xyz,
                occ=atom.occ,
                bfac=atom.bfac,
                charge=atom.charge
            )
            if atom_data['xyz'] != (0.000, 0.000, 0.000):
                all_xyz.add(atom_data['xyz'])
                if atom_data['element'] > 1:
                    heavy_xyz.add(atom_data['xyz'])
                common_atoms[atom_data['xyz']] = atom_data
    return common_atoms, all_xyz, heavy_xyz

def extract_biopython_atoms(structure, periodic_table, first_model_only = False):
    common_atoms = {}
    all_xyz = set()
    heavy_xyz = set()
    if first_model_only:
        structure.child_list = structure.child_list[:1]
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_data = create_common_atom(
                        periodic_table,
                        chain_id=chain.id,
                        residue_name=atom.get_parent().get_resname(),
                        atom_name=atom.name,
                        element=atom.element,
                        xyz_coords=atom.get_coord(),
                        occ=atom.occupancy,
                        bfac=atom.bfactor,
                        charge=atom.pqr_charge
                    )
                    all_xyz.add(atom_data['xyz'])
                    if atom_data['element'] > 1:
                        heavy_xyz.add(atom_data['xyz'])
                    common_atoms[atom_data['xyz']] = atom_data
    return common_atoms, all_xyz, heavy_xyz

@pytest.mark.parametrize("pdb_id", ["1lys", "6dmg", "1a8o", "6dmh", "1a8o"])
def test_atom_coordinates(pdb_id):
    filename = f'/databases/rcsb/cif/{pdb_id[1:3]}/{pdb_id}.cif.gz'
    periodic_table = parser_utils.PeriodicTable()
    
    # Parse with cifutils_legacy
    chains_legacy, asmb_legacy, covale_legacy, meta_legacy, modres_legacy = parse_with_cifutils_legacy(filename)
    cifutils_legacy_common_atoms, cifutils_legacy_all_xyz_legacy, cifutils_legacy__heavy_xyz_legacy = extract_legacy_cifutils_atoms(chains_legacy, periodic_table)
    
    # Parse with cifutils_extended
    structure_cifutils_extended, asmb_extended, modified_residues_extended = parse_with_cifutils_extended(filename)
    cifutils_extended_common_atoms, cifutils_extended_all_xyz, cifutils_extended_heavy_xyz = extract_extended_cifutils_atoms(structure_cifutils_extended, periodic_table)
    
    # Parse with biopython
    structure_biopython = parse_with_biopython(filename, pdb_id)
    biopython_common_atoms_for_legacy, biopython_all_xyz_for_legacy, biopython_heavy_xyz_for_legacy = extract_biopython_atoms(structure_biopython, periodic_table, first_model_only = True)
    biopython_common_atoms_for_extended, biopython_all_xyz_for_extended, biopython_heavy_xyz_for_extended = extract_biopython_atoms(structure_biopython, periodic_table, first_model_only = False)
    
    # Compare cifutils_legacy with biopython
    logging.info(f" ---- Comparing {pdb_id} with cifutils_legacy and biopython ----")
    compare_parsers(pdb_id, cifutils_legacy_all_xyz_legacy, cifutils_legacy__heavy_xyz_legacy, biopython_all_xyz_for_legacy, biopython_heavy_xyz_for_legacy, cifutils_legacy_common_atoms, biopython_common_atoms_for_legacy, {}, {})

    # Compare cifutils_extended with biopython
    logging.info(f" ---- Comparing {pdb_id} with cifutils_extended and biopython ----")
    compare_parsers(pdb_id, cifutils_extended_all_xyz, cifutils_extended_heavy_xyz, biopython_all_xyz_for_extended, biopython_heavy_xyz_for_extended, cifutils_extended_common_atoms, biopython_common_atoms_for_extended, {}, {})

    # Compare cifutils_legacy with cifutils_extended
    logging.info(f" ---- Comparing {pdb_id} with cifutils_legacy and cifutils_extended ----")
    compare_parsers(pdb_id, cifutils_legacy_all_xyz_legacy, cifutils_legacy__heavy_xyz_legacy, cifutils_extended_all_xyz, cifutils_extended_heavy_xyz, cifutils_legacy_common_atoms, cifutils_extended_common_atoms, {}, {})

    # Perform a deep compare of cifutils legacy and extended
    compare_cifutils_legacy_and_extended(chains_legacy, asmb_legacy, covale_legacy, meta_legacy, modres_legacy, structure_cifutils_extended, asmb_extended, modified_residues_extended)
    

if __name__ == "__main__":
    # test_non_canonical_amino_acids("3k4a")
    test_atom_coordinates("4js1") # Oligosaccharide
    test_atom_coordinates("1ivo") # (3) residues unobserved in the middle, a handful of modified amino acids (MSE)
    test_atom_coordinates("1lys")
    test_atom_coordinates("1a8o") # (3) residues unobserved in the middle, a handful of modified amino acids (MSE)
    test_atom_coordinates("6dmg")
    test_atom_coordinates("1cbn")
    test_atom_coordinates("6wjc") # Starts on label_seq_id 26, since first 25 were unobserved
    test_atom_coordinates("1y1w") # Protein-nucleic acid complex
    test_atom_coordinates("133d") # DNA
    test_atom_coordinates("6dmh") # Multiconformer ligand
    test_atom_coordinates("6wtf")
    test_atom_coordinates("1azx")
    # test_atom_coordinates("1zy8") # Polypeptide with FAD ligand assigned to polypeptide chain 0 (causes error when importing). Asymmetric unit is smaller than biological unit.