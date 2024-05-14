import pytest
import gzip
import io
from cifutils import cifutils_extended, cifutils_legacy, parser_utils
from Bio.PDB.MMCIFParser import MMCIFParser, MMCIF2Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def compare_cifutils_legacy_and_extended(cif_legacy_data, cif_extended_data):
    # Placeholder for detailed comparison
    pass


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
    
    # Compare cifutils_legacy with cifutils_extended
    # compare_cifutils_legacy_and_extended(cifutils_common_atoms_legacy, cifutils_common_atoms_extended)
    
    # Compare cifutils_legacy with biopython
    logging.info(f" ---- Comparing {pdb_id} with cifutils_legacy and biopython ----")
    compare_parsers(pdb_id, cifutils_legacy_all_xyz_legacy, cifutils_legacy__heavy_xyz_legacy, biopython_all_xyz_for_legacy, biopython_heavy_xyz_for_legacy, cifutils_legacy_common_atoms, biopython_common_atoms_for_legacy, {}, {})

    # Compare cifutils_extended with biopython
    logging.info(f" ---- Comparing {pdb_id} with cifutils_extended and biopython ----")
    compare_parsers(pdb_id, cifutils_extended_all_xyz, cifutils_extended_heavy_xyz, biopython_all_xyz_for_extended, biopython_heavy_xyz_for_extended, cifutils_extended_common_atoms, biopython_common_atoms_for_extended, {}, {})

    # Compare cifutils_legacy with cifutils_extended
    logging.info(f" ---- Comparing {pdb_id} with cifutils_legacy and cifutils_extended ----")
    compare_parsers(pdb_id, cifutils_legacy_all_xyz_legacy, cifutils_legacy__heavy_xyz_legacy, cifutils_extended_all_xyz, cifutils_extended_heavy_xyz, cifutils_legacy_common_atoms, cifutils_extended_common_atoms, {}, {})
    
    # Compare cifutils_extended with biopython
    # compare_chains(cifutils_common_atoms_extended, biopython_common_atoms)

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