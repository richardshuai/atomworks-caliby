""" 
Areas not covered by these unit tests:
- Atom charges (there were discrepancies, and we're not currently using them in RF2AA)
"""

import pytest
import gzip
import io
from cifutils import cifutils_extended, cifutils_legacy, parser_utils
from Bio.PDB.MMCIFParser import MMCIFParser, MMCIF2Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_common_atom(periodic_table, chain_id, residue_name, atom_name, element, xyz_coords, occ, bfac, charge):
    """ 
    Create a dictionary enforcing consistent types across all elements.
    Used to compare between cifutils and BioPython outputs.
    """
    # Format XYZ coordinates
    formatted_xyz = tuple(round(float(coord), 3) for coord in xyz_coords)
    
    # Strip spaces for residue_name, atom_name, and chain_id
    formatted_residue_name = residue_name.strip()
    formatted_atom_name = atom_name.strip()
    formatted_chain_id = chain_id.strip()
    
    # Convert element to its atomic number; handle both string and numeric inputs
    if isinstance(element, str):
        element_number = periodic_table.get_atomic_number(element.strip())
    elif isinstance(element, int):
        element_number = element  # Assume it's already an atomic number if integer
    
    # Format occupancy, b-factor and charge
    formatted_occ = round(float(occ), 1)
    formatted_bfac = round(float(bfac), 2)
    formatted_charge = int(charge) if charge is not None else 0
    
    # Create and return the atom dictionary
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


@pytest.mark.parametrize("pdb_id", ["1lys", "6dmg", "1a8o", "6dmh", "1a8o"])
def test_non_canonical_amino_acids(pdb_id):
    # Parse filename
    filename = f'/databases/rcsb/cif/{pdb_id[1:3]}/{pdb_id}.cif.gz'

    # Initialize CIFParser and parse data
    cif_parser = cifutils_extended.CIFParser()
    chains, asmb, covale, meta, modres = cif_parser.parse(filename)
    
    if pdb_id == "3k4a":
        # Check if modres is {'MSE': 'MET'}
        assert modres == {'MSE': 'MET'}, f"MODRES dictionary does not match expected output for PDB ID: {pdb_id}"
        # Ensure that the MSE residue is present in the chains
        assert any(residue.name == 'MSE' for chain in chains.values() for residue in chain.residues.values()), f"MSE residue not found in chains for PDB ID: {pdb_id}"

    print("Hmm.")


@pytest.mark.parametrize("pdb_id", ["1lys", "6dmg", "1a8o", "6dmh", "1a8o"])
def test_atom_coordinates(pdb_id):
    # Parse filename
    filename = f'/databases/rcsb/cif/{pdb_id[1:3]}/{pdb_id}.cif.gz'
    periodic_table = parser_utils.PeriodicTable()

    # Initialize CIFParser legacy and parse data
    cif_parser_legacy = cifutils_legacy.CIFParser()
    chains, _, _, _, _ = cif_parser_legacy.parse(filename)

    # Extracting coordinates from CIFParser output
    cifutils_common_atoms = {}
    cifutils_original_atoms = {} # for debugging
    cifutils_all_xyz = set()
    cifutils_heavy_xyz = set()
    for chain in chains.values():
        for atom in chain.atoms.values():
            # Check to make sure the coordinates aren't all zeros
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
            if atom_data['xyz'] != (0.000, 0.000, 0.000): # All coordinates are rounded to 3 decimals
                cifutils_all_xyz.add(atom_data['xyz'])
                if atom_data['element'] > 1:
                    cifutils_heavy_xyz.add(atom_data['xyz'])
                cifutils_common_atoms[atom_data['xyz']] = atom_data
                cifutils_original_atoms[atom_data['xyz']] = atom

    # Initialize Biopython PDB parser and parse the same file
    biopython_parser = MMCIFParser(QUIET=True)
    with gzip.open(filename, 'rt') as file:
        structure = biopython_parser.get_structure(pdb_id, file)

    # Extract coordinates from BioPython output
    biopython_common_atoms = {}
    biopython_original_atoms = {} # for debugging
    biopython_all_xyz = set()
    biopython_heavy_xyz = set()

    # Use the first model in the structure
    model = structure.child_list[0]
    for chain in model.get_chains():
        for residue in chain.get_residues():
            for atom in residue.get_atoms():
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
                biopython_all_xyz.add(atom_data['xyz'])
                if atom_data['element'] > 1:
                    biopython_heavy_xyz.add(atom_data['xyz'])
                biopython_common_atoms[atom_data['xyz']] = atom_data
                biopython_original_atoms[atom_data['xyz']] = atom

    # Asserting that XYZ coordinates match
    all_xyz_mismatches = cifutils_all_xyz.symmetric_difference(biopython_all_xyz)
    heavy_xyz_mismatches = cifutils_heavy_xyz.symmetric_difference(biopython_heavy_xyz)

    # Log warning for non-heavy XYZ mismatches
    if all_xyz_mismatches:
        # Loop through mismatches
        for mismatch in all_xyz_mismatches:
            biopython_atom = biopython_original_atoms[mismatch] if mismatch in biopython_original_atoms else None
            cifutils_atom = cifutils_original_atoms[mismatch] if mismatch in cifutils_original_atoms else None
            logger.warning(f"XYZ coordinate mismatch at {mismatch}: BioPython {biopython_atom} vs CIFParser {cifutils_atom}")
        logger.warning(f"Non-heavy XYZ coordinate mismatches ({len(all_xyz_mismatches)} errors), e.g., {next(iter(all_xyz_mismatches), None)}")

    assert not heavy_xyz_mismatches, f"Heavy XYZ coordinate mismatches ({len(heavy_xyz_mismatches)} errors), e.g., {next(iter(heavy_xyz_mismatches), None)}"

    # Mismatch counters
    mismatches = {
        "chain_id": 0,
        "charge": 0
    }

    # Compare the atomic data for common elements
    for xyz in cifutils_heavy_xyz & biopython_heavy_xyz:
        cif_atom = cifutils_common_atoms[xyz]
        bio_atom = biopython_common_atoms[xyz]
        
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
        logger.info("All atomic data matches between CIFParser and BioPython for PDB ID: %s", pdb_id)
    

if __name__ == "__main__":
    # test_non_canonical_amino_acids("3k4a")
    # test_atom_coordinates("4js1") # Oligosaccharide
    # test_atom_coordinates("1ivo") # (3) residues unobserved in the middle, a handful of modified amino acids (MSE)
    # test_atom_coordinates("1lys")
    # test_atom_coordinates("1a8o") # (3) residues unobserved in the middle, a handful of modified amino acids (MSE)
    # test_atom_coordinates("6dmg")
    # test_atom_coordinates("1cbn")
    # test_atom_coordinates("6wjc") # Starts on label_seq_id 26, since first 25 were unobserved
    # test_atom_coordinates("1y1w") # Protein-nucleic acid complex
    # test_atom_coordinates("133d") # DNA
    # test_atom_coordinates("6dmh") # Multiconformer ligand
    # test_atom_coordinates("6wtf")
    # test_atom_coordinates("1azx")
    # test_atom_coordinates("1zy8") # Polypeptide with FAD ligand assigned to polypeptide chain 0 (causes error when importing). Asymmetric unit is smaller than biological unit.