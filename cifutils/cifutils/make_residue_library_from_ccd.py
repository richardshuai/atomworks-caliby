"""Pre-process the Chemical Component Dictionary (CCD) with OpenBabel to create a residue library for use in the CIF parser."""

import glob
import logging
import numpy as np
import pickle
import time
from openbabel import openbabel
from cifutils import cifutils_legacy
import networkx as nx
import cifutils.obutils as obutils

logging.basicConfig(level=logging.INFO)

def get_leaving_atoms_from_graph(atom_name, graph):
    """
    Identifies leaving groups connected to a given atom in the molecular graph.
    
    Parameters:
    atom_name (str): The name of the atom.
    graph (networkx.Graph): The molecular graph.
    
    Returns:
    list: A list of atom names in the leaving group.
    """
    if graph.nodes[atom_name]['leaving']:
        return []

    leaving_group = set()
    
    for neighbor in graph.neighbors(atom_name):
        if graph.nodes[neighbor]['leaving']:
            leaving_group.add(neighbor)
            subgraph = graph.subgraph(set(graph.nodes) - {neighbor})
            connected_components = nx.connected_components(subgraph)
            for component in connected_components:
                if atom_name not in component:
                    leaving_group.update(component)

    return list(leaving_group)

def read_sdf_file(sdfname, obConversion):
    """
    Reads an SDF file and returns an OBMol object.
    
    Parameters:
    sdfname (str): The name of the SDF file.
    obConversion (openbabel.OBConversion): Open Babel conversion object.
    
    Returns:
    openbabel.OBMol: The molecule read from the SDF file.
    """
    obmol = openbabel.OBMol()
    obConversion.ReadFile(obmol, sdfname)
    return obmol

def validate_coordinates(obmol, cif):
    """
    Validates the coordinates of the molecule against the CIF data.
    
    Parameters:
    obmol (openbabel.OBMol): The molecule object.
    cif (dict): The CIF data.
    
    Returns:
    bool: True if validation is successful, False otherwise.
    """
    xyz = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in openbabel.OBMolAtomIter(obmol)])
    if obmol.NumAtoms() != cif['xyz'].shape[0]:
        return False
    return ((xyz - cif['xyz'])[~np.isnan(cif['xyz'])] < 1e-3).all()  # Check if coordinates are within 1e-3

def extract_residue_details(obmol, atom_id, leaving, pdbx_align):
    """
    Extracts detailed information about the residue from the OBMol object.
    
    Parameters:
    obmol (openbabel.OBMol): The molecule object.
    atom_id (List[str]): List of atom IDs.
    leaving (List[bool]): List of leaving flags for each atom.
    pdbx_align (List[int]): List of alignment indices.
    
    Returns:
    dict: Dictionary containing residue details.
    """
    obmol_ph = openbabel.OBMol(obmol)
    obmol_ph.CorrectForPH()
    obmol_ph.DeleteHydrogens()
    ha_iter = openbabel.OBMolAtomIter(obmol_ph) # Iterate over all atoms (non-hydrogen)
    
    residue_atoms = {}
    for atom_name, is_leaving, align, atom in zip(atom_id, leaving, pdbx_align, openbabel.OBMolAtomIter(obmol)):
        charge = atom.GetFormalCharge() # Not corrected for pH / including hydrogens
        nhyd = atom.ExplicitHydrogenCount()
        if atom.GetAtomicNum() > 1:
            ha = next(ha_iter)
            charge = ha.GetFormalCharge() # Corrected for pH
            nhyd = ha.GetTotalDegree() - ha.GetHvyDegree()
        
        residue_atoms[atom_name] = {
            'id': atom_name,
            'leaving_atom_flag': is_leaving,
            'leaving_group': [],
            'element': atom.GetAtomicNum(),
            'is_metal': atom.IsMetal(),
            'charge': charge,
            'hyb': atom.GetHyb(),
            'nhyd': nhyd,
            'hvydeg': atom.GetHvyDegree(),
            'align': align,
        }
    
    bonds = []
    for b in openbabel.OBMolBondIter(obmol):
        atom_a_name = atom_id[b.GetBeginAtom().GetIndex()]
        atom_b_name = atom_id[b.GetEndAtom().GetIndex()]
        bonds.append({
            'atom_a_id': atom_a_name,
            'atom_b_id': atom_b_name,
            'is_aromatic': b.IsAromatic(),
            'in_ring': b.IsInRing(),
            'order': b.GetBondOrder(),
            'length': b.GetLength(),
        })
    
    automorphisms = obutils.FindAutomorphisms(obmol, heavy=True)
    
    mask = (automorphisms[:1] == automorphisms).all(dim=0)
    automorphisms = automorphisms[:, ~mask]
    
    chirals = obutils.GetChirals(obmol, heavy=True)
    planars = obutils.GetPlanars(obmol, heavy=True)
    
    G = nx.Graph()
    G.add_nodes_from([(a['id'], {'leaving': a['leaving_atom_flag']}) for a in residue_atoms.values()])
    G.add_edges_from([(bond['atom_a_id'], bond['atom_b_id']) for bond in bonds])
    for k, v in residue_atoms.items():
        v['leaving_group'] = get_leaving_atoms_from_graph(k, G)
    
    anames = np.array(atom_id)
    return {
        'name': obmol.GetTitle(),
        'intra_residue_bonds': bonds,
        'automorphisms': anames[automorphisms].tolist(),
        'chirals': anames[chirals].tolist(),
        'planars': anames[planars].tolist(),
        'atoms': residue_atoms
    }

def process_ligands(sdfnames):
    """
    Processes a list of SDF files to create a dictionary of ligands with their details.
    
    Parameters:
    sdfnames (list): List of SDF file names.
    
    Returns:
    dict: Dictionary containing ligands information.
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("sdf")
    ligands = {}
    total_ligands = len(sdfnames)
    start_time = time.time()

    for i, sdfname in enumerate(sdfnames):
        if i > 0 and i % (total_ligands // 10) == 0:
            logging.info(f"Processing {i}/{total_ligands} ligands ({(i / total_ligands) * 100:.1f}%)")
        
        obmol = read_sdf_file(sdfname, obConversion)
        cifname = sdfname.replace('_model.sdf', '.cif')
        
        try:
            cif = cifutils_legacy.ParsePDBLigand(cifname)
        except Exception as e:
            logging.error(f"FAILED: {sdfname}, due to error: {str(e)}")
            continue

        if not validate_coordinates(obmol, cif):
            logging.error(f"FAILED: {sdfname}")
            continue

        id = cifname.split('/')[-1][:-4]
        ligand_details = extract_residue_details(
            obmol=obmol,
            atom_id=cif['atom_id'].tolist(),
            leaving=cif['leaving'].tolist(),
            pdbx_align=cif['pdbx_align'].tolist()
        )
        ligands[id] = ligand_details
    
    end_time = time.time()
    logging.info(f"Finished processing all ligands. Total time: {end_time - start_time:.2f} seconds")
    return ligands

def save_ligands_to_pickle(ligands, output_filename="ligands.pkl"):
    """
    Saves the ligands dictionary to a pickle file.
    
    Parameters:
    ligands (dict): Dictionary containing ligands information.
    output_filename (str): The output pickle file name.
    """
    with open(output_filename, "wb") as outfile:
        pickle.dump(ligands, outfile)
    logging.info(f"Ligands saved to {output_filename}")

if __name__ == "__main__":
    sdfnames = glob.glob('/projects/ml/ligand_datasets/pdb/ligands/?/*_model.sdf')
    ligands = process_ligands(sdfnames)
    save_ligands_to_pickle(ligands)
