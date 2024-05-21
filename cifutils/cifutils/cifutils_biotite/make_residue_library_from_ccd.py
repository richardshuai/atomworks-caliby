"""Pre-process the Chemical Component Dictionary (CCD) with OpenBabel to create a residue library for use in the CIF parser."""

import glob
import logging
import numpy as np
import pandas as pd
import pickle
import time
from openbabel import openbabel
import cifutils.cifutils_legacy.cifutils_legacy as cifutils_legacy
import networkx as nx
import cifutils.cifutils_legacy.obutils as obutils
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

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
    visited = set()

    def explore(neighbor, original_atom, visited):
        if neighbor in visited:
            return
        visited.add(neighbor)
        if graph.nodes[neighbor]['leaving']:
            leaving_group.add(neighbor)
            subgraph = graph.subgraph(set(graph.nodes) - {neighbor})
            connected_components = nx.connected_components(subgraph)
            for component in connected_components:
                if original_atom not in component:
                    leaving_group.update(component)

    for neighbor in graph.neighbors(atom_name):
        if neighbor not in visited:
            explore(neighbor, atom_name, visited)

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
    return ((xyz - cif['xyz'])[~np.isnan(cif['xyz'])] < 1e-3).all()

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
    ha_iter = openbabel.OBMolAtomIter(obmol_ph)
    
    residue_atoms = {}
    for atom_name, is_leaving, align, atom in zip(atom_id, leaving, pdbx_align, openbabel.OBMolAtomIter(obmol)):
        charge = atom.GetFormalCharge()
        nhyd = atom.ExplicitHydrogenCount()
        if atom.GetAtomicNum() > 1:
            ha = next(ha_iter)
            charge = ha.GetFormalCharge()
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
    
    automorphisms = obutils.FindAutomorphisms(obmol, heavy=True, maxmem=20*(2**30), max_automorphs=10**4)
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

def process_single_ligand(sdfname, obConversion):
    """
    Processes a single SDF file to extract ligand details.
    
    Parameters:
    sdfname (str): The name of the SDF file.
    obConversion (openbabel.OBConversion): Open Babel conversion object.
    
    Returns:
    tuple: A tuple containing the ligand ID and its details.
    """
    obmol = read_sdf_file(sdfname, obConversion)
    cifname = sdfname.replace('_model.sdf', '.cif')

    try:
        cif = cifutils_legacy.ParsePDBLigand(cifname)
    except Exception as e:
        logging.error(f"FAILED: {sdfname}, due to error: {str(e)}")
        return None

    if not validate_coordinates(obmol, cif):
        logging.error(f"FAILED: {sdfname}")
        return None

    id = cifname.split('/')[-1][:-4]
    ligand_details = extract_residue_details(
        obmol=obmol,
        atom_id=cif['atom_id'].tolist(),
        leaving=cif['leaving'].tolist(),
        pdbx_align=cif['pdbx_align'].tolist()
    )
    return id, ligand_details

def process_ligands(sdfnames, num_threads=5, timeout=300):
    """
    Processes a list of SDF files to create a dictionary of ligands with their details.
    
    Parameters:
    sdfnames (list): List of SDF file names.
    num_threads (int): Number of threads to use.
    timeout (int): Timeout in seconds for processing each ligand.
    
    Returns:
    list: List containing tuples of ligand ID and ligand details.
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("sdf")
    ligands = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_single_ligand, sdfname, obConversion): sdfname for sdfname in sdfnames}
        
        for future in tqdm(as_completed(futures), total=len(sdfnames), desc="Processing ligands"):
            sdfname = futures[future]
            try:
                result = future.result(timeout=timeout)
                if result:
                    id, ligand_details = result
                    ligands.append((id, ligand_details))
            except TimeoutError:
                logging.error(f"FAILED: {sdfname}, due to timeout (exceeded {timeout} seconds)")
            except Exception as e:
                logging.error(f"FAILED: {sdfname}, due to error: {str(e)}")
    
    end_time = time.time()
    logging.info(f"Finished processing all ligands. Total time: {end_time - start_time:.2f} seconds")
    return ligands

def save_ligands_to_pickles(ligands, by_residues_filename="ligands_by_residue.pkl", by_atoms_filename="ligands_by_atom.pkl"):
    """
    Saves the ligands information to a pickle file using HIGHEST_PROTOCOL.
    Saves two files:
    1. ligands_by_residue.pkl: Contains ligand details grouped by residue.
    2. ligands_by_atom.pkl: Contains ligand details grouped by atom.
    
    Parameters:
    ligands (list): List containing tuples of ligand ID and ligand details.
    by_residues_filename (str): The name of the file to save ligands grouped by residue.
    by_atoms_filename (str): The name of the file to save ligands grouped by atom.
    """
    ligand_records = []
    for id, details in ligands:
        record = {
            'id': id,
            'name': details['name'],
            'intra_residue_bonds': details['intra_residue_bonds'],
            'automorphisms': details['automorphisms'],
            'chirals': details['chirals'],
            'planars': details['planars'],
            'atoms': details['atoms']
        }
        ligand_records.append(record)
    
    by_residue_df = pd.DataFrame(ligand_records)
    with open(by_residues_filename, 'wb') as outfile:
        pickle.dump(by_residue_df, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Ligands grouped by residue saved to {by_residues_filename}")

    # Function to process each row and extract necessary information
    def process_row(row):
        residue_name = row['id']
        atoms_data = row['atoms']
        chiral_centers = set(chiral[0] for chiral in row['chirals'])

        atom_records = []
        for atom_id, atom_info in atoms_data.items():
            atom_records.append({
                'residue_name': residue_name,
                'atom_id': atom_id,
                'is_chiral_center': atom_id in chiral_centers,
                'leaving_atom_flag': atom_info['leaving_atom_flag'],
                'leaving_group': atom_info['leaving_group'],
                'is_metal': atom_info['is_metal'],
                'charge': atom_info['charge'],
                'hybridization': atom_info['hyb'],
                'n_hydrogens': atom_info['nhyd'],
                'hvy_degree': atom_info['hvydeg'],
                'pdbx_align': atom_info['align'],
            })
        return atom_records

    # Apply the function to each row and concatenate the results into a DataFrame
    by_atom_df = pd.concat([pd.DataFrame(process_row(row)) for _, row in by_residue_df.iterrows()], ignore_index=True)
    with open(by_atoms_filename, 'wb') as outfile:
        pickle.dump(by_atom_df, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Ligands grouped by atoms saved to {by_atoms_filename}")

if __name__ == "__main__":
    sdfnames = glob.glob('/projects/ml/ligand_datasets/pdb/ligands/?/*_model.sdf')
    ligands = process_ligands(sdfnames, num_threads=15)
    save_ligands_to_pickles(ligands)
    logging.info("################## Complete. ##################")