"""Pre-process the Chemical Component Dictionary (CCD) with OpenBabel to create a residue library for use in the CIF parser."""

# TODO: Save as a parquet rather than a pickle

from __future__ import annotations
import glob
import logging
import pickle
import time
import traceback

import networkx as nx
import numpy as np
import pandas as pd
from openbabel import openbabel
from tqdm import tqdm
from fire import Fire
from datetime import datetime

import cifutils.cifutils_legacy.cifutils_legacy as cifutils_legacy
import cifutils.cifutils_legacy.obutils as obutils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_leaving_atoms_from_graph(atom_name, graph):
    """
    Identifies leaving groups connected to a given atom in the molecular graph.

    Parameters:
    atom_name (str): The name of the atom.
    graph (networkx.Graph): The molecular graph.

    Returns:
    list: A list of atom names in the leaving group.
    """
    if graph.nodes[atom_name]["leaving"]:
        return []

    leaving_group = set()
    visited = set()

    def explore(neighbor, original_atom, visited):
        if neighbor in visited:
            return
        visited.add(neighbor)
        if graph.nodes[neighbor]["leaving"]:
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


def _coords_are_valid(obmol, cif):
    """
    Validates the coordinates of the molecule against the CIF data.

    Parameters:
    obmol (openbabel.OBMol): The molecule object.
    cif (dict): The CIF data.

    Returns:
    bool: True if validation is successful, False otherwise.
    """
    xyz = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in openbabel.OBMolAtomIter(obmol)])
    if obmol.NumAtoms() != cif["xyz"].shape[0]:
        return False
    return ((xyz - cif["xyz"])[~np.isnan(cif["xyz"])] < 1e-3).all()


def extract_residue_details(obmol, atom_id, leaving, pdbx_align, params):
    """
    Extracts detailed information about the residue from the OBMol object.

    Parameters:
    obmol (openbabel.OBMol): The molecule object.
    atom_id (List[str]): List of atom IDs.
    leaving (List[bool]): List of leaving flags for each atom.
    pdbx_align (List[int]): List of alignment indices.
    params (dict): Dictionary of parameters.

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
            "id": atom_name,
            "leaving_atom_flag": is_leaving,
            "leaving_group": [],
            "element": atom.GetAtomicNum(),
            "is_metal": atom.IsMetal(),
            "charge": charge,
            "hyb": atom.GetHyb(),
            "nhyd": nhyd,
            "hvydeg": atom.GetHvyDegree(),
            "align": align,
        }

    bonds = []
    for b in openbabel.OBMolBondIter(obmol):
        atom_a_name = atom_id[b.GetBeginAtom().GetIndex()]
        atom_b_name = atom_id[b.GetEndAtom().GetIndex()]
        bonds.append(
            {
                "atom_a_id": atom_a_name,
                "atom_b_id": atom_b_name,
                "is_aromatic": b.IsAromatic(),
                "in_ring": b.IsInRing(),
                "order": b.GetBondOrder(),
                "length": b.GetLength(),
            }
        )

    automorphisms = []
    chirals = []
    planars = []

    if params["include_automorphisms"]:
        automorphisms = obutils.FindAutomorphisms(
            obmol, heavy=True, maxmem=20 * (2**30), max_automorphs=params["max_automorphisms"]
        )
        mask = (automorphisms[:1] == automorphisms).all(dim=0)
        automorphisms = automorphisms[:, ~mask]

    if params["include_chirals"]:
        chirals = obutils.GetChirals(obmol, heavy=True)

    if params["include_planars"]:
        planars = obutils.GetPlanars(obmol, heavy=True)

    G = nx.Graph()
    G.add_nodes_from([(a["id"], {"leaving": a["leaving_atom_flag"]}) for a in residue_atoms.values()])
    G.add_edges_from([(bond["atom_a_id"], bond["atom_b_id"]) for bond in bonds])
    for k, v in residue_atoms.items():
        v["leaving_group"] = get_leaving_atoms_from_graph(k, G)

    anames = np.array(atom_id)
    return {
        "name": obmol.GetTitle(),
        "intra_residue_bonds": bonds,
        "automorphisms": anames[automorphisms].tolist() if params["include_automorphisms"] else [],
        "chirals": anames[chirals].tolist() if params["include_chirals"] else [],
        "planars": anames[planars].tolist() if params["include_planars"] else [],
        "atoms": residue_atoms,
    }


def process_single_ligand(sdfname, obConversion, params, debug: bool = True):
    """
    Processes a single SDF file to extract ligand details.

    Parameters:
    sdfname (str): The name of the SDF file.
    obConversion (openbabel.OBConversion): Open Babel conversion object.
    params (dict): Dictionary of parameters.

    Returns:
    tuple: A tuple containing the ligand ID and its details.
    """
    obmol = read_sdf_file(sdfname, obConversion)
    if "_model" in sdfname:
        cifname = sdfname.replace("_model.sdf", ".cif")
    elif "_ideal" in sdfname:
        cifname = sdfname.replace("_ideal.sdf", ".cif")
    else:
        raise ValueError(f"Invalid SDF file name: {sdfname}")

    try:
        cif = cifutils_legacy.ParsePDBLigand(cifname)
        if debug:
            logger.debug(f"Successfully parsed: {sdfname}")
    except Exception as e:
        logger.error(f"FAILED: {sdfname}, due to {type(e).__name__}: {str(e)}")
        if debug:
            # log stack trace
            logger.debug(f"Stack trace: \n{traceback.format_exc()}\n" + "=" * 100)
            raise e
        return None

    if "_ideal" not in sdfname and (not _coords_are_valid(obmol, cif)):
        logger.error(f"FAILED: {sdfname}")
        return None

    id = cifname.split("/")[-1][:-4]
    ligand_details = extract_residue_details(
        obmol=obmol,
        atom_id=cif["atom_id"].tolist(),
        leaving=cif["leaving"].tolist(),
        pdbx_align=cif["pdbx_align"].tolist(),
        params=params,
    )
    return id, ligand_details


def process_ligands(sdfnames, params, debug: bool = True):
    """
    Processes a list of SDF files to create a dictionary of ligands with their details.

    Parameters:
    sdfnames (list): List of SDF file names.
    params (dict): Dictionary of parameters.

    Returns:
    list: List containing tuples of ligand ID and ligand details.
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("sdf")
    ligands = []
    start_time = time.time()
    logger.info(f"Processing {len(sdfnames)} ligands with ...")

    for sdfname in tqdm(sdfnames, desc="Processing ligands"):
        result = process_single_ligand(sdfname, obConversion, params, debug=debug)
        if result:
            id, ligand_details = result
            ligands.append((id, ligand_details))

    end_time = time.time()
    logger.info(f"Finished processing all ligands. Total time: {end_time - start_time:.2f} seconds")
    return ligands


def save_ligands_to_pickles(ligands, params):
    """
    Saves the ligands information to a pickle file using HIGHEST_PROTOCOL.
    Saves two files:
    1. ligands_by_residue.pkl: Contains ligand details grouped by residue.
    2. ligands_by_atom.pkl: Contains ligand details grouped by atom.

    Parameters:
    ligands (list): List containing tuples of ligand ID and ligand details.
    params (dict): Dictionary of parameters.
    """
    ligand_records = []
    for id, details in ligands:
        record = {
            "id": id,
            "name": details["name"],
            "intra_residue_bonds": details["intra_residue_bonds"],
            "atoms": details["atoms"],
        }
        if params["include_automorphisms"]:
            record["automorphisms"] = details["automorphisms"]
        if params["include_chirals"]:
            record["chirals"] = details["chirals"]
        if params["include_planars"]:
            record["planars"] = details["planars"]

        ligand_records.append(record)

    by_residue_df = pd.DataFrame(ligand_records)
    with open(params["by_residues_filename"], "wb") as outfile:
        pickle.dump(by_residue_df, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Ligands grouped by residue saved to {params['by_residues_filename']}")

    # Function to process each row and extract necessary information
    def process_row(row):
        residue_name = row["id"]
        atoms_data = row["atoms"]
        chiral_centers = set()
        if params["include_chirals"]:
            chiral_centers = set(chiral[0] for chiral in row["chirals"])

        atom_records = []
        for atom_id, atom_info in atoms_data.items():
            atom_record = {
                "residue_name": residue_name,
                "atom_id": atom_id,
                "leaving_atom_flag": atom_info["leaving_atom_flag"],
                "leaving_group": atom_info["leaving_group"],
                "is_metal": atom_info["is_metal"],
                "charge": atom_info["charge"],
                "hybridization": atom_info["hyb"],
                "n_hydrogens": atom_info["nhyd"],
                "hvy_degree": atom_info["hvydeg"],
                "pdbx_align": atom_info["align"],
            }
            if params["include_chirals"]:
                atom_record["is_chiral_center"] = atom_id in chiral_centers
            atom_records.append(atom_record)
        return atom_records

    # Apply the function to each row and concatenate the results into a DataFrame
    by_atom_df = pd.concat([pd.DataFrame(process_row(row)) for _, row in by_residue_df.iterrows()], ignore_index=True)
    with open(params["by_atoms_filename"], "wb") as outfile:
        pickle.dump(by_atom_df, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Ligands grouped by atoms saved to {params['by_atoms_filename']}")


def main(
    ligand_dir: str = "/projects/ml/RF2_allatom/cifutils_biotite/ccd_ligands_2024_05_31/ccd",
    out_dir: str = ".",
    max_automorphisms: int = 2000,
    include_automorphisms: bool = True,
    include_planars: bool = True,
    include_chirals: bool = True,
    use_ideal: bool = True,
    debug: bool = False,
):
    """
    Pre-process the Chemical Component Dictionary (CCD) with OpenBabel.

    Args:
    ligand_dir (str): Directory containing the ligand SDF files.
    out_dir (str): Directory to save the pickle files.
    max_automorphisms (int): Maximum number of automorphisms to consider.
    include_automorphisms (bool): Include automorphisms in the output.
    include_planars (bool): Include planars in the output.
    include_chirals (bool): Include chirals in the output.
    use_ideal (bool): Use ideal coordinates. If False, use model coordinates.
        c.f. https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/small-molecule-ligands
    """
    # Get datetime date stamp in format YYYY_MM_DD
    date_stamp = datetime.now().strftime("%Y_%m_%d")
    if use_ideal:
        # Ideal coordinates are calculated by the PDB by software based on the known covalent geometry
        # ... often (always?) RDKit
        sdf_origin = "ideal"
    else:
        # Model coordinates are taken from the first entry in which the component was observed, and as
        # such can represent the conformation that the ligand adopts upon binding to a macromolecule
        sdf_origin = "model"

    logger.info(f"Processing `{sdf_origin}` ligands.")

    by_residues_filename = f"{out_dir}/ligands_by_residue_{sdf_origin}_v{date_stamp}.pkl"
    by_atoms_filename = f"{out_dir}/ligands_by_atom_{sdf_origin}_v{date_stamp}.pkl"

    # Set log level to DEBUG if debug is True
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    params = {
        "max_automorphisms": max_automorphisms,
        "include_automorphisms": include_automorphisms,
        "include_chirals": include_chirals,
        "include_planars": include_planars,
        "by_residues_filename": by_residues_filename,
        "by_atoms_filename": by_atoms_filename,
    }

    logger.info(f"Processing {ligand_dir}. Parameters: \n{params}")

    sdfnames = glob.glob(f"{ligand_dir}/?/*/*_{sdf_origin}.sdf")
    logger.info(f"Found {len(sdfnames)} ligand SDF files.")

    ligands = process_ligands(sdfnames, params, debug=debug)
    save_ligands_to_pickles(ligands, params)
    logger.info("################## Complete. ##################")


if __name__ == "__main__":
    Fire(main)
