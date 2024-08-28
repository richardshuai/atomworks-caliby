"""Pre-process the Chemical Component Dictionary (CCD) with OpenBabel to create a residue library for use in the CIF parser."""

# TODO: Update to reflect new cifutils structure, where we save individual residue files to a directory, indexed by three-letter residue code

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
import biotite.structure.io.pdbx as pdbx
from os import PathLike

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_ccd_ligand_from_cif(cif_path: PathLike) -> dict:
    """Parse a single molecule from the PDB-Ligands set using Biotite"""

    # ... read the CIF file
    block = pdbx.CIFFile.read(cif_path).block

    # ...get the the "chem_comp_atom" category
    chem_comp_atom = block["chem_comp_atom"]

    # ...retrieve the "atom_id" column from the "chem_comp_atom" category (i.e., standard atom name)
    atom_id = chem_comp_atom["atom_id"].as_array()

    # ...retrieve the "type_symbol" column from the "chem_comp_atom" category (i.e., element)
    symbol = chem_comp_atom["type_symbol"].as_array()

    # ...parse leaving flags (whether an atom is a leaving group)
    leaving = chem_comp_atom["pdbx_leaving_atom_flag"].as_array()
    leaving = np.array([True if flag == "Y" else False for flag in leaving], dtype=bool)

    # ...parse xyz coordinates
    x = chem_comp_atom["model_Cartn_x"].as_array(float)
    y = chem_comp_atom["model_Cartn_y"].as_array(float)
    z = chem_comp_atom["model_Cartn_z"].as_array(float)
    xyz = np.stack((x, y, z), axis=-1)

    # ...replace missing with NaN
    xyz[xyz == "."] = np.nan

    out = {"atom_id": atom_id, "leaving": leaving, "symbol": symbol, "xyz": xyz}

    return out


def get_leaving_atoms_from_graph(atom_name: str, graph: nx.Graph) -> list:
    """
    Identifies leaving groups connected to a given atom in the molecular graph.

    Args:
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


def read_sdf_file(sdfname: str, obConversion: openbabel.OBConversion) -> openbabel.OBMol:
    """
    Reads an SDF file and returns an OBMol object.

    Args:
        sdfname (str): The name of the SDF file.
        obConversion (openbabel.OBConversion): Open Babel conversion object.

    Returns:
        openbabel.OBMol: The molecule read from the SDF file.
    """
    obmol = openbabel.OBMol()
    obConversion.ReadFile(obmol, sdfname)
    return obmol


def _coords_are_valid(obmol: openbabel.OBMol, cif: dict) -> bool:
    """
    Validates the coordinates of the molecule against the CIF data.

    Args:
        obmol (openbabel.OBMol): The molecule object.
        cif (dict): The CIF data.

    Returns:
        bool: True if validation is successful, False otherwise.
    """
    xyz = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in openbabel.OBMolAtomIter(obmol)])
    if obmol.NumAtoms() != cif["xyz"].shape[0]:
        return False
    return ((xyz - cif["xyz"])[~np.isnan(cif["xyz"])] < 1e-3).all()


def extract_residue_details(
    obmol: openbabel.OBMol,
    atom_id: list[str],
    leaving: list[bool],
) -> dict:
    """
    Extracts detailed information about the residue from the OBMol object.

    Args:
        obmol (openbabel.OBMol): The molecule object.
        atom_id (List[str]): List of atom IDs.
        leaving (List[bool]): List of leaving flags for each atom.

    Returns:
        dict: Dictionary containing residue details.
    """
    obmol_ph = openbabel.OBMol(obmol)
    obmol_ph.CorrectForPH()
    obmol_ph.DeleteHydrogens()
    ha_iter = openbabel.OBMolAtomIter(obmol_ph)

    residue_atoms = {}
    for atom_name, is_leaving, atom in zip(atom_id, leaving, openbabel.OBMolAtomIter(obmol)):
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

    G = nx.Graph()
    G.add_nodes_from([(a["id"], {"leaving": a["leaving_atom_flag"]}) for a in residue_atoms.values()])
    G.add_edges_from([(bond["atom_a_id"], bond["atom_b_id"]) for bond in bonds])
    for k, v in residue_atoms.items():
        v["leaving_group"] = get_leaving_atoms_from_graph(k, G)

    return {
        "name": obmol.GetTitle(),
        "intra_residue_bonds": bonds,
        "atoms": residue_atoms,
    }


def process_single_ligand(sdfname: str, obConversion: openbabel.OBConversion, debug: bool = True):
    """
    Processes a single SDF file to extract ligand details.

    Args:
        sdfname (str): The name of the SDF file.
        obConversion (openbabel.OBConversion): Open Babel conversion object.

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
        cif = parse_ccd_ligand_from_cif(cifname)
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
    )
    return id, ligand_details


def process_ligands(sdfnames: list, debug: bool = True):
    """
    Processes a list of SDF files to create a dictionary of ligands with their details.

    Args:
        sdfnames (list): List of SDF file names.

    Returns:
        list: List containing tuples of ligand ID and ligand details.
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("sdf")
    ligands = []
    start_time = time.time()
    logger.info(f"Processing {len(sdfnames)} ligands with ...")

    for sdfname in tqdm(sdfnames, desc="Processing ligands"):
        result = process_single_ligand(sdfname, obConversion, debug=debug)
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

    Args:
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
        ligand_records.append(record)

    by_residue_df = pd.DataFrame(ligand_records)
    with open(params["by_residues_filename"], "wb") as outfile:
        pickle.dump(by_residue_df, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Ligands grouped by residue saved to {params['by_residues_filename']}")


def main(
    ligand_dir: str = "/projects/ml/RF2_allatom/cifutils_biotite/ccd_ligands_2024_05_31/ccd",
    out_dir: str = ".",
    use_ideal: bool = True,
    debug: bool = False,
):
    """
    Pre-process the Chemical Component Dictionary (CCD) with OpenBabel.

    Args:
        ligand_dir (str): Directory containing the ligand SDF files.
        out_dir (str): Directory to save the pickle files.
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
        "by_residues_filename": by_residues_filename,
        "by_atoms_filename": by_atoms_filename,
    }

    logger.info(f"Processing {ligand_dir}. Parameters: \n{params}")

    sdfnames = glob.glob(f"{ligand_dir}/?/*/*_{sdf_origin}.sdf")
    logger.info(f"Found {len(sdfnames)} ligand SDF files.")

    ligands = process_ligands(sdfnames, debug=debug)
    save_ligands_to_pickles(ligands, params)
    logger.info("################## Complete. ##################")


if __name__ == "__main__":
    Fire(main)
