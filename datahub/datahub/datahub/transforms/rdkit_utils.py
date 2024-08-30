import logging
from functools import cache, wraps
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdchem import Mol
from rf2aa.data_new.utils import default

from rf2aa.data_new.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_does_not_contain_keys,
    check_is_instance,
    check_nonzero_length,
)
from rf2aa.data_new.transforms.base import Transform

logger = logging.getLogger(__name__)

_RDKIT_BOND_TYPE_TO_BIOTITE = {
    Chem.BondType.UNSPECIFIED: struc.bonds.BondType.ANY,
    Chem.BondType.SINGLE: struc.bonds.BondType.SINGLE,
    Chem.BondType.DOUBLE: struc.bonds.BondType.DOUBLE,
    Chem.BondType.TRIPLE: struc.bonds.BondType.TRIPLE,
    Chem.BondType.QUADRUPLE: struc.bonds.BondType.QUADRUPLE,
    Chem.BondType.AROMATIC: struc.bonds.BondType.AROMATIC_SINGLE,
    Chem.BondType.ONEANDAHALF: struc.bonds.BondType.AROMATIC_SINGLE,
    Chem.BondType.TWOANDAHALF: struc.bonds.BondType.AROMATIC_DOUBLE,
    Chem.BondType.THREEANDAHALF: struc.bonds.BondType.AROMATIC_TRIPLE,
}
"""
Mapping from RDKit bond types to Biotite bond types.
Unspecified bonds are mapped to `ANY` bond type.
"""

_BIOTITE_BOND_TYPE_TO_RDKIT = {
    struc.bonds.BondType.ANY: Chem.BondType.UNSPECIFIED,
    struc.bonds.BondType.SINGLE: Chem.BondType.SINGLE,
    struc.bonds.BondType.DOUBLE: Chem.BondType.DOUBLE,
    struc.bonds.BondType.TRIPLE: Chem.BondType.TRIPLE,
    struc.bonds.BondType.QUADRUPLE: Chem.BondType.QUADRUPLE,
    struc.bonds.BondType.AROMATIC_SINGLE: Chem.BondType.AROMATIC,
    struc.bonds.BondType.AROMATIC_DOUBLE: Chem.BondType.AROMATIC,
    struc.bonds.BondType.AROMATIC_TRIPLE: Chem.BondType.AROMATIC,
}
"""Mapping from Biotite bond types to RDKit bond types."""

_BIOTITE_DEFAULT_ANNOTATIONS = ["chain_id", "res_id", "res_name", "atom_name"]


def _is_bond_aromatic(bond_type: struc.bonds.BondType) -> bool:
    return bond_type in {
        struc.bonds.BondType.AROMATIC_SINGLE,
        struc.bonds.BondType.AROMATIC_DOUBLE,
        struc.bonds.BondType.AROMATIC_TRIPLE,
    }


@cache
def _element_to_atomic_number(element: str | int) -> int:
    """
    Convert an element string or atomic number to an atomic number.

    Args:
        - element (str | int): The element symbol (e.g., 'C') or atomic number.

    Returns:
        - int: The atomic number of the element.

    Examples:
        >>> _element_to_atomic_number("C")
        6
        >>> _element_to_atomic_number("8")
        8
        >>> _element_to_atomic_number(1)
        1
    """
    try:
        return int(element)
    except ValueError:
        return Chem.GetPeriodicTable().GetAtomicNumber(element)


def _preserve_annotations(func: callable) -> callable:
    """
    Decorator to copy annotations from an RDKit molecule to a new molecule.

    This decorator ensures that any custom annotations stored in the `_annotations`
    attribute of an RDKit molecule are preserved when the molecule is modified or
    converted.

    Args:
        - func (callable): The function to be decorated. Must accept an RDKit molecule as
          positional argument or keyword argument with keyword 'mol'.

    Returns:
        callable: The decorated function that preserves annotations.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        # Find the first RDKit molecule in the arguments or keyword arguments
        if "mol" in kwargs:
            mol = kwargs["mol"]
        else:
            mol = next(arg for arg in args if isinstance(arg, Mol))

        if hasattr(mol, "_annotations"):
            annotations = mol._annotations
            new_mol = func(*args, **kwargs)
            new_mol._annotations = annotations
        else:
            new_mol = func(*args, **kwargs)

        return new_mol

    return wrapped


@_preserve_annotations
def generate_conformers(
    mol: Mol,
    seed: int | None = None,
    n_conformers: int = 1,
    infer_hydrogens: bool = True,
    optimize_conformers: bool = True,
    **uff_optimize_kwargs: dict,
) -> Mol:
    """
    Generate conformations for the given molecule.

    Args:
        - mol (rdkit.Chem.Mol): The RDKit molecule to generate conformations for.
        - seed (int | None): Random seed for reproducibility. If None, a random seed is used.
        - n_conformers (int): Number of conformations to generate.
        - infer_hydrogens (bool): Whether to add hydrogens if they are not present. This is
            recommended, since having hydrogens improves the accuracy of the conformer
            generation.
        - optimize_conformers (bool): Whether to optimize the generated conformers using UFF.
        - **uff_optimize_kwargs (dict): Additional keyword arguments for UFF optimization:
            - maxIters (int): Maximum number of iterations (default 200).
            - vdwThresh (float): Used to exclude long-range van der Waals interactions
              (default 10.0).
            - confId (int): ID of the conformer to optimize (default -1, meaning all).
            - ignoreInterfragInteractions (bool): If True, nonbonded terms between
              fragments will not be added to the forcefield (default True).

    Returns:
        rdkit.Chem.Mol: The molecule with generated conformations.

    Note:
        - If `infer_hydrogens` is False, make sure the input molecule already has
          the desired hydrogen representation.
        - Adding hydrogens (infer_hydrogens=True) is generally recommended for more
          accurate conformer generation, as it provides a more complete representation
          of the molecule's structure.
        - Optimizing conformers (optimize_conformers=True) is recommended for obtaining
          more realistic and lower-energy conformations. However, it may increase
          computation time.
        - The ETKDG method is used for conformer generation, which incorporates
          torsion angle preferences and basic knowledge terms for improved accuracy.
        - For macrocycles or complex ring systems, you may need to increase the number
          of conformers generated to ensure good sampling of the conformational space
          (if a representative ensemble of conformers is what you are after).

    Best Practices:
        1. Always add hydrogens before generating conformers unless you have a specific
           reason not to (e.g., you're working with a protein structure where hydrogens
           are already correctly placed).
        2. Use a non-zero seed for reproducibility in research or production environments.
        3. Generate multiple conformers (e.g., 50-100) for flexible molecules to sample
           the conformational space more thoroughly.
        4. Optimize conformers using UFF or MMFF94 for more realistic geometries, especially
           if the conformers will be used for further calculations or analysis.
        5. For very large or complex molecules, you may need to adjust parameters such as
           maxIterations or use more advanced sampling techniques.

    References:
        1. RDKit Cookbook: https://www.rdkit.org/docs/Cookbook.html
        2. Riniker and Landrum, "Better Informed Distance Geometry: Using What We Know To
           Improve Conformation Generation", JCIM, 2015.

    """

    # Infer hydrogens if needed, i.e. if they are not present in the molecule. Normally this should
    # always be set to `True` to generate realistic conformations. The only reason to set this to `False`
    # is if you are already providing hydrogens in the molecule.
    if infer_hydrogens:
        mol = Chem.AddHs(mol)

    seed = -1 if seed is None else seed
    try:
        Chem.EmbedMultipleConfs(mol, numConfs=n_conformers, enforceChirality=True, randomSeed=seed)
    except Exception as e:
        Chem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=seed, enforceChirality=False)

    if optimize_conformers:
        success = Chem.UFFOptimizeMoleculeConfs(mol, **uff_optimize_kwargs)
        if not success:
            logger.warning("Conformer optimization did not converge.")

    return mol


def get_chiral_centers(mol: Mol) -> list[int]:
    """
    Identify and return the tetrahedral chiral centers in an RDKit molecule.

    This function finds all tetrahedral chiral centers in the given molecule
    and returns their information, including the chiral center atom index and
    the indices of the atoms bonded to it.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule to analyze.

    Returns:
        - list[dict]: A list of dictionaries, where each dictionary contains:
            - "chiral_center_idx" (int): The index of the chiral center atom.
            - "bonded_explicit_atom_idxs" (list[int]): A list of indices of the atoms
              bonded to the chiral center.
            - "chirality" (str): The chirality of the center ('R' or 'S').

    Note:
        This function will generate a 3D conformation if one is not present, as
        chirality assignment requires 3D coordinates in RDKit to break the conditional
        tie between multiple possible chirality centers.
    """
    # Infer 3D coordinates if not present
    if mol.GetNumConformers() == 0:
        Chem.EmbedMolecule(mol, enforceChirality=True, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
        Chem.UFFOptimizeMolecule(mol)

    # Assign chiral tags based on the 3D structure
    Chem.AssignAtomChiralTagsFromStructure(mol)

    # Identify chiral centers
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    # Filter chiral centers with tetrahedral geometry
    tetrahedral_chiral_centers = []
    for center in chiral_centers:
        idx, chirality = center
        atom = mol.GetAtomWithIdx(idx)
        if (
            atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
            or atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
        ):
            tetrahedral_chiral_centers.append(
                {
                    "chiral_center_idx": idx,
                    "bonded_explicit_atom_idxs": [bond.GetOtherAtomIdx(idx) for bond in atom.GetBonds()],
                    "chirality": chirality,
                }
            )

    return tetrahedral_chiral_centers


def smiles_to_rdkit(smile: str) -> Mol:
    """
    Generate an RDKit molecule from a SMILES string.

    This function creates an RDKit molecule object from a SMILES string,
    performs sanitization, and perceives aromaticity.

    Args:
        - smile (str): The SMILES string representing the molecule.

    Returns:
        - rdkit.Chem.Mol: The RDKit molecule generated from the SMILES string.

    Note:
        The returned molecule is sanitized and has aromaticity perceived.
    """
    mol = Chem.MolFromSmiles(smile)
    # ... perform sanitization to perceive aromaticity
    Chem.SanitizeMol(mol)
    return mol


def atom_array_from_rdkit(
    mol: Mol,
    set_coord_if_available: bool = True,
    conformer_id: int | None = None,
    elements_as_int: bool = True,
    remove_hydrogens: bool = True,
    remove_inferred_atoms: bool = False,
) -> AtomArray:
    """
    Convert an RDKit molecule to a Biotite AtomArray object.

    This function takes an RDKit Mol object and converts it into a Biotite AtomArray object,
    matching and preserving the (optional) atom-level annotations in `mol._annotations` and
    bond information.

    Args:
        - mol (rdkit.Chem.Mol): The RDKit molecule to convert.
        - set_coord_if_available (bool): Whether to set the coordinates from the RDKit molecule if
            a conformer is available.
        - conformer_id (int | None): The conformer ID to use for coordinates. If None, the first
          conformer is used.
        - elements_as_int (bool): Whether to store the elements as integers (atomic numbers) or strings.
        - remove_hydrogens (bool): Whether to remove any explicit hydrogen atoms.
        - remove_inferred_atoms (bool): Whether to remove any atoms that do not carry the `rdkit_atom_id` annotation.

    Returns:
        - biotite.structure.AtomArray: A Biotite AtomArray object containing the atoms and bonds from the input Mol object.

    Example:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> atom_array = atom_array_from_rdkit(mol)
        >>> print(atom_array)
        AtomArray([Atom(element='6', coord=array([0.0, 0.0, 0.0]), ...)])
    """
    n_atoms = mol.GetNumAtoms()

    # Get coordinates (if available)
    coords = np.full((mol.GetNumAtoms(), 3), np.nan)
    n_conformers = mol.GetNumConformers()
    if set_coord_if_available and n_conformers > 0:
        conformer_id = conformer_id if conformer_id is not None else -1
        if conformer_id >= n_conformers:
            raise ValueError(f"Conformer ID {conformer_id} out of range for molecule with {n_conformers} conformers")
        coords = mol.GetConformer(conformer_id).GetPositions()

    # Set atoms
    atoms = []
    for idx, rdatom in enumerate(mol.GetAtoms()):
        atoms.append(
            struc.Atom(
                element=rdatom.GetAtomicNum() if elements_as_int else rdatom.GetSymbol(),
                coord=coords[idx],
                charge=rdatom.GetFormalCharge(),
                hyb=str(rdatom.GetHybridization()),
                nhyd=rdatom.GetTotalNumHs(),
                hvydeg=rdatom.GetDegree() - rdatom.GetTotalNumHs(),
                rdkit_atom_id=rdatom.GetIntProp("rdkit_atom_id") if rdatom.HasProp("rdkit_atom_id") else -1,
            )
        )
    atom_array = struc.array(atoms)

    # Set bonds
    bond_list = []
    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        bond_type = _RDKIT_BOND_TYPE_TO_BIOTITE[bond.GetBondType()]
        bond_list.append((begin_atom_idx, end_atom_idx, bond_type))
    atom_array.bonds = struc.bonds.BondList(n_atoms, np.array(bond_list))

    # Set extra annotations
    annotations = mol._annotations if hasattr(mol, "_annotations") else {}
    if len(annotations) > 0:
        # Create mapping of array idx <> annotation idx via the openbabel atom id:
        _rdkit_id_to_annotation_idx = {
            rdkit_atom_id: idx for idx, rdkit_atom_id in enumerate(annotations["rdkit_atom_id"])
        }

        array_idx_to_annotation_idx = []
        for idx, atom in enumerate(atoms):
            if atom.rdkit_atom_id in _rdkit_id_to_annotation_idx:
                array_idx_to_annotation_idx.append((idx, _rdkit_id_to_annotation_idx[atom.rdkit_atom_id]))
        array_idx_to_annotation_idx = np.array(array_idx_to_annotation_idx)

        for key, val in annotations.items():
            if key in ["coord", "charge"]:
                logger.warning(f"Found built-in annotation: {key} as custom annotation. Skipping.")
                continue

            if np.issubdtype(val.dtype, np.integer):
                default = -1
            elif np.issubdtype(val.dtype, np.floating):
                default = np.nan
            elif np.issubdtype(val.dtype, np.str_):
                default = ""
            else:
                logger.warning(f"Unsupported annotation dtype: {val.dtype}. Skipping.")
                continue

            vals = np.full(len(atom_array), default, dtype=val.dtype)
            vals[array_idx_to_annotation_idx[:, 0]] = annotations[key][array_idx_to_annotation_idx[:, 1]]
            atom_array.set_annotation(key, vals)

    if remove_hydrogens:
        atom_array = atom_array[~np.isin(atom_array.element, ["H", "D", "T", "1"])]

    if remove_inferred_atoms:
        atom_array = atom_array[atom_array.rdkit_atom_id != -1]

    return atom_array


def atom_array_to_rdkit(
    atom_array: AtomArray,
    set_coord: bool = False,
    infer_hydrogens: bool = True,
    annotations_to_keep: list[str] = _BIOTITE_DEFAULT_ANNOTATIONS,
    strict_sanitization: bool = True,
) -> Mol:
    """
    Generate an RDKit molecule from a Biotite AtomArray object.

    Args:
        - atom_array (biotite.structure.AtomArray): The Biotite AtomArray to convert.
        - set_coord (bool): Whether to set atomic coordinates in the RDKit molecule.
        - infer_hydrogens (bool): Whether to infer hydrogens in the RDKit molecule.
        - annotations_to_keep (list[str]): List of atom annotations to preserve from the AtomArray.

    Returns:
        - rdkit.Chem.Mol: RDKit Molecule generated from the AtomArray.

    Note:
        Aromaticity, hybridization states, and other properties are automatically
        perceived by RDKit's SanitizeMol during the conversion process.
    """
    # Initialize the RDKit molecule
    mol = Chem.RWMol()

    # Set atoms
    # ... we use an internal `rdkit_atom_id` property to keep track of atoms that were originally
    #     in the AtomArray from atoms that might have been added by RDKit implicitly later
    #     (implicit hydrogens)
    rdkit_atom_ids = []

    is_hydrogen = np.isin(atom_array.element, ["H", "D", "T", "1", 1])
    if np.any(is_hydrogen) and infer_hydrogens:
        logger.info(f"Found {np.sum(is_hydrogen)} hydrogen atoms in the AtomArray. Removing them.")
    atom_array = atom_array[~is_hydrogen] if infer_hydrogens else atom_array

    for atom_id, atom in enumerate(atom_array):
        atomic_number = _element_to_atomic_number(atom.element)

        rdatom = Chem.Atom(atomic_number)
        if hasattr(atom, "charge"):
            # ... set formal charge if available (otherwise RDKit will assume it is 0
            #  and assign a charge state in SanitizeMol if it is required to satisfy
            #  valence constraints)
            rdatom.SetFormalCharge(int(atom.charge))

        rdatom.SetIntProp("rdkit_atom_id", atom_id)
        rdatom.SetProp("atom_name", atom.atom_name)
        rdkit_atom_ids.append(atom_id)
        mol.AddAtom(rdatom)

    # Set bonds
    for bond in atom_array.bonds.as_array():
        atom1, atom2, bond_type = list(map(int, bond))
        if bond_type == struc.bonds.BondType.ANY:
            # ... warn if underspecified bonds are encountered
            logger.warning("Encountered BondType.ANY. Interpreting as single bond.")

        mol.AddBond(atom1, atom2, order=_BIOTITE_BOND_TYPE_TO_RDKIT[bond_type])

    # Set coordinates
    if set_coord:
        # ... add conformer (at id 0)
        conf_id = mol.AddConformer(Chem.Conformer(len(atom_array)), assignId=True)
        # ... fill in coordinates
        for atom_id, coord in enumerate(atom_array.coord):
            mol.GetConformer(conf_id).SetAtomPosition(atom_id, coord.tolist())

    # Clean up the molecule and infer various properties
    if strict_sanitization:
        sanitize_result = Chem.SanitizeMol(mol)
    else:
        sanitize_ops = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE
        sanitize_ops = 268435447
        sanitize_result = Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops, catchErrors=True)
    
    if sanitize_result != Chem.SanitizeFlags.SANITIZE_NONE:
        logger.warning(f"Sanitization failed with flags: {sanitize_result}")

    # Turn into a non-editable molecule
    mol = mol.GetMol()

    # Attach custom atom-level annotations from the atom array
    mol._annotations = {"rdkit_atom_id": np.array(rdkit_atom_ids)}
    for annotation in annotations_to_keep:
        if annotation in atom_array.get_annotation_categories():
            mol._annotations[annotation] = atom_array._annot[annotation]

    return mol


def sample_rdkit_conformer_for_atom_array(atom_array: AtomArray, seed: int | None = None) -> Mol:
    """
    Sample a conformer for a Biotite AtomArray using RDKit.

    Args:
        - atom_array (AtomArray): The Biotite AtomArray to sample a conformer for.
        - seed (int | None): Random seed for reproducibility.

    Returns:
        - AtomArray: The AtomArray with updated coordinates from the sampled conformer.

    Note:
        This function preserves the original atom order and properties of the input AtomArray.
    """
    atom_array = atom_array.copy()
    mol = atom_array_to_rdkit(atom_array, infer_hydrogens=False)
    mol = generate_conformers(mol, seed=seed, n_conformers=1)
    new_atom_array = atom_array_from_rdkit(
        mol, set_coord_if_available=True, elements_as_int=True, remove_inferred_atoms=True, remove_hydrogens=False
    )
    assert new_atom_array.array_length() == atom_array.array_length()
    assert np.all(new_atom_array.atom_name == atom_array.atom_name)
    assert np.all(new_atom_array.res_id == atom_array.res_id)
    assert np.all(new_atom_array.res_name == atom_array.res_name)
    atom_array.coord = new_atom_array.coord

    return atom_array


# -------------------------------------------------------------------------------------------------
# ---------------------  RDKit related transforms  ------------------------------------------------
# -------------------------------------------------------------------------------------------------


class AddRDKitMoleculesForAtomizedMolecules(Transform):
    """
    Add RDKit molecules for atomized molecules in the atom array.

    This transform converts atomized molecules in the atom array to RDKit Mol objects and stores them in the
    `data` dictionary under the "rdkit" key. Each molecule is identified by its `pn_unit_iid`.

    Note:
        This transform requires the `AtomizeResidues` transform to be applied previously.

    Args:
        data (dict[str, Any]): A dictionary containing the input data, including the atom array.

    Returns:
        dict[str, Any]: The updated `data` dictionary with the added RDKit molecules under the
            `"rdkit"` key.

    Example:
        >>> data = {
        >>>     "atom_array": AtomArray(...),  # Your atom array here
        >>> }
        >>> transform = AddRDKitMoleculesForAtomizedMolecules()
        >>> data = transform(data)
        >>> print(data["rdkit"])
        {
            'A_1': <rdkit.Chem.rdchem.Mol object at 0x...>,
            'B_1': <rdkit.Chem.rdchem.Mol object at 0x...>,
            ...
        }
    """

    requires_previous_transforms = ["AtomizeResidues"]
    incompatible_previous_transforms = ["CropContiguousLikeAF3", "CropSpatialLikeAF3"]

    def __init__(self, infer_hydrogens: bool = True):
        self.infer_hydrogens = infer_hydrogens

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["atom_array"])
        check_does_not_contain_keys(data, ["rdkit"])
        check_is_instance(data, "atom_array", AtomArray)
        check_nonzero_length(data, "atom_array")
        check_atom_array_annotation(data, ["atomize", "pn_unit_iid"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array: AtomArray = data["atom_array"]

        # Subset to atomized molecules
        _atom_array = atom_array[atom_array.atomize]

        # Iterate over unique pn_unit_iids
        data["rdkit"] = {}
        for pn_unit_iid in np.unique(_atom_array.pn_unit_iid):
            pn_unit_mask = _atom_array.pn_unit_iid == pn_unit_iid
            molecule = _atom_array[pn_unit_mask]
            try: 
                rdmol = atom_array_to_rdkit(
                    molecule,
                    infer_hydrogens=self.infer_hydrogens,
                    annotations_to_keep=["chain_id", "res_id", "res_name", "atom_name", "atom_id", "pn_unit_iid"],
                )
            except Exception as e:
                logger.error(f"Failed to convert molecule {pn_unit_iid} to RDKit: {str(e)}")
                rdmol = atom_array_to_rdkit(
                    molecule,
                    infer_hydrogens=self.infer_hydrogens,
                    annotations_to_keep=["chain_id", "res_id", "res_name", "atom_name", "atom_id", "pn_unit_iid"],
                    strict_sanitization=False,
                )
            # Store by pn_unit_iid as key
            data["rdkit"][pn_unit_iid] = rdmol

        return data


class GenerateRDKitConformers(Transform):
    """
    Generate conformers for RDKit molecules stored in the `data["rdkit"]` dictionary.

    This transform generates conformers for each RDKit molecule in the data dictionary and updates
    the molecules with the new conformers. The random seed for conformer generation is derived from
    the global numpy RNG state.

    Args:
        data (dict[str, Any]): A dictionary containing the input data, including RDKit molecules
            under the `"rdkit"` key.
        n_conformers (int): Number of conformations to generate for each molecule. Default is 1.

    Returns:
        dict[str, Any]: The updated `data` dictionary with RDKit molecules containing generated conformers.

    Example:
        >>> data = {
        >>>     "rdkit": {
        >>>         'A_1': <rdkit.Chem.rdchem.Mol object at 0x...>,
        >>>         'B_1': <rdkit.Chem.rdchem.Mol object at 0x...>,
        >>>     }
        >>> }
        >>> transform = GenerateRDKitConformers(n_conformers=3)
        >>> data = transform(data)
        >>> print(data["rdkit"]["A_1"].GetNumConformers())
        3
    """

    requires_previous_transforms = ["AddRDKitMoleculesForAtomizedMolecules"]

    def __init__(
        self, n_conformers: int = 1, optimize_conformers: bool = True, optimize_kwargs: dict[str, Any] | None = None
    ):
        self.n_conformers = n_conformers
        self.optimize_conformers = optimize_conformers
        self.optimize_kwargs = optimize_kwargs

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["rdkit"])
        check_is_instance(data, "rdkit", dict)
        check_nonzero_length(data, "rdkit")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        for pn_unit_iid, rdmol in data["rdkit"].items():
            try:
                # Generate a random seed using numpy's global RNG
                random_seed = np.random.randint(0, 2**32 - 1)

                rdmol_with_conformers = generate_conformers(
                    rdmol,
                    seed=random_seed,
                    n_conformers=self.n_conformers,
                    infer_hydrogens=True,
                    optimize_conformers=self.optimize_conformers,
                    optimize_kwargs=self.optimize_kwargs,
                )
                data["rdkit"][pn_unit_iid] = rdmol_with_conformers
            except Exception as e:
                logger.warning(f"Failed to generate conformers for molecule {pn_unit_iid}: {e}")

        return data