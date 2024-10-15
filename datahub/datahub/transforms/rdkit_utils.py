import copy
import logging
import time
from functools import cache, lru_cache, wraps
from os import PathLike
from pathlib import Path
from typing import Any

import biotite.structure as struc
import numpy as np
import toolz
from biotite.structure import AtomArray
from cifutils.constants import METAL_ELEMENTS
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Mol, rdDistGeom, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.DataStructs import ExplicitBitVect

from datahub.common import default, exists
from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_does_not_contain_keys,
    check_is_instance,
    check_nonzero_length,
)
from datahub.transforms.base import Transform
from datahub.utils.timeout import timeout as timeout_decorator

logger = logging.getLogger(__name__)
# ... disable RDKit logging
RDLogger.DisableLog("rdApp.*")

# Set default pickle properties to all properties, otherwise
#  annotations get lost when pickling/unpickling molecules
# https://github.com/rdkit/rdkit/issues/1320
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

_RDKIT_HYBRIDIZATION_TO_INT = {
    Chem.rdchem.HybridizationType.S: 0,
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
    Chem.rdchem.HybridizationType.OTHER: 6,
    Chem.rdchem.HybridizationType.UNSPECIFIED: -1,
}
"""Mapping from RDKit hybridization types to integers."""

_RDKIT_BOND_TYPE_TO_BIOTITE = {
    # (rdkit bond type, is_aromatic) -> biotite bond type
    (Chem.BondType.UNSPECIFIED, False): struc.bonds.BondType.ANY,
    (Chem.BondType.SINGLE, False): struc.bonds.BondType.SINGLE,
    (Chem.BondType.DOUBLE, False): struc.bonds.BondType.DOUBLE,
    (Chem.BondType.TRIPLE, False): struc.bonds.BondType.TRIPLE,
    (Chem.BondType.QUADRUPLE, False): struc.bonds.BondType.QUADRUPLE,
    (Chem.BondType.SINGLE, True): struc.bonds.BondType.AROMATIC_SINGLE,
    (Chem.BondType.DOUBLE, True): struc.bonds.BondType.AROMATIC_DOUBLE,
    (Chem.BondType.TRIPLE, True): struc.bonds.BondType.AROMATIC_TRIPLE,
}
"""
Mapping from RDKit bond types to Biotite bond types.
Maps (rdkit bond type, is_aromatic) -> biotite bond type

Unspecified bonds are mapped to `ANY` bond type.
"""

_CONVERTIBLE_RDKIT_BOND_TYPES = toolz.keymap(lambda x: x[0], _RDKIT_BOND_TYPE_TO_BIOTITE)
"""List of RDKit Bond types that can be converted to Biotite bond types."""

_BIOTITE_BOND_TYPE_TO_RDKIT = {
    # biotite bond type -> (rdkit bond type, is_aromatic)
    struc.bonds.BondType.ANY: (Chem.BondType.UNSPECIFIED, False),
    struc.bonds.BondType.SINGLE: (Chem.BondType.SINGLE, False),
    struc.bonds.BondType.DOUBLE: (Chem.BondType.DOUBLE, False),
    struc.bonds.BondType.TRIPLE: (Chem.BondType.TRIPLE, False),
    struc.bonds.BondType.QUADRUPLE: (Chem.BondType.QUADRUPLE, False),
    # NOTE: We map aromatics to single/double/triple instead of Chem.BondType.AROMATIC
    #       because the PDB specified bond-order (from a kekulized form of the molecule)
    #       is lost when we map to aromatic, which can lead to incorrect bond-order
    #       perception in RDKit.
    struc.bonds.BondType.AROMATIC_SINGLE: (Chem.BondType.SINGLE, True),
    struc.bonds.BondType.AROMATIC_DOUBLE: (Chem.BondType.DOUBLE, True),
    struc.bonds.BondType.AROMATIC_TRIPLE: (Chem.BondType.TRIPLE, True),
}
"""Mapping from Biotite bond types to RDKit bond types.
Maps (biotite bond type) -> (rdkit bond type, is_aromatic)
"""

_BIOTITE_DEFAULT_ANNOTATIONS = ["chain_id", "res_id", "res_name", "atom_name", "hetero"]


class ChEMBLNormalizer:
    """
    Normalize an RDKit molecule like the ChEMBL structure pipeline does.
    This is useful for `rescuing` molecules that failed to be sanitized by RDKit
    alone.

    References:
        - https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/standardizer.py#L33C1-L73C15
    """

    def __init__(self):
        with open(Path(__file__).parent / "chembl_transformations.smirks", "r") as f:
            self._normalization_transforms = f.read()
        self._normalizer_params = rdMolStandardize.CleanupParameters()
        self._normalizer = rdMolStandardize.NormalizerFromData(
            paramData=self._normalization_transforms, params=self._normalizer_params
        )

    def normalize_in_place(self, mol: Mol) -> Mol:
        self._normalizer.normalizeInPlace(mol)
        return mol


@cache
def _periodic_table() -> Chem.PeriodicTable:
    return Chem.GetPeriodicTable()


@cache
def _valence_checker() -> rdMolStandardize.RDKitValidation:
    return rdMolStandardize.RDKitValidation()


@cache
def _chembl_normalizer() -> rdMolStandardize.Normalizer:
    return ChEMBLNormalizer()


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
        return Chem.GetPeriodicTable().GetAtomicNumber(element.capitalize())


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


def _mol_has_correct_valence(mol: Mol) -> bool:
    """
    Check if an RDKit molecule has correct valences.
    """
    mol.UpdatePropertyCache(False)
    return len(_valence_checker().validate(mol)) == 0


def _fix_valence_by_changing_formal_charge(mol: Mol) -> Mol:
    """
    Attempt to fix the valence of an RDKit molecule by changing the formal charge of atoms.
    """
    if not _mol_has_correct_valence(mol):
        for rdatom in mol.GetAtoms():
            n_electron = (
                rdatom.GetImplicitValence()
                + rdatom.GetExplicitValence()
                - _periodic_table().GetDefaultValence(rdatom.GetSymbol())
            )
            rdatom.SetFormalCharge(n_electron)
    return mol


def metal_to_dative_bonds(mol: Mol) -> Mol:
    """
    Change all single bonds to a metal to be dative bonds (coordination bonds).
    This is useful since most bonds between metals and organic atoms are dative.

    Args:
        mol (Mol): The input RDKit molecule

    Returns:
        Mol: The molecule with metal bonds converted to dative bonds
    """
    if not isinstance(mol, Chem.RWMol):
        # ... create a writable copy of the molecule if not writable in-place
        mol = Chem.RWMol(mol)

    metal_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in METAL_ELEMENTS]

    for metal_idx in metal_indices:
        for bond in mol.GetAtomWithIdx(metal_idx).GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                other_idx = bond.GetOtherAtomIdx(metal_idx)
                mol.RemoveBond(metal_idx, other_idx)
                mol.AddBond(other_idx, metal_idx, Chem.BondType.DATIVE)

    return mol


@_preserve_annotations
def fix_mol(
    mol: Mol,
    attempt_fix_valence_by_changing_formal_charge: bool = True,
    attempt_fix_by_normalizing_like_chembl: bool = True,
    attempt_fix_by_normalizing_like_rdkit: bool = True,
    in_place: bool = True,
    raise_on_failure: bool = True,
) -> Mol:
    """
    Fix an RDKit molecule (in-place).

    This function attempts to infer aromaticity, valences, implicit hydrogens, and
    formal charges to result in a molecule that can be successfully sanitized. It
    does **not** change the heavy atoms or bonds in the molecule.

    # TODO:
    #  - Add sanitization for aromatic systems with incorrect formal charges (that cannot be kekulized) (https://github.com/datamol-io/datamol/issues/231)
    #    - This may be done via Hueckel's rule (https://en.wikipedia.org/wiki/H%C3%BCckel%27s_rule): Find all aromatic systems, compute Hueckel's rule,
    #      if the number of pi electrons is not equal to 4*n+2, where n is an integer, then the aromatic system is not valid with the given formal charges.
    #      Try to adjust the formal charges on non-carbon atoms to make the aromatic system valid.
    #  - Add sanifix4 style sanitization (https://github.com/datamol-io/datamol/blob/0312388b956e2b4eeb72d791167cfdb873c7beab/datamol/_sanifix4.py#L114)
    #  - Add attempt to fix valences by changing the bond orders (c.f. https://github.com/datamol-io/datamol)
    #  - Add further ChEMBL style sanitization: https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/standardizer.py#L33C1-L73C15


    References:
        - https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
        - https://github.com/chembl/ChEMBL_Structure_Pipeline/blob/master/chembl_structure_pipeline/standardizer.py
        - https://github.com/datamol-io/datamol/blob/0312388b956e2b4eeb72d791167cfdb873c7beab/datamol/mol.py

    """
    if not in_place:
        mol = copy.deepcopy(mol)

    # ... try to fixe the molecule by automatically performing some standardization
    #  steps to infer formal charges and perceive aromaticity
    sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)

    if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
        # ... do not fix molecules that are not broken
        return mol

    logger.warning(
        f"Molecule failed sanitization: {sanitize_result}. Attempting to fix by inferring valences and aromaticity."
    )

    # ... recompute current valences
    mol.UpdatePropertyCache(strict=False)

    if attempt_fix_by_normalizing_like_chembl:
        # ... perform normalization steps recommended by the ChEMBL team to "rescue"
        _chembl_normalizer().normalize_in_place(mol)
        # ... recompute valences after attempted fixing
        mol.UpdatePropertyCache(strict=False)
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
            return mol

    if attempt_fix_by_normalizing_like_rdkit:
        # ... perform normalization steps recommended by the RDKit team.
        rdMolStandardize.NormalizeInPlace(mol)
        # ... recompute valences after attempted fixing
        mol.UpdatePropertyCache(strict=False)
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
            return mol

    if attempt_fix_valence_by_changing_formal_charge:
        _fix_valence_by_changing_formal_charge(mol)
        mol.UpdatePropertyCache(strict=False)
        sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
        if sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE:
            return mol

    if sanitize_result != Chem.SanitizeFlags.SANITIZE_NONE:
        logger.warning(f"Could not fix molecule, final sanitization result: {sanitize_result}")
        if raise_on_failure:
            raise Chem.MolSanitizeException(f"Molecule failed sanitization: {sanitize_result}")

    return mol


@_preserve_annotations
def add_hydrogens(mol: Mol) -> Mol:
    """
    Add hydrogens to an RDKit molecule.
    """
    return Chem.AddHs(mol)


@_preserve_annotations
def remove_hydrogens(mol: Mol) -> Mol:
    """
    Remove hydrogens from an RDKit molecule.
    """
    return Chem.RemoveHs(mol)


@_preserve_annotations
@timeout_decorator(strategy="subprocess")
def generate_conformers(
    mol: Mol,
    *,
    seed: int | None = None,
    n_conformers: int = 1,
    method: str = "ETKDGv3",
    num_threads: int = 1,
    infer_hydrogens: bool = True,
    optimize: bool = False,
    attempts_with_distance_geometry: int = 10,
    attempts_with_random_coordinates: int = 10_000,
    **uff_optimize_kwargs: dict,
) -> Mol:
    """
    Generate conformations for the given molecule.

    Args:
        - mol (rdkit.Chem.Mol): The RDKit molecule to generate conformations for.
        - seed (int | None): Random seed for reproducibility. If None, a random seed is used.
        - n_conformers (int): Number of conformations to generate.
        - method (str): The method to use for conformer generation. Default is "ETKDGv3".
            Allowed methods are: "ETDG", "ETKDG", "ETKDGv2", "ETKDGv3", "srETKDGv3"
            See https://rdkit.org/docs/RDKit_Book.html#conformer-generation for details.
        - num_threads (int): Number of threads to use for parallel computation. Default is 1.
        - infer_hydrogens (bool): Whether to add hydrogens if they are not present. This is
            recommended, since having hydrogens improves the accuracy of the conformer
            generation. Default is True.
        - optimize (bool): Whether to optimize the generated conformers using UFF.
            Default is True.
        - **uff_optimize_kwargs (dict): Additional keyword arguments for UFF optimization:
            - maxIters (int): Maximum number of iterations (default 200).
            - vdwThresh (float): Used to exclude long-range van der Waals interactions
              (default 10.0).
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
        - The ETKDGv3 method is used for conformer generation, which incorporates
          torsion angle preferences and basic knowledge (e.g. aromatic rings are planar)
          for improved accuracy.
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
        1. Conformer tutorial: https://rdkit.org/docs/RDKit_Book.html#conformer-generation
        1. RDKit Cookbook: https://www.rdkit.org/docs/Cookbook.html
        2. Riniker and Landrum, "Better Informed Distance Geometry: Using What We Know To
           Improve Conformation Generation", JCIM, 2015.

    """
    # Ensure that all properties are being pickled (needed when we use timeout)
    assert (
        Chem.GetDefaultPickleProperties() == Chem.PropertyPickleOptions.AllProps
    ), "Default pickle properties are not set to all properties. Annotation loss will occur."
    assert attempts_with_distance_geometry > 0, "Attempts with distance geometry must be greater than 0."
    assert attempts_with_random_coordinates > 0, "Attempts with random coordinates must be greater than 0."

    # Infer hydrogens if needed, i.e. if they are not present in the molecule. Normally this should
    # always be set to `True` to generate realistic conformations. The only reason to set this to `False`
    # is if you are already providing hydrogens in the molecule.
    mol = add_hydrogens(mol) if infer_hydrogens else mol

    # Setup the parameters for the coordinate embedding
    params = getattr(rdDistGeom, method)()
    params.clearConfs = True
    params.randomSeed = default(seed, -1)
    params.enforceChirality = True
    params.useRandomCoords = False
    params.numThreads = num_threads
    params.maxAttempts = attempts_with_distance_geometry

    try:
        successful_cids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params)
        if len(successful_cids) < n_conformers:
            logger.warning(
                f"Initial conformer generation based on distance geometry failed. Successful: {len(successful_cids)}. "
                "Falling back to generating a conformer starting from random coordinates."
            )
            raise RuntimeError("Failed to generate enough conformers.")
    except RuntimeError:
        # Addresses issues with bad conformers, which happens when distance embeddings fail due to
        #  too many constraints or rotatable bonds, see for example:
        # https://github.com/rdkit/rdkit/issues/1433#issuecomment-305097888
        params.useRandomCoords = True
        params.enforceChirality = False
        params.maxAttempts = attempts_with_random_coordinates
        successful_cids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=int(n_conformers), params=params)
        if len(successful_cids) < n_conformers:
            raise RuntimeError(
                f"Requested {n_conformers} conformers, but only {mol.GetNumConformers()} were generated."
            )

    if optimize:
        mol = optimize_conformers(mol, **uff_optimize_kwargs)

    mol = remove_hydrogens(mol) if infer_hydrogens else mol
    return mol


@_preserve_annotations
@timeout_decorator(strategy="subprocess")
def optimize_conformers(
    mol: Mol,
    numThreads: int = 1,
    maxIters: int = 200,
    vdwThresh: float = 10.0,
    ignoreInterfragInteractions: bool = True,
) -> Mol:
    """
    Optimize the conformers of an RDKit molecule.

    Args:
        - mol (Mol): The RDKit molecule to optimize.
        - numThreads (int): Number of threads to use for parallel computation. Default is 1.
        - maxIters (int): Maximum number of iterations for UFF optimization. Defaults to 200.
        - vdwThresh (float): Used to exclude long-range van der Waals interactions. Defaults to 10.0.
        - ignoreInterfragInteractions (bool): If True, nonbonded terms between fragments will not be added to the
            forcefield. Defaults to True.

    Returns:
        Mol: The optimized RDKit molecule.
    """
    success = AllChem.UFFOptimizeMoleculeConfs(
        mol,
        numThreads=numThreads,
        maxIters=maxIters,
        vdwThresh=vdwThresh,
        ignoreInterfragInteractions=ignoreInterfragInteractions,
    )
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
        generate_conformers(mol, n_conformers=1)

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


def find_automorphisms(mol: Chem.Mol, max_automorphs: int = 1000, timeout: float | None = None) -> np.ndarray:
    """
    Find automorphisms of a given RDKit molecule.

    This function identifies the automorphisms (symmetry-related atom swaps) of the input molecule
    and returns them as a numpy array. If the search for automorphisms times out, it returns a single
    automorphism representing the identity (no swaps).

    Args:
        mol (Chem.Mol): The RDKit molecule for which to find automorphisms.
        max_automorphs (int): The maximum number of automorphisms to return. These are deterministically
            set to be the first `max_automorphs` automorphisms found by RDKit.
            For model training it is recommended to deterministically select the automorphisms
            to be used (as done in this transform) as a model might otherwise be nudged towards a specific
            automorph in one training step, but that automorph then does not show up in the next training
            step, leading to a moving target problem.

    Returns:
        automorphs (np.ndarray): A numpy array of shape [n_automorphs, n_atoms, 2], where each element
            represents an automorphism as list of paired atom indices (from_idx, to_idx).
            If the search fails (e.g. due to running out of memory), returns an array with
            a single automorphism representing the identity (no swaps).

    References:
        - https://sourceforge.net/p/rdkit/mailman/message/27897393/

    Example:
        >>> from openbabel import pybel
        >>> mol = pybel.readstring("smi", "c1c(O)cccc1(O)").OBMol
        >>> automorphisms = find_automorphisms(mol)
        >>> print(automorphisms)
            [[[0 0]
              [1 1]
              [2 2]
              [3 3]
              [4 4]
              [5 5]
              [6 6]
              [7 7]]

             [[0 0]
              [1 6]
              [2 7]
              [3 5]
              [4 4]
              [5 3]
              [6 1]
              [7 2]]]
    """

    # NOTE: We compute the automorphisms via a substructure match. This may not be the computationally most
    #  efficient way, but still works well even for highly symmetric molecules (e.g. 60C).
    #  (c.f. https://sourceforge.net/p/rdkit/mailman/message/27897393/)
    #  The probably optimal way to do this would be to access internal symmetry labels for models
    #  (c.f. https://sourceforge.net/p/rdkit/mailman/message/27902778/)
    #  but this would require using an underlying graph librarly like nauty to determine the automorphisms of
    #  the coloured graph. Until we run into performance issues, we will stick with the current approach.
    @timeout_decorator(default_timeout=timeout, strategy="subprocess")
    def _find_automorphisms() -> tuple:
        return mol.GetSubstructMatches(mol, uniquify=False, maxMatches=max_automorphs, useChirality=False)

    _start = time.time()
    try:
        automorphs_tuple = _find_automorphisms()
    except TimeoutError:
        logger.warning(
            f"Automorphism search timed out after {time.time() - _start:.2f}s. Returning identity automorphism."
        )
        automorphs_tuple = (tuple(range(mol.GetNumAtoms())),)

    # Turn the tuple of automorphisms into a numpy array of shape [n_automorphs, n_atoms, 2]
    automorphs = np.array(automorphs_tuple)
    n_automorphs, n_atoms = automorphs.shape
    identity = np.tile(np.arange(n_atoms), (n_automorphs, 1))

    return np.stack([identity, automorphs], axis=-1)


def smiles_to_rdkit(smile: str, sanitize: bool = True) -> Mol:
    """
    Generate an RDKit molecule from a SMILES string.

    This function creates an RDKit molecule object from a SMILES string,
    performs sanitization, and perceives aromaticity.

    Args:
        - smile (str): The SMILES string representing the molecule.
        - sanitize (bool): Whether to sanitize the molecule.

    Returns:
        - rdkit.Chem.Mol: The RDKit molecule generated from the SMILES string.

    Note:
        The returned molecule is sanitized and has aromaticity perceived.
    """
    mol = Chem.MolFromSmiles(smile, sanitize=sanitize)
    if mol is None:
        raise Chem.MolSanitizeException(
            f"Failed to create molecule from SMILES string: {smile}. Try setting `sanitize=False`."
        )
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
    mol = copy.deepcopy(mol)

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
                element=str(rdatom.GetAtomicNum()) if elements_as_int else rdatom.GetSymbol(),
                coord=coords[idx],
                charge=rdatom.GetFormalCharge(),
                hyb=_RDKIT_HYBRIDIZATION_TO_INT[rdatom.GetHybridization()],
                nhyd=rdatom.GetTotalNumHs(),
                hvydeg=rdatom.GetDegree() - rdatom.GetTotalNumHs(),
                rdkit_atom_id=rdatom.GetIntProp("rdkit_atom_id") if rdatom.HasProp("rdkit_atom_id") else -1,
                hetero=True,  # per default, set all atoms to be hetero atoms
            )
        )
    atom_array = struc.array(atoms)

    # Set bonds
    # ... kekulize to ensure aromaticity is perceived correctly and assign integer bond orders
    Chem.Kekulize(mol)
    # ... create bond list with integer bond orders
    bond_list = []
    for bond in mol.GetBonds():
        rdkit_bond_type = bond.GetBondType()

        if rdkit_bond_type not in _CONVERTIBLE_RDKIT_BOND_TYPES:
            # ... skip undesired bonds, e.g. dative bonds (=metal coordination bonds)
            logger.warning(f"Skipping {rdkit_bond_type=}. Not in convertible bond types.")
            continue

        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        is_bond_aromatic = bond.GetIsAromatic()
        # ... after kekulize, order is guaranteed to be integer
        bond_type = _RDKIT_BOND_TYPE_TO_BIOTITE[(bond.GetBondType(), is_bond_aromatic)]
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

        # Exit if there are no annotations to set
        if len(array_idx_to_annotation_idx) == 0:
            logger.warning(
                "No rdkit atoms match any annotation. You may want to check that you are "
                "using the @_preserve_annotations decorator correctly. And set the "
                "rdkit pickle options to preserve properties. Returning."
            )
            return atom_array

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
            elif np.issubdtype(val.dtype, bool):
                default = False
            else:
                logger.warning(f"Unsupported annotation dtype: {val.dtype} for annotation: {key}. Skipping.")
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
    sanitize: bool = True,
    attempt_fixing_corrupted_molecules: bool = True,
    assume_metal_bonds_are_dative: bool = False,  # NOTE: This messes up RDKit conformer generation
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
        logger.debug(f"Found {np.sum(is_hydrogen)} hydrogen atoms in the AtomArray. Removing them.")
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
    _should_be_aromatic = set()

    if exists(atom_array.bonds):
        for bond in atom_array.bonds.as_array():
            atom1, atom2, bond_type = list(map(int, bond))
            if bond_type == struc.bonds.BondType.ANY:
                # ... warn if underspecified bonds are encountered
                logger.warning("Encountered BondType.ANY. Interpreting as single bond.")
            bond_order, bond_is_aromatic = _BIOTITE_BOND_TYPE_TO_RDKIT[bond_type]
            mol.AddBond(atom1, atom2, order=bond_order)
            if bond_is_aromatic and not attempt_fixing_corrupted_molecules:
                # ... set aromaticity explicitly (and require the molecule makes sense later)
                mol.GetAtomWithIdx(atom1).SetIsAromatic(True)
                mol.GetAtomWithIdx(atom2).SetIsAromatic(True)
            _should_be_aromatic.union({atom1, atom2})

    # Set coordinates
    if set_coord:
        # ... add conformer (at id 0)
        conf_id = mol.AddConformer(Chem.Conformer(len(atom_array)), assignId=True)
        # ... fill in coordinates
        for atom_id, coord in enumerate(atom_array.coord):
            mol.GetConformer(conf_id).SetAtomPosition(atom_id, coord.tolist())

    # Clean up organometallics:
    # TODO: The CCD unfortunatley only supplies all metal bonds as single bonds. For now we assume
    #  all bonds with metals are coordination bonds in the PDB. This will
    #  likely be true for most ligands but not all. Revisit this later.
    # Change all bonds to a metal to be dative bonds (= coordination bonds)
    if assume_metal_bonds_are_dative:
        mol = metal_to_dative_bonds(mol)

    if attempt_fixing_corrupted_molecules:
        # ... fix_mol has no effect if the molecule is already sanitized
        mol = fix_mol(
            mol,
            attempt_fix_by_normalizing_like_chembl=True,
            attempt_fix_by_normalizing_like_rdkit=True,
            attempt_fix_valence_by_changing_formal_charge=True,
            in_place=True,
            raise_on_failure=False,
        )

    # Clean up the molecule and infer various properties
    #  (we always sanitize when attempting to fix corrupted molecules)
    if sanitize or attempt_fixing_corrupted_molecules:
        # ... verify validity of the molecule (according to Lewis octet rule)
        Chem.SanitizeMol(mol)

        # ... verify that atoms that are labelled as `_should_be_aromatic` are aromatic
        for atom_idx in _should_be_aromatic:
            assert mol.GetAtomWithIdx(
                atom_idx
            ).GetIsAromatic(), f"Atom {atom_idx} is not aromatic but was labelled as aromatic."

    # Turn into a non-editable molecule
    mol = mol.GetMol()

    # Attach custom atom-level annotations from the atom array
    mol._annotations = {"rdkit_atom_id": np.array(rdkit_atom_ids)}
    for annotation in annotations_to_keep:
        if annotation in atom_array.get_annotation_categories():
            mol._annotations[annotation] = atom_array._annot[annotation]

    return mol


@timeout_decorator(strategy="subprocess")
def sample_rdkit_conformer_for_atom_array(atom_array: AtomArray, seed: int | None = None) -> AtomArray:
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


@lru_cache(maxsize=1000)
def res_name_to_rdkit(
    res_name: str,
    set_coord: bool = True,
    infer_hydrogens: bool = True,
    ccd_dir: PathLike | None = Path("/projects/ml/RF2_allatom/cifutils_biotite/ccd_ligands_2024_05_31/ccd"),
    **atom_array_to_rdkit_kwargs,
) -> Mol:
    """
    Convert a CCD residue name to an RDKit molecule.

    This function retrieves an RDKit molecule corresponding to a given CCD residue name.
    If `ccd_dir` is not provided, Biotite's internal CCD is used. Otherwise, the specified local CCD directory is used.
    By default, the function returns the 'ideal' conformer from the CCD entry.

    Args:
        res_name (str): The residue name to convert. I.e, 'ALA', 'GLY', '9RH', etc.
        set_coord (bool): Whether to set coordinates for the molecule. Defaults to True.
        infer_hydrogens (bool): Whether to infer missing hydrogens. Defaults to True.
        ccd_dir (PathLike): Path to the local CCD directory. If None, Biotite's internal CCD is used.
        **atom_array_to_rdkit_kwargs: Additional keyword arguments passed to the `atom_array_to_rdkit` function.

    Returns:
        Mol: The RDKit molecule corresponding to the given residue name.
    """
    if ccd_dir is None:
        # ...use Biotite's internal CCD (WARNING: may be outdated)
        return atom_array_to_rdkit(
            struc.info.residue(res_name),
            set_coord=set_coord,
            infer_hydrogens=infer_hydrogens,
            **atom_array_to_rdkit_kwargs,
        )
    elif Path(ccd_dir).exists():
        # ...use the local CCD directory, which we assume is sharded by the first character of the res_name (as it should be, since we're using `rsync`)
        path = ccd_dir / res_name[0] / res_name / f"{res_name}.cif"
        cif_file = struc.io.pdbx.CIFFile.read(path)
        return atom_array_to_rdkit(
            struc.io.pdbx.get_component(cif_file),
            set_coord=set_coord,
            infer_hydrogens=infer_hydrogens,
            **atom_array_to_rdkit_kwargs,
        )
    else:
        raise FileNotFoundError(f"CCD directory not found: {ccd_dir}")


def get_morgan_fingerprint_from_rdkit_mol(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> ExplicitBitVect:
    """
    Generates the Morgan fingerprint for an RDKit molecule.

    From the AF-3 supplementary material:
        > We measure ligand Tanimoto similarity using RDKit v.2023_03_3 Morgan fingerprints (radius 2, 2048 bits)

    References:
        - AF-3 Supplement
        - https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    """
    morgan_fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint = morgan_fingerprint_generator.GetFingerprint(mol)
    return fingerprint


def res_name_to_rdkit_with_conformers(
    res_name: str, n_conformers: int, *, timeout_seconds: float = 2.0, **generate_conformers_kwargs
) -> Chem.Mol:
    """
    Generate an RDKit molecule with conformers for a given residue name.

    This function attempts to generate the specified number of conformers for the given residue
    using RDKit's conformer generation (based on ETKDGv3 per default).
    If conformer generation fails or times out, it falls back to using the idealized conformer
    from the CCD entry if one is available.

    Args:
        res_name (str): The residue name to generate conformers for. E.g. 'ALA' or 'GLY', '9RH' etc.
        n_conformers (int): The number of conformers to generate for the given residue.
        timeout_seconds (float, optional): The maximum time allowed for conformer generation.
            Defaults to 10.0 seconds, which is sufficient for most residues.
        **generate_conformers_kwargs: Additional keyword arguments to pass to the
            generate_conformers function.

    Returns:
        Chem.Mol: An RDKit molecule with the specified number of conformers.
    """
    # ... get molecule from CCD with its idealized conformer (default conformer 0)
    mol = res_name_to_rdkit(res_name)

    # ... get idealized conformer from CCD entry
    idealized_conformer = Chem.Conformer(mol.GetConformer(0))  # creates a copy

    # ... try generating `count` conformers within a given time limit
    try:
        mol = generate_conformers(mol, n_conformers=n_conformers, timeout=timeout_seconds, **generate_conformers_kwargs)
    except (TimeoutError, RuntimeError) as e:
        logger.warning(
            f"Failed to generate conformers for {res_name=}. Falling back to idealized conformer. Error message: {e}"
        )

    # ... if conformer generation fails or is incomplete, return the idealized conformer (set `count` conformers)
    missing_conformers = n_conformers - mol.GetNumConformers()
    if missing_conformers > 0:
        for _ in range(missing_conformers):
            mol.AddConformer(Chem.Conformer(idealized_conformer), assignId=True)

    return mol


# -------------------------------------------------------------------------------------------------
# ---------------------  RDKit related transforms  ------------------------------------------------
# -------------------------------------------------------------------------------------------------


class AddRDKitMoleculesForAtomizedMolecules(Transform):
    """
    Add RDKit molecules for atomized molecules in the atom array.

    This transform converts atomized molecules in the atom array to RDKit Mol objects and stores them in the
    `data` dictionary under the "rdkit" key. Each molecule is identified by its `pn_unit_iid`.

    Note:
        This transform requires the `AtomizeByCCDName` transform to be applied previously.

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

    requires_previous_transforms = ["AtomizeByCCDName"]
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
                    sanitize=True,
                    attempt_fixing_corrupted_molecules=False,
                )
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                logger.error(
                    f"Failed to convert molecule {pn_unit_iid} to RDKit: {str(e)}. Trying again and attempting to fix the corrupted molecule."
                )
                rdmol = atom_array_to_rdkit(
                    molecule,
                    infer_hydrogens=self.infer_hydrogens,
                    annotations_to_keep=["chain_id", "res_id", "res_name", "atom_name", "atom_id", "pn_unit_iid"],
                    sanitize=True,
                    attempt_fixing_corrupted_molecules=True,
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
        self.optimize_kwargs = default(optimize_kwargs, {})

    def check_input(self, data: dict[str, Any]):
        check_contains_keys(data, ["rdkit"])
        check_is_instance(data, "rdkit", dict)
        check_nonzero_length(data, "rdkit")

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        for pn_unit_iid, rdmol in data["rdkit"].items():
            try:
                # Generate a random seed using numpy's global RNG
                random_seed = np.random.randint(0, 2**16 - 1)

                rdmol_with_conformers = generate_conformers(
                    rdmol,
                    seed=random_seed,
                    n_conformers=self.n_conformers,
                    infer_hydrogens=True,
                    optimize=self.optimize_conformers,
                    optimize_kwargs=self.optimize_kwargs,
                )
                data["rdkit"][pn_unit_iid] = rdmol_with_conformers
            except Exception as e:
                logger.warning(f"Failed to generate conformers for molecule {pn_unit_iid}: {e}")

        return data
