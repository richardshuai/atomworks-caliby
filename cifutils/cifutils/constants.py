"""Constants used in the `cifutils` package."""

import os
import logging
from typing import Final
from toolz import keymap
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _load_env_var(var_name: str) -> str | None:
    """Load an environment variable, returning None if it is not set."""
    try:
        return os.environ[var_name]
    except KeyError:
        logger.warning(
            f"Environment variable {var_name} not set. "
            "Will not be able to use function requiring this variable. "
            "To set it you may:\n"
            "  (1) add the line 'export VAR_NAME=path/to/variable' to your .bashrc or .zshrc file\n"
            "  (2) set it in your current shell with 'export VAR_NAME=path/to/variable'\n"
            "  (3) write it to a .env file in the root of the cifutils repository"
        )
        return None


CCD_MIRROR_PATH: Final[str] = _load_env_var("CCD_MIRROR_PATH")
"""A path to a carbon-copy mirror of the CCD ligands in the RCSB CCD."""

CCD_PICKLED_PATH: Final[str] = _load_env_var("CCD_PICKLED_PATH")
"""A path to a processed version of the CCD ligands in the RCSB CCD as '.pkl' files."""

UNKNOWN_ELEMENT: Final[str] = "X"
"""The element name for an unknown element."""

UNKNOWN_ATOMIC_NUMBER: Final[int] = 0
"""The atomic number for an unknown element."""

# fmt: off
ELEMENT_NAME_TO_ATOMIC_NUMBER: Final[dict[str, int]] = keymap(str.upper, {
    "H": 1,    "He": 2,   "Li": 3,   "Be": 4,   "B": 5,   "C": 6,   "N": 7,    "O": 8,    "F": 9,   "Ne": 10,  
    "Na": 11,  "Mg": 12,  "Al": 13,  "Si": 14,  "P": 15,  "S": 16,  "Cl": 17,  "Ar": 18,  "K": 19,  "Ca": 20,  
    "Sc": 21,  "Ti": 22,  "V": 23,   "Cr": 24,  "Mn": 25, "Fe": 26, "Co": 27,  "Ni": 28,  "Cu": 29, "Zn": 30,  
    "Ga": 31,  "Ge": 32,  "As": 33,  "Se": 34,  "Br": 35, "Kr": 36, "Rb": 37,  "Sr": 38,  "Y": 39,  "Zr": 40,  
    "Nb": 41,  "Mo": 42,  "Tc": 43,  "Ru": 44,  "Rh": 45, "Pd": 46, "Ag": 47,  "Cd": 48,  "In": 49, "Sn": 50, 
    "Sb": 51,  "Te": 52,  "I": 53,   "Xe": 54,  "Cs": 55, "Ba": 56, "La": 57,  "Ce": 58,  "Pr": 59, "Nd": 60,  
    "Pm": 61,  "Sm": 62,  "Eu": 63,  "Gd": 64,  "Tb": 65, "Dy": 66, "Ho": 67,  "Er": 68,  "Tm": 69, "Yb": 70, 
    "Lu": 71,  "Hf": 72,  "Ta": 73,  "W": 74,   "Re": 75, "Os": 76, "Ir": 77,  "Pt": 78,  "Au": 79, "Hg": 80, 
    "Tl": 81,  "Pb": 82,  "Bi": 83,  "Po": 84,  "At": 85, "Rn": 86, "Fr": 87,  "Ra": 88,  "Ac": 89, "Th": 90,  
    "Pa": 91,  "U": 92,   "Np": 93,  "Pu": 94,  "Am": 95, "Cm": 96, "Bk": 97,  "Cf": 98,  "Es": 99, "Fm": 100, 
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,"Sg": 106, "Bh": 107,"Hs": 108, "Mt": 109,"Ds": 110, 
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
    UNKNOWN_ELEMENT: UNKNOWN_ATOMIC_NUMBER
})
"""Map canonical *UPPERCASE* 2 letter element names to their atomic numbers. WARNING: Case-sensitive."""

ATOMIC_NUMBER_TO_ELEMENT: Final[dict[int | str, str]] = (
    {v: k for k, v in ELEMENT_NAME_TO_ATOMIC_NUMBER.items()} | 
    {str(v): k for k, v in ELEMENT_NAME_TO_ATOMIC_NUMBER.items()}
)
"""Map atomic numbers (int/str) to their canonical *UPPERCASE* 2 letter element names. WARNING: Case-sensitive."""

METAL_ELEMENTS: Final[frozenset[str]] = frozenset(map(str.upper, [
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi",
]))
"""A set of all metal elements, all *UPPERCASE*. WARNING: Case-sensitive."""
# fmt: on

CHEM_COMP_TYPES: Final[tuple[str, ...]] = tuple(
    [
        chemtype.upper()
        for chemtype in (
            "D-beta-peptide, C-gamma linking",
            "D-gamma-peptide, C-delta linking",
            "D-peptide COOH carboxy terminus",
            "D-peptide NH3 amino terminus",
            "D-peptide linking",
            "D-saccharide",
            "D-saccharide, alpha linking",
            "D-saccharide, beta linking",
            "DNA OH 3 prime terminus",
            "DNA OH 5 prime terminus",
            "DNA linking",
            "L-DNA linking",
            "L-RNA linking",
            "L-beta-peptide, C-gamma linking",
            "L-gamma-peptide, C-delta linking",
            "L-peptide COOH carboxy terminus",
            "L-peptide NH3 amino terminus",
            "L-peptide linking",
            "L-saccharide",
            "L-saccharide, alpha linking",
            "L-saccharide, beta linking",
            "RNA OH 3 prime terminus",
            "RNA OH 5 prime terminus",
            "RNA linking",
            "non-polymer",
            "other",
            "peptide linking",
            "peptide-like",
            "saccharide",
        )
    ]
)
"""Allowed Chemical Component Types for residues in the PDB + `mask`.
All uppercase.

Reference:
    - http://mmcif.rcsb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp.type.html
"""

AA_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset(
    [
        chemtype.upper()
        for chemtype in (
            "D-beta-peptide, C-gamma linking",
            "D-gamma-peptide, C-delta linking",
            "D-peptide COOH carboxy terminus",
            "D-peptide NH3 amino terminus",
            "D-peptide linking",
            "L-beta-peptide, C-gamma linking",
            "L-gamma-peptide, C-delta linking",
            "L-peptide COOH carboxy terminus",
            "L-peptide NH3 amino terminus",
            "L-peptide linking",
            "peptide linking",
            "peptide-like",
        )
    ]
)
"""Set of amino acid-like chemical component types. All uppercase."""

POLYPEPTIDE_L_CHEM_TYPES: Final[frozenset[str]] = frozenset(
    [
        chemtype.upper()
        for chemtype in (
            "L-beta-peptide, C-gamma linking",
            "L-gamma-peptide, C-delta linking",
            "L-peptide COOH carboxy terminus",
            "L-peptide NH3 amino terminus",
            "L-peptide linking",
        )
    ]
)
"""Set of polypeptide-L (left-handed amino acids) chemical component types. All uppercase."""

POLYPEPTIDE_D_CHEM_TYPES: Final[frozenset[str]] = frozenset(
    [
        chemtype.upper()
        for chemtype in (
            "D-beta-peptide, C-gamma linking",
            "D-gamma-peptide, C-delta linking",
            "D-peptide COOH carboxy terminus",
            "D-peptide NH3 amino terminus",
            "D-peptide linking",
        )
    ]
)
"""Set of polypeptide-D (right-handed amino acids) chemical component types. All uppercase."""

RNA_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset(
    [
        chemtype.upper()
        for chemtype in (
            "L-RNA linking",
            "RNA OH 3 prime terminus",
            "RNA OH 5 prime terminus",
            "RNA linking",
        )
    ]
)
"""Set of RNA-like chemical component types. All uppercase."""

DNA_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset(
    [
        chemtype.upper()
        for chemtype in (
            "DNA OH 3 prime terminus",
            "DNA OH 5 prime terminus",
            "DNA linking",
            "L-DNA linking",
        )
    ]
)
"""Set of DNA-like chemical component types. All uppercase."""

CARBOHYDRATE_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset(
    [
        chemtype.upper()
        for chemtype in (
            "D-saccharide",
            "D-saccharide, alpha linking",
            "D-saccharide, beta linking",
            "L-saccharide",
            "L-saccharide, alpha linking",
            "L-saccharide, beta linking",
            "saccharide",
        )
    ]
)
"""Set of carbohydrate-like chemical component types. All uppercase."""

LIGAND_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset([chemtype.upper() for chemtype in ("non-polymer", "other")])
"""Set of ligand-like chemical component types. All uppercase."""

MASK_LIKE_CHEM_TYPES: Final[frozenset[str]] = frozenset([chemtype.upper() for chemtype in ("mask",)])
"""Set of mask-like chemical component types. All uppercase."""

CRYSTALLIZATION_AIDS: Final[list[str]] = [
    "SO4",
    "GOL",
    "EDO",
    "PO4",
    "ACT",
    "PEG",
    "DMS",
    "TRS",
    "PGE",
    "PG4",
    "FMT",
    "EPE",
    "MPD",
    "MES",
    "CD",
    "IOD",
]
"""A list of CCD codes of common crystallization aids used in the crystallization of proteins.

Reference:
    - AF3 (Supp. Table 9) https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
"""

AF3_EXCLUDED_LIGANDS: Final[list[str]] = [
    "144",
    "15P",
    "1PE",
    "2F2",
    "2JC",
    "3HR",
    "3SY",
    "7N5",
    "7PE",
    "9JE",
    "AAE",
    "ABA",
    "ACE",
    "ACN",
    "ACT",
    "ACY",
    "AZI",
    "BAM",
    "BCN",
    "BCT",
    "BDN",
    "BEN",
    "BME",
    "BO3",
    "BTB",
    "BTC",
    "BU1",
    "C8E",
    "CAD",
    "CAQ",
    "CBM",
    "CCN",
    "CIT",
    "CL",
    "CLR",
    "CM",
    "CMO",
    "CO3",
    "CPT",
    "CXS",
    "D10",
    "DEP",
    "DIO",
    "DMS",
    "DN",
    "DOD",
    "DOX",
    "EDO",
    "EEE",
    "EGL",
    "EOH",
    "EOX",
    "EPE",
    "ETF",
    "FCY",
    "FJO",
    "FLC",
    "FMT",
    "FW5",
    "GOL",
    "GSH",
    "GTT",
    "GYF",
    "HED",
    "IHP",
    "IHS",
    "IMD",
    "IOD",
    "IPA",
    "IPH",
    "LDA",
    "MB3",
    "MEG",
    "MES",
    "MLA",
    "MLI",
    "MOH",
    "MPD",
    "MRD",
    "MSE",
    "MYR",
    "N",
    "NA",
    "NH2",
    "NH4",
    "NHE",
    "NO3",
    "O4B",
    "OHE",
    "OLA",
    "OLC",
    "OMB",
    "OME",
    "OXA",
    "P6G",
    "PE3",
    "PE4",
    "PEG",
    "PEO",
    "PEP",
    "PG0",
    "PG4",
    "PGE",
    "PGR",
    "PLM",
    "PO4",
    "POL",
    "POP",
    "PVO",
    "SAR",
    "SCN",
    "SEO",
    # "SEP", # Phosphoserine; a commonly occuring PTM in proteins, useful in cellular signaling pathways
    "SIN",
    "SO4",
    "SPD",
    "SPM",
    "SR",
    "STE",
    "STO",
    "STU",
    "TAR",
    "TBU",
    "TME",
    # "TPO", # Phosphothreonine; a commonly occuring PTM in proteins, useful in cellular signaling pathways
    "TRS",
    "UNK",
    "UNL",
    "UNX",
    "UPL",
    "URE",
]
"""A list of CCD codes of ligands that were excluded in AF3.

Reference:
    - AF3 (Supp. Table 10) https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf
"""

AF3_EXCLUDED_LIGANDS_REGEX: Final[str] = "(?:^|,)\s*(?:" + "|".join(AF3_EXCLUDED_LIGANDS) + ")\s*(?:,|$)"
"""A regex pattern that matches any of the ligands in `AF3_EXCLUDED_LIGANDS`. Used for filtering out ligands from the assembled dataframes."""

# TODO: Replace this by general mapping of CCD codes to one-letter codes.
DICT_THREE_TO_ONE: Final[dict[str, str]] = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
    "*": " * ",
}
"""A dictionary that maps three-letter amino acid codes to one-letter codes.

Reference:
    - Biotite: https://github.com/biotite-dev/biotite/blob/v0.41.0/src/biotite/sequence/seqtypes.py#L348-L556
"""

UNKNOWN_LIGAND: Final[str] = "UNL"
"""The CCD code for unknown ligands."""

UNKNOWN_AA: Final[str] = "UNK"
"""The CCD code for unknown amino acids."""

# TODO: Change these to something unique.
UNKNOWN_RNA: Final[str] = "X"
"""The (non-standard) CCD code for unknown RNA nucleotides."""

UNKNOWN_DNA: Final[str] = "DX"
"""The (non-standard) CCD code for unknown DNA nucleotides."""

GAP: Final[str] = "<G>"
"""The (non-standard) code for a gap token."""


STANDARD_AA: Final[tuple[str, ...]] = tuple(
    sorted(
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]
    )
)
"""Tuple of the CCD codes for the standard 20 amino acids, alphabetically sorted by their three-letter CCD codes."""

STANDARD_AA_ONE_LETTER: Final[tuple[str, ...]] = tuple(map(DICT_THREE_TO_ONE.get, STANDARD_AA))
"""Tuple of the one-letter symbols for the standard 20 amino acids, alphabetically sorted by their three-letter CCD codes."""

STANDARD_RNA: Final[tuple[str, ...]] = tuple(sorted(["A", "C", "G", "U"]))
"""Tuple of the CCD codes for the standard 4 RNA nucleotides. These happen to be the same as the one-letter symbols."""

STANDARD_DNA: Final[tuple[str, ...]] = tuple(sorted(["DA", "DC", "DG", "DT"]))
"""Tuple of the CCD codes for the standard 4 DNA nucleotides."""

STANDARD_DNA_ONE_LETTER: Final[tuple[str, ...]] = tuple(sorted(["A", "C", "G", "T"]))
"""Tuple of the one-letter symbols for the standard 4 DNA nucleotides."""

BIOTITE_DEFAULT_ANNOTATIONS: Final[tuple[str, ...]] = (
    "chain_id",
    "res_id",
    "res_name",
    "atom_name",
    "hetero",
    "element",
)
"""The default mandatory annotations for Biotite AtomArrays."""

STANDARD_PYRAMIDINE_RESIDUES: Final[tuple[str, ...]] = ("C", "U", "DC", "DT")
"""Tuple of the CCD codes for the 4 standard pyramidine nucleotides."""

STANDARD_PURINE_RESIDUES: Final[tuple[str, ...]] = ("A", "G", "DA", "DG")
"""Tuple of the CCD codes for the 4 standard purine nucleotides."""

HYDROGEN_LIKE_SYMBOLS: Final[tuple[str, ...]] = ("H", "H2", "D", "T", "1", 1)
"""
A tuple of symbols for (isotopes of) hydrogen.

WARNING: It is important that this remains a tuple, as it is used by `np.isin`
 downstream, which does not play well with sets.
"""

PEPTIDE_MAX_RESIDUES: Final[int] = 20
"""The maximum number of residues until which we consider a protein-like sequence to be a peptide."""
