from data.data_constants import ChainType

FULL_PDB_EDGE_CASE_LIST = [
    "1A8O",  # Single-chain protein complex with non-trivial symmetry
    "1IVO",  # Protein with glycosylation, with covalently attached ligands
    "3K4A",  # Proteins with modified amino acids (selenomethionine)
    "3KFA",  # No `struct_conn` field, which can cause errors
    "6WJC",  # Molecule  with "Subject of Investigation" label; missing residues up until label_seq_id 26
    "1EN2",  # Proteins with sequence heterogeneity
    "1CBN",  # Proteins with sequence heterogeneity
    "133D",  # Very simple DNA molecule (6 bases)
    "4JS1",  # Glycol (oligosaccharide) covalently bound to a protein (oligosaccharide with multiple residues)
    "1L2Y",  # Simple, single-residue protein solved through NMR (multiple models)
    "2K0A",  # Heinous multi-chain NMR structure
    "4CPA",  # Molecule with unknown or ambiguous element (marked with 'X')
    "1ZY8",  # Incorrect in the legacy parser. An FAD ligand, (P, 4750, FAD), has two alternative locations; in the label-assigned ID's (and in PyMol) those are correctly noted, but they have different author residue ID's and thus are both present in the legacy parser.
    "6DMH",  # Incorrect in the legacy parser. Waters with multiple occupancies not resolved correctly.
    "1FU2",  # Simple, small example with multiple chains
    "6DMG",  # Multiconformer ligand
    "1Y1W",  # Protein-nucleic-acid complex
    "5XNL",  # Another ribosomal monstrous molecule to limit-test loading speeds
    "2E2H",  # Complex with protein, DNA, and RNA
    "4NDZ",  # Large assembly with two ligand chains covalently bound together; also has non-biological bonds
    "3NE7",  # Small assembly with two ligand chains covalently bound together. Also has an "UNL" residue (unknown ligand). Further complicated by the fact that only atoms in one occupancy state are covalently bonded between the two ligands, but not in the other.
    "3NEZ",  # Sequence heterogeneity with non-standard residues (CH6/NRQ), with equal occupancy. Requires additional checks when loading covalent bonds
    "1RXZ",  # Single protein chain in complex with a short peptide (11 residues) with non-trivial symmetry (multiple transformations)
    "3J31",  # Enormous molecule (viral capsid - very large bioassembly)
    "7MUB",  # Has a clashing molecule, which is also a LOI - has a potassium at symmetry center
    "1QK0",  # Has 3 LOIs declared in the cif file, but 1 LOI on the PDB summary page / FTP service
    "1DYL",  # Non-numeric transformation ID
    "7SBV",  # Oligosaccharide defined as separate chains
    "3EPC",  # Invalid index to scalar
    "6O7K",  # Empty coordinates after filtering
    "104D",  # DNA/RNA Hybrid
    "5X3O",  # polypeptide(D)
    "5GAM",  # Complex with proteins and RNA; used in MSA tests
    "6A5J",  # Small peptide, used in MSA tests (ensure no MSA)
    "3NE2",  # Manageable-size example with two simple proteins
    "1MNA",  # Simple homomer (for MSA testing)
    "1HGE",  # Simple heteromer (for MSA testing)
    "3EJJ",  # Simple heteromer (for MSA testing)
    "112M",  # Protein-ligand, no LOI, heme ligand
    "1A3G",  # Involves covalent modification, protein-ligand
    "1A2N",  # Protein-protein homeric interface
    "1A2Y",  # Protein-protein heteromeric interface
    "1BDV",  # Protein-nucleic acid interface
    "184D",  # DNA strands with MG ions around
    "4HF4",  # Includes zinc ion, ligand, protein
    "3LPV",  # DNA
    "2NVZ",  # RNA (large structure, overall)
    "7KF1",  # Used to test data loading pipeline
    "7CJG",  # Used to test data loading pipeline
    "7B1W",  # Used to test data loading pipeline
    "6ZIE",  # Used to test data loading pipeline
    "7NMJ",  # Used to test data loading pipeline
    "6M2Z",  # Used to test data loading pipeline
    "5OCM",  # Domain swapped homo-dimer with small molecule ligands, used in many test cases
    "3SJM",  # For regression test against legacy dataloader
    "4I7Z",  # For regression test against legacy dataloader
    "4OLB",  # For regression test against legacy dataloader
    "4RES",  # For regression test against legacy dataloader
    "6BGN",  # For regression test against legacy dataloader
    "6VET",  # For regression test against legacy dataloader
    "6ZSJ",  # For regression test against legacy dataloader
    "5RX1",  # For chirals featurization test
    "7D9H",  # Used to test datasets
    "5S4P",  # Used to test datasets
    "4GQA",  # Used to test atom arrays
]
"""
A list of manually selected PDB IDs that are known to be tricky for a large variety of 
reasons. They mostly serve as test cases for the data preprocessing pipeline."""

UNSUPPORTED_CHAIN_TYPE_TEST_CASES = [
    "104D",  # DNA/RNA Hybrid
    "5X3O",  # polypeptide(D)
]

FIND_CONTACTS_TEST_CASES = [
    # Defined with:
    # - contact_distance = 5
    # - close_distance = 30
    {
        # Simple protein complex
        "pdb_id": "1fu2",
        "contact_information": [
            {
                "assembly_id": "1",
                "pn_unit_iid": "A_1",
                "num_contacting_pn_units": 2,
                "num_contacts": 461,  # 458 for (A,B) and 3 for (A, D)
                "num_close_pn_units": 11,  # NOTE: 8 if excluding AF-3 exclusion ligands (Na, Cl)
            }
        ],
    },
    {
        # RNA complex
        "pdb_id": "4gxy",
        "contact_information": [
            {
                "assembly_id": "1",
                "pn_unit_iid": "C_1",
                "num_contacting_pn_units": 1,
                "num_contacts": 94,  # 94 for (C, A)
                "num_close_pn_units": 10,
            }
        ],
    },
]

CHAIN_TYPE_TEST_CASES = [
    {
        # Simple polymer & non-polymers
        "pdb_id": "6qhp",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.POLYPEPTIDE_L,
            "C": ChainType.NON_POLYMER,  # fluoroacetic acid
            "D": ChainType.NON_POLYMER,  # fluoroacetic acid
        },
    },
    {
        # DNA and RNA, separately
        "pdb_id": "1fix",
        "chain_types": {
            "A": ChainType.RNA,
            "B": ChainType.DNA,
        },
    },
    {
        # DNA and RNA hybrid
        "pdb_id": "1d9d",
        "chain_types": {
            "A": ChainType.DNA_RNA_HYBRID,
            "B": ChainType.POLYPEPTIDE_L,
            "C": ChainType.NON_POLYMER,  # Zinc ion
            "F": ChainType.NON_POLYMER,  # Magnesium ion
        },
    },
    {
        # Oligosaccharides
        "pdb_id": "1ivo",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.POLYPEPTIDE_L,
            "C": ChainType.POLYPEPTIDE_L,
            "D": ChainType.POLYPEPTIDE_L,
            "E": ChainType.NON_POLYMER,  # Oligosaccharide
            "F": ChainType.NON_POLYMER,  # Monosaccharide
            "G": ChainType.NON_POLYMER,  # Monosaccharide
        },
    },
    {
        # Covalently bonded ligands
        "pdb_id": "3ne7",
        "chain_types": {
            "A": ChainType.POLYPEPTIDE_L,
            "B": ChainType.NON_POLYMER,  # Nickel
            "C": ChainType.NON_POLYMER,  # CoA
        },
    },
]

FILTERING_CRITERIA_TEST_CASES = [
    # Clashing test cases
    {"pdb_id": "6qhp", "pn_units_to_keep": ["A_1"], "pn_units_to_remove": ["C_1"]},
    {
        "pdb_id": "3voz",  # NOTE: In this example, the pn_units are bonded via a sulfer bridge, but still clash since we superimpose across the axis of symmetry
        "pn_units_to_keep": ["B_1", "B_3"],
        "pn_units_to_remove": ["B_2", "B_4"],
    },
    {
        "pdb_id": "2voy",
        "pn_units_to_keep": ["D_1", "J_1", "F_1", "C_1", "I_1"],
        "pn_units_to_remove": ["H_1", "B_1", "A_1"],
    },
    # Non-biological bond test cases
    {"pdb_id": "1pak", "pn_units_to_keep": ["A_1"], "pn_units_to_remove": ["B_1"]},
    # 2pnf - non-biological bonds
]

PN_UNIT_IID_TEST_CASES = [
    {
        "pdb_id": "4ndz",
        "assembly_id": "1",
        "pn_unit_iids": ["A_1", "B_1", "C_1", "G_1,R_1", "H_1", "J_1", "Q_1", "S_1", "I_1,T_1", "U_1"],
        "q_pn_unit_iid": "C_1",
    },  # Covalently bonded ligands
    {
        "pdb_id": "4ndz",
        "assembly_id": "2",
        "q_pn_unit_iid": "D_1",
        "pn_unit_iids": [
            "D_1",
            "E_1",
            "F_1",
            "K_1",
            "L_1,W_1",
            "M_1",
            "N_1,Y_1",
            "O_1",
            "P_1",
            "V_1",
            "X_1",
            "Z_1",
        ],
    },  # Covalently bonded ligands, query PN unit of C_1, near the center
    {
        "pdb_id": "4ndz",
        "assembly_id": "1",
        "q_pn_unit_iid": "Q_1",
        "pn_unit_iids": ["A_1", "B_1", "C_1", "G_1,R_1", "Q_1"],
    },  # Covalently bonded ligands, query PN unit of Q_1, off to the side
    {
        "pdb_id": "3ne7",
        "assembly_id": "1",
        "q_pn_unit_iid": "A_1",
        "pn_unit_iids": ["A_1", "B_1", "C_1,D_1", "E_1", "H_1"],  # NOTE: E, H, and D are AF-3 excluded ligands
    },  # Covalently bonded ligands
    {
        "pdb_id": "1ivo",
        "assembly_id": "1",
        "q_pn_unit_iid": "A_1",
        "pn_unit_iids": [
            "A_1",
            "B_1",
            "C_1",
            "D_1",
            "E_1",
            "F_1",
            "G_1",
            "H_1",
            "I_1",
            "J_1",
            "K_1",
            "L_1",
            "M_1",
        ],
    },  # Oligosaccharides covalently bonded to residues (glycoprotein)
    {
        "pdb_id": "1a8o",
        "assembly_id": "1",
        "q_pn_unit_iid": "A_1",
        "pn_unit_iids": ["A_1", "A_2"],
    },  # Simple assembly with non-trivial symmetry
    {
        "pdb_id": "1rxz",
        "assembly_id": "1",
        "q_pn_unit_iid": "A_1",
        "pn_unit_iids": [
            "A_1",
            "A_2",
            "A_3",
            "B_1",
            "B_2",
            "B_3",
        ],  # More complex assembly with non-trivial symmetry
    },
]

LOI_EXTRACTION_TEST_CASES = [
    {
        # TEST CASE 0
        "pdb_id": "7lad",
        "loi": set(["HEM", "XRD"]),
    },
    {
        # TEST CASE 1
        "pdb_id": "5ocm",
        "loi": set(),
    },
    {
        # TEST CASE 2
        "pdb_id": "6wjc",
        "loi": set(["Y01", "OIN"]),
    },
    {
        # TEST CASE 3
        "pdb_id": "7mub",
        "loi": set(["K"]),
    },
    {
        # TEST CASE 4
        "pdb_id": "1qk0",
        "loi": set(["IOB", "GLC", "XYS"]),
        # NOTE: GLC & XYS are only specified in the cif file, not on
        #  the PDB summary page
        "has_covalently_bonded_loi": True,
    },
]

PDB_PROCESSING_TEST_CASES = [
    {
        # Protein-ligand with LOI
        "pdb_id": "6wjc",
        "require_matching_interfaces": True,  # Whether we are enumerating all interfaces in `expected_interfaces` or only relevant ones
        "num_pn_units": 8,
        "expected_interfaces": [
            # Protein-protein
            {"pn_unit_1": "A_1", "pn_unit_2": "B_1"},
            # Protein-ligand
            {"pn_unit_1": "A_1", "pn_unit_2": "C_1", "involves_covalent_modification": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "D_1", "involves_loi": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "E_1", "involves_loi": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "F_1", "involves_loi": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "G_1", "involves_loi": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "H_1", "involves_loi": True},
            # Ligand-ligand
            {"pn_unit_1": "F_1", "pn_unit_2": "G_1", "involves_loi": True},
        ],
    },
    {
        # Covalent modifications
        "pdb_id": "1ivo",
        "require_matching_interfaces": False,
        "num_pn_units": 13,
        "expected_interfaces": [
            {"pn_unit_1": "A_1", "pn_unit_2": "B_1"},
            {"pn_unit_1": "B_1", "pn_unit_2": "K_1", "involves_covalent_modification": True},
            {"pn_unit_1": "A_1", "pn_unit_2": "J_1", "involves_covalent_modification": True},
        ],
    },
    {
        # Homomeric symmetry
        "pdb_id": "1a8o",
        "require_matching_interfaces": True,
        "num_pn_units": 2,
        "expected_interfaces": [
            {"pn_unit_1": "A_1", "pn_unit_2": "A_2"},
        ],
    },
    # TODO: Add additional test cases, including clashes, DNA, RNA, multi-chain ligands, etc.
]
