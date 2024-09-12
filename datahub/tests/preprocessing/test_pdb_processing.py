import shutil
import tempfile
from pathlib import Path

import pytest

from scripts.preprocessing.pdb.generate_interfaces_df import generate_interfaces_df
from scripts.preprocessing.pdb.generate_pn_units_df import generate_pn_units_df
from scripts.preprocessing.pdb.process_pdbs import run_pipeline as process_pdb

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


# Define a fixture for the temporary directory
@pytest.fixture(scope="module")
def temp_dir():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)


@pytest.fixture(scope="module")
def processed_pdb_files(temp_dir):
    # Define a pre-defined list of PDB IDs
    pdb_ids = [test_case["pdb_id"] for test_case in PDB_PROCESSING_TEST_CASES]

    # Run the process_pdb function to process the PDB IDs and save the output in the temp directory
    process_pdb(
        base_cif_dir="/databases/rcsb/cif",
        pdb_selection=",".join(pdb_ids),
        out_dir=temp_dir,
    )

    yield temp_dir, pdb_ids


def test_pdb_processing_saves_individual_csvs(processed_pdb_files):
    temp_dir, pdb_ids = processed_pdb_files
    csv_dir = Path(temp_dir) / "csv"

    # Check that each PDB ID has a corresponding CSV file
    for pdb_id in pdb_ids:
        csv_file = csv_dir / f"{pdb_id}.csv"
        assert csv_file.exists(), f"CSV file for {pdb_id} does not exist"


def test_generate_pn_units_and_interfaces_dfs(processed_pdb_files):
    temp_dir, _ = processed_pdb_files
    input_dir = Path(temp_dir) / "csv"

    # Run the generate_pn_units_df function to concatenate the CSV files in the temp directory
    pn_units_df = generate_pn_units_df(input_dir, max_workers=4)
    # We don't explicitly valide the `pn_units_df`, since the `components` tests cover this aim

    for test_case in PDB_PROCESSING_TEST_CASES:
        pdb_id = test_case["pdb_id"]
        num_pn_units = test_case["num_pn_units"]

        # Check that the concatenated DataFrame contains the correct number of PN units for the PDB ID
        pdb_df = pn_units_df[pn_units_df["pdb_id"] == pdb_id]
        assert len(pdb_df) == num_pn_units, f"Number of PN units for {pdb_id} is incorrect"

    # NOTE: We can't test clustering in this workflow, since we don't have enough data to cluster with mmseqs2

    # Now run the new script to create the interfaces DataFrame
    interfaces_df = generate_interfaces_df(pn_units_df)
    validate_interfaces_dataframe(interfaces_df, PDB_PROCESSING_TEST_CASES)


def validate_interfaces_dataframe(interfaces_df, test_cases):
    for test_case in test_cases:
        pdb_id = test_case["pdb_id"]
        expected_interfaces = test_case["expected_interfaces"]

        if test_case["require_matching_interfaces"]:
            assert len(expected_interfaces) == len(
                interfaces_df[interfaces_df["pdb_id"] == pdb_id]
            ), f"{pdb_id}: Number of interfaces is incorrect."

        for expected_interface in expected_interfaces:
            pn_unit_1 = expected_interface["pn_unit_1"]
            pn_unit_2 = expected_interface["pn_unit_2"]
            involves_loi = expected_interface.get("involves_loi", False)
            involves_covalent_modification = expected_interface.get("involves_covalent_modification", False)
            involves_metal = expected_interface.get("involves_metal", False)

            interface_df = interfaces_df[
                (interfaces_df["pdb_id"] == pdb_id)
                & (interfaces_df["pn_unit_1_iid"] == pn_unit_1)
                & (interfaces_df["pn_unit_2_iid"] == pn_unit_2)
            ]

            assert (
                len(interface_df) == 1
            ), f"{pdb_id}: Interface between {pn_unit_1} and {pn_unit_2} in {pdb_id} not found or multiple found"
            assert (
                interface_df["involves_loi"].iloc[0] == involves_loi
            ), f"{pdb_id}: LOI involvement for interface between {pn_unit_1} and {pn_unit_2} in {pdb_id} is incorrect"
            assert (
                interface_df["involves_covalent_modification"].iloc[0] == involves_covalent_modification
            ), f"{pdb_id}: Covalent modification involvement for interface between {pn_unit_1} and {pn_unit_2} in {pdb_id} is incorrect"
            assert (
                interface_df["involves_metal"].iloc[0] == involves_metal
            ), f"{pdb_id}: Metal involvement for interface between {pn_unit_1} and {pn_unit_2} in {pdb_id} is incorrect"


if __name__ == "__main__":
    pytest.main(["-v", "-x", "--log-cli-level=WARNING", __file__])
