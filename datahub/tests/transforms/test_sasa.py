from typing import Any

import numpy as np
import pytest

# Assuming CalculateSASA is in transforms/sasa.py
from datahub.transforms.sasa import CalculateSASA
from datahub.utils.testing import cached_parse

# Define test cases
# (all values for radii and SASA are from "WhatIF")
SASA_TEST_CASES = [
    {
        "pdb_id": "1fu2",  # (multi-chain protein)
        "probe_radius": 1.4,  # default radius for water as a solvent
        "atom_radii": "ProtOr",
        "point_number": 100,
        "spot_checks": [
            {"atom_name": "H"},  # should be nan
            {"atom_name": "ZN"},  # should be nan
            {"atom_name": "N"},  # should be not nan
        ],
    },
    {
        "pdb_id": "3p42",  # testing protein with certain NaN coordinates
        "probe_radius": 1.4,  # default radius for water as a solvent
        "atom_radii": "ProtOr",
        "point_number": 100,
        "spot_checks": [
            {"atom_name": "H"},  # should be nan
            {"atom_name": "N"},  # should be not nan
        ],
    },
]


@pytest.mark.parametrize("test_case", SASA_TEST_CASES)
def test_calculate_sasa(test_case: dict[str, Any]):
    """
    Test the CalculateSASA transform using a multi-chain protein.
    Checks:
    - The SASA of atoms that should not have SASA calculated are NaN
    - The SASA of heavy atoms that should have SASA are >=0
    """

    # Load the atom array
    data = cached_parse(test_case["pdb_id"])

    # Apply the transform
    transform = CalculateSASA(
        probe_radius=test_case["probe_radius"],
        atom_radii=test_case["atom_radii"],
        point_number=test_case["point_number"],
    )
    data = transform(data)

    # Check SASA values of specific atoms

    for spot_check in test_case["spot_checks"]:
        atom_mask = data["atom_array"].atom_name == spot_check["atom_name"]
        if data["atom_array"][atom_mask].atom_name[0] in (["ZN", "NA", "H"]):
            assert np.isnan(data["atom_array"][atom_mask].sasa).all()
        else:
            valid_mask = ~np.isnan(data["atom_array"][atom_mask].coord).all(axis=1)
            assert np.all(data["atom_array"][atom_mask][valid_mask].sasa >= 0)
