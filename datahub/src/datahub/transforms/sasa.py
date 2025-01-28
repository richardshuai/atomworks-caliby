from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from datahub.transforms._checks import (
    check_atom_array_annotation,
    check_contains_keys,
    check_is_instance,
)
from datahub.transforms.base import Transform


def calculate_atomwise_sasa(
    atom_array: AtomArray, probe_radius: float = 1.4, atom_radii: str | np.ndarray = "ProtOr", point_number: int = 100
) -> np.ndarray:
    """
    Calculate the SASA for each atom in `atom_array`, excluding those
    with NaN coordinates. The output will have the same length as the
    input AtomArray, with NaN values for excluded (invalid) atoms.

     Args:
        probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
        atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr". "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions or certain atoms with charges
        point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.

    """
    # 1) Create a boolean vector for valid atoms (no NaNs in their coordinates)
    has_resolved_coordinates = ~np.isnan(atom_array.coord).any(axis=-1)

    # 2) Slice the array to keep only valid atoms
    valid_atom_array = atom_array[has_resolved_coordinates]

    # 3) Compute SASA on only the valid atoms
    valid_sasa = struc.sasa(
        valid_atom_array, probe_radius=probe_radius, vdw_radii=atom_radii, point_number=point_number
    )

    # 4) Create a full-length result array, fill with NaNs
    full_sasa = np.full(atom_array.array_length(), np.nan, dtype=float)

    # 5) Place valid SASA values back into their original positions
    full_sasa[has_resolved_coordinates] = valid_sasa

    return full_sasa


class CalculateSASA(Transform):
    """Transform for calculating Solvent-Accessible Surface Area (SASA) for each atom in an AtomArray."""

    def __init__(
        self,
        probe_radius: float = 1.4,
        atom_radii: str | np.ndarray = "ProtOr",
        point_number: int = 100,
    ):
        """
        Initialize the CalculateSASA transform.

        Args:
            probe_radius (float, optional): Van-der-Waals radius of the probe in Angstrom. Defaults to 1.4 (for water).
            atom_radii (str | np.ndarray, optional): Atom radii set to use for calculation. Defaults to "ProtOr". "ProtOr" will not get sasa's for hydrogen atoms and some other atoms, like ions or certain atoms with charges
            point_number (int, optional): Number of points in the Shrake-Rupley algorithm to sample for calculating SASA. Defaults to 100.
        """
        self.probe_radius = probe_radius
        self.atom_radii = atom_radii
        self.point_number = point_number

    def check_input(self, data: dict[str, Any]) -> None:
        check_contains_keys(data, ["atom_array"])
        check_is_instance(data, "atom_array", AtomArray)
        check_atom_array_annotation(data, ["res_name"])

    def forward(self, data: dict, key_to_add_sasa_to: str = "atom_array") -> dict:
        """
        Calculates SASA and adds it to the data dictionary under the key `atom_array`.
        Args:
            data: dict
                A dictionary containing the input data atomarray.
            key_to_add_sasa_to: str
                The key in the data dictionary to add the SASA values to.

        Returns:
            dict: The data dictionary with SASA values added.
        """
        atom_array: AtomArray = data[key_to_add_sasa_to]
        sasa = calculate_atomwise_sasa(
            atom_array,
            self.probe_radius,
            self.atom_radii,
            self.point_number,
        )
        atom_array.set_annotation("sasa", sasa)
        data[key_to_add_sasa_to] = atom_array
        return data
