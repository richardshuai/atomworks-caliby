import numpy as np
import py3Dmol
import pytest

from biotite.structure import AtomArray
from cifutils.utils.visualize import view


@pytest.fixture
def sample_atom_array():
    """Create a sample AtomArray for testing."""
    atoms = AtomArray(10)
    atoms.set_annotation("chain_id", ["A"] * 5 + ["B"] * 5)
    atoms.set_annotation("element", [6] * 5 + [7] * 5)  # Carbon and Nitrogen
    atoms.set_annotation("res_id", list(range(1, 11)))
    atoms.set_annotation("res_name", ["ALA"] * 10)
    atoms.set_annotation("atom_name", ["CA"] * 10)
    atoms.coord = np.random.rand(10, 3)
    return atoms


def test_view_basic(sample_atom_array):
    """Test basic functionality of the view function."""
    result = view(sample_atom_array)
    assert isinstance(result, py3Dmol.view)


def test_view_custom_dimensions():
    """Test view function with custom width and height."""
    atoms = AtomArray(5)
    result = view(atoms, width=800, height=600)
    assert isinstance(result, py3Dmol.view)
    # Note: We can't directly check the dimensions of the view object


def test_view_zoom_to_selection(sample_atom_array):
    """Test zooming to a specific selection."""
    result = view(sample_atom_array, zoom_to_selection={"chain": "A"})
    assert isinstance(result, py3Dmol.view)


def test_view_show_unoccupied(sample_atom_array):
    """Test showing unoccupied atoms."""
    sample_atom_array.set_annotation("occupancy", [1.0] * 8 + [0.0] * 2)
    result_hidden = view(sample_atom_array, show_unoccupied=False)
    result_shown = view(sample_atom_array, show_unoccupied=True)
    assert isinstance(result_hidden, py3Dmol.view)
    assert isinstance(result_shown, py3Dmol.view)


def test_view_custom_colors():
    """Test using custom colors for visualization."""
    atoms = AtomArray(5)
    atoms.set_annotation("chain_id", ["A"] * 5)
    custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
    result = view(atoms, colors=custom_colors)
    assert isinstance(result, py3Dmol.view)


def test_view_polymer_types(sample_atom_array):
    """Test visualization of different polymer types."""
    # Modify sample_atom_array to include protein, nucleic acid, and ion
    sample_atom_array.res_name[:3] = ["ALA", "GLY", "SER"]
    sample_atom_array.res_name[3:6] = ["A", "C", "G"]
    sample_atom_array.element[6:] = [11, 12, 13, 14]  # Some metal ions
    result = view(sample_atom_array, min_polymer_size=1)
    assert isinstance(result, py3Dmol.view)


@pytest.mark.parametrize("show_surface", [True, False])
def test_view_surface_option(sample_atom_array, show_surface):
    """Test the show_surface option."""
    result = view(sample_atom_array, show_surface=show_surface)
    assert isinstance(result, py3Dmol.view)


@pytest.mark.parametrize("show_hover", [True, False])
def test_view_hover_option(sample_atom_array, show_hover):
    """Test the show_hover option."""
    result = view(sample_atom_array, show_hover=show_hover)
    assert isinstance(result, py3Dmol.view)


def test_view_invalid_input():
    """Test view function with invalid input."""
    with pytest.raises(AttributeError):
        view("not an AtomArray")


# Additional tests can be added for more specific scenarios or edge cases
