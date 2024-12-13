"""Utility functions to visualize atom arrays with py3Dmol in Jupyter notebooks."""

__all__ = ["view"]

import logging
from itertools import cycle

import biotite.structure as struc
import numpy as np
import py3Dmol
from biotite.structure import AtomArray

from cifutils.constants import ATOMIC_NUMBER_TO_ELEMENT, METAL_ELEMENTS
from cifutils.utils.io_utils import to_cif_string

logger = logging.getLogger("cifutils")

IPD_PYMOL_COLORS = [
    "#888888",  # pymol_gray
    "#FAC72C",  # good_yellow
    "#29B0C1",  # good_teal
    "#AAC32F",  # good_green
    "#EC72A4",  # good_pink
    "#4499E7",  # good_blue
    "#DCDCDC",  # good_gray
    "#E44A3E",  # good_red
    "#65B37C",  # good_light_green
    "#4FB9AF",  # paper_teal
    "#FFE0AC",  # paper_navaho
    "#FFC6B2",  # paper_melon
    "#FFACB7",  # paper_pink
    "#D59AB5",  # paper_purple
    "#9596C6",  # paper_lightblue
    "#6686C5",  # paper_blue
    "#4B5FAA",  # paper_darkblue
    "#222222",  # pymol_black
]

_is_metal = np.vectorize(lambda x: ATOMIC_NUMBER_TO_ELEMENT.get(x, x.capitalize()) in METAL_ELEMENTS)


def view(
    structure: AtomArray,
    *,
    zoom_to_selection: dict[str, int | str] | None = None,
    show_hover: bool = True,
    show_unoccupied: bool = False,
    show_surface: bool = True,
    width: int = 600,
    height: int = 400,
    ligand_linewidth: float = 0.2,
    polymer_sidechain_linewidth: float = 0.05,
    min_polymer_size: int = 1,
    colors: list[str] = IPD_PYMOL_COLORS,
) -> py3Dmol.view:
    """Visualize an AtomArray structure using py3Dmol for display in jupyter notebooks.

    Args:
        - structure (AtomArray): The atomic structure to be visualized.
        - zoom_to_selection (dict[str, int | str] | None, optional): A dictionary specifying the
            selection to zoom into. Defaults to None. Here are some examples:
                - `{'serial': 35}` - will zoom to the atom with index 35 in the atom array
                - `{'chain': 'A', 'resi': 35}` - will zoom to the residue id 35 in chain A
                - `{'chain': 'C'} - will zoom to the entire chain C
            !WARNING! If the selection is wrong, the visualization will be empty.
        - show_hover (bool, optional): Whether to enable hover functionality to display atom details.
            Defaults to True.
        - show_unoccupied (bool, optional): Whether to show unoccupied atoms. Defaults to False.
        - show_surface (bool, optional): Whether to show the surface. Defaults to False.
        - width (int, optional): The width of the visualization window. Defaults to 400.
        - height (int, optional): The height of the visualization window. Defaults to 300.
        - ligand_linewidth (float, optional): The linewidth for ligand representation. Defaults to 0.2.
        - polymer_sidechain_linewidth (float, optional): The linewidth for polymer sidechain representation. Defaults to 0.05.
        - min_polymer_size (int, optional): The minimum size for a chain to be displayed as a polymer. Defaults to 1.
        - colors (list[str], optional): A list of colors to cycle through for different chains. Defaults to IPD_PYMOL_COLORS.

    Returns:
        py3Dmol.view: The py3Dmol view object for the structure visualization.
    """

    # Initialize the py3Dmol view with specified width and height
    view = py3Dmol.view(width=width, height=height)

    # Handle unoccupied atoms
    if not show_unoccupied and ("occupancy" in structure.get_annotation_categories()):
        structure = structure[structure.occupancy > 0]

    # Convert the structure to a temporary CIF string for interacting with py3Dmol
    _tmp_cif_str = to_cif_string(structure, _allow_ambiguous_bond_annotations=True)
    # ... add the structure model to the view in mmCIF format
    view.addModel(_tmp_cif_str, "structure", format="mmcif")

    # Get the chain IDs from the structure
    chain_ids = struc.get_chains(structure)

    # Iterate over each chain and assign styles based on the type of polymer
    for chain_id, color in zip(chain_ids, cycle(colors)):
        is_protein = np.all(
            struc.filter_polymer(
                structure[structure.chain_id == chain_id], pol_type="peptide", min_size=min_polymer_size
            )
            & struc.filter_amino_acids(structure[structure.chain_id == chain_id])
        )
        is_nucleic = np.any(
            struc.filter_polymer(
                structure[structure.chain_id == chain_id], pol_type="nucleotide", min_size=min_polymer_size
            )
            & struc.filter_nucleotides(structure[structure.chain_id == chain_id])
        )
        is_ion = np.all(_is_metal(structure[structure.chain_id == chain_id].element))

        if is_protein or is_nucleic:
            # Apply protein or nucleic acid style
            view.setStyle(
                {"chain": chain_id},
                {
                    "cartoon": {"color": color, "arrows": True},
                    "stick": {"radius": polymer_sidechain_linewidth, "style": "outline"},
                },
            )
        elif is_ion:
            # Apply ion style
            view.setStyle(
                {"chain": chain_id},
                {"sphere": {"scale": 0.8}},
            )
        else:
            # Apply ligand style
            # ... first, set the style for carbon atoms colored by chain
            view.setStyle(
                {"chain": chain_id, "elem": "C"},
                {"stick": {"color": color, "radius": ligand_linewidth}},
            )
            # ... then, set the style for all other atoms based on the element
            view.setStyle(
                {"chain": chain_id, "not": {"elem": "C"}},
                {"stick": {"colorscheme": "element", "radius": ligand_linewidth}},
            )

    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.4, "color": "gray"})

    # Add hover functionality to display atom details on hover
    if show_hover:
        js_script = """function(atom,viewer) {
                    if(!atom.label) {
                        atom.label = viewer.addLabel(
                            atom.chain + ':' +
                            atom.resn + '(' + atom.resi + '):' +
                            atom.atom + '(idx' + atom.serial + ')',
                            {position: atom, backgroundColor:"white", fontColor:"black"}
                        );
                    }
                }"""
        view.setHoverable(
            {},
            True,
            js_script,
            """function(atom,viewer) {
                    if(atom.label) {
                        viewer.removeLabel(atom.label);
                        delete atom.label;
                    }
                    }""",
        )

    # Zoom to the entire structure or to a specific selection if provided
    view.zoomTo()
    if zoom_to_selection is not None:
        view.zoomTo(zoom_to_selection)

    return view
