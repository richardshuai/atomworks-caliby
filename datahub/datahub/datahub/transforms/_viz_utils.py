"""Utility functions to visualize atom arrays with py3Dmol in Jupyter notebooks."""

import io
from itertools import cycle

import biotite.structure as struc
import numpy as np
import py3Dmol
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from cifutils.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER

ATOMIC_NUMBER_TO_ELEMENT = {str(v): k for k, v in ELEMENT_NAME_TO_ATOMIC_NUMBER.items()}

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


def to_cif(structure: AtomArray) -> str:
    """Convert an AtomArray structure to a CIF formatted string.

    Args:
        structure (AtomArray): The atomic structure to be converted.

    Returns:
        str: The CIF formatted string representation of the structure.
    """
    structure = structure.copy()
    buffer = io.StringIO()
    cif_file = pdbx.CIFFile()
    structure.element = np.vectorize(lambda x: ATOMIC_NUMBER_TO_ELEMENT.get(x, x))(structure.element)
    pdbx.set_structure(cif_file, structure, data_block="structure", include_bonds=True)
    cif_file.write(buffer)
    return buffer.getvalue()


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
        - colors (list[str], optional): A list of colors to cycle through for different chains. Defaults to IPD_PYMOL_COLORS.

    Returns:
        py3Dmol.view: The py3Dmol view object for the structure visualization.
    """

    # Initialize the py3Dmol view with specified width and height
    view = py3Dmol.view(width=width, height=height)

    # Handle unoccupied atoms
    if not show_unoccupied:
        structure = structure[structure.occupancy > 0]

    # Convert the structure to a temporary CIF string for interacting with py3Dmol
    _tmp_cif_str = to_cif(structure)
    # ... add the structure model to the view in mmCIF format
    view.addModel(_tmp_cif_str, "structure", format="mmcif")

    # Get the chain IDs from the structure
    chain_ids = struc.get_chains(structure)

    # Iterate over each chain and assign styles based on the type of polymer
    for chain_id, color in zip(chain_ids, cycle(colors)):
        is_protein = np.all(
            struc.filter_polymer(structure[structure.chain_id == chain_id], pol_type="peptide", min_size=1)
        )
        is_nucleic = np.any(
            struc.filter_polymer(structure[structure.chain_id == chain_id], pol_type="nucleotide", min_size=1)
        )
        is_ion = len(np.unique(structure[structure.chain_id == chain_id].element)) == 1

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
            view.setStyle(
                {"chain": chain_id, "elem": "C"},
                {"stick": {"radius": ligand_linewidth, "style": "outline", "color": color}},
            )
            view.setStyle(
                {"chain": chain_id, "not": {"elem": "C"}},
                {"stick": {"radius": ligand_linewidth, "style": "outline", "colorscheme": "element"}},
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
