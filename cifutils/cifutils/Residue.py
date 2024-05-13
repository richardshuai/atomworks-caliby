"""Residue class, the main representation of amino acids, nucleic acid bases, and ligands.
Modified from BioPython's Residue class."""

import numpy as np
from cifutils.data_structures import Chain
from cifutils.MultiChildComponent import MultiChildComponent

class Residue(MultiChildComponent):
    """ Defines a Residue class.
    
    Residue objects contain information about their respective amino acid, nucleic base, or ligand, as well as maintaining a list of constituent Atom objects.
    """
    
    def __init__(self, 
        id, # The residue number in the parent chain's sequence, e.g., 25
        name = None, # e.g., LYS
        intra_residue_bonds = None, # List of Bond objects representing bonds between atoms within the residue
        automorphisms = None,
        chirals = None,
        planars = None,
        alternative_residue_names = [],
        hetero = False,
        unmatched_heavy_atom = False
    ):
        self.name = name
        self.intra_residue_bonds = intra_residue_bonds
        self.automorphisms = automorphisms
        self.chirals = chirals
        self.planars = planars
        self.alternative_residue_names = alternative_residue_names
        self.hetero = hetero
        self.unmatched_heavy_atom = unmatched_heavy_atom
        MultiChildComponent.__init__(self, id)

    # Overridden methods
    
    # Public methods
    
    def get_full_id(self):
        """Return the full id of the residue.
        
        The full id of a residue is a tuple of (chain_id, id [sequence index], name [residue name])
        We need the residue name to handle cases of sequence heterogeneity.
        """
    def set_parent(self, parent):
        """Set the parent residue and update the full_id accordingly.

        Arguments:
        - parent - Residue object

        """
        self.parent = parent
    
    def get_parent_chain_id(self):
        """Return the parent chain id."""
        return self.parent.id
    
    def get_atom(self, id):
        """Fetch a single atom based on its id"""
        return self[id]

    def get_atoms(self):
        """Returns atoms as an iterator"""
        yield from self
    
    def add_atom(self, atom):
        """Atom a single atom to both the atom list and atom dictionary"""
        MultiChildComponent.add_child(self, atom)
    
    def add_multiple_atoms(self, atoms):
        """Add a list or dictionary of atoms"""
        MultiChildComponent.add_multiple_children(self, atoms)