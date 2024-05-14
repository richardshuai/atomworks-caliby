"""Atom class, used to represent atoms in a molecule. Modified from BioPython's Atom class."""

import numpy as np
from cifutils.Residue import Residue

class Atom:
    """ Define atom class.

    """

    def __init__(
        self,
        id: str,
        xyz: np.ndarray,
        occupancy: float,
        altloc: str,
        bfactor,
        leaving_atom_flag,
        leaving_group,
        parent_heavy_atom,
        element,
        is_metal,
        charge,
        hyb,
        nhyd,
        hvydeg,
        align,
        hetero,
        parent: Residue = None
    ):
        self.id = id # e.g., CA
        self.name = id # e.g., CA
        self.xyz = xyz # list of x, y, z coordinates
        self.occupancy = occupancy
        self.altloc = altloc
        self.bfactor = bfactor
        self.leaving_atom_flag = leaving_atom_flag
        self.leaving_group = leaving_group
        self.parent_heavy_atom = parent_heavy_atom
        self.element = element
        self.is_metal = is_metal
        self.charge = charge
        self.hyb = hyb
        self.nhyd = nhyd
        self.hvydeg = hvydeg
        self.align = align
        self.hetero = hetero
        self.full_id = None
        self.parent = parent
    
    # Overridden methods
    
    # standard across different objects and allows direct comparison
    def __eq__(self, other):
        """Test equality."""
        if isinstance(other, Atom):
            return self.full_id[1:] == other.full_id[1:] # We skip the first ID, which is the structure ID, so we can compare across structures
        else:
            return NotImplemented

    def __ne__(self, other):
        """Test inequality."""
        if isinstance(other, Atom):
            return self.full_id[1:] != other.full_id[1:]
        else:
            return NotImplemented
    
    # Hash method to allow uniqueness (set)
    def __hash__(self):
        """Return atom full identifier."""
        return hash(self.full_id or self.get_full_id())
    
    def _reset_full_id(self):
        """Reset the full id."""
        self.full_id = self.get_full_id(recompute = True)
    
    
    # Public methods

    def set_parent(self, parent):
        """Set the parent residue and update the full_id accordingly.

        Arguments:
        - parent - Residue object
        """
        self.parent = parent
        self.full_id = self.get_full_id(recompute = True)

    def get_parent_chain_id(self):
        """Return the parent chain id."""
        return self.parent.parent.id
    
    def get_parent_residue_name(self):
        """Return the parent residue name."""
        return self.parent.name
    
    def get_parent_residue_id(self):
        """Return the parent residue sequence number."""
        return self.parent.id

    def get_full_id(self, recompute = False):
        """Return the full id of the atom.

        The full id of an atom is a tuple used to uniquely identify
        the atom and consists of the following elements:
        (structure id, model id, chain id, residue id, atom id)
        """
        if not recompute and self.full_id is not None:
            return self.full_id
        else:
            try:
                return self.parent.get_full_id() +  (self.id,) #((self.name, self.altloc),)
            except AttributeError:
                return (None, None, None, None, self.name)
    
    def transform(self, rot, tran):
        """Apply rotation and translation to the atomic coordinates.

        :param rot: A right multiplying rotation matrix
        :type rot: 3x3 NumPy array

        :param tran: the translation vector
        :type tran: size 3 NumPy array

        Examples
        --------
        This is an incomplete but illustrative example::

            from numpy import pi, array
            from Bio.PDB.vectors import Vector, rotmat
            rotation = rotmat(pi, Vector(1, 0, 0))
            translation = array((0, 0, 1), 'f')
            atom.transform(rotation, translation)

        """
        self.coord = np.dot(self.coord, rot) + tran