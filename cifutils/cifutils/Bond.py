
"""Bond class, used to represent a bond between two atoms. Invariant to the order of the atoms."""

class Bond:
    def __init__(self, atom_a, atom_b, is_aromatic=False, in_ring=False, order=1, length=0.0, type=None):
        self._atom_a = atom_a
        self._atom_b = atom_b
        self.is_aromatic = is_aromatic
        self.in_ring = in_ring
        self.order = order
        self.length = length
        self.type = type  # One of 'intra_residue', 'inter_residue', 'inter_chain'

    @property
    def atom_a(self):
        return self._atom_a

    @atom_a.setter
    def atom_a(self, value):
        self._atom_a = value

    @property
    def atom_b(self):
        return self._atom_b

    @atom_b.setter
    def atom_b(self, value):
        self._atom_b = value

    @property
    def id(self):
        return tuple(sorted([self._atom_a.get_full_id(), self._atom_b.get_full_id()]))

    def __eq__(self, other):
        if isinstance(other, Bond):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)