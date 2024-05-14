import collections

Bond = collections.namedtuple('Bond', [
    'id', # unique bond ID (tuple of (a.id, b.id))
    'a','b', # names of atoms forming the bond (str)
    'is_aromatic', # is the bond aromatic? (bool)
    'in_ring', # is the bond in a ring? (bool)
    'order', # bond order (int)
    'intra_residue', # is the bond intra-residue? (bool)
    'length' # reference bond length from openbabel (float)
])


# ============================================================