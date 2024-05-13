import collections

Bond = collections.namedtuple('Bond', [
    'a','b', # names of atoms forming the bond (str)
    'is_aromatic', # is the bond aromatic? (bool)
    'in_ring', # is the bond in a ring? (bool)
    'order', # bond order (int)
    'intra_residue', # is the bond intra-residue? (bool)
    'length' # reference bond length from openbabel (float)
])

Residue = collections.namedtuple('Residue', [
    'name',
    'atoms',
    'bonds',
    'automorphisms',
    'chirals',
    'planars',
    'alternative_residue_ids'
])

Chain = collections.namedtuple('Chain', [
    'id',
    'type',
    'canonical_sequence',
    'non_canonical_sequence',
    'residues',
    'atoms',
    'bonds',
    'chirals',
    'planars',
    'automorphisms'
])


# ============================================================