"""Chain class, the main representation of collections of amino acids, nucleic acid bases, and single ligands.
Modified from BioPython's Chain class."""

from cifutils.MultiChildComponent import MultiChildComponent

class Chain(MultiChildComponent):
    """ Defines a Chain class.
    
    Chain objects contain information about their type, as well as maintaing a list of their respective polymers or ligand(s)
    """
    
    def __init__(self, 
        id, # Chain asymmetric ID (asym_id)
        entity_id = None, # ID mapping to unique CIF entity
        entity_details = None, # Details about the entity
        pdb_strand_id = None,
        # For polymers this is the entity type from the `_entity_poly` category. 
        # Namely, "other", "polydeoxyribonucleotide", "polypeptide(D)", "polypeptide(L)", "polyribonucleotide", "plysaccharide(D)", "polysaccharide(L)"
        # For non-polymers, this is set to "nonpoly"
        type = None, 
        canonical_sequence = None, # Sequence of monomers without any modified or non-standard amino acids
        non_canonical_sequence = None, # Sequence of monomers with modified or non-standard amino acids (e.g., MSE)
        symmetric_id = None, # ID mapping to unique symmetrical copy of the chain
    ):
        self.entity_id = entity_id
        self.entity_details = entity_details
        self.pdb_strand_id = pdb_strand_id
        self.type = type
        self.canonical_sequence = canonical_sequence
        self.non_canonical_sequence = non_canonical_sequence
        self.symmetric_id = symmetric_id
        MultiChildComponent.__init__(self, id)

    # Overridden methods
    
    # Public methods
    
    def get_residue(self, id):
        """Fetch a single residue based on its sequence id"""
        return self[id]

    def get_residues(self):
        """Returns residues as an iterator"""
        yield from self
    
    def add_residue(self, residue):
        """Atom a single residue to both the residue list and residue dictionary"""
        MultiChildComponent.add_child(self, residue)
    
    def add_multiple_residues(self, residue):
        """Add a list or dictionary of residues"""
        MultiChildComponent.add_multiple_children(self, residue)
    
    def get_atom(self, residue_id, atom_id):
        """Fetch a single atom based on its identifying information"""
        return self[residue_id][atom_id]
    
    def get_atoms(self):
        """Returns all atoms as an iterator"""
        for residue in self:
            yield from residue.get_atoms()
    
    def add_inter_residue_bond(self, bond):
        """Add a single inter-residue bond"""
        MultiChildComponent.add_bond(self, bond)
        bond.type = "inter_residue"
    
    def add_inter_residue_bonds(self, bonds):
        """Add a list of inter-residue bonds"""
        MultiChildComponent.add_multiple_bonds(self, bonds)
        for bond in bonds:
            bond.type = "inter_residue"
    
    def get_inter_residue_bonds(self):
        """Returns inter-residue bonds as an iterator"""
        yield from self.component_level_bond_list

    def get_intra_residue_bonds(self):
        """Returns intra-residue bonds as an iterator"""
        for residue in self:
            yield from residue.get_intra_residue_bonds()
    
    def remove_inter_residue_bond(self, atom_a, atom_b):
        """Remove a single inter-residue bond"""
        MultiChildComponent.remove_bond(self, atom_a, atom_b)