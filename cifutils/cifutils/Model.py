"""Model class, which holds one experimentally viable arrangement of chains. 
Modified from BioPython's Model class."""

from cifutils.MultiChildComponent import MultiChildComponent

class Model(MultiChildComponent):
    """ Defines a Model class.
    
    Model classes contain one possible model from the mmCIF file, comprising one or more chains. Most files, except NMR files, have one model.
    """
    
    def __init__(self, 
        id, # Numeric
    ):
        MultiChildComponent.__init__(self, id)

    # Public methods
    
    def add_chain(self, chain):
        """Atom a single chain to both the chain list and chain dictionary"""
        MultiChildComponent.add_child(self, chain)
    
    def add_multiple_chains(self, chain):
        """Add a list or dictionary of chains"""
        MultiChildComponent.add_multiple_children(self, chain)
    
    def get_chains_by_entity_id(self, entity_id):
        """Return a list of chains by entity_id"""
        return [chain for chain in self if chain.entity_id == entity_id]
    
    def get_chain(self, id):
        """Fetch a single chain based on its id"""
        return self[id]
    
    def get_chains(self):
        """Returns chains as an iterator"""
        yield from self
    
    def get_atom(self, chain_id, residue_id, atom_id):
        """Fetch a single atom based on its identifying information"""
        return self[chain_id][residue_id][atom_id]
    
    def get_atoms(self):
        """Returns all atoms as an iterator"""
        for chain in self:
            for residue in chain:
                yield from residue.get_atoms()
    
    def remove_atom(self, chain_id, residue_id, atom_id):
        """Remove a single atom based on its identifying information"""
        del self[chain_id][residue_id][atom_id]
    
    def get_residue(self, chain_id, residue_id):
        """Fetch a single residue based on its identifying information"""
        return self[chain_id][residue_id]
    
    def add_inter_chain_bond(self, bond):
        """Add a single inter-chain bond"""
        MultiChildComponent.add_bond(self, bond)
        bond.type = 'inter_chain'
    
    def add_inter_chain_bonds(self, bonds):
        """Add a list of inter-chain bonds"""
        MultiChildComponent.add_multiple_bonds(self, bonds)
        for bond in bonds:
            bond.type = 'inter_chain'
    
    def get_inter_chain_bonds(self):
        """Returns inter-chain bonds as an iterator"""
        yield from self.component_level_bond_list
    
    def get_inter_residue_bonds(self):
        """Returns inter-residue bonds as an iterator"""
        for chain in self:
            yield from chain.get_inter_residue_bonds()
    
    def get_intra_residue_bonds(self):
        """Returns intra-residue bonds as an iterator"""
        for chain in self:
            yield from chain.get_intra_residue_bonds()
    
    def remove_inter_chain_bond(self, atom_a, atom_b):
        """Remove a single inter-chain bond"""
        MultiChildComponent.remove_bond(self, atom_a, atom_b)
    
    def get_bonds(self):
        """Returns all bonds as an iterator"""
        yield from self.get_inter_chain_bonds()
        yield from self.get_inter_residue_bonds()
        yield from self.get_intra_residue_bonds()