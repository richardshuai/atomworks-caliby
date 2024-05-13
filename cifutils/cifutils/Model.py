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
    
    def set_parent(self, parent):
        """Set the parent residue and update the full_id accordingly.

        Arguments:
        - parent - Residue object

        """
        self.parent = parent
        # self.full_id = self.get_full_id()

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
    
    def get_residue(self, chain_id, residue_id):
        """Fetch a single residue based on its identifying information"""
        return self[chain_id][residue_id]