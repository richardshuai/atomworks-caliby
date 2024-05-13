""" Base class for Chain and Residue classes. Modified from BioPython's Entity class. """


class MultiChildComponent():
    """ Simple container that defines basic fields and functions for Chain and Residue classes. """

    def __init__(self, id):
        """Initialize the wrapper class"""
        self.id = id # Unique id within the parent object
        self.full_id = None # Globalley unique id (assuming the parent is a Structure object)
        self.parent = None
        self.child_list = []
        self.child_dict = {}
    
    # Private methods
    
    def _reset_full_id(self):
        """Reset the full_id (PRIVATE).
        
        Resets the full_id of this entity and recursively all of its children based on their ID.
        """
        for child in self:
            try:
                child._reset_full_id()
            except AttributeError:
                pass
        self.full_id = self._generate_full_id()


    def _generate_full_id(self):
        """Generate full_id (PRIVATE).

        Generate the full_id of the MultiChildComponent based on its
        Id and the IDs of the parents.
        """
        component_id = self.id
        parts = [component_id]
        parent = self.parent
        while parent is not None:
            component_id = parent.id
            parts.append(component_id)
            parent = parent.parent
        parts.reverse()
        return tuple(parts)

    # Overridden methods

    def __len__(self):
        """Return the number of children."""
        return len(self.child_list)

    def __iter__(self):
        """Iterate over children."""
        yield from self.child_list
    
    def __hash__(self):
        """Hash method to allow uniqueness (set)."""
        return hash(self.full_id)
    
    def __getitem__(self, id):
        """Return the child with given id."""
        return self.child_dict[id]
    
    def __contains__(self, id):
        """Check if there is a child element with the given id."""
        return id in self.child_dict
    
    def __delitem__(self, id):
        """Remove a child."""
        return self.detach_child(id)
    
    # Generic id-based comparison methods considers all parents as well as children
    # Works for all MultiChildComponents - Atoms have comparable custom operators
    def __eq__(self, other):
        """Test for equality. This compares full_id including the IDs of all parents."""
        if isinstance(other, type(self)):
            if self.parent is None:
                return self.id == other.id
            else:
                return self.full_id[1:] == other.full_id[1:]
        else:
            return NotImplemented
    
    # Public methods

    def set_parent(self, parent):
        """Set the parent and update the full_id accordingly."""
        self.parent = parent
        self._reset_full_id()

    def has_id(self, id):
        """Check if a child with given id exists."""
        return id in self.child_dict
    
    def get_full_id(self):
        """Return the full id.

        TODO: Rewrite this description.
        The full id is a tuple containing all id's starting from
        the top object (Structure) down to the current object. A full id for
        a Residue object e.g. is something like:

        ("1abc", 0, "A", (" ", 10, "A"))

        This corresponds to:

        Structure with id "1abc"
        Model with id 0
        Chain with id "A"
        Residue with id (" ", 10, "A")

        The Residue id indicates that the residue is not a hetero-residue
        (or a water) because it has a blank hetero field, that its sequence
        identifier is 10 and its insertion code "A".
        """
        if self.full_id is None:
            self.full_id = self._generate_full_id()
        return self.full_id
    
    def add_child(self, child):
        """Add a single child to the MultiChildComponent"""
        assert not self.has_id(child.id), "Child ID already exists within MultiChildComponent"
        child.set_parent(self)
        self.child_list.append(child)
        self.child_dict[child.id] = child

    def add_multiple_children(self, children):
        """Add a child or multiple children to the MultiChildComponent"""
        if isinstance(children, list):
            for child_item in children:
                self.add_child(child_item)
        elif isinstance(children, dict):
            for child_id, child_item in children.items():
                assert child_id == child_item.id, "Child ID does not match dictionary key"
                self.add_child(child_item)
        else:
            # Raise an error
            raise TypeError("Children must be a list or a dictionary mapping child_id to child objects")

    def detach_child(self, id):
        """Remove a child."""
        child = self.child_dict[id]
        child.parent = None
        del self.child_dict[id]
        self.child_list.remove(child)