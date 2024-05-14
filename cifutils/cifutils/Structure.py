"""Defines a Structure class, which holds one or several experimentally feasible Models. 
Modified from BioPython's Structure class."""

from cifutils.MultiChildComponent import MultiChildComponent

class Structure(MultiChildComponent):
    """ Defines a Structure class.
    
    Structures classes hold at least one experimentally feasible Model.
    Structure classes also contain experiment metadata.
    """
    
    def __init__(self, 
        id, # Numeric
        method = None, # E.g., NMR, Cryo-EM, X-ray Diffraction
        initial_deposition_date = None, # Date of initial deposition
        resolution = None # Experimental resolution, adjusted for method type
    ):
        self.method = method
        self.initial_deposition_date = initial_deposition_date
        self.resolution = resolution
        MultiChildComponent.__init__(self, id)

    # Overridden methods
    
    # Public methods
    
    def get_model(self, model_id):
        """Fetch a single model based on its id"""
        return self[model_id]
    
    def get_models(self):
        """Returns models as an iterator"""
        yield from self

    def add_model(self, model):
        """Atom a single model to both the model list and model dictionary"""
        MultiChildComponent.add_child(self, model)
    
    def add_multiple_models(self, model):
        """Add a list or dictionary of models"""
        MultiChildComponent.add_multiple_children(self, model)