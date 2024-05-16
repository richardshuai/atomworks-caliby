import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb

# The maximum distance between an atom in the repressor and an atom in
# the DNA for them to be considered 'in contact'
THRESHOLD_DISTANCE = 4.0

# Fetch and load structure
pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch("4js1", "bcif"))
structure = pdbx.get_structure(pdbx_file, model=1, use_author_fields=False)

for chain in struc.chain_iter(structure):
    for residue in struc.residue_iter(chain):
        print("Here.")
    
print("done.")