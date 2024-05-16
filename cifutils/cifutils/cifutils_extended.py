import os
import json
import gzip
import re
import copy
import random
from openbabel import openbabel
import itertools
from typing import Dict,List
import numpy as np
import pandas as pd
import networkx as nx
import torch
import cifutils.pdbx as pdbx
from cifutils.Structure import Structure
from cifutils.Model import Model
from cifutils.Chain import Chain
from cifutils.Residue import Residue
from cifutils.Atom import Atom
from cifutils.Bond import Bond
from cifutils.pdbx.reader.PdbxReader import PdbxReader
import cifutils.obutils as obutils


def parse_pdb_ligand(cifname : str) -> Dict:
    '''Parse a single molecule from the PDB-Ligands set'''
    data = []
    with open(cifname,'r') as cif:
        reader = PdbxReader(cif)
        reader.read(data)
    data = data[0]
    chem_comp_atom = data.getObj('chem_comp_atom')
    rows = chem_comp_atom.getRowList()

    # Parse atom names
    idx = chem_comp_atom.getIndex('atom_id')
    atom_id = np.array([r[idx] for r in rows])
    
    # Parse element symbols
    idx = chem_comp_atom.getIndex('type_symbol')
    symbol = np.array([r[idx] for r in rows])

    # Parse leaving flags
    idx = chem_comp_atom.getIndex('pdbx_leaving_atom_flag')
    leaving = [r[idx] for r in rows]
    leaving = np.array([True if flag=='Y' else False for flag in leaving], dtype=bool)

    # Atom name alignment offset in PDB atom field
    idx = chem_comp_atom.getIndex('pdbx_align')
    pdbx_align = np.array([int(r[idx]) for r in rows])
    
    # parse xyz coordinates
    i = chem_comp_atom.getIndex('model_Cartn_x')
    j = chem_comp_atom.getIndex('model_Cartn_y')
    k = chem_comp_atom.getIndex('model_Cartn_z')
    xyz = [(r[i],r[j],r[k]) for r in rows]
    xyz = np.array([[float(c) if c!='?' else np.nan for c in p] for p in xyz])

    return {'atom_id': atom_id, 'leaving': leaving, 'symbol': symbol, 'pdbx_align': pdbx_align, 'xyz': xyz}

# ============================================================
class CIFParser:
    
    def __init__(self, skip_res : List[str] = None):
        # Parse pre-compiled library of all residues observed in the PDB
        DIR = os.path.dirname(__file__)
        with gzip.open(f'{DIR}/ligands.json.gz','rt') as file:
            self.precomputed_ccd_data = json.load(file)
            # Residues to be ignored during parsing are deleted from the precomputed library
            if skip_res is not None:
                self.precomputed_ccd_data = {k: v for k, v in self.precomputed_ccd_data.items() if k not in skip_res}
            
        # Parse quasi-symmetric groups table
        df = pd.read_csv(f'{DIR}/data/quasisym.csv')
        df.indices = df.indices.apply(lambda x : [int(xi) for xi in x.split(',')])
        df['matcher'] = df.apply(lambda x : openbabel.OBSmartsPattern(), axis=1)
        df.apply(lambda x : x.matcher.Init(x.smarts), axis=1)
        self.quasisym = {smarts:(matcher,torch.tensor(indices))
                         for smarts,matcher,indices 
                         in zip(df.smarts,df.matcher,df.indices)}
        
        # Parse periodic table
        with open(f'{DIR}/data/elements.txt','r') as f:
            self.i2a = [l.strip().split()[:2] for l in f.readlines()]
            self.i2a = {int(i):a for i,a in self.i2a}

        # initialize dictionary to store residues information dictionaries
        self.residue_name_to_data_map = {}
    

    def initialize_residue_from_name(self, id : str, name : str, parent: Chain = None) -> Residue:
        """
        Initializes and pre-populates with Atoms and Bonds a Residue object from a given residue name.
        Loads bond and atom information from Open Babel.

        Parameters:
        id (int): The ID of the residue.
        name (str): The name of the residue.
        parent (Chain): The parent Chain object to which the Residue belongs. If provided, the Residue will be added to the parent Chain.

        Returns:
        Residue: The initialized Residue object, or None if the residue data is not available in the library.
        """

        ccd_data = self.precomputed_ccd_data.get(name)
        # If we didn't include the result in our library (e.g., we skipped it), we return None, and skip the residue downstream
        if ccd_data is None:
            return None
        
        # If we haven't already loaded the residue, we load it now
        if name not in self.residue_name_to_data_map:
            self.residue_name_to_data_map[name] = self.load_residue_details_and_atoms_from_open_babel(
                sdfstring=ccd_data['sdf'],
                atom_id=ccd_data['atom_id'],
                leaving=ccd_data['leaving'],
                pdbx_align=ccd_data['pdbx_align']
            )
        
        residue_template = self.residue_name_to_data_map[name]
        
        new_residue = Residue(
            id=id,
            name=residue_template['name'],
            automorphisms=residue_template['automorphisms'],
            chirals=residue_template['chirals'],
            planars=residue_template['planars']
        )
        new_residue.set_parent(parent)

        # Create and add atoms to the new Residue
        atoms = {
            aname: Atom(
                id=a['id'],
                leaving_atom_flag=a['leaving_atom_flag'],
                leaving_group=a['leaving_group'],
                parent_heavy_atom=a['parent_heavy_atom'],
                element=a['element'],
                is_metal=a['is_metal'],
                charge=a['charge'],
                hyb=a['hyb'],
                nhyd=a['nhyd'],
                hvydeg=a['hvydeg'],
                align=a['align'],
            ) for aname, a in residue_template['atoms'].items()
        }
        new_residue.add_multiple_atoms(atoms)
        
        # Create and add bonds to the new Residue
        bonds = [
            Bond(
                atom_a=atoms[bond['atom_a_id']],
                atom_b=atoms[bond['atom_b_id']],
                is_aromatic=bond['is_aromatic'],
                in_ring=bond['in_ring'],
                order=bond['order'],
                length=bond['length'],
            ) for bond in residue_template['intra_residue_bonds']
        ]
        new_residue.add_intra_residue_bonds(bonds)# Then, add bonds, once the atom parents are set correctly

        return new_residue
        
    def add_quasi_symmetries(self, 
                           obmol : openbabel.OBMol,
                           automorphisms : torch.Tensor) -> torch.Tensor:
        '''add quasisymmetries to automorphisms
        '''

        renum = []
        for smarts,(matcher,indices) in self.quasisym.items():
            res = openbabel.vectorvInt()
            if matcher.Match(obmol,res,0):
                res = torch.tensor(res)[:,indices]-1
                res = res.sort(-1)[0]
                res = torch.unique(res,dim=0)
                for res_i in res:
                    res_i = torch.tensor(list(itertools.permutations(res_i,indices.shape[0])))
                    renum.append(res_i)
                
        if len(renum)<1:
            return automorphisms
        elif len(renum)==1:
            renum = renum[0]
        else:
            random.shuffle(renum)
            renum = renum[:4]
            renum = torch.stack([torch.cat(ijk) for ijk in itertools.product(*renum)])

        L = automorphisms.shape[-1]
        modified = automorphisms[:,None].repeat(1,renum.shape[0],1)
        modified[...,renum[0]]=automorphisms[:,renum]
        modified = modified.reshape(-1,L)
        modified = torch.unique(modified, dim=0)
        
        return modified


    @staticmethod
    def get_leaving_atoms_from_ob_atom(a, leaving, s):
        for b in openbabel.OBAtomAtomIter(a):
            if leaving[b.GetIndex()]==True:
                if b.GetIndex() not in s:
                    s.append(b.GetIndex())
                    CIFParser.get_leaving_atoms_from_ob_atom(b,leaving,s)


    @staticmethod
    def get_leaving_atoms_from_graph(aname, G):

        leaving_group = set()
    
        if G.nodes[aname]['leaving']==True:
            return []

        for m in G.neighbors(aname):
            if G.nodes[m]['leaving']==False:
                continue
            leaving_group.update({m})
            H = G.subgraph(set(G.nodes)-{m})
            ccs = list(nx.connected_components(H))
            if len(ccs)>1:
                for cc in ccs:
                    if aname not in cc:
                        leaving_group.update(cc)

        return list(leaving_group)


    #@staticmethod
    def load_residue_details_and_atoms_from_open_babel(self,
                    sdfstring : str,
                    atom_id : List[str],
                    leaving : List[bool],
                    pdbx_align : List[int],) -> Residue:
        """
        """
        # create molecule from the sdf string
        obmol = openbabel.OBMol()
        obConversion = openbabel.OBConversion()
        obConversion.SetInFormat("sdf")
        obConversion.ReadString(obmol,sdfstring)
        #obmol.DeleteHydrogens()
        
        # correct for pH to get some charged groups
        obmol_ph = openbabel.OBMol(obmol)
        obmol_ph.CorrectForPH()
        obmol_ph.DeleteHydrogens()
        ha_iter = openbabel.OBMolAtomIter(obmol_ph)                
        
        # get atoms and their features (as a dictionary)
        residue_atoms = {}
        for aname,aleaving,align,a in zip(atom_id,leaving,pdbx_align,openbabel.OBMolAtomIter(obmol)):

            # parent heavy atoms
            parent = None
            for b in openbabel.OBAtomAtomIter(a):
                if b.GetAtomicNum()>1:
                    parent = atom_id[b.GetIndex()]
            
            charge = a.GetFormalCharge()
            nhyd = a.ExplicitHydrogenCount()
            if a.GetAtomicNum()>1:
                ha = next(ha_iter)
                charge = ha.GetFormalCharge()
                nhyd = ha.GetTotalDegree()-ha.GetHvyDegree()
            
            # Store the information needed to create the atom
            residue_atoms[aname] = {
                'id': aname, # Atom ID's are their CCD "names"
                'leaving_atom_flag': aleaving,
                'leaving_group': [],
                'parent_heavy_atom': parent,
                'element': a.GetAtomicNum(),
                'is_metal': a.IsMetal(),
                'charge': charge,
                'hyb': a.GetHyb(),
                'nhyd': nhyd,
                'hvydeg': a.GetHvyDegree(),
                'align': align,
            }
            
        # get bonds and their features
        bonds = []
        for b in openbabel.OBMolBondIter(obmol):
            atom_a_name = atom_id[b.GetBeginAtom().GetIndex()]
            atom_b_name = atom_id[b.GetEndAtom().GetIndex()]
            bonds.append(
                {
                    'atom_a_id': atom_a_name,
                    'atom_b_id': atom_b_name,
                    'is_aromatic': b.IsAromatic(),
                    'in_ring': b.IsInRing(),
                    'order': b.GetBondOrder(),
                    'length': b.GetLength(),
                }
            )

        # get automorphisms
        automorphisms = obutils.FindAutomorphisms(obmol, heavy=True)
        
        # add quasi-symmetric groups
        automorphisms = self.add_quasi_symmetries(obmol, automorphisms)
        
        # only retain atoms with alternative mappings
        mask = (automorphisms[:1]==automorphisms).all(dim=0)
        automorphisms = automorphisms[:,~mask]

        # get chirals and planars
        chirals = obutils.GetChirals(obmol, heavy=True)
        planars = obutils.GetPlanars(obmol, heavy=True)

        # add leaving groups to atoms
        G = nx.Graph()
        G.add_nodes_from([(a['id'],{'leaving':a['leaving_atom_flag']}) for a in residue_atoms.values()])
        G.add_edges_from([(bond['atom_a_id'],bond['atom_b_id']) for bond in bonds])
        for k, v in residue_atoms.items():
            v['leaving_group'] = CIFParser.get_leaving_atoms_from_graph(k,G)
        
        # Put everything into a dictionary we will later use to initialize the Residue object
        anames = np.array(atom_id)
        return {
            'name': obmol.GetTitle(),
            'intra_residue_bonds': bonds,
            'automorphisms': anames[automorphisms].tolist(),
            'chirals': anames[chirals].tolist(),
            'planars': anames[planars].tolist(),
            'atoms': residue_atoms
        }

    
    @staticmethod
    def parse_operation_expression(expression : str) -> List:
        '''a function to parse _pdbx_struct_assembly_gen.oper_expression 
        into individual operations'''

        expression = expression.strip('() ')
        operations = []
        for e in expression.split(','):
            e = e.strip()
            pos = e.find('-')
            if pos>0:
                start = int(e[0:pos])
                stop = int(e[pos+1:])
                operations.extend([str(i) for i in range(start,stop+1)])
            else:
                operations.append(e)

        return operations


    @staticmethod
    def parse_assemblies(data : pdbx.reader.PdbxContainers.DataContainer) -> Dict:
        '''parse biological assembly data'''
        
        assembly_data = data.getObj("pdbx_struct_assembly")
        assembly_gen = data.getObj("pdbx_struct_assembly_gen")
        oper_list = data.getObj("pdbx_struct_oper_list")

        if (assembly_data is None) or (assembly_gen is None) or (oper_list is None):
            return {}

        # save all basic transformations in a dictionary
        opers = {}
        for k in range(oper_list.getRowCount()):
            key = oper_list.getValue("id", k)
            val = np.eye(4)
            for i in range(3):
                val[i,3] = float(oper_list.getValue("vector[%d]"%(i+1), k))
                for j in range(3):
                    val[i,j] = float(oper_list.getValue("matrix[%d][%d]"%(i+1,j+1), k))
            opers.update({key:val})

        chains,ids = [],[]
        xforms = []
        #details,method = [],[]

        for index in range(assembly_gen.getRowCount()):

            # Retrieve the assembly_id attribute value for this assembly
            assemblyId = assembly_gen.getValue("assembly_id", index)
            ids.append(assemblyId)

            # Retrieve the operation expression for this assembly from the oper_expression attribute	
            oper_expression = assembly_gen.getValue("oper_expression", index)

            oper_list = [CIFParser.parse_operation_expression(expression) 
                         for expression in re.split('\(|\)', oper_expression) if expression]

            # chain IDs which the transform should be applied to
            chains.append(assembly_gen.getValue("asym_id_list", index).split(','))

            if len(oper_list)==1:
                xforms.append(np.stack([opers[o] for o in oper_list[0]]))
            elif len(oper_list)==2:
                xforms.append(np.stack([opers[o1]@opers[o2] 
                                        for o1 in oper_list[0] 
                                        for o2 in oper_list[1]]))
            else:
                print('Error in processing assembly')           
                return xforms

        # return xforms as a dict {asmb_id:[(chain_id,xform[4,4])]}
        out = {i:[] for i in set(ids)}
        for key,c,x in zip(ids,chains,xforms):
            out[key].extend(itertools.product(c,x))
            
        return out


    def parse_entities(self, data : pdbx.reader.PdbxContainers.DataContainer) -> Dict:
        '''Parse entity data'''

        entities = data.getObj('entity')
        if entities is None:
            return {}

        entity_map = {}
        for row in entities.getRowList():
            entity_id = row[entities.getIndex('id')]
            entity_map[entity_id] = {
                'type': row[entities.getIndex('type')],
                'pdbx_description': row[entities.getIndex('pdbx_description')],
                'formula_weight': row[entities.getIndex('formula_weight')],
                'pdbx_number_of_molecules': row[entities.getIndex('pdbx_number_of_molecules')],
                'pdbx_ec': row[entities.getIndex('pdbx_ec')],
                'pdbx_mutation': row[entities.getIndex('pdbx_mutation')],
                'pdbx_fragment': row[entities.getIndex('pdbx_fragment')],
                'pdbx_detail': row[entities.getIndex('pdbx_detail')]
            }

        return entity_map

    def parse_modified_residues(self, data : pdbx.reader.PdbxContainers.DataContainer) -> Dict:
        '''Parse modified residues data'''
        pdbx_struct_mod_residue = data.getObj('pdbx_struct_mod_residue')
        if pdbx_struct_mod_residue is None:
            return {}
        modified_residues = {(r[pdbx_struct_mod_residue.getIndex('label_comp_id')],
                            r[pdbx_struct_mod_residue.getIndex('parent_comp_id')])
                            for r in pdbx_struct_mod_residue.getRowList()}
        return {k: v for k, v in modified_residues if k != v}

    def parse_polymeric_chains(self, data, model, entity_map):
        '''Parse polymeric chains from the CIF file and add them to the model'''
        pdbx_poly_seq_scheme = data.getObj('pdbx_poly_seq_scheme')
        if pdbx_poly_seq_scheme is not None:
            
            # Initialize Chain objects with mapping asym_id <--> entity_id, pdb_strand_id 
            # Note: We only initialize polymer chains this way; we initialize non-polymers within the atom_site loop
            poly_chains = {}
            for row in pdbx_poly_seq_scheme.getRowList():
                asym_id = row[pdbx_poly_seq_scheme.getIndex('asym_id')]
                if asym_id not in poly_chains:
                    entity_id = row[pdbx_poly_seq_scheme.getIndex('entity_id')]
                    poly_chains[asym_id] = Chain(
                        id = asym_id,
                        entity_id=entity_id,
                        entity_details=entity_map.get(entity_id),
                        pdb_strand_id=row[pdbx_poly_seq_scheme.getIndex('pdb_strand_id')],
                    )
            
            # Add the Chain objects to the Model
            model.add_multiple_chains(poly_chains)

            # Add information stored in the `_entity_poly` category, organized by entity_id, to the chains
            # Namely, the type, canonical sequence, and non-canonical sequence
            entity_poly = data.getObj('entity_poly')
            if entity_poly is not None:
                for row in entity_poly.getRowList():
                    entity_id = row[entity_poly.getIndex('entity_id')]
                    entity_type = row[entity_poly.getIndex('type')]
                    canonical_sequence = row[entity_poly.getIndex('pdbx_seq_one_letter_code_can')].replace('\n','')
                    non_canonical_sequence = row[entity_poly.getIndex('pdbx_seq_one_letter_code')].replace('\n','')
                    for chain in model.get_chains_by_entity_id(entity_id):
                        chain.type = entity_type
                        chain.canonical_sequence = canonical_sequence
                        chain.non_canonical_sequence = non_canonical_sequence

            # Parse polymer residues from the `_entity_poly_seq` category; these include proteins and nucleic acids
            entity_poly_seq = data.getObj('entity_poly_seq')
            for row in entity_poly_seq.getRowList():
                entity_id = row[entity_poly_seq.getIndex('entity_id')]
                sequence_id = row[entity_poly_seq.getIndex('num')]
                residue_name = row[entity_poly_seq.getIndex('mon_id')]
                is_hetero_residue = row[entity_poly_seq.getIndex('hetero')] in {'y','yes'}
                for chain in model.get_chains_by_entity_id(entity_id):
                    # When there are alternative residues at the same position (sequence heterogeneity), pick the one which occurs last
                    # This scheme is consistent with OpenSource solutions (e.g., BioPython), and anecdotally results in higher occupancies
                    residue_to_add = self.initialize_residue_from_name(id=sequence_id, name=residue_name, parent=chain)
                    if residue_to_add: # Ensure we aren't including any intentionally skipped residues
                        residue_to_add.hetero = is_hetero_residue
                        if not chain.has_id(sequence_id):
                            chain.add_residue(residue_to_add)
                        else:
                            old_residue = chain.get_residue(sequence_id)
                            del chain[sequence_id]
                            # Set the new residue's alternative residue names to the old residue's name combined with the old residue's alternative residue names
                            residue_to_add.alternative_residue_names.update(old_residue.alternative_residue_names, [old_residue.name])
                            chain.add_residue(residue_to_add)

    def populate_residues_with_coordinates(self, data, structure, entity_map):
        atom_site = data.getObj('atom_site')
        # Mapping of descriptive keys to their respective column indices in the atom_site object
        column_map = {
            'entity_id': 'label_entity_id',
            'atom_or_hetatm': 'group_PDB',
            'symbol': 'type_symbol',
            'atom_name': 'label_atom_id',
            'alt_location_id': 'label_alt_id',
            'residue_name': 'label_comp_id',
            'chain_id': 'label_asym_id',
            'sequence_number': 'label_seq_id',
            'x_coord': 'Cartn_x',
            'y_coord': 'Cartn_y',
            'z_coord': 'Cartn_z',
            'occupancy': 'occupancy',
            'b_factor': 'B_iso_or_equiv',
            'formal_charge': 'pdbx_formal_charge',
            'author_sequence_number': 'auth_seq_id',
            'model_number': 'pdbx_PDB_model_num'
        }

        # Create a dictionary to hold the index of each column for quick access
        column_indices = {key: atom_site.getIndex(val) for key, val in column_map.items()}

        for row in atom_site.getRowList():
            parsed_row = {
                key: (float(row[column_indices[key]]) if 'coord' in key or key in ['occupancy', 'b_factor']
                    else int(row[column_indices[key]]) if key in ['model_number']
                    else str(row[column_indices[key]]))
                for key in column_indices
            }
            
            # Check if the model exists; if not, create it (e.g., for NMR strucutres)
            if structure.has_id(parsed_row['model_number']):
                model = structure.get_model(parsed_row['model_number'])
            else:
                model = Model(id = parsed_row['model_number'])
                structure.add_model(model)
            
            # Check if the model has the relevant chain
            if model.has_id(parsed_row['chain_id']):
                chain = model.get_chain(parsed_row['chain_id'])
            else:
                # If we're working with a non-polymeric chain, create a new chain
                chain = Chain(
                    id = parsed_row['chain_id'],
                    entity_id=parsed_row['entity_id'],
                    entity_details=entity_map.get(parsed_row['entity_id']),
                    type='nonpoly',
                )
                model.add_chain(chain)

            # We use author assigned residue numbers for non-polymeric chains
            if chain.type =='nonpoly':
                sequence_id = parsed_row['author_sequence_number']
            else:
                sequence_id = parsed_row['sequence_number']

            if sequence_id == '.': # !!! Fixes 1ZY8 is which FAD ligand is assigned to a polypeptide chain O !!!
                continue

            # TODO: Handle multiple models; we may need to copy one
            # Check if the model has the relevant residue
            if chain.has_id(sequence_id):
                residue = chain.get_residue(sequence_id)
                # Set residue to None if the name doesn't match
                # This occurs for chains with sequence heterogeneity
                if residue.name != parsed_row['residue_name']:
                    residue = None
            elif parsed_row['atom_or_hetatm'] == 'HETATM': 
                # If we're working with a non-polymeric chain, create a new residue
                residue = self.initialize_residue_from_name(id=sequence_id, name=parsed_row['residue_name'], parent=chain)
                chain.add_residue(residue)
            else: 
                # Raise exception
                raise ValueError(f"Residue {sequence_id} not found in chain {chain.id} in model {model.id}")
                residue = None
            
            if residue:
                # If any heavy atom in a residue cannot be matched, then mask the whole residue
                atom_id = parsed_row['atom_name']
                atom_symbol = parsed_row['symbol']
                atom_occupancy = parsed_row['occupancy']
                if residue.has_id(atom_id):
                    atom = residue.get_atom(atom_id)
                    if atom_occupancy>atom.occupancy or (atom.occupancy == 0 and atom_occupancy == 0): # Handle examples where atoms have information (e.g., coordinates) in the CIF file, but no occupancy. For instance, `1a8o`
                        atom.occupancy = atom_occupancy
                        atom.hetero = (parsed_row['atom_or_hetatm'] == 'HETATM')
                        atom.bfactor = parsed_row['b_factor']
                        atom.xyz = np.array([parsed_row['x_coord'], parsed_row['y_coord'], parsed_row['z_coord']])
                else:
                    # We do not worry about unmatched hydrogen or deuterium atoms 
                    # NOTE: Do we want to be throwing out this coordinate information here, vs. processing later?
                    # NOTE: Do we want to make any attempt to correct the atom name mismatch?
                    # TODO: Attempt to correct atom name mismatch (look at BioPython documentation)
                    if atom_symbol != 'H' and atom_symbol != 'D': 
                        # Raise error
                        raise ValueError(f"Atom {atom_id} not found in residue {residue.id} in chain {chain.id} in model {model.id}")
                        residue.unmatched_heavy_atom = True

    def parse_covalent_connections(self, data, structure):
        struct_conn = data.getObj('struct_conn')
        #  Retrieve all necessary indices once to avoid repeated calls to getIndex
        indices = {
            'ptnr1_label_asym_id': struct_conn.getIndex('ptnr1_label_asym_id'),
            'ptnr1_label_seq_id': struct_conn.getIndex('ptnr1_label_seq_id'),
            'ptnr1_auth_seq_id': struct_conn.getIndex('ptnr1_auth_seq_id'),
            'ptnr1_label_comp_id': struct_conn.getIndex('ptnr1_label_comp_id'),
            'ptnr1_label_atom_id': struct_conn.getIndex('ptnr1_label_atom_id'),
            'ptnr2_label_asym_id': struct_conn.getIndex('ptnr2_label_asym_id'),
            'ptnr2_label_seq_id': struct_conn.getIndex('ptnr2_label_seq_id'),
            'ptnr2_auth_seq_id': struct_conn.getIndex('ptnr2_auth_seq_id'),
            'ptnr2_label_comp_id': struct_conn.getIndex('ptnr2_label_comp_id'),
            'ptnr2_label_atom_id': struct_conn.getIndex('ptnr2_label_atom_id'),
            'conn_type_id': struct_conn.getIndex('conn_type_id')
        }

        for row in struct_conn.getRowList():
            if row[indices['conn_type_id']] == 'covale': # Only process covalent connections (but excluding disulfide bridges and coordination covalent bonds, which are technically covalent, as well as hydrogen bonds)
                
                # Loop through all models
                for model in structure.get_models():
                    # Get the chains
                    chain_a = model.get_chain(row[indices['ptnr1_label_asym_id']])
                    chain_b = model.get_chain(row[indices['ptnr2_label_asym_id']])

                    # Get the appropriate sequence ID for each chain (either author or label)
                    sequence_id_a = row[indices['ptnr1_auth_seq_id']] if chain_a.type == 'nonpoly' else row[indices['ptnr1_label_seq_id']]
                    sequence_id_b = row[indices['ptnr2_auth_seq_id']] if chain_b.type == 'nonpoly' else row[indices['ptnr2_label_seq_id']]

                    # Check to make sure that none of the residues involved are heterogenous in sequence
                    # If they are, raise an error
                    residue_a = chain_a.get_residue(sequence_id_a)
                    residue_b = chain_b.get_residue(sequence_id_b)
                    if residue_a.hetero or residue_b.hetero:
                        raise ValueError(f"Residue {residue_a.id} or {residue_b.id} is heterogenous in sequence in model {model.id}")
                    
                    atom_a_name = row[indices['ptnr1_label_atom_id']]
                    atom_b_name = row[indices['ptnr2_label_atom_id']]

                    # Get the two partner atoms involved in the covalent bond
                    atom_a = residue_a.get_atom(atom_a_name)
                    atom_b = residue_b.get_atom(atom_b_name) 
                    
                    # Ensure the bond is inter-residue
                    # NOTE: This check was always evaluating to true in the original code
                    if(atom_a.parent != atom_b.parent):
                        # Create a bond object
                        bond = Bond(
                            atom_a = atom_a,
                            atom_b = atom_b,
                            is_aromatic = False,
                            in_ring = False,
                            order = 1, # We assume that all inter-residue bonds are single
                            length = 1.5, # This is a rough guesstimate # NOTE: This should be different for proteins, nucleic acids, and oligosaccharides
                        )

                        # If the bond is within a chain, add to the chain's `inter_residue_bonds` list. Otherwise, add to the structure's `inter_chain_bonds` list
                        if chain_a == chain_b:
                            chain_a.add_inter_residue_bond(bond)
                            chain_b.add_inter_residue_bond(bond)
                        else:
                            model.add_inter_chain_bond(bond)

    def add_inter_residue_bonds(self, structure):
        '''Add inter-residue connections in polymers'''
        # Possible types given at: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_entity_poly.type.html
        atom_pairs = {
            'polydeoxyribonucleotide': ("O3'", 'P'),
            'polydeoxyribonucleotide/polyribonucleotide hybrid': ("O3'", 'P'),
            'polypeptide(D)': ('C', 'N'),
            'polypeptide(L)': ('C', 'N'),
            'polyribonucleotide': ("O3'", 'P'),
        }          

        for model in structure.get_models():
            for chain in model.get_chains():
                # Determine atom pair for the chain type
                bond_atoms = atom_pairs.get(chain.type, ())
                
                if(len(bond_atoms)>0):
                    current_residues = list(chain.get_residues())[:-1] # Skip the last residue
                    next_residues = list(chain.get_residues())[1:] # Skip the first residue

                    for current_residue, next_residue in zip(current_residues , next_residues): # E.g., process residues 1 and 2, 2 and 3, 3 and 4, etc.
                        if current_residue.unmatched_heavy_atom or next_residue.unmatched_heavy_atom:
                            continue

                        # Get the two atoms involved in the bond
                        atom_a = current_residue.get_atom(bond_atoms[0])
                        atom_b = next_residue.get_atom(bond_atoms[1])
                        
                        # Create the bond between the two atoms and add the object to the Chain's list of inter-residue bonds
                        bond = Bond(
                            atom_a = atom_a,
                            atom_b = atom_b,
                            is_aromatic = False,
                            in_ring = False,
                            order = 1, # We assume that all inter-residue bonds are single
                            length=1.5, # This is a rough guesstimate # NOTE: This should be different for proteins, nucleic acids, and oligosaccharides
                        )
                        chain.add_inter_residue_bond(bond)
            
    def remove_leaving_groups_and_bonds(self, structure):
        """Remove leaving atoms and corresponding bonds from the model"""
        model = structure.child_list[0] # TODO: Make robust for multi-structure
        # Create a list of atoms to ignore that are leaving groups of atoms that are part of a covalent bond
        atoms_to_remove = set()
        for bond in model.get_inter_residue_bonds():
            atoms_to_remove.update(bond.atom_a.parent.get_atom(atom) for atom in bond.atom_a.leaving_group)
            atoms_to_remove.update(bond.atom_b.parent.get_atom(atom) for atom in bond.atom_b.leaving_group)
        
        for bond in model.get_inter_chain_bonds():
            atoms_to_remove.update(bond.atom_a.parent.get_atom(atom) for atom in bond.atom_a.leaving_group)
            atoms_to_remove.update(bond.atom_b.parent.get_atom(atom) for atom in bond.atom_b.leaving_group)
        
        # Remove leaving atoms from the model
        for atom in atoms_to_remove:
            # Remove bonds to the atom we will delete
            atom.parent.remove_bonds_involving_atom(atom)
            del atom.parent[atom.id] # Invokes overloaded __delitem__ method in Residue class

    def parse(self, filename : str) -> Dict:
        """ TODO: Rewrite this docstring
        Parses the given CIF file and returns a dictionary.

        The method performs the following steps:
        1. Load the CIF file.
        2. Build the entity map dictionary.
        3. Parse mappings of modified residues to their standard counterparts.

        Args:
            filename (str): The path to the CIF file to parse.

        Returns:
            Dict: A dictionary containing XYZ
        """
        
        #### 1. Load the CIF file ####
        data = []
        if filename.endswith('.gz'):
            with gzip.open(filename,'rt') as cif:
                reader = PdbxReader(cif)
                reader.read(data)
        else:
            with open(filename,'r') as cif:
                reader = PdbxReader(cif)
                reader.read(data)
        data = data[0]
        
        #### 2. Build the entity map dictionary ####
        entity_map = self.parse_entities(data)

        #### 3. Parse mappings of modified residues to their standard counterparts ####
        modified_residues = self.parse_modified_residues(data)
        
        # Create a Structure and Model wrapper class TODO: Make the structure ID the pdb ID
        structure = Structure(id = 0)

        # Get the first model number
        atom_site = data.getObj('atom_site')
        first_model_number = int(atom_site.getRow(0)[atom_site.getIndex("pdbx_PDB_model_num")])
        model = Model(id = first_model_number)

        # Load the model into the structure
        structure.add_model(model)

        #### 4. Parse polymeric chains ####
        # We can't parse non-polymers this way, as the residue ordering for glycans is not guaranteed to be correct
        self.parse_polymeric_chains(data, model, entity_map)
                
        #### 5. Populate residues with coordinates ####
        # If we come across a non-polymeric chain, we will create a new Chain object
        self.populate_residues_with_coordinates(data, structure, entity_map)
        
        #### 6. Parse covalent connections ####
        # NOTE: We make an implicit assumption that covalent connections across atoms are shared between models. The RCSB documentation does not include a `model` flag for `_struct_conn`.
        self.parse_covalent_connections(data, structure)
        
        #### 6. build connected chains ####
        self.add_inter_residue_bonds(structure)

        #### 7. Remove leaving atoms and corresponding bonds from the model ####
        self.remove_leaving_groups_and_bonds(structure)
            
        #### 6. Parse assemblies ####
        asmb = self.parse_assemblies(data)

        # Filter to ensure the chains exist
        asmb = {
            k:[vi for vi in v if structure.get_model(1).has_id(vi[0])]
            for k,v in asmb.items()
        }

        # Fix inter-chain and inter-residue charges
        for bond in itertools.chain(model.get_inter_residue_bonds(), model.get_inter_chain_bonds()):
            for atom in (bond.atom_a, bond.atom_b):
                if atom.element==7 and atom.charge==1 and atom.hyb==3 and atom.nhyd==2 and atom.hvydeg==2: # -(NH2+)-
                    atom.charge = 0
                    atom.hyb = 2
                    atom.nhyd = 0
                elif atom.element==7 and atom.charge==1 and atom.hyb==3 and atom.nhyd==3 and atom.hvydeg==0: # free NH3+ group
                    atom.charge = 0
                    atom.hyb = 2
                    atom.nhyd = 2
                elif atom.element==8 and atom.charge==-1 and atom.hyb==3 and atom.nhyd==0:
                    atom.charge = 0
                elif atom.element==8 and atom.charge==-1 and atom.hyb==2 and atom.nhyd==0: # O-linked connections
                    atom.charge = 0
        
        # Load metadata
        resolution = None
        if data.getObj('refine') is not None:
            try:
                resolution = float(data.getObj('refine').getValue('ls_d_res_high',0))
            except:
                resolution = None
        if (data.getObj('em_3d_reconstruction') is not None) and (resolution is None):
            try:
                resolution = float(data.getObj('em_3d_reconstruction').getValue('resolution',0))
            except:
                resolution = None

        meta = {
            'method' : data.getObj('exptl').getValue('method',0).replace(' ','_'),
            'date' : data.getObj('pdbx_database_status').getValue('recvd_initial_deposition_date',0),
            'resolution' : resolution
        }

        structure.method = meta['method']
        structure.initial_deposition_date = meta['date']
        structure.resolution = meta['resolution']

        return structure, asmb, modified_residues

    #@staticmethod
    def save(self, chain : Chain, filename : str):
        '''save a single chain'''
        
        with open(filename, 'w') as f:
            acount = 1
            a2i = {}
            for r,a in chain.atoms.items():
                if a.occ>0:
                    element = self.i2a[a.element] if a.element in self.i2a.keys() else 'X'
                    f.write ("%-6s%5s %-4s %3s%2s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n"%(
                        "HETATM" if a.hetero==True else "ATOM",
                        acount, ' '*a.align+a.full_id[2], r[2], r[0], int(r[1]),
                        a.xyz[0], a.xyz[1], a.xyz[2], a.occupancy, 0.0, element, a.charge) )
                    a2i[r] = acount
                    acount += 1
            for bond in chain.bonds:
                if chain.atoms[bond.a].occ==0.0:
                    continue
                if chain.atoms[bond.b].occ==0.0:
                    continue
                if chain.atoms[bond.a].hetero==False and chain.atoms[bond.b].hetero==False:
                    continue
                f.write ("%-6s%5d%5d\n"%("CONECT", a2i[bond.a], a2i[bond.b]))


    #@staticmethod
    def save_all(self,
                 chains : Dict[str,Chain],
                 covale : List[Bond],
                 filename : str):
        '''save multiple chains'''

        #'''
        with open(filename, 'w') as f:
            acount = 1
            a2i = {}
            for chain_id,chain in chains.items():
                for r,a in chain.atoms.items():
                    if a.occ>0:
                        element = self.i2a[a.element] if a.element in self.i2a.keys() else 'X'
                        f.write ("%-6s%5s %-4s %3s%2s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n"%(
                            "HETATM" if a.hetero==True else "ATOM",
                            acount, ' '*a.align+a.full_id[2], r[2], chain_id, int(r[1]),
                            a.xyz[0], a.xyz[1], a.xyz[2], a.occ, 0.0, element, a.charge) )
                        a2i[r] = acount
                        acount += 1
                for bond in chain.bonds:
                    a = chain.atoms[bond.a]
                    b = chain.atoms[bond.b]
                    if a.occ==0.0 or b.occ==0.0 or (a.hetero==False and b.hetero==False):
                        continue
                    f.write ("%-6s%5d%5d\n"%("CONECT", a2i[bond.a], a2i[bond.b]))
                f.write('TER\n')
            
            for bond in covale:
                a = chains[bond.a[0]].atoms[bond.a]
                b = chains[bond.b[0]].atoms[bond.b]
                if a.occ==0.0 or b.occ==0.0:
                    continue
                f.write ("%-6s%5d%5d\n"%("CONECT", a2i[bond.a], a2i[bond.b]))
