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
from cifutils.data_structures import Bond
from cifutils.Structure import Structure
from cifutils.Model import Model
from cifutils.Chain import Chain
from cifutils.Residue import Residue
from cifutils.Atom import Atom
from cifutils.pdbx.reader.PdbxReader import PdbxReader
import cifutils.obutils as obutils


def ParsePDBLigand(cifname : str) -> Dict:
    '''Parse a single molecule from the PDB-Ligands set
    '''

    data = []
    with open(cifname,'r') as cif:
        reader = PdbxReader(cif)
        reader.read(data)
    data = data[0]
    chem_comp_atom = data.getObj('chem_comp_atom')
    rows = chem_comp_atom.getRowList()

    # parse atom names
    idx = chem_comp_atom.getIndex('atom_id')
    atom_id = np.array([r[idx] for r in rows])
    
    # parse element symbols
    idx = chem_comp_atom.getIndex('type_symbol')
    symbol = np.array([r[idx] for r in rows])

    # parse leaving flags
    idx = chem_comp_atom.getIndex('pdbx_leaving_atom_flag')
    leaving = [r[idx] for r in rows]
    leaving = np.array([True if flag=='Y' else False for flag in leaving], dtype=bool)

    # atom name alignment offset in PDB atom field
    idx = chem_comp_atom.getIndex('pdbx_align')
    pdbx_align = np.array([int(r[idx]) for r in rows])
    
    # parse xyz coordinates
    i = chem_comp_atom.getIndex('model_Cartn_x')
    j = chem_comp_atom.getIndex('model_Cartn_y')
    k = chem_comp_atom.getIndex('model_Cartn_z')
    xyz = [(r[i],r[j],r[k]) for r in rows]
    xyz = np.array([[float(c) if c!='?' else np.nan for c in p] for p in xyz])

    out = {'atom_id' : atom_id,
           'leaving' : leaving,
           'symbol' : symbol,
           'pdbx_align' : pdbx_align,
           'xyz' : xyz}

    return out


# ============================================================
class CIFParser:
    
    def __init__(self, skip_res : List[str] = None):
        
        # parse pre-compiled library of all residues observed in the PDB
        DIR = os.path.dirname(__file__)
        with gzip.open(f'{DIR}/ligands.json.gz','rt') as file:
            self.precomputed_open_babel_residue_data = json.load(file)
            
            # Residues to be ignored during parsing are deleted from the precomputed library
            if skip_res is not None:
                self.precomputed_open_babel_residue_data = {k: v for k, v in self.precomputed_open_babel_residue_data.items() if k not in skip_res}
            
        # parse quasi-symmetric groups table
        df = pd.read_csv(f'{DIR}/data/quasisym.csv')
        df.indices = df.indices.apply(lambda x : [int(xi) for xi in x.split(',')])
        df['matcher'] = df.apply(lambda x : openbabel.OBSmartsPattern(), axis=1)
        df.apply(lambda x : x.matcher.Init(x.smarts), axis=1)
        self.quasisym = {smarts:(matcher,torch.tensor(indices))
                         for smarts,matcher,indices 
                         in zip(df.smarts,df.matcher,df.indices)}
        
        # parse pesiodic table
        with open(f'{DIR}/data/elements.txt','r') as f:
            self.i2a = [l.strip().split()[:2] for l in f.readlines()]
            self.i2a = {int(i):a for i,a in self.i2a}

        # initialize dictionary to store residues information dictionaries
        self.residue_name_to_data_map = {}
    

    def initialize_residue_from_name(self, id : str, name : str) -> Residue:
        """
        Initializes and pre-populates with Atoms and Bonds a Residue object from a given residue name.
        Loads bond and atom information from Open Babel.

        Parameters:
        id (int): The ID of the residue.
        name (str): The name of the residue.

        Returns:
        Residue: The initialized Residue object, or None if the residue data is not available in the library.
        """

        open_babel_residue_data = self.precomputed_open_babel_residue_data.get(name)
        # If we didn't include the result in our library (e.g., we skipped it), we return None, and skip the residue downstream
        if open_babel_residue_data is None:
            return None
        
        # If we haven't already loaded the residue, we load it now
        if name not in self.residue_name_to_data_map:
            self.residue_name_to_data_map[name] = self.load_residue_details_and_atoms_from_open_babel(
                sdfstring=open_babel_residue_data['sdf'],
                atom_id=open_babel_residue_data['atom_id'],
                leaving=open_babel_residue_data['leaving'],
                pdbx_align=open_babel_residue_data['pdbx_align']
            )
        
        # Copy the residue data to avoid modifying the original
        residue_data = copy.deepcopy(self.residue_name_to_data_map[name])
        
        new_residue = Residue(
            id=id,
            name=residue_data['name'],
            intra_residue_bonds=residue_data['intra_residue_bonds'],
            automorphisms=residue_data['automorphisms'],
            chirals=residue_data['chirals'],
            planars=residue_data['planars'],
            alternative_residue_names=set()
        )

        # Add atoms to set parents correctly
        new_residue.add_multiple_atoms(residue_data['atoms'])
        return new_residue
        
    def AddQausisymmetries(self, 
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
    def getLeavingAtoms(a,leaving,s):
        for b in openbabel.OBAtomAtomIter(a):
            if leaving[b.GetIndex()]==True:
                if b.GetIndex() not in s:
                    s.append(b.GetIndex())
                    CIFParser.getLeavingAtoms(b,leaving,s)


    @staticmethod
    def getLeavingAtoms2(aname, G):

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
        
        # get atoms and their features
        residue_atoms = {}
        for aname,aleaving,align,a in zip(atom_id,leaving,pdbx_align,openbabel.OBMolAtomIter(obmol)):

            '''
            leaving_group = []
            if aleaving==False:
                CIFParser.getLeavingAtoms(a,leaving,leaving_group)
                leaving_group = [atom_id[i] for i in leaving_group]
            '''
            
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
            
            # Create the Atom object
            residue_atoms[aname] = Atom(
                id=aname, # Atom ID's are their CCD "names"
                xyz=np.array([0.0,0.0,0.0]),
                occupancy=0.0,
                altloc='',
                bfactor=0.0,
                leaving_atom_flag=aleaving,
                leaving_group=[],
                parent_heavy_atom=parent,
                element=a.GetAtomicNum(),
                is_metal=a.IsMetal(),
                charge=charge,
                hyb=a.GetHyb(),
                nhyd=nhyd,
                hvydeg=a.GetHvyDegree(),
                align=align,
                hetero=False
            )
        
        # get bonds and their features
        bonds = []
        for b in openbabel.OBMolBondIter(obmol):
            bonds.append(Bond(a=atom_id[b.GetBeginAtom().GetIndex()],
                              b=atom_id[b.GetEndAtom().GetIndex()],
                              is_aromatic=b.IsAromatic(),
                              in_ring=b.IsInRing(),
                              order=b.GetBondOrder(),
                              intra_residue=True,
                              length=b.GetLength()))

        # get automorphisms
        automorphisms = obutils.FindAutomorphisms(obmol, heavy=True)
        
        # add quasi-symmetric groups
        automorphisms = self.AddQausisymmetries(obmol, automorphisms)
        
        # only retain atoms with alternative mappings
        mask = (automorphisms[:1]==automorphisms).all(dim=0)
        automorphisms = automorphisms[:,~mask]

        # skip automorphisms which include leaving atoms
        '''
        if automorphisms.shape[0]>1:
            mask = torch.tensor(leaving)[automorphisms].any(dim=0)
            #print(mask, torch.tensor(leaving)[automorphisms])
            automorphisms = automorphisms[:,~mask]
            if automorphisms.shape[-1]>0:
                automorphisms = torch.unique(automorphisms,dim=0)
            else:
                automorphisms = automorphisms.flatten()
        '''
                
        # get chirals and planars
        chirals = obutils.GetChirals(obmol, heavy=True)
        planars = obutils.GetPlanars(obmol, heavy=True)

        # add leaving groups to atoms
        G = nx.Graph()
        G.add_nodes_from([(a.id,{'leaving':a.leaving_atom_flag}) for a in residue_atoms.values()])
        G.add_edges_from([(bond.a,bond.b) for bond in bonds])
        for k, v in residue_atoms.items():
            v.leaving_group = CIFParser.getLeavingAtoms2(k,G)
        
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
    def parseOperationExpression(expression : str) -> List:
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
    def parseAssemblies(data : pdbx.reader.PdbxContainers.DataContainer) -> Dict:
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

            oper_list = [CIFParser.parseOperationExpression(expression) 
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


    def parse(self, filename : str) -> Dict:
        
        ########################################################
        # 0. read a .cif file
        ########################################################
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
        
        ########################################################
        # 0.5. parse entities into a dictionary
        ########################################################
        entities = data.getObj('entity')
        entity_map = {}
        if entities is not None:
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

        ########################################################
        # 1. parse mappings of modified residues to their 
        #    standard counterparts
        ########################################################
        pdbx_struct_mod_residue = data.getObj('pdbx_struct_mod_residue')
        if pdbx_struct_mod_residue is None:
            modified_residues = {}
        else:
            modified_residues = {(r[pdbx_struct_mod_residue.getIndex('label_comp_id')],
                       r[pdbx_struct_mod_residue.getIndex('parent_comp_id')])
                      for r in pdbx_struct_mod_residue.getRowList()}
            modified_residues = {k:v for k,v in modified_residues if k!=v}
        
        # Create a Structure and Model wrapper class TODO: Make these real
        structure = Structure(id = 0)

        # Get the first model number
        atom_site = data.getObj('atom_site')
        first_model_number = int(atom_site.getRow(0)[atom_site.getIndex("pdbx_PDB_model_num")])
        model = Model(id = first_model_number)

        # Load the model into the structure
        structure.add_model(model)

        ########################################################
        # 2. parse polymeric chains
        # We can't parse non-polymers this way, as the residue ordering for glycans is not guaranteed to be correct
        ########################################################
        pdbx_poly_seq_scheme = data.getObj('pdbx_poly_seq_scheme')
        if pdbx_poly_seq_scheme is not None:
            
            # Initialize Chain objects with mapping asym_id <--> entity_id, pdb_strand_id 
            # Note: We only initialize polymer chains this way; we initialize non-polymers within the atom_site loop
            poly_chains = {}
            for row in pdbx_poly_seq_scheme.getRowList():
                asym_id = row[pdbx_poly_seq_scheme.getIndex('asym_id')]
                if asym_id not in poly_chains:
                    poly_chains[asym_id] = Chain(
                        id = asym_id,
                        entity_id=row[pdbx_poly_seq_scheme.getIndex('entity_id')],
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
                    residue_to_add = self.initialize_residue_from_name(id=sequence_id, name=residue_name)
                    if residue_to_add: # Ensure we aren't including any intentionally skipped residues
                        residue_to_add.hetero = is_hetero_residue
                        if not chain.has_id(sequence_id):
                            chain.add_residue(residue_to_add)
                        else:
                            old_residue = chain.get_residue(sequence_id)
                            del chain[sequence_id]
                            # Set the new residue's alternative residue names to the old residue's name combined with the old residue's alternative residue names
                            residue_to_add.alternative_residue_names = old_residue.alternative_residue_names.append(old_residue.name)
                            chain.add_residue(residue_to_add)
                
        ########################################################
        # 4. populate residues with coordinates
        #    If we come across a non-polymeric chain, we will create a new Chain object
        ########################################################
        

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

            # Check if the model has the relevant residue
            if chain.has_id(sequence_id):
                residue = chain.get_residue(sequence_id)
            elif parsed_row['atom_or_hetatm'] == 'HETATM': # If we're working with a non-polymer, create a new chain
                residue = Residue(id=sequence_id, name=parsed_row['residue_name'])
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

        ########################################################
        # 5. parse covalent connections
        ########################################################

        # NOTE: We make an implicit assumption that covalent connections across atoms are shared between models. The RCSB documentation does not include a `model` flag for `_struct_conn`
        # TODO: Is this a fair assumption?
        
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

        covale = []
        for row in struct_conn.getRowList():
            if row[indices['conn_type_id']] == 'covale': # Only process covalent connections (but excluding disulfide bridges and coordination covalent bonds, which are technically covalent, as well as hydrogen bonds)
                
                # Loop through all models
                for model in structure.get_models():
                    # Check to make sure that none of the residues involved are heterogenous in sequence
                    # If they are, raise an error
                    residue_a = model.get_residue(row[indices['ptnr1_label_asym_id']], row[indices['ptnr1_label_seq_id']])
                    residue_b = model.get_residue(row[indices['ptnr1_label_asym_id']], row[indices['ptnr1_label_seq_id']])
                    if residue_a.hetero or residue_b.hetero:
                        raise ValueError(f"Residue {residue_a.id} or {residue_b.id} is heterogenous in sequence in model {model.id}")
                    
                    # Get the two partner atoms involved in the covalent bond
                    atom_a = model.get_atom(row[indices['ptnr1_label_asym_id']], row[indices['ptnr1_label_seq_id']], row[indices['ptnr1_label_atom_id']])
                    atom_b = model.get_atom(row[indices['ptnr2_label_asym_id']], row[indices['ptnr2_label_seq_id']], row[indices['ptnr2_label_atom_id']])
                    
                    # Ensure the bond is inter-residue
                    if(atom_a.residue != atom_b.residue):
                        # Add the bond to the list of covalent bonds
                        covale.append((atom_a, atom_b))

                if p1[:4] != p2[:4]:  # Skip intra-residue covalent bonds
                    sequence_id_or_auth = lambda x: x[2] if chains[x[0]]['type'] == 'nonpoly' else x[1]
                    covale.append(
                        ((p1[0], sequence_id_or_auth(p1), p1[3], p1[4]),
                        (p2[0], sequence_id_or_auth(p2), p2[3], p2[4]))
                    )

        if struct_conn is not None:
            covale = [(r[struct_conn.getIndex('ptnr1_label_asym_id')],
                       r[struct_conn.getIndex('ptnr1_label_seq_id')],
                       r[struct_conn.getIndex('ptnr1_auth_seq_id')],
                       r[struct_conn.getIndex('ptnr1_label_comp_id')],
                       r[struct_conn.getIndex('ptnr1_label_atom_id')],
                       r[struct_conn.getIndex('ptnr2_label_asym_id')],
                       r[struct_conn.getIndex('ptnr2_label_seq_id')],
                       r[struct_conn.getIndex('ptnr2_auth_seq_id')],
                       r[struct_conn.getIndex('ptnr2_label_comp_id')],
                       r[struct_conn.getIndex('ptnr2_label_atom_id')])
                      for r in struct_conn.getRowList() if r[struct_conn.getIndex('conn_type_id')]=='covale']
            F = lambda x : x[2] if chains[x[0]]['type']=='nonpoly' else x[1]
            # here we skip intra-residue covalent bonds assuming that
            # they are properly handled by parsing from the residue library
            covale = [((c[0],F(c[:4]),c[3],c[4]),(c[5],F(c[5:]),c[8],c[9])) 
                      for c in covale if c[:4]!=c[5:8]]
                        
        ########################################################
        # 6. build connected chains
        ########################################################
        return_chains = {}
        for chain_id,chain in chains.items():
                        
            residues = list(chain['res'].items())
            atoms,bonds,skip_atoms = [],[],[]
            
            # (a) add inter-residue connections in polymers
            if 'polypept' in chain['type']:
                ab = ('C','N')
            elif 'polyribo' in chain['type'] or 'polydeoxyribo' in chain['type']:
                ab = ("O3'",'P')
            else:
                ab = ()

            if len(ab)>0:
                for residue_a, residue_b in zip(residues[:-1],residues[1:]):
                    # check for skipped residues (the ones failed in step 4)
                    if residue_a[1] is None or residue_b[1] is None:
                        continue
                    atom_a = residue_a[1][ab[0]]
                    atom_b = residue_b[1][ab[1]]
                    if atom_a is not None and atom_b is not None:
                        bonds.append(Bond(
                            a=(chain_id,residue_a[0],residue_a[1].name, atom_a.id),
                            b=(chain_id,residue_b[0],residue_b[1].name, atom_b.id),
                            is_aromatic=False,
                            in_ring=False,
                            order=1, # !!! we assume that all inter-residue bonds are single !!!
                            intra_residue=False,
                            length=1.5 # !!! this is a rough guesstimate !!!
                        ))
                        skip_atoms.extend([(chain_id,residue_a[0],residue_a[1].name,ai) for ai in atom_a.leaving_group])
                        skip_atoms.extend([(chain_id,residue_b[0],residue_b[1].name,bi) for bi in atom_b.leaving_group])

            # (b) add connections parsed from mmcif's struct_conn record
            # TODO: Refactor to be less terrible, using my new class
            for residue_a, residue_b in covale:
                atom_a = atom_b = None
                if residue_a[0] == chain_id and chain['res'][residue_a[1]] is not None and chain['res'][residue_a[1]].name == residue_a[2]:
                    atom_a = chain['res'][residue_a[1]].get_atom(residue_a[3])
                    skip_atoms.extend([(chain_id,*residue_a[1:3],ai) for ai in atom_a.leaving_group])
                if residue_b[0]==chain_id and chain['res'][residue_b[1]] is not None and chain['res'][residue_b[1]].name==residue_b[2]:
                    atom_b = chain['res'][residue_b[1]].get_atom(residue_b[3])
                    skip_atoms.extend([(chain_id,*residue_b[1:3],bi) for bi in atom_b.leaving_group])
                if atom_a is not None and atom_b is not None:
                    bonds.append(Bond(
                        a=(chain_id,*residue_a[1:3],atom_a.id),
                        b=(chain_id,*residue_b[1:3],atom_b.id),
                        is_aromatic=False,
                        in_ring=False,
                        order=1, # !!! we assume that all inter-residue bonds are single !!!
                        intra_residue=False,
                        length=1.5 # !!! this is a rough guesstimate !!!
                    ))
                    
            # (c) collect atoms
            skip_atoms = set(skip_atoms)
            # Create a dictionary indexed by Atom tuples (chain_id, seq_id, res_name, atom_name) and containing Atom objects

            # atoms = {}
            # for r in residues:
            #     if r[1] is not None:
            #         for atom in r[1].get_atoms():
            #             atoms[(chain_id, r[0], r[1].name, atom.id)] = atom

            atoms = {
                (chain_id, r[0], r[1].name, atom.id): atom for r in residues if r[1] is not None for atom in r[1].get_atoms()
            }
            
            filtered_atoms = {}
            for atom_full_id, atom in atoms.items():
                if atom_full_id not in skip_atoms:
                    atom.full_id = atom_full_id
                    filtered_atoms[atom_full_id] = atom
                    
            # atoms = {aname: a._replace(name=aname) for aname,a in atoms.items() if aname not in skip_atoms}
            atoms = None  # FOR TESTING
            # (d) collect intra-residue bonds
            bonds_intra = [bond._replace(a=(chain_id,r[0],r[1].name,bond.a),
                                         b=(chain_id,r[0],r[1].name,bond.b))
                           for r in residues if r[1] is not None
                           for bond in r[1].intra_residue_bonds]
            bonds_intra = [bond for bond in bonds_intra 
                           if bond.a not in skip_atoms and \
                           bond.b not in skip_atoms]

            bonds.extend(bonds_intra)
            
            # (e) double check whether bonded atoms actually exist:
            #     some could be part of the skip_atoms set and thus removed
            bonds = [bond for bond in bonds if bond.a in filtered_atoms.keys() and bond.b in filtered_atoms.keys()]
            bonds = list(set(bonds))
            
            # (f) fix charges and hydrogen counts for cases when
            #     charged atoms are connected by an inter-residue bond
            '''
            for bond in bonds:
                if bond.intra==False:
                    a = atoms[bond.a]
                    b = atoms[bond.b]

                    # nitrogen (involved in a peptide bond)
                    if a.element==7 and a.charge==1 and a.hyb==3 and a.nhyd==3:
                        atoms[bond.a] = a._replace(charge=0, hyb=2, nhyd=1)
                    if b.element==7 and b.charge==1 and b.hyb==3 and b.nhyd==3:
                        atoms[bond.b] = b._replace(charge=0, hyb=2, nhyd=1)
            
                    # oxygen (nucletides)
                    if a.element==8 and a.charge==-1 and a.hyb==3 and a.nhyd==0:
                        atoms[bond.a] = a._replace(charge=0)
                    if b.element==8 and b.charge==-1 and b.hyb==3 and b.nhyd==0:
                        atoms[bond.b] = b._replace(charge=0)
            '''
            
            # (g) relabel chirals, planars and automorphisms 
            #     to include residue indices and names
            chirals = [[(chain_id,r[0],r[1].name,c) for c in chiral] 
                       for r in residues if r[1] is not None for chiral in r[1].chirals]
            
            planars = [[(chain_id,r[0],r[1].name,c) for c in planar] 
                       for r in residues if r[1] is not None for planar in r[1].planars]
            
            automorphisms = [[[(chain_id,r[0],r[1].name,a) 
                               for a in auto] for auto in r[1].automorphisms] 
                             for r in residues if r[1] is not None and len(r[1].automorphisms)>1]

            chirals = [c for c in chirals if all([ci in filtered_atoms.keys() for ci in c])]
            planars = [c for c in planars if all([ci in filtered_atoms.keys() for ci in c])]

            if len(filtered_atoms)>0:
                # return_chains[chain_id] = Chain(id=chain_id,
                #                                 type=chain['type'],
                #                                 canonical_sequence=chain.get('seq'),
                #                                 non_canonical_sequence=None, # TODO
                #                                 atoms=filtered_atoms,
                #                                 bonds=bonds,
                #                                 chirals=chirals,
                #                                 planars=planars,
                #                                 automorphisms=automorphisms)

                return_chains[chain_id] = Chain(id=chain_id,
                                                type=chain['type'],
                                                canonical_sequence=chain.get('seq'),
                                                non_canonical_sequence=None, # TODO
                                                entity_id=None,
                                                symmetric_id=0
                                                )
                
        ########################################################
        # 6. parse assemblies
        ########################################################
        asmb = self.parseAssemblies(data)
        asmb = {k:[vi for vi in v if vi[0] in return_chains.keys()]
                for k,v in asmb.items()}

        
        # convert covalent links to Bonds
        covale = [Bond(a=c[0],
                       b=c[1],
                       is_aromatic=False,
                       in_ring=False,
                       order=1,
                       intra_residue=False,
                       length=1.5)
                  for c in covale if c[0][0]!=c[1][0]]

        # make sure covale atoms exist
        covale = [c for c in covale if \
                  c.a[0] in return_chains.keys() and \
                  c.b[0] in return_chains.keys() and \
                  c.a in return_chains[c.a[0]].atoms.keys() and \
                  c.b in return_chains[c.b[0]].atoms.keys()]

        
        # fix charges and hydrogen counts for cases when
        # charged a atom is connected by an inter-residue bond
        for bond in bonds+covale:
            if bond.intra_residue==False:
                for i in (bond.a,bond.b):
                    atom = return_chains[i[0]].atoms[i]
                    
                    #if a.element==7 and a.charge==1 and a.hyb==3 and a.nhyd==3 and a.hvydeg==1: # -NH3+
                    #    return_chains[i[0]].atoms[i] = a._replace(charge=0, hyb=2, nhyd=1)
                    if atom.element==7 and atom.charge==1 and atom.hyb==3 and atom.nhyd==2 and atom.hvydeg==2: # -(NH2+)-
                        return_chains[i[0]].atoms[i] = atom._replace(charge=0, hyb=2, nhyd=0)
                    elif atom.element==7 and atom.charge==1 and atom.hyb==3 and atom.nhyd==3 and atom.hvydeg==0: # free NH3+ group
                        return_chains[i[0]].atoms[i] = atom._replace(charge=0, hyb=2, nhyd=2)
                    elif atom.element==8 and atom.charge==-1 and atom.hyb==3 and atom.nhyd==0:
                        return_chains[i[0]].atoms[i] = atom._replace(charge=0)
                    elif atom.element==8 and atom.charge==-1 and atom.hyb==2 and atom.nhyd==0: # O-linked connections
                        return_chains[i[0]].atoms[i] = atom._replace(charge=0)
                    elif atom.charge!=0:
                        #print(filename, i, a)
                        pass

                '''
                a = return_chains[bond.a[0]].atoms[bond.a]
                b = return_chains[bond.b[0]].atoms[bond.b]

                # nitrogen (involved in a peptide bond)
                if a.element==7 and a.charge==1 and a.hyb==3 and (a.nhyd==3 or (a.nhyd==2 and bond.a[2]=='PRO')):
                    return_chains[bond.a[0]].atoms[bond.a] = a._replace(charge=0, hyb=2, nhyd=1)
                if b.element==7 and b.charge==1 and b.hyb==3 and (b.nhyd==3 or (b.nhyd==2 and bond.b[2]=='PRO')):
                    return_chains[bond.b[0]].atoms[bond.b] = b._replace(charge=0, hyb=2, nhyd=1)
            
                # oxygen (nucletides)
                if a.element==8 and a.charge==-1 and a.hyb==3 and a.nhyd==0:
                    return_chains[bond.a[0]].atoms[bond.a] = a._replace(charge=0)
                if b.element==8 and b.charge==-1 and b.hyb==3 and b.nhyd==0:
                    return_chains[bond.b[0]].atoms[bond.b] = b._replace(charge=0)
                    
                a = return_chains[bond.a[0]].atoms[bond.a]
                b = return_chains[bond.b[0]].atoms[bond.b]
                if a.charge!=0:
                    print(filename, bond.a,bond.b, a)
                if b.charge!=0:
                    print(filename, bond.b,bond.a, b)
                '''

        residue_id_obj_map = None
        if data.getObj('refine') is not None:
            try:
                residue_id_obj_map = float(data.getObj('refine').getValue('ls_d_res_high',0))
            except:
                residue_id_obj_map = None
        if (data.getObj('em_3d_reconstruction') is not None) and (residue_id_obj_map is None):
            try:
                residue_id_obj_map = float(data.getObj('em_3d_reconstruction').getValue('resolution',0))
            except:
                residue_id_obj_map = None

        meta = {
            'method' : data.getObj('exptl').getValue('method',0).replace(' ','_'),
            'date' : data.getObj('pdbx_database_status').getValue('recvd_initial_deposition_date',0),
            'resolution' : residue_id_obj_map
        }

        return return_chains,asmb,covale,meta, modified_residues
    
    
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
