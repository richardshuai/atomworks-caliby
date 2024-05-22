import numpy as np
import itertools
from biotite.structure.atoms import repeat
from biotite.structure.util import matrix_rotate
import gzip
import os
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import pandas as pd
from collections import OrderedDict
from pathlib import Path

def category_to_df(cif_block, category):
    """Convert a CIF block to a pandas DataFrame."""
    return pd.DataFrame({key: column.as_array() for key, column in cif_block[category].items()}) if category in cif_block.keys() else None

def deduplicate_iterator(iterator):
    """Deduplicate an iterator while preserving order."""
    return iter(OrderedDict.fromkeys(iterator))

def get_bond_type_from_order_and_is_aromatic(order, is_aromatic):
    """Get the biotite struc.BondType from the bond order and aromaticity."""
    aromatic_bond_types = {
        1: struc.BondType.AROMATIC_SINGLE,
        2: struc.BondType.AROMATIC_DOUBLE,
        3: struc.BondType.AROMATIC_TRIPLE,
    }

    non_aromatic_bond_types = {
        1: struc.BondType.SINGLE,
        2: struc.BondType.DOUBLE,
        3: struc.BondType.TRIPLE,
        4: struc.BondType.QUADRUPLE,
    }

    return aromatic_bond_types.get(order, struc.BondType.ANY) if is_aromatic else non_aromatic_bond_types.get(order, struc.BondType.ANY)

def read_cif_file(filename):
    """Reads a CIF, BCIF, or gzipped CIF/BCIF file and returns its contents."""
    if not isinstance(filename, Path):
        filename = Path(filename)
    
    file_ext = filename.suffix
    
    if file_ext == '.gz':
        with gzip.open(filename, 'rt') as f:
            # Handle gzipped CIF files
            if filename.name.endswith('.cif.gz'):
                cif_file = pdbx.CIFFile.read(f)
            elif filename.name.endswith('.bcif.gz'):
                with gzip.open(filename, 'rb') as bf:
                    cif_file = pdbx.BinaryCIFFile.read(bf)
            else:
                raise ValueError("Unsupported file format for gzip compressed file")
    elif file_ext == '.bcif':
        # Handle BinaryCIF files
        cif_file = pdbx.BinaryCIFFile.read(filename)
    elif file_ext == '.cif':
        # Handle plain CIF files
        cif_file = pdbx.CIFFile.read(filename)
    else:
        raise ValueError("Unsupported file format")
    
    return cif_file

def parse_transformations(struct_oper):
    """
    Get transformation operation in terms of rotation matrix and
    translation for each operation ID in ``pdbx_struct_oper_list``.
    
    Copied from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398 
    """
    transformation_dict = {}
    for index, id in enumerate(struct_oper["id"].as_array(str)):
        rotation_matrix = np.array(
            [
                [
                    struct_oper[f"matrix[{i}][{j}]"].as_array(float)[index]
                    for j in (1, 2, 3)
                ]
                for i in (1, 2, 3)
            ]
        )
        translation_vector = np.array([
            struct_oper[f"vector[{i}]"].as_array(float)[index]
            for i in (1, 2, 3)
        ])
        transformation_dict[id] = (rotation_matrix, translation_vector)
    return transformation_dict

def parse_operation_expression(expression):
    """
    Get successive operation steps (IDs) for the given
    ``oper_expression``.
    Form the cartesian product, if necessary.

    Copied from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398 
    """
    # Split groups by parentheses:
    # use the opening parenthesis as delimiter
    # and just remove the closing parenthesis
    expressions_per_step = expression.replace(")", "").split("(")
    expressions_per_step = [e for e in expressions_per_step if len(e) > 0]
    # Important: Operations are applied from right to left
    expressions_per_step.reverse()

    operations = []
    for expr in expressions_per_step:
        if "-" in expr:
            # Range of operation IDs, they must be integers
            first, last = expr.split("-")
            operations.append(
                [str(id) for id in range(int(first), int(last) + 1)]
            )
        elif "," in expr:
            # List of operation IDs
            operations.append(expr.split(","))
        else:
            # Single operation ID
            operations.append([expr])

    # Cartesian product of operations
    return list(itertools.product(*operations))

def apply_transformations(structure, transformation_dict, operations):
    """
    Get subassembly by applying the given operations to the input
    structure containing affected asym IDs.

    Copied from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398
    """
    # Additional first dimesion for 'structure.repeat()'
    assembly_coord = np.zeros((len(operations),) + structure.coord.shape)

    # Apply corresponding transformation for each copy in the assembly
    for i, operation in enumerate(operations):
        coord = structure.coord
        # Execute for each transformation step
        # in the operation expression
        for op_step in operation:
            rotation_matrix, translation_vector = transformation_dict[op_step]
            # Rotate
            coord = matrix_rotate(coord, rotation_matrix)
            # Translate
            coord += translation_vector
        assembly_coord[i] = coord

    return repeat(structure, assembly_coord)

def fix_bonded_atom_charges(atom):
    """
    Fix charges and hydrogen counts for cases when
    charged a atom is connected by an inter-residue bond.

    Args:
        atom (Atom): The atom object to be modified.

    Returns:
        dict: A dictionary with updated 'charge', 'hyb', and 'nhyd' values.
    """
    if atom.element == 7 and atom.charge == 1 and atom.hyb == 3 and atom.nhyd == 2 and atom.hvydeg == 2:  # -(NH2+)-
        return {"charge": 0, "hyb": 2, "nhyd": 0}
    elif atom.element == 7 and atom.charge == 1 and atom.hyb == 3 and atom.nhyd == 3 and atom.hvydeg == 0:  # free NH3+ group
        return {"charge": 0, "hyb": 2, "nhyd": 2}
    elif atom.element == 8 and atom.charge == -1 and atom.hyb == 3 and atom.nhyd == 0:
        return {"charge": 0, "hyb": atom.hyb, "nhyd": atom.nhyd}
    elif atom.element == 8 and atom.charge == -1 and atom.hyb == 2 and atom.nhyd == 0:  # O-linked connections
        return {"charge": 0, "hyb": atom.hyb, "nhyd": atom.nhyd}
    elif atom.charge != 0:
        # Additional logic for other cases if needed
        pass
    return {"charge": atom.charge, "hyb": atom.hyb, "nhyd": atom.nhyd}

