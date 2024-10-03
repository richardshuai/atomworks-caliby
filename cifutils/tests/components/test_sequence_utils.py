import numpy as np

from cifutils.utils.sequence_utils import is_pyramidine, is_purine, is_unknown_nucleotide, is_protein, is_glycine, is_protein_not_glycine, is_protein_unknown

def test_is_pyramidine():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA",
        "DC", 
        "DG",
        "DT", 
        "UNK",
        "A",
        "C", 
        "G",
        "U", 
        "DX",
        "X",
    ]

    expected = [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
    ]
    assert (is_pyramidine(sequence) == np.array(expected)).all()

def test_is_purine():
    
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA", 
        "DC", 
        "DG",
        "DT", 
        "UNK",
        "A", 
        "C", 
        "G", 
        "U", 
        "DX",
        "X",
    ]

    expected = [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
    ]
    assert (is_purine(sequence) == np.array(expected)).all()

def test_is_unknown_nucleotide():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA", 
        "DC", 
        "DG",
        "DT", 
        "UNK",
        "A", 
        "C", 
        "G", 
        "U", 
        "DX",
        "X",
    ]

    expected = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]

    assert (is_unknown_nucleotide(sequence) == np.array(expected)).all()

def test_is_protein():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA", 
        "DC", 
        "DG",
        "DT", 
        "UNK",
        "A", 
        "C", 
        "G", 
        "U", 
        "DX",
        "X",
    ]

    expected = [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    assert (is_protein(sequence) == np.array(expected)).all()

def test_is_glycine():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA", 
        "DC", 
        "DG",
        "DT", 
        "UNK",
        "A", 
        "C", 
        "G", 
        "U", 
        "DX",
        "X",
    ]

    expected = [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    assert (is_glycine(sequence) == np.array(expected)).all()

def test_is_protein_not_glycine():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA", 
        "DC", 
        "DG",
        "DT", 
        "UNK",
        "A", 
        "C", 
        "G", 
        "U", 
        "DX",
        "X",
    ]

    expected = [
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    assert (is_protein_not_glycine(sequence) == np.array(expected)).all()

def test_is_protein_unknown():
    sequence = [
        "MET",
        "LEU",
        "GLY",
        "VAL",
        "ALA",
        "DA", 
        "DC", 
        "DG",
        "DT", 
        "UNK",
        "A", 
        "C", 
        "G", 
        "U", 
        "DX",
        "X",
    ]

    expected = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    assert (is_protein_unknown(sequence) == np.array(expected)).all()