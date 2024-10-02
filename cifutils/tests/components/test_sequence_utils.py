import numpy as np

from cifutils.utils.sequence_utils import is_pyramidine, is_purine, is_unknown_nucleotide


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
    assert (is_purine(np.array(sequence)) == np.array(expected)).all()


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
    assert (is_purine(np.array(sequence)) == np.array(expected)).all()


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
    assert (is_unknown_nucleotide(np.array(sequence)) == np.array(expected)).all()
